"""Evolve the Schwarzschild Proca cloud. Edit simulation_parameters.py to configure the run."""

from functools import partial
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.dark_photon_superradiant_cloud import simulation_parameters as p
from demos.dark_photon_superradiant_cloud.initial_cloud import initial_cloud
from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.initial_data import schwarzschild_trumpet_metric
from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.apparent_horizon import find_apparent_horizon
from RadiShPICR.Z4C.time_evolve import rk4_step
from RadiShPICR.Z4C.utils import generate_r_grid


def run_simulation():
    jax.config.update("jax_enable_x64", True)
    if p.SAMPLE_EVERY < 1 or p.SNAPSHOT_EVERY < 1:
        raise ValueError("SAMPLE_EVERY and SNAPSHOT_EVERY must be positive step counts.")
    output = p.OUTPUT_DIRECTORY
    if (output / "bssn_fields.h5").exists() or (output / "probe.csv").exists():
        raise FileExistsError(f"Cloud data already exists in {output}. Select a new OUTPUT_DIRECTORY.")
    r = generate_r_grid(0.0, p.R_MAX, p.NUM_CELLS)
    metric = schwarzschild_trumpet_metric(r, p.MASS, kappa=p.KAPPA, eta=p.ETA, nu=p.NU)
    dark_fields = initial_cloud(metric, p.BOHR, p.MU, p.CLOUD_ENERGY)
    horizon, _ = find_apparent_horizon(metric)
    if not np.isfinite(float(horizon)):
        raise ValueError("The radial grid must resolve the trumpet horizon to place the exterior probe.")
    R = r * jnp.sqrt(metric.conformal_gt / metric.chi)
    probe = int(jnp.argmin(jnp.where(r > horizon, jnp.abs(R - 2.0 * p.BOHR), jnp.inf)))

    dt = p.CFL * float(metric.dr)
    num_steps = round(p.EVOLVE_TIME / dt)
    empty = jnp.empty(0)
    vacuum = particle_species("vacuum", 0.0, 1.0, empty, empty, empty, empty, empty, "quadratic")
    zeros = jnp.zeros_like(r)

    @partial(jax.jit, static_argnames=("count",))
    def advance(metric, fields, start_step, count):
        # Emit only the probe scalar per step. Full fields stay on the device
        # until the end of the snapshot interval.
        def step(state, step_number):
            metric, fields = state
            _, metric, _, _, fields = rk4_step(
                vacuum, metric, dt, E_r=zeros, EM_on=False, GR_on=p.BACKREACTION,
                dark_fields=fields, mu=p.MU, sin_chi=0.0, dark_ko=p.DARK_KO,
                time=step_number * dt,
            )
            return (metric, fields), fields[1][probe]

        return jax.lax.scan(step, (metric, fields), start_step + jnp.arange(count))

    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        mass=p.MASS, mu=p.MU, bohr=p.BOHR, cloud_energy=p.CLOUD_ENERGY,
        backreaction=p.BACKREACTION, kappa=p.KAPPA, eta=p.ETA, nu=p.NU, dark_ko=p.DARK_KO,
        r_max=p.R_MAX, num_cells=p.NUM_CELLS, cfl=p.CFL, evolve_time=p.EVOLVE_TIME,
        sample_every=p.SAMPLE_EVERY, snapshot_every=p.SNAPSHOT_EVERY,
        probe_index=probe, probe_r=float(r[probe]), probe_initial_R=float(R[probe]),
        initial_horizon_r=float(horizon), completed=False,
    )
    with (output / "probe.csv").open("x") as probe_file, BSSNFieldSeries(output, metadata=metadata) as series:
        probe_file.write("time,A_prime_r\n")
        np.savetxt(probe_file, [[0.0, float(dark_fields[1][probe])]], delimiter=",")
        series.write(metric, 0, time=0.0, dt=dt, dark_fields=dark_fields)
        for start in tqdm(range(0, num_steps, p.SNAPSHOT_EVERY), desc="Cloud snapshots"):
            end = min(start + p.SNAPSHOT_EVERY, num_steps)
            (metric, dark_fields), probe_values = advance(metric, dark_fields, start, end - start)
            steps = np.arange(start + 1, end + 1)
            sampled = steps % p.SAMPLE_EVERY == 0
            samples = np.column_stack((steps[sampled] * dt, np.asarray(probe_values)[sampled]))
            np.savetxt(probe_file, samples, delimiter=",")
            probe_file.flush()
            series.write(metric, end, time=end * dt, dt=dt, dark_fields=dark_fields)
        series.set_metadata(completed=True, final_step=num_steps, final_time=num_steps * dt)
    print(f"Saved {num_steps} steps to {output} (final t/M = {num_steps * dt / p.MASS:g}).")


if __name__ == "__main__":
    run_simulation()
