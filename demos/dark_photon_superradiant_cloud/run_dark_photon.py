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
from RadiShPICR.initial_data import (
    cold_plasma_shell,
    neutral_plasma_shell,
    schwarzschild_trumpet_metric,
)
from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.apparent_horizon import find_apparent_horizon
from RadiShPICR.Z4C.time_evolve import rk4_step
from RadiShPICR.Z4C.utils import generate_r_grid


def _build_plasma(metric):
    """Return the kinetically mixed shell and its normalization diagnostics.

    The shell is specified by its plasma frequency, so no geometrized electron
    charge or mass is ever formed. Its rest-mass density follows as
    EPSILON_0 * omega_p**2 / alpha_s**2 and is reported for the mass budget.
    """

    if p.NEUTRALIZE:
        return neutral_plasma_shell(
            metric, p.PLASMA_FREQUENCY, r_min=p.PLASMA_R_MIN,
            r_max=p.PLASMA_R_MAX, particles_per_cell=p.PARTICLES_PER_CELL,
            charge_to_mass=p.PLASMA_CHARGE / p.PLASMA_MASS,
            epsilon_0=p.EPSILON_0,
        )
    return cold_plasma_shell(
        metric, "electrons", p.PLASMA_FREQUENCY, charge=p.PLASMA_CHARGE,
        mass=p.PLASMA_MASS, r_min=p.PLASMA_R_MIN, r_max=p.PLASMA_R_MAX,
        particles_per_cell=p.PARTICLES_PER_CELL, epsilon_0=p.EPSILON_0,
    )


def _plasma_metadata(plasma, prefix="plasma"):
    """Flatten the helper diagnostics into scalar series attributes.

    set_metadata stores bool/int/float/str only, so the nested electron and
    ion entries are flattened rather than passed through.
    """

    if plasma is None:
        return {}
    flattened = {}
    for key, value in plasma.items():
        if isinstance(value, dict):
            flattened.update(_plasma_metadata(value, f"{prefix}_{key}"))
        else:
            flattened[f"{prefix}_{key}"] = value
    return flattened


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
    zeros = jnp.zeros_like(r)
    if p.PLASMA_FREQUENCY is None:
        particles = particle_species(
            "vacuum", 0.0, 1.0, empty, empty, empty, empty, empty, "quadratic")
        # The vacuum cloud is normalized in G = c = epsilon_0 = 1; EPSILON_0 is
        # the rescaled eps0/lambda**2 and applies only once a plasma is present.
        sin_chi, plasma, epsilon_0 = 0.0, None, 1.0
    else:
        particles, plasma = _build_plasma(metric)
        sin_chi, epsilon_0 = p.SIN_CHI, p.EPSILON_0
        # Plasma oscillations are the fastest timescale once the shell is on.
        if p.PLASMA_FREQUENCY * dt > 0.1:
            raise ValueError(
                f"omega_p * dt = {p.PLASMA_FREQUENCY * dt:.3g} does not resolve "
                "the plasma oscillation; lower CFL or PLASMA_FREQUENCY."
            )
        # EPSILON_0 sets the plasma rest mass at fixed omega_p, and the ions
        # carry mass_ratio times the electron density. It is easy to ask for a
        # shell heavier than the hole without noticing, so say so plainly.
        rest_mass = float(jnp.sum(particles.get_mass()))
        print(f"Plasma rest mass {rest_mass:.3e} = {rest_mass / p.MASS:.3g} M "
              f"(omega_p={p.PLASMA_FREQUENCY:g}, epsilon_0={p.EPSILON_0:g}).")
        if p.BACKREACTION and rest_mass > 0.01 * p.MASS:
            raise ValueError(
                f"The shell weighs {rest_mass / p.MASS:.3g} M and would dominate "
                "the spacetime. Lower EPSILON_0 (rest mass scales with it at "
                "fixed omega_p), shrink the shell, or set BACKREACTION = False."
            )

    @partial(jax.jit, static_argnames=("count",))
    def advance(metric, fields, start_step, count):
        # Emit only the probe scalar per step. Full fields stay on the device
        # until the end of the snapshot interval.
        def step(state, step_number):
            metric, fields = state
            _, metric, _, _, fields = rk4_step(
                particles, metric, dt, E_r=zeros, EM_on=False, GR_on=p.BACKREACTION,
                dark_fields=fields, mu=p.MU, sin_chi=sin_chi,
                epsilon_0=epsilon_0, dark_ko=p.DARK_KO,
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
        sin_chi=sin_chi, epsilon_0=epsilon_0,
        plasma_frequency=p.PLASMA_FREQUENCY, **_plasma_metadata(plasma),
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
