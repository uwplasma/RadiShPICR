"""Vacuum Reissner-Nordstrom puncture at the charged star's mass and charge.

A comparison baseline for the charged collapse: the same M and Q, no matter.
Initial data follow arXiv:1311.6483v2 equations (2.8)-(2.13).

The puncture carries constant electric flux rather than an evolved field, so
``E_r`` is reconstructed from the current metric at each snapshot; the series
records that as the attribute ``electric_field_update``.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import jax
import jax.numpy as jnp
from tqdm import tqdm

from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.diagnostics.read_bssn import final_snapshot, read_metadata
from RadiShPICR.initial_data import puncture_electric_field, puncture_metric
from RadiShPICR.Z4C.time_evolve import electrovac_rk4_step

INPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/z4c_charged_star_evolution_300M"
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/reissner_nordstrom_puncture_evolution_300M"

DURATION = 300.0  # Final evolution t/M.
CFL = 0.2
SNAPSHOT_COUNT = 500
PUNCTURE_CHI_FLOOR = 1e-12

# None inherits the value the charged-star run recorded.
ZERO_SHIFT = None
KAPPA = None
ETA = None
NU = None


def read_source_run(input_directory):
    """Take M, Q and the evolution parameters from the charged-star run."""

    parameters = read_metadata(input_directory)
    missing = {"total_mass", "initial_charge", "epsilon_0", "kappa", "eta", "nu",
               "zero_shift"} - set(parameters)
    if missing:
        raise ValueError(
            f"Missing charged-star run attributes: {', '.join(sorted(missing))}"
        )

    mass = float(parameters["total_mass"])
    charge = float(parameters["initial_charge"])
    epsilon_0 = float(parameters["epsilon_0"])
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("total_mass must be finite and positive.")
    if not np.isfinite(epsilon_0) or epsilon_0 <= 0.0:
        raise ValueError("epsilon_0 must be finite and positive.")
    if mass <= abs(charge) / math.sqrt(4.0 * math.pi * epsilon_0):
        raise ValueError("The RN puncture requires subextremal M > |Q|/sqrt(4 pi eps0).")

    # The puncture and origin-parity stencils require the original half-cell grid.
    r = np.asarray(final_snapshot(input_directory, with_particles=False).metric.r)
    dr = r[1] - r[0]
    if (not np.all(np.isfinite(r)) or dr <= 0.0
            or not np.allclose(np.diff(r), dr) or not np.isclose(r[0], 0.5 * dr)):
        raise ValueError("The source grid must be uniform and half-cell centered.")

    return r, parameters


def advance_puncture_steps(metric, charge, dt, num_steps, epsilon_0=1.0,
                           zero_shift=0, puncture_chi_floor=0.0):
    """Return the last acceptable state, accepted step count, and failure code."""

    def advance_one_step(carry, unused):
        def advance(carry):
            metric, accepted, failure = carry
            trial = electrovac_rk4_step(
                metric, charge, dt, epsilon_0=epsilon_0, zero_shift=zero_shift,
                puncture_chi_floor=puncture_chi_floor,
            )
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(field)) for field in trial]))
            positive = jnp.all((trial.chi > 0.0) & (trial.conformal_grr > 0.0)
                               & (trial.conformal_gt > 0.0))
            valid = finite & positive
            last_valid = jax.lax.cond(valid, lambda _: trial, lambda _: metric, operand=None)
            failure = jnp.where(valid, 0, jnp.where(finite, 2, 1)).astype(jnp.int32)
            return last_valid, accepted + valid.astype(jnp.int32), failure

        carry = jax.lax.cond(carry[2] == 0, advance, lambda state: state, carry)
        return carry, None

    state = (metric, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    state, _ = jax.lax.scan(advance_one_step, state, xs=None, length=num_steps)

    return state


def evolve_puncture(metric, mass, charge, epsilon_0, fields, duration, cfl,
                    snapshot_count, zero_shift, puncture_chi_floor):
    target_time = duration * mass
    num_steps = max(1, math.ceil(target_time / (cfl * float(metric.dr))))
    dt = target_time / num_steps
    interval = max(1, math.ceil(num_steps / snapshot_count))
    advance = jax.jit(advance_puncture_steps, static_argnames=("num_steps",))

    completed_steps, failure = 0, 0
    started = time.monotonic()
    with tqdm(total=num_steps, desc="evolving RN puncture") as progress:
        while True:
            evolution_time = (target_time if completed_steps == num_steps
                              else completed_steps * dt)
            fields.write(
                metric, completed_steps, time=evolution_time, dt=dt,
                E_r=puncture_electric_field(metric, charge, epsilon_0),
            )
            if failure or completed_steps == num_steps:
                break

            chunk_steps = min(interval, num_steps - completed_steps)
            metric, accepted, failure = advance(
                metric, charge, dt, num_steps=chunk_steps, epsilon_0=epsilon_0,
                zero_shift=zero_shift, puncture_chi_floor=puncture_chi_floor,
            )
            accepted, failure = int(accepted), int(failure)
            completed_steps += accepted
            progress.update(accepted)

    return {
        "completed": completed_steps == num_steps,
        "final_step": completed_steps,
        "final_time": evolution_time,
        "final_time_over_mass": evolution_time / mass,
        "target_time": target_time, "duration_over_mass": duration,
        "dt": dt, "cfl": cfl, "snapshot_interval_steps": interval,
        "stop_reason": (("nonfinite_state" if failure == 1 else "nonpositive_geometry")
                        if failure else "target_time"),
        "first_invalid_step": completed_steps + 1 if failure else -1,
        "wall_runtime_seconds": time.monotonic() - started,
    }


def main():
    if (not np.isfinite(DURATION) or not np.isfinite(CFL)
            or DURATION <= 0.0 or CFL <= 0.0 or SNAPSHOT_COUNT < 1):
        raise ValueError("DURATION, CFL, and SNAPSHOT_COUNT must be positive and finite.")
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {OUTPUT_DIRECTORY}")
    if not np.isfinite(PUNCTURE_CHI_FLOOR) or PUNCTURE_CHI_FLOOR < 0.0:
        raise ValueError("PUNCTURE_CHI_FLOOR must be finite and nonnegative.")

    jax.config.update("jax_enable_x64", True)
    r, parameters = read_source_run(INPUT_DIRECTORY)
    for name, value in (("zero_shift", ZERO_SHIFT), ("kappa", KAPPA),
                        ("eta", ETA), ("nu", NU)):
        if value is not None:
            parameters[name] = value

    mass, charge, epsilon_0 = (float(parameters[name]) for name in
                               ("total_mass", "initial_charge", "epsilon_0"))
    metric = puncture_metric(r, mass, charge, epsilon_0, parameters["kappa"],
                             parameters["eta"], parameters["nu"])
    print(f"RN reference: M={mass:.12g}, Q={charge:.12g}; "
          f"source: {INPUT_DIRECTORY.resolve()}", flush=True)

    provenance = dict(
        source_run=str(INPUT_DIRECTORY.resolve()),
        mass=mass, total_mass=mass, initial_charge=charge, epsilon_0=epsilon_0,
        geometric_charge=charge / math.sqrt(4.0 * math.pi * epsilon_0),
        mass_source="initial_gravitational_mass", charge_source="initial_charge",
        time_coordinate="evolution",
        num_cells=int(r.size), dr=float(metric.dr),
        zero_shift=int(parameters["zero_shift"]), kappa=float(parameters["kappa"]),
        eta=float(parameters["eta"]), nu=float(parameters["nu"]),
        metric_boundary="sommerfeld", initial_lapse="psi^-2",
        electric_field_update="constant_puncture_flux",
        initial_solver="analytic_single_puncture_u_1",
        initial_data_reference="arXiv:1311.6483v2 equations (2.8)-(2.13)",
        puncture_chi_floor=PUNCTURE_CHI_FLOOR,
        puncture_chi_floor_cells=3 if PUNCTURE_CHI_FLOOR > 0.0 else 0,
        puncture_chi_floor_placement="each_RK_stage_and_final_state",
    )

    with BSSNFieldSeries(OUTPUT_DIRECTORY, metadata=provenance) as fields:
        result = evolve_puncture(
            metric, mass, charge, epsilon_0, fields, DURATION, CFL,
            SNAPSHOT_COUNT, parameters["zero_shift"], PUNCTURE_CHI_FLOOR,
        )
        fields.set_metadata(**result)

    print(json.dumps({**provenance, **result}, indent=2))
    if not result["completed"]:
        raise SystemExit("RN evolution stopped; the last valid checkpoint is preserved.")


if __name__ == "__main__":
    main()
