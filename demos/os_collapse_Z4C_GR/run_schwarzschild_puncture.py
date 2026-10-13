"""Vacuum Schwarzschild puncture at the OS run's initial gravitational mass.

A comparison baseline for the Oppenheimer-Snyder collapse: the same mass, the
same grid, no matter.  Reads the mass and grid from the OS run's openPMD
series and writes its own.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import jax
from tqdm import tqdm

from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.diagnostics.read_bssn import final_snapshot, read_metadata
from RadiShPICR.initial_data import puncture_metric
from RadiShPICR.Z4C.time_evolve import advance_vacuum_steps


def read_source_run(input_directory):
    """Take the gravitational mass, grid and shift choice from the OS run."""

    metadata = read_metadata(input_directory)
    missing = {"total_mass", "zero_shift"} - set(metadata)
    if missing:
        raise ValueError(
            f"Missing OS run attributes: {', '.join(sorted(missing))}"
        )

    mass = float(metadata["total_mass"])
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Initial total_mass must be finite and positive.")

    # The puncture and origin-parity stencils require the original half-cell grid.
    r = np.asarray(final_snapshot(input_directory, with_particles=False).metric.r)
    dr = r[1] - r[0]
    if (not np.all(np.isfinite(r)) or dr <= 0.0
            or not np.allclose(np.diff(r), dr) or not np.isclose(r[0], 0.5 * dr)):
        raise ValueError("The source grid must be uniform and half-cell centered.")

    return mass, r, int(metadata["zero_shift"])


def evolve_puncture(metric, mass, fields, duration, cfl, snapshot_count,
                    zero_shift):
    """Evolve with Sommerfeld boundaries, saving only finite checkpoints."""

    target_time = duration * mass
    num_steps = max(1, math.ceil(target_time / (cfl * float(metric.dr))))
    dt = target_time / num_steps
    interval = max(1, math.ceil(num_steps / snapshot_count))
    advance = jax.jit(advance_vacuum_steps, static_argnames=("num_steps",))

    completed_steps = 0
    failed_step = None
    fields.write(metric, 0, time=0.0, dt=dt)

    with tqdm(total=num_steps, desc="evolving initial-mass vacuum puncture") as progress:
        while completed_steps < num_steps:
            chunk_steps = min(interval, num_steps - completed_steps)
            trial_metric, first_nonfinite = advance(
                metric, dt, num_steps=chunk_steps, zero_shift=zero_shift,
            )
            first_nonfinite = int(jax.device_get(first_nonfinite))
            if first_nonfinite >= 0:
                failed_step = completed_steps + first_nonfinite + 1
                break

            metric = trial_metric
            completed_steps += chunk_steps
            fields.write(metric, completed_steps,
                         time=completed_steps * dt, dt=dt)
            progress.update(chunk_steps)

    return {
        "completed": completed_steps == num_steps,
        "final_step": completed_steps,
        "final_time": completed_steps * dt,
        "final_time_over_mass": completed_steps * dt / mass,
        "target_time": target_time, "duration_over_mass": duration,
        "time_coordinate": "evolution", "dt": dt, "cfl": cfl,
        "snapshot_interval_steps": interval,
        "first_nonfinite_step": -1 if failed_step is None else failed_step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path,
                        default=DEMO_DIRECTORY / "outputs/z4c_oppenheimer_snyder_consistent_initial_data")
    parser.add_argument("--output-directory", type=Path,
                        default=DEMO_DIRECTORY / "outputs/schwarzschild_puncture_initial_mass")
    parser.add_argument("--duration", type=float, default=150.0,
                        help="Final evolution time in initial stellar masses (default: 150).")
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--snapshot-count", type=int, default=500)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--nu", type=float, default=0.02)
    parser.add_argument("--zero-shift", type=int, choices=(0, 1),
                        help="Default: use the shift setting of the OS run.")
    args = parser.parse_args()
    if args.duration <= 0 or args.cfl <= 0 or args.snapshot_count < 1:
        parser.error("duration, cfl, and snapshot-count must be positive")
    if args.output_directory.exists():
        parser.error(f"Refusing to overwrite existing run: {args.output_directory}")

    jax.config.update("jax_enable_x64", True)
    mass, r, saved_zero_shift = read_source_run(args.input_directory)
    zero_shift = saved_zero_shift if args.zero_shift is None else args.zero_shift
    print(f"Initial stellar gravitational mass: {mass:.12g}; "
          f"source: {args.input_directory.resolve()}", flush=True)

    metric = puncture_metric(r, mass, kappa=args.kappa, eta=args.eta, nu=args.nu)
    provenance = {
        "mass": mass, "total_mass": mass,
        "mass_source": "initial_gravitational_mass",
        "source_run": str(args.input_directory.resolve()),
        "kappa": args.kappa, "eta": args.eta, "nu": args.nu,
        "zero_shift": zero_shift, "metric_boundary": "sommerfeld",
        "initial_data": "schwarzschild_puncture",
        "num_cells": int(r.size), "dr": float(metric.dr),
    }

    with BSSNFieldSeries(args.output_directory, metadata=provenance) as fields:
        result = evolve_puncture(metric, mass, fields, args.duration, args.cfl,
                                 args.snapshot_count, zero_shift)
        fields.set_metadata(**result)

    print(json.dumps({**provenance, **result}, indent=2))
    if not result["completed"]:
        raise SystemExit(
            "Vacuum evolution became nonfinite; the last accepted checkpoint is preserved."
        )


if __name__ == "__main__":
    main()
