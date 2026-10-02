"""Evolve a vacuum puncture with the saved initial OS gravitational mass.

Defaults read outputs/z4c_oppenheimer_snyder beside this script and write
outputs/schwarzschild_puncture_initial_mass. The duration is evolution time
in units of that initial mass; no final-horizon measurement is used.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from RadiShPICR.Z4C.boundary_conditions import (
    METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    METRIC_BOUNDARY_SOMMERFELD,
)
from RadiShPICR.Z4C.time_evolve import advance_vacuum_steps
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


SNAPSHOT_NAMES = {
    "alpha": "alpha", "beta": "beta", "chi": "chi",
    "conformal_grr": "conformal_grr", "conformal_gt": "conformal_gT",
    "Kh": "Kh", "Arr": "Arr", "At": "AT", "theta": "theta", "Gamma": "Gamma",
}
BOUNDARIES = {
    "sommerfeld": METRIC_BOUNDARY_SOMMERFELD,
    "constraint_preserving": METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
}


def read_initial_data(input_directory):
    """Read gravitational mass, evolution grid, and the saved shift choice."""
    path = Path(input_directory) / "initial_data_metadata.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing OS initial metadata: {path}")
    with np.load(path) as data:
        missing = {"total_mass", "z4c_r", "zero_shift"} - set(data.files)
        if missing:
            raise ValueError(f"Missing OS initial metadata fields: {', '.join(sorted(missing))}")
        mass = float(data["total_mass"])
        r = np.asarray(data["z4c_r"], dtype=float)
        zero_shift = int(data["zero_shift"])

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Initial total_mass must be finite and positive.")
    # The puncture and origin-parity stencils require the original half-cell grid.
    dr = r[1] - r[0]
    if (not np.all(np.isfinite(r)) or dr <= 0.0
            or not np.allclose(np.diff(r), dr) or not np.isclose(r[0], 0.5 * dr)):
        raise ValueError("z4c_r must be a uniform radial grid starting half a cell from the origin.")
    return mass, r, zero_shift


def initialize_puncture(r, mass, kappa=1.0, eta=2.0, nu=0.02):
    r = jnp.asarray(r)
    zeros, ones = jnp.zeros_like(r), jnp.ones_like(r)
    psi = 1.0 + mass / (2.0 * r)
    return Z4C_Metric(
        alpha=psi**-2, beta=zeros, chi=psi**-4,
        conformal_grr=ones, conformal_gt=ones,
        Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        r=r, dr=r[1] - r[0], kappa=jnp.asarray(kappa),
        eta=jnp.asarray(eta), nu=jnp.asarray(nu),
    )


def save_snapshot(metric, directory, step, time, mass):
    fields = {key: np.asarray(getattr(metric, name))
              for name, key in SNAPSHOT_NAMES.items()}
    np.savez_compressed(
        directory / f"metric_step_{step:06d}.npz", **fields,
        r=np.asarray(metric.r),
        areal_radius=np.asarray(metric.r * jnp.sqrt(metric.conformal_gt / metric.chi)),
        step=step, time=time, time_over_mass=time / mass,
        mass=mass, saved_coordinates="evolution",
    )


def evolve_puncture(metric, mass, output_directory, duration=150.0, cfl=0.2,
                    snapshot_count=500, zero_shift=0,
                    metric_boundary=METRIC_BOUNDARY_SOMMERFELD):
    """Save finite chunks; a failed chunk never replaces an accepted checkpoint."""
    metric_directory = Path(output_directory) / "metric"
    metric_directory.mkdir()
    target_time = duration * mass
    num_steps = max(1, math.ceil(target_time / (cfl * float(metric.dr))))
    dt = target_time / num_steps
    interval = max(1, math.ceil(num_steps / snapshot_count))
    advance = jax.jit(advance_vacuum_steps, static_argnames=("num_steps",))
    completed_steps = 0
    failed_step = None
    save_snapshot(metric, metric_directory, 0, 0.0, mass)

    with tqdm(total=num_steps, desc="evolving initial-mass vacuum puncture") as progress:
        while completed_steps < num_steps:
            chunk_steps = min(interval, num_steps - completed_steps)
            trial_metric, first_nonfinite = advance(
                metric, dt, num_steps=chunk_steps,
                zero_shift=zero_shift, metric_boundary=metric_boundary,
            )
            first_nonfinite = int(jax.device_get(first_nonfinite))
            if first_nonfinite >= 0:
                failed_step = completed_steps + first_nonfinite + 1
                break

            metric = trial_metric
            completed_steps += chunk_steps
            save_snapshot(metric, metric_directory, completed_steps,
                          completed_steps * dt, mass)
            progress.update(chunk_steps)

    return {
        "completed": completed_steps == num_steps,
        "final_step": completed_steps,
        "final_time": completed_steps * dt,
        "final_time_over_mass": completed_steps * dt / mass,
        "target_time": target_time, "duration_over_mass": duration,
        "time_coordinate": "evolution", "dt": dt, "cfl": cfl,
        "snapshot_interval_steps": interval,
        "first_nonfinite_step": failed_step,
        "first_nonfinite_time": None if failed_step is None else failed_step * dt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path,
                        default=DEMO_DIRECTORY / "outputs/z4c_oppenheimer_snyder")
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
                        help="Default: use the shift setting in the OS initial metadata.")
    parser.add_argument("--metric-boundary", choices=tuple(BOUNDARIES), default="sommerfeld")
    args = parser.parse_args()
    if args.duration <= 0 or args.cfl <= 0 or args.snapshot_count < 1:
        parser.error("duration, cfl, and snapshot-count must be positive")
    if args.output_directory.exists():
        parser.error(f"Refusing to overwrite existing run: {args.output_directory}")

    jax.config.update("jax_enable_x64", True)
    mass, r, saved_zero_shift = read_initial_data(args.input_directory)
    zero_shift = saved_zero_shift if args.zero_shift is None else args.zero_shift
    source = (args.input_directory / "initial_data_metadata.npz").resolve()
    print(f"Initial stellar gravitational mass: {mass:.12g}; source: {source}", flush=True)
    metric = initialize_puncture(r, mass, args.kappa, args.eta, args.nu)
    provenance = {
        "mass": mass, "mass_source": "initial_gravitational_mass",
        "source_initial_metadata": str(source), "mass_metadata_key": "total_mass",
        "kappa": args.kappa, "eta": args.eta, "nu": args.nu,
        "zero_shift": zero_shift, "metric_boundary": args.metric_boundary,
        "damping_parameters_source": "CLI values; defaults match the current OS demo, not recorded in old OS output",
        "num_cells": int(r.size), "dr": float(metric.dr),
    }
    args.output_directory.mkdir(parents=True)
    summary_path = args.output_directory / "run_summary.json"
    summary_path.write_text(json.dumps({**provenance, "completed": False}, indent=2) + "\n")
    result = evolve_puncture(
        metric, mass, args.output_directory, args.duration, args.cfl,
        args.snapshot_count, zero_shift, BOUNDARIES[args.metric_boundary],
    )
    summary_path.write_text(json.dumps({**provenance, **result}, indent=2) + "\n")
    print(json.dumps({**provenance, **result}, indent=2))
    if not result["completed"]:
        raise SystemExit("Vacuum evolution became nonfinite; the last accepted checkpoint is preserved.")


if __name__ == "__main__":
    main()
