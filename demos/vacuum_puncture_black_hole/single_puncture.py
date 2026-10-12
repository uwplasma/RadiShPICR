"""Z4C single-puncture black-hole vacuum evolution.

Time-symmetric Schwarzschild puncture data,

    psi = 1 + M / 2r,    chi = psi**-4,    alpha = psi**-2,

evolved as vacuum with Sommerfeld outer boundaries.  Snapshots go to one
openPMD series; `plot_final_bssn.py` reads them back.
"""

import math
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import jax
from tqdm import tqdm

from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.initial_data import puncture_metric
from RadiShPICR.Z4C.time_evolve import advance_vacuum_steps
from RadiShPICR.Z4C.utils import generate_r_grid

DEMO_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/vacuum_single_puncture"

MASS = 1.0
R_MIN = 0.0
R_MAX = 100.0
NUM_POINTS = 8000

CFL = 0.2
DURATION = 150.0  # Final evolution t/M.
SNAPSHOT_COUNT = 500

KAPPA = 1.0
ETA = 2.0
NU = 0.02


def run_simulation():
    jax.config.update("jax_enable_x64", True)

    r = generate_r_grid(R_MIN, R_MAX, NUM_POINTS)
    metric = puncture_metric(r, MASS, kappa=KAPPA, eta=ETA, nu=NU)

    # Compile several timesteps together and synchronize only at output
    # boundaries.
    advance = jax.jit(advance_vacuum_steps, static_argnames=("num_steps",))
    target_time = DURATION * MASS
    num_steps = max(1, math.ceil(target_time / (CFL * float(metric.dr))))
    dt = target_time / num_steps
    snapshot_interval = max(1, num_steps // SNAPSHOT_COUNT)

    metadata = dict(
        mass=MASS, duration_over_mass=DURATION, cfl=CFL, dt=dt,
        num_cells=NUM_POINTS, dr=float(metric.dr),
        kappa=KAPPA, eta=ETA, nu=NU,
        initial_data="schwarzschild_puncture",
        metric_boundary="sommerfeld", time_coordinate="evolution",
        snapshot_interval_steps=snapshot_interval,
    )

    completed_steps = 0
    nonfinite_step = -1
    with BSSNFieldSeries(OUTPUT_DIRECTORY, metadata=metadata) as fields, \
            tqdm(total=num_steps) as progress:
        fields.write(metric, 0, time=0.0, dt=dt)

        while completed_steps < num_steps and nonfinite_step < 0:
            chunk = min(snapshot_interval, num_steps - completed_steps)
            chunk_start = completed_steps
            metric, first_nonfinite = advance(metric, dt, num_steps=chunk)
            completed_steps += chunk
            progress.update(chunk)

            first_nonfinite = int(jax.device_get(first_nonfinite))
            if first_nonfinite >= 0:
                nonfinite_step = chunk_start + first_nonfinite + 1
                print(
                    "Non-finite fields encountered at step "
                    f"{nonfinite_step}, time {nonfinite_step * dt:.8e}"
                )
                break

            fields.write(metric, completed_steps,
                         time=completed_steps * dt, dt=dt)

        fields.set_metadata(
            completed=nonfinite_step < 0,
            final_step=completed_steps,
            final_time=completed_steps * dt,
            final_time_over_mass=completed_steps * dt / MASS,
            first_nonfinite_step=nonfinite_step,
        )
        print(f"Wrote {fields.path}")


if __name__ == "__main__":
    run_simulation()
