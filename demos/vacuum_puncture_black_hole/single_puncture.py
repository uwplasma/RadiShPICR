# # Z4C single-puncture black-hole vacuum evolution
# $$
# \psi = 1 + \frac{M}{2r},\qquad
# \chi = \psi^{-4},\qquad
# \alpha = \psi^{-2}.
# $$


from __future__ import annotations
import os
import sys
from pathlib import Path
# import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import pickle

jax.config.update("jax_enable_x64", True)
os.environ["JAX_ENABLE_X64"] = "True"

cwd = Path.cwd().resolve()
package_root = next(
    candidate
    for root in (cwd, *cwd.parents)
    for candidate in (root / "code" / "RadiShPICR", root)
    if (candidate / "RadiShPICR" / "Z4C").is_dir()
)
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from RadiShPICR.Z4C.time_evolve import advance_vacuum_steps
from RadiShPICR.Z4C.utils import generate_r_grid
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

################# PARAMETERS #################
MASS = 1.0
R_MIN = 0.0
R_MAX = 100.0
NUM_POINTS = 8000

CFL = 0.2
FINAL_TIME = 100.0
SNAPSHOT_COUNT = 500
PLOT_R_MAX = 100.0

KAPPA = 0.02
ETA = 2.0
NU = 0.02
###############################################

r = generate_r_grid(R_MIN, R_MAX, NUM_POINTS)
dr = r[1] - r[0]
zeros = jnp.zeros_like(r)
ones = jnp.ones_like(r)

psi = 1.0 + MASS / (2.0 * r)
chi = psi**(-4)
alpha = psi**(-2)
# build the initial metric object with the analytic single-puncture solution

metric = Z4C_Metric(
    alpha=alpha,
    beta=zeros,
    conformal_grr=ones,
    conformal_gt=ones,
    chi=chi,
    Kh=zeros,
    Arr=zeros,
    At=zeros,
    theta=zeros,
    Gamma=zeros,
    kappa=jnp.asarray(KAPPA, dtype=r.dtype),
    eta=jnp.asarray(ETA, dtype=r.dtype),
    nu=jnp.asarray(NU, dtype=r.dtype),
    r=r,
    dr=dr,
)

jitted_advance_vacuum_steps = jax.jit(
    advance_vacuum_steps,
    static_argnames=("num_steps",),
)
# Compile several timesteps together and synchronize only at output boundaries.
dt = CFL * float(dr)
# compute the time step based on the CFL condition and the radial grid spacing
Nt = int( FINAL_TIME / dt )
# compute the number of time steps needed to reach the final time
run_data = {
    "snapshots": [],
    "final_metric": None,
}

snapshot_interval = max(1, Nt // SNAPSHOT_COUNT)
completed_steps = 0

with tqdm(total=Nt) as progress_bar:
    while completed_steps < Nt:
        snapshot_step = completed_steps
        metric, first_nonfinite_step = jitted_advance_vacuum_steps(
            metric,
            dt,
            num_steps=1,
        )
        host_metric = jax.device_get(metric)
        first_nonfinite_step = jax.device_get(first_nonfinite_step)
        completed_steps += 1
        progress_bar.update(1)

        if first_nonfinite_step >= 0:
            print(
                "Non-finite fields encountered at "
                f"step {snapshot_step}, time {snapshot_step*dt:.8e}"
            )
            break

        snapshot = {
            "time": float(snapshot_step * dt),
            "fields": {
                "alpha": host_metric.alpha,
                "beta": host_metric.beta,
                "conformal_grr": host_metric.conformal_grr,
                "conformal_gt": host_metric.conformal_gt,
                "chi": host_metric.chi,
                "Kh": host_metric.Kh,
                "Arr": host_metric.Arr,
                "At": host_metric.At,
                "theta": host_metric.theta,
                "Gamma": host_metric.Gamma,
            },
        }
        run_data["snapshots"].append(snapshot)

        remaining_chunk_steps = min(
            snapshot_interval - 1,
            Nt - completed_steps,
        )
        if remaining_chunk_steps == 0:
            continue

        metric, first_nonfinite_step = jitted_advance_vacuum_steps(
            metric,
            dt,
            num_steps=remaining_chunk_steps,
        )
        first_nonfinite_step = jax.device_get(first_nonfinite_step)
        completed_steps += remaining_chunk_steps
        progress_bar.update(remaining_chunk_steps)

        if first_nonfinite_step >= 0:
            nonfinite_step = (
                snapshot_step + 1 + int(first_nonfinite_step)
            )
            print(
                "Non-finite fields encountered at "
                f"step {nonfinite_step}, time {nonfinite_step*dt:.8e}"
            )
            break

run_data["final_metric"] = metric
# save the final metric after the evolution is complete

with open("run_data.pkl", "wb") as f:
    pickle.dump(run_data, f)
# save the run data to a pickle file for later analysis
