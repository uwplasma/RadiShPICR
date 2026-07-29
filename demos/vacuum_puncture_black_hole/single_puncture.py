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

from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
from RadiShPICR.Z4C.time_evolve import rk4_step
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

def metric_fields_finite(metric):
    alpha = metric.alpha
    beta = metric.beta
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    chi = metric.chi
    Kh = metric.Kh
    Arr = metric.Arr
    At = metric.At
    theta = metric.theta
    Gamma = metric.Gamma

    bools = jnp.stack([
        jnp.isfinite(alpha),
        jnp.isfinite(beta),
        jnp.isfinite(conformal_grr),
        jnp.isfinite(conformal_gt),
        jnp.isfinite(chi),
        jnp.isfinite(Kh),
        jnp.isfinite(Arr),
        jnp.isfinite(At),
        jnp.isfinite(theta),
        jnp.isfinite(Gamma),
    ])
    return jnp.all(bools)
    # define a function to check if all metric fields are finite


jitted_rk4_step = jax.jit(rk4_step)
# jit the RK4 step function for performance
dt = CFL * float(dr)
# compute the time step based on the CFL condition and the radial grid spacing
Nt = int( FINAL_TIME / dt )
# compute the number of time steps needed to reach the final time
run_data = {
    "snapshots": [],
    "final_metric": None,
}

for t in tqdm(range(Nt)):
    metric = jitted_rk4_step(metric, initialize_vacuum_matter_terms(metric), dt)
    # perform a single RK4 time step to evolve the metric fields
    jax.block_until_ready(metric.alpha)
    # block until the metric fields are ready to ensure synchronization
    if not metric_fields_finite(metric):
        print(f"Non-finite fields encountered at step {t}, time {t*dt:.8e}")
        break

    if t % (Nt // SNAPSHOT_COUNT) == 0:
        snapshot = {
            "time": float(t * dt),
            "fields": {
                "alpha": jnp.asarray(metric.alpha),
                "beta": jnp.asarray(metric.beta),
                "conformal_grr": jnp.asarray(metric.conformal_grr),
                "conformal_gt": jnp.asarray(metric.conformal_gt),
                "chi": jnp.asarray(metric.chi),
                "Kh": jnp.asarray(metric.Kh),
                "Arr": jnp.asarray(metric.Arr),
                "At": jnp.asarray(metric.At),
                "theta": jnp.asarray(metric.theta),
                "Gamma": jnp.asarray(metric.Gamma),
            },
        }
        run_data["snapshots"].append(snapshot)
        # save a snapshot of the metric fields at regular intervals

run_data["final_metric"] = metric
# save the final metric after the evolution is complete

with open("run_data.pkl", "wb") as f:
    pickle.dump(run_data, f)
# save the run data to a pickle file for later analysis