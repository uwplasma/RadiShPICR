import math

import jax.numpy as jnp
import numpy as np
from jax import jit

from RadiShPICR.Z4C.energy_momentum_tensor import compute_radial_matter_terms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def unit_determinant_conformal_metric(conformal_grr, conformal_gt):
    det_gamma_tilde = ( conformal_grr * conformal_gt**2 )**(1/3)

    conformal_grr = conformal_grr / det_gamma_tilde
    conformal_gt = conformal_gt / det_gamma_tilde

    return conformal_grr, conformal_gt


def trace_free_curvature(Arr, At, metric: Z4C_Metric):
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    chi = metric.chi
    # unpack the metric components

    trace = Arr / conformal_grr + 2.0 * At / conformal_gt
    # compute the trace of the curvature

    Arr_trace_free = Arr - (1.0 / 3.0) * conformal_grr * trace
    At_trace_free = At - (1.0 / 3.0) * conformal_gt * trace
    # compute the trace-free parts

    return Arr_trace_free, At_trace_free

def generate_r_grid(r_min, r_max, Nr):
    dr = (r_max - r_min) / Nr 
    # get the radial grid spacing
    r_grid = r_min + (jnp.arange(Nr) + 0.5) * dr
    # shift the grid points to be at the center of each cell
    return r_grid


FREE_FALL_FRACTION = 0.05


def freefall_collapse_time_step(particles, metric, fraction=FREE_FALL_FRACTION,
                                inner_open=True):
    """Return a timestep limit from the local free-fall time of the matter.

    A collapsing dust sphere outruns the CFL limit as it compactifies, so the
    runners take the smaller of this and the CFL step.  Returns ``inf`` where
    there is no matter, leaving the CFL limit in charge.
    """

    matter_terms = compute_radial_matter_terms(particles, metric,
                                               inner_open=inner_open)
    rho_max = float(np.max(np.asarray(matter_terms.rho)))
    if rho_max <= 0.0:
        return math.inf

    return fraction * math.sqrt(3.0 * math.pi / (32.0 * rho_max))
