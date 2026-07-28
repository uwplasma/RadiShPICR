# Christopher Woolford Jul 28, 2026
# This file contains my solver for the electric field
# in a spherically symmetric Z4C numerical relativity simulation.

# dE^r/dr = (-Gamma^r_{rr} E^r + rho )
# equation for the radial electric field in a spherically symmetric
# spacetime, where rho is the charge density and Gamma^r_{rr} is the
# Christoffel symbol for the radial coordinate.

# In this formulation, should be:
# dE^r/dr = (-1/(2 * g_rr) * dg_rr/dr * E^r + rho )


import jax
import jax.numpy as jnp
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative, sixth_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def dE_dr(metric: Z4C_Metric, E_r, rho):
    """
    Compute the radial derivative of the electric field in a spherically symmetric spacetime.

    Parameters:
    metric (Z4C_Metric): The Z4C metric object containing the metric components.
    E_r (array): The radial electric field.
    rho (array): The charge density.

    Returns:
    array: The radial derivative of the electric field.
    """
    grr = metric.conformal_grr
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    
    # Compute the radial derivative of the electric field
    dE_dr = (-1 / (2 * grr) * dgrrdr * E_r + rho)
    
    return dE_dr


def compute_E_r(metric: Z4C_Metric, rho):
    # E_r(r=0) = 0
    # RK2 integration of dE^r/dr = (-Gamma^r_{rr} E^r + rho )
    # with initial condition E^r(r=0) = 0

    dr = metric.dr
    r = metric.r

    def heun_step(E_r, i):
        k1 = dE_dr(metric, E_r, rho[i])
        k2 = dE_dr(metric, E_r + dr * k1, rho[i])
        return E_r + (dr / 2) * (k1 + k2)
    # Use Heun's method (RK2) to integrate the electric field equation

    vmapped_heun_step = jax.vmap(heun_step, in_axes=(0, 0), out_axes=0)
    # Initialize the electric field array with zeros and perform the integration

    E_r = jnp.zeros_like(r)
    E_r = vmapped_heun_step(E_r, jnp.arange(len(r)))
    # Return the computed radial electric field
    return E_r