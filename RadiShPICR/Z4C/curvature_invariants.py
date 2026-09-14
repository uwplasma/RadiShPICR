"""Spherical geometric mass diagnostics."""

import jax.numpy as jnp

from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _areal_geometry(metric: Z4C_Metric):
    physical_grr = metric.conformal_grr / metric.chi
    areal_radius = metric.r * jnp.sqrt(metric.conformal_gt / metric.chi)

    dareal_radius_dr = first_derivative(
        areal_radius,
        metric.dr,
        parity=-1,
    )
    dareal_radius_dl = dareal_radius_dr / jnp.sqrt(physical_grr)

    return areal_radius, dareal_radius_dl


def misner_sharp_mass(metric: Z4C_Metric):
    """Return the Misner-Sharp mass on every radial grid sphere.

    The Z4C trace variable is ``Kh = K - 2 theta``.  The normal derivative
    of areal radius therefore follows from the physical angular extrinsic
    curvature reconstructed with ``K = Kh + 2 theta``.
    """

    areal_radius, dareal_radius_dl = _areal_geometry(metric)

    trace_extrinsic_curvature = metric.Kh + 2.0 * metric.theta
    angular_extrinsic_curvature = (
        metric.At / metric.conformal_gt
        + trace_extrinsic_curvature / 3.0
    )
    normal_areal_radius_derivative = (
        -areal_radius * angular_extrinsic_curvature
    )

    return 0.5 * areal_radius * (
        1.0
        - dareal_radius_dl**2
        + normal_areal_radius_derivative**2
    )
