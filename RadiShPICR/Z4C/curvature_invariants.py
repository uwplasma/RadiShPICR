"""Spherical geometric mass and spacetime-curvature diagnostics."""

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
    d2areal_radius_dl2 = first_derivative(
        dareal_radius_dl,
        metric.dr,
        parity=1,
    ) / jnp.sqrt(physical_grr)

    return (
        physical_grr,
        areal_radius,
        dareal_radius_dl,
        d2areal_radius_dl2,
    )


def misner_sharp_mass(metric: Z4C_Metric):
    """Return the Misner-Sharp mass on every radial grid sphere.

    The Z4C trace variable is ``Kh = K - 2 theta``.  The normal derivative
    of areal radius therefore follows from the physical angular extrinsic
    curvature reconstructed with ``K = Kh + 2 theta``.
    """

    _, areal_radius, dareal_radius_dl, _ = _areal_geometry(metric)

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


def kretschmann_scalar(metric: Z4C_Metric, matter_terms):
    """Return ``R_abcd R^abcd`` from spherical 3+1 data and matter sources.

    The magnetic Weyl tensor vanishes in spherical symmetry.  The electric
    Weyl tensor is built from the trace-free part of the Gauss projection,
    while the Ricci invariants follow from Einstein's equation in ``G=c=1``
    units.  ``matter_terms`` must contain the total particle and field stress.
    """

    (
        physical_grr,
        areal_radius,
        dareal_radius_dl,
        d2areal_radius_dl2,
    ) = _areal_geometry(metric)

    spatial_ricci_radial = -2.0 * d2areal_radius_dl2 / areal_radius
    spatial_ricci_angular = (
        -d2areal_radius_dl2 / areal_radius
        + (1.0 - dareal_radius_dl**2) / areal_radius**2
    )

    trace_extrinsic_curvature = metric.Kh + 2.0 * metric.theta
    radial_extrinsic_curvature = (
        metric.Arr / metric.conformal_grr
        + trace_extrinsic_curvature / 3.0
    )
    angular_extrinsic_curvature = (
        metric.At / metric.conformal_gt
        + trace_extrinsic_curvature / 3.0
    )

    radial_stress = matter_terms.Srr / physical_grr
    angular_stress = (
        matter_terms.Stt * metric.chi / metric.conformal_gt
    )
    stress_trace = radial_stress + 2.0 * angular_stress

    radial_gauss_projection = (
        spatial_ricci_radial
        + trace_extrinsic_curvature * radial_extrinsic_curvature
        - radial_extrinsic_curvature**2
        - 4.0 * jnp.pi * radial_stress
    )
    angular_gauss_projection = (
        spatial_ricci_angular
        + trace_extrinsic_curvature * angular_extrinsic_curvature
        - angular_extrinsic_curvature**2
        - 4.0 * jnp.pi * angular_stress
    )
    electric_weyl_difference = (
        radial_gauss_projection - angular_gauss_projection
    )
    weyl_squared = 16.0 * electric_weyl_difference**2 / 3.0

    momentum_squared = physical_grr * matter_terms.Sr**2
    stress_squared = radial_stress**2 + 2.0 * angular_stress**2
    ricci_tensor_squared = (8.0 * jnp.pi) ** 2 * (
        matter_terms.rho**2
        - 2.0 * momentum_squared
        + stress_squared
    )
    ricci_scalar = 8.0 * jnp.pi * (matter_terms.rho - stress_trace)

    return (
        weyl_squared
        + 2.0 * ricci_tensor_squared
        - ricci_scalar**2 / 3.0
    )
