"""Outer-boundary choices for the spherical Z4C evolution system."""

import jax.numpy as jnp

from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


METRIC_BOUNDARY_SOMMERFELD = 0
METRIC_BOUNDARY_CONSTRAINT_PRESERVING = 1

SHIFT_GAUGE_COEFFICIENT = 5.0 / 2.0
CHI_FLOOR_VALUE = 1.0e-12


def conformal_connection_constraint(metric: Z4C_Metric):
    """Return ``Gamma^r - (Gamma_d)^r`` in the regular spherical basis."""

    grr = metric.conformal_grr
    gt = metric.conformal_gt
    dgrr_dr = first_derivative(grr, metric.dr, parity=1)

    # The algebraic determinant constraint reduces the reference-metric
    # contracted conformal Christoffel symbol to this regular radial form.
    Gamma_d = (
        dgrr_dr / grr**2
        - 2.0 / (metric.r * grr)
        + 2.0 / (metric.r * gt)
    )

    return metric.Gamma - Gamma_d


def constraint_preserving_characteristic_speeds(metric: Z4C_Metric):
    """Return outward physical, lapse, and longitudinal-shift speeds."""

    physical_radial_metric = metric.conformal_grr / metric.chi
    normal_speed = 1.0 / jnp.sqrt(physical_radial_metric)

    physical_speed = -metric.beta + metric.alpha * normal_speed
    lapse_speed = (
        -metric.beta
        + jnp.sqrt(2.0 * metric.alpha) * normal_speed
    )
    shift_speed = (
        -metric.beta
        + jnp.sqrt(
            4.0
            * SHIFT_GAUGE_COEFFICIENT
            / (3.0 * metric.conformal_grr)
        )
    )

    return physical_speed, lapse_speed, shift_speed


def apply_constraint_preserving_boundary(
    metric: Z4C_Metric,
    bulk_derivatives: Z4C_Metric,
):
    """Replace the incoming scalar Z4C modes at the outer radial point.

    This is the spherical specialization of Eqs. (17), (20)--(22) of
    Hilditch et al., Phys. Rev. D 88, 084057 (2013).  These nonlinear
    conditions have the constraint-preserving principal part of the earlier
    spherical construction, but deliberately retain the later paper's gauge
    and lower-order terms rather than the flat-space Appendix-C completion.
    The outer boundary is assumed to be in vacuum.  Metric fields keep their
    bulk equations there; only ``Kh``, ``Gamma``, ``theta``, ``Arr``, and the
    trace-related ``At`` derivative are replaced.
    """

    alpha = metric.alpha
    beta = metric.beta
    chi = jnp.maximum(metric.chi, CHI_FLOOR_VALUE)
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Kh = metric.Kh
    Arr = metric.Arr
    At = metric.At
    theta = metric.theta
    r = metric.r

    dalpha_dr = first_derivative(alpha, metric.dr, parity=1)
    d2alpha_dr2 = second_derivative(alpha, metric.dr, parity=1)
    dbeta_dr = first_derivative(beta, metric.dr, parity=-1)
    d2beta_dr2 = second_derivative(beta, metric.dr, parity=-1)
    dgrr_dr = first_derivative(grr, metric.dr, parity=1)
    dgt_dr = first_derivative(gt, metric.dr, parity=1)
    dchi_dr = first_derivative(chi, metric.dr, parity=1)
    dKh_dr = first_derivative(Kh, metric.dr, parity=1)
    dArr_dr = first_derivative(Arr, metric.dr, parity=1)
    dtheta_dr = first_derivative(theta, metric.dr, parity=1)
    dGamma_dr = first_derivative(metric.Gamma, metric.dr, parity=-1)

    physical_grr = grr / chi
    physical_gt = gt / chi
    dphysical_grr_dr = first_derivative(
        physical_grr,
        metric.dr,
        parity=1,
    )
    dphysical_gt_dr = first_derivative(
        physical_gt,
        metric.dr,
        parity=1,
    )
    d2physical_gt_dr2 = second_derivative(
        physical_gt,
        metric.dr,
        parity=1,
    )
    normal_speed = 1.0 / jnp.sqrt(physical_grr)

    # The lapse condition uses mu_L = 2 / alpha.  Tangential Cartesian
    # derivatives of a radial scalar contribute twice its radial 1/r term.
    tangential_partial_laplacian_alpha = (
        2.0 * dalpha_dr / (physical_gt * r)
    )
    cp_Kh = (
        -jnp.sqrt(2.0 * alpha)
        * (normal_speed * dKh_dr + Kh / r)
        - tangential_partial_laplacian_alpha
        + beta * dKh_dr
    )

    # alpha**2 * mu_S = 5/2 is constant for the live Gamma-driver.  Equation
    # (20) therefore stays valid without changing the production shift gauge.
    eta = jnp.broadcast_to(metric.eta, r.shape)
    cp_Gamma = (
        eta * beta * dGamma_dr
        + eta * beta**2 * d2beta_dr2 / SHIFT_GAUGE_COEFFICIENT
        + eta * beta * dbeta_dr**2 / SHIFT_GAUGE_COEFFICIENT
        - eta**2 * beta * dbeta_dr / SHIFT_GAUGE_COEFFICIENT
    )

    cp_theta = (-alpha * normal_speed + beta) * dtheta_dr

    # Physical Ricci tensor for ds^2 = physical_grr dr^2
    #                              + physical_gt r^2 dOmega^2.
    radial_connection = dphysical_grr_dr / (2.0 * physical_grr)
    angular_connection = 1.0 / r + dphysical_gt_dr / (2.0 * physical_gt)
    dangular_connection_dr = (
        -1.0 / r**2
        + d2physical_gt_dr2 / (2.0 * physical_gt)
        - dphysical_gt_dr**2 / (2.0 * physical_gt**2)
    )
    Ricci_rr = (
        -2.0 * dangular_connection_dr
        + 2.0 * radial_connection * angular_connection
        - 2.0 * angular_connection**2
    )

    angular_numerator = (
        r * physical_gt + 0.5 * r**2 * dphysical_gt_dr
    )
    dangular_numerator_dr = (
        physical_gt
        + 2.0 * r * dphysical_gt_dr
        + 0.5 * r**2 * d2physical_gt_dr2
    )
    radial_angular_connection = -angular_numerator / physical_grr
    d_radial_angular_connection_dr = (
        -dangular_numerator_dr / physical_grr
        + angular_numerator
        * dphysical_grr_dr
        / physical_grr**2
    )
    Ricci_tt = (
        d_radial_angular_connection_dr
        + radial_connection * radial_angular_connection
        + 1.0
    )
    Ricci_ss = Ricci_rr / physical_grr
    Ricci_qq = 2.0 * Ricci_tt / (r**2 * physical_gt)

    # Divergence of the conformal trace-free curvature.  The angular
    # components are coordinate components A_theta_theta = r^2 At.
    conformal_radial_connection = dgrr_dr / (2.0 * grr)
    conformal_angular_connection = 1.0 / r + dgt_dr / (2.0 * gt)
    conformal_radial_angular_connection = -(
        r * gt + 0.5 * r**2 * dgt_dr
    ) / grr
    Arr_raised = Arr / grr**2
    At_raised = At / (r**2 * gt**2)
    dArr_raised_dr = dArr_dr / grr**2 - 2.0 * Arr * dgrr_dr / grr**3
    divergence_A_r = (
        dArr_raised_dr
        + conformal_radial_connection * Arr_raised
        + 2.0 * conformal_radial_angular_connection * At_raised
        + (
            conformal_radial_connection
            + 2.0 * conformal_angular_connection
        )
        * Arr_raised
    )
    physical_s_covariant = jnp.sqrt(physical_grr)
    divergence_A_s = physical_s_covariant * divergence_A_r

    d_two_Kh_plus_theta_dr = first_derivative(
        2.0 * Kh + theta,
        metric.dr,
        parity=1,
    )
    conformal_gradient_two_Kh_plus_theta_s = (
        physical_s_covariant
        * d_two_Kh_plus_theta_dr
        / grr
    )

    connection_constraint = conformal_connection_constraint(metric)
    dconnection_constraint_dr = first_derivative(
        connection_constraint,
        metric.dr,
        parity=-1,
    )
    radial_connection_constraint_derivative = dconnection_constraint_dr
    tangential_connection_constraint_derivative = (
        2.0 * connection_constraint / r
    )

    conformal_gradient_ln_chi_A_s = (
        physical_s_covariant
        * Arr
        * dchi_dr
        / (grr**2 * chi)
    )
    kappa = jnp.broadcast_to(metric.kappa, r.shape)

    scalar_constraint_terms = (
        2.0 * divergence_A_s
        - (2.0 / 3.0) * conformal_gradient_two_Kh_plus_theta_s
        - (2.0 / 3.0) * Ricci_ss
        + (2.0 / 3.0) * chi * radial_connection_constraint_derivative
        - (1.0 / 3.0) * chi * tangential_connection_constraint_derivative
        + (1.0 / 3.0) * Ricci_qq
        - 3.0 * conformal_gradient_ln_chi_A_s
        - kappa * physical_s_covariant * connection_constraint
    )

    A_ss = Arr / physical_grr
    A_i_s_A_is = chi * Arr**2 / grr**2
    lapse_hessian_ss = (
        d2alpha_dr2 - radial_connection * dalpha_dr
    ) / physical_grr
    lapse_hessian_AA = (
        2.0
        * angular_numerator
        * dalpha_dr
        / (physical_grr * r**2 * physical_gt)
    )

    Lie_beta_Arr = (
        beta * dArr_dr
        + (4.0 / 3.0) * Arr * dbeta_dr
        - (4.0 / 3.0) * Arr * beta / r
    )
    Lie_beta_A_ss = Lie_beta_Arr / physical_grr

    cp_A_ss = (
        -alpha * chi * scalar_constraint_terms
        + alpha * (A_ss * (Kh + 2.0 * theta) - 2.0 * A_i_s_A_is)
        - (2.0 / 3.0) * chi * lapse_hessian_ss
        + (1.0 / 3.0) * chi * lapse_hessian_AA
        + Lie_beta_A_ss
    )
    cp_Arr = physical_grr * cp_A_ss

    # Make the tangential curvature RHS tangent to the algebraic trace
    # constraint using the bulk conformal-metric derivatives retained above.
    cp_At = (
        -gt * cp_Arr / (2.0 * grr)
        + gt * Arr * bulk_derivatives.conformal_grr / (2.0 * grr**2)
        + At * bulk_derivatives.conformal_gt / gt
    )

    return bulk_derivatives._replace(
        Kh=bulk_derivatives.Kh.at[-1].set(cp_Kh[-1]),
        Gamma=bulk_derivatives.Gamma.at[-1].set(cp_Gamma[-1]),
        theta=bulk_derivatives.theta.at[-1].set(cp_theta[-1]),
        Arr=bulk_derivatives.Arr.at[-1].set(cp_Arr[-1]),
        At=bulk_derivatives.At.at[-1].set(cp_At[-1]),
    )
