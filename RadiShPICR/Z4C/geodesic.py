import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.utils import safe_radius
from RadiShPICR.particles.particle_shapes import interpolate_fields_to_particles
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.derivatives import first_derivative


def _radial_grid_from_metric(metric: Z4C_Metric):
    return RadialGrid(
        r_full=metric.r,
        r_interior=metric.r,
        dr=metric.dr,
        r_max=metric.r[-1],
    )


def compute_geodesic_terms(particles, metric: Z4C_Metric):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()
    # ur and uphi are the physical covariant spatial momenta u_r and u_phi.

    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    dr = metric.dr
    grid = _radial_grid_from_metric(metric)

    dalphadr = first_derivative(alpha, dr, parity=1)
    dbetadr = first_derivative(beta, dr, parity=-1)
    dchidr = first_derivative(chi, dr, parity=1)
    dgrrdr = first_derivative(conformal_grr, dr, parity=1)
    dgtdr = first_derivative(conformal_gt, dr, parity=1)

    (
        alpha_p,
        beta_p,
        chi_p,
        conformal_grr_p,
        conformal_gt_p,
        dalphadr_p,
        dbetadr_p,
        dchidr_p,
        dgrrdr_p,
        dgtdr_p,
    ) = interpolate_fields_to_particles(
        jnp.stack(
            (
                alpha,
                beta,
                chi,
                conformal_grr,
                conformal_gt,
                dalphadr,
                dbetadr,
                dchidr,
                dgrrdr,
                dgtdr,
            )
        ),
        r_particle,
        grid,
        shape_mode=particle_shape,
    )

    safe_r_particle = safe_radius(r_particle, 0.5 * dr)

    # The convention is conformal_gamma_ij = chi * gamma_ij, so the
    # physical inverse metric is gamma^ij = chi * conformal_gamma^ij.
    gamma_rr_inv_p = chi_p / conformal_grr_p
    gamma_phiphi_inv_p = chi_p / (
        safe_r_particle**2 * conformal_gt_p
    )

    dgamma_rr_inv_dr_p = (
        dchidr_p / conformal_grr_p
        - chi_p * dgrrdr_p / conformal_grr_p**2
    )
    dgamma_phiphi_inv_dr_p = (
        dchidr_p / (safe_r_particle**2 * conformal_gt_p)
        - 2.0 * chi_p / (safe_r_particle**3 * conformal_gt_p)
        - chi_p * dgtdr_p
        / (safe_r_particle**2 * conformal_gt_p**2)
    )

    lorentz_factor = jnp.sqrt(
        1.0
        + gamma_rr_inv_p * ur**2
        + gamma_phiphi_inv_p * uphi**2
    )

    dr_dt = alpha_p * gamma_rr_inv_p * ur / lorentz_factor - beta_p
    dphi_dt = (
        alpha_p * gamma_phiphi_inv_p * uphi / lorentz_factor
    )

    du_r_dt = -lorentz_factor * dalphadr_p + ur * dbetadr_p
    du_r_dt -= alpha_p / (2.0 * lorentz_factor) * (
        ur**2 * dgamma_rr_inv_dr_p
        + uphi**2 * dgamma_phiphi_inv_dr_p
    )
    du_phi_dt = jnp.zeros_like(uphi)

    return du_r_dt, du_phi_dt, dr_dt, dphi_dt
