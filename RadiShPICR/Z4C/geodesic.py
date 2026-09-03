import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles.particle_shapes import (
    _interpolate_cell_centered_fields_to_particles,
)
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _radial_grid_from_metric(metric: Z4C_Metric):
    return RadialGrid(
        r_full=metric.r,
        r_interior=metric.r,
        dr=metric.dr,
        r_max=metric.r[-1],
    )


def _angular_geodesic_terms(
    uphi,
    r_particle,
    alpha_p,
    chi_p,
    conformal_gt_p,
    dchidr_p,
    dgtdr_p,
    lorentz_factor,
):
    """Return angular terms without evaluating the radial ``0 / 0`` limit."""

    def angular_terms_at_particle(
        uphi_p,
        r_p,
        alpha_at_p,
        chi_at_p,
        gt_at_p,
        dchi_at_p,
        dgt_at_p,
        gamma_at_p,
    ):
        def zero_angular_momentum():
            zero = jnp.zeros_like(uphi_p)
            return zero, zero

        def nonzero_angular_momentum():
            gamma_phiphi_inv = chi_at_p / (r_p**2 * gt_at_p)
            dgamma_phiphi_inv_dr = (
                dchi_at_p / (r_p**2 * gt_at_p)
                - 2.0 * chi_at_p / (r_p**3 * gt_at_p)
                - chi_at_p * dgt_at_p / (r_p**2 * gt_at_p**2)
            )
            angular_force = uphi_p**2 * dgamma_phiphi_inv_dr
            dphi_dt = (
                alpha_at_p * gamma_phiphi_inv * uphi_p / gamma_at_p
            )

            return angular_force, dphi_dt

        return jax.lax.cond(
            uphi_p == 0.0,
            zero_angular_momentum,
            nonzero_angular_momentum,
        )

    return jax.vmap(angular_terms_at_particle)(
        uphi,
        r_particle,
        alpha_p,
        chi_p,
        conformal_gt_p,
        dchidr_p,
        dgtdr_p,
        lorentz_factor,
    )


def compute_geodesic_terms(particles, metric: Z4C_Metric):
    """Evaluate the Z4C geodesic RHS in ordinary covariant momenta."""

    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()
    grid = _radial_grid_from_metric(metric)

    dalphadr = first_derivative(metric.alpha, metric.dr, parity=1)
    dbetadr = first_derivative(metric.beta, metric.dr, parity=-1)
    dchidr = first_derivative(metric.chi, metric.dr, parity=1)
    dgrrdr = first_derivative(
        metric.conformal_grr,
        metric.dr,
        parity=1,
    )
    dgtdr = first_derivative(metric.conformal_gt, metric.dr, parity=1)

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
    ) = _interpolate_cell_centered_fields_to_particles(
        jnp.stack(
            (
                metric.alpha,
                metric.beta,
                metric.chi,
                metric.conformal_grr,
                metric.conformal_gt,
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
        field_parities=jnp.asarray(
            (1, -1, 1, 1, 1, -1, 1, -1, -1, -1)
        ),
    )

    # The convention is conformal_gamma_ij = chi * gamma_ij, so the
    # physical inverse metric is gamma^ij = chi * conformal_gamma^ij.
    gamma_rr_inv_p = chi_p / conformal_grr_p
    dgamma_rr_inv_dr_p = (
        dchidr_p / conformal_grr_p
        - chi_p * dgrrdr_p / conformal_grr_p**2
    )

    def angular_lorentz_term(uphi_p, r_p, chi_at_p, gt_at_p):
        return jax.lax.cond(
            uphi_p == 0.0,
            lambda: jnp.zeros_like(uphi_p),
            lambda: chi_at_p * uphi_p**2 / (r_p**2 * gt_at_p),
        )

    angular_lorentz = jax.vmap(angular_lorentz_term)(
        uphi,
        r_particle,
        chi_p,
        conformal_gt_p,
    )
    lorentz_factor = jnp.sqrt(
        1.0 + gamma_rr_inv_p * ur**2 + angular_lorentz
    )

    angular_force, dphi_dt = _angular_geodesic_terms(
        uphi,
        r_particle,
        alpha_p,
        chi_p,
        conformal_gt_p,
        dchidr_p,
        dgtdr_p,
        lorentz_factor,
    )

    dr_dt = alpha_p * gamma_rr_inv_p * ur / lorentz_factor - beta_p
    du_r_dt = -lorentz_factor * dalphadr_p + ur * dbetadr_p
    du_r_dt -= alpha_p / (2.0 * lorentz_factor) * (
        ur**2 * dgamma_rr_inv_dr_p + angular_force
    )
    du_phi_dt = jnp.zeros_like(uphi)

    return du_r_dt, du_phi_dt, dr_dt, dphi_dt
