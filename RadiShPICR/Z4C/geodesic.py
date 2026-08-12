import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles.particle_shapes import interpolate_fields_to_particles
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _radial_grid_from_metric(metric: Z4C_Metric):
    return RadialGrid(
        r_full=metric.r,
        r_interior=metric.r,
        dr=metric.dr,
        r_max=metric.r[-1],
    )


def normal_particle_state(particles, metric: Z4C_Metric):
    """Return the ordinary isotropic ``r`` and covariant ``u_r`` arrays."""

    rs, _ = particles.get_positions()
    ubar, _ = particles.get_velocities()
    grid = _radial_grid_from_metric(metric)

    rs_grid = metric.r / metric.chi**(3.0 / 4.0)
    r_particle = jnp.interp(rs, rs_grid, metric.r)
    chi_p, conformal_grr_p = interpolate_fields_to_particles(
        jnp.stack((metric.chi, metric.conformal_grr)),
        r_particle,
        grid,
        shape_mode=particles.get_shape(),
    )
    ur = ubar * jnp.sqrt(conformal_grr_p / chi_p)

    return r_particle, ur


def lapse_freezing_particle_state(r_particle, ur, metric, shape_mode):
    """Convert ordinary isotropic particle variables to ``r_s`` and ``ubar``."""

    grid = _radial_grid_from_metric(metric)
    chi_p, conformal_grr_p = interpolate_fields_to_particles(
        jnp.stack((metric.chi, metric.conformal_grr)),
        r_particle,
        grid,
        shape_mode=shape_mode,
    )
    rs_grid = metric.r / metric.chi**(3.0 / 4.0)
    rs = jnp.interp(r_particle, metric.r, rs_grid)
    ubar = ur * jnp.sqrt(chi_p / conformal_grr_p)

    return rs, ubar


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


def compute_normal_geodesic_terms(particles, metric: Z4C_Metric):
    """Evaluate the Z4c geodesic RHS in ordinary ``(r, u_r)`` variables."""

    r_particle, ur = normal_particle_state(particles, metric)

    return _normal_geodesic_terms(particles, metric, r_particle, ur)


def _normal_geodesic_terms(particles, metric, r_particle, ur):
    _, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()

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


def compute_geodesic_terms(
    particles,
    metric: Z4C_Metric,
    metric_derivative: Z4C_Metric,
):
    """Evaluate the particle RHS for stored lapse-freezing variables."""

    ubar, _ = particles.get_velocities()
    particle_shape = particles.get_shape()
    grid = _radial_grid_from_metric(metric)

    r_particle, ur = normal_particle_state(particles, metric)
    du_r_dt, du_phi_dt, dr_dt, dphi_dt = _normal_geodesic_terms(
        particles,
        metric,
        r_particle,
        ur,
    )

    (
        chi_p,
        conformal_grr_p,
        dchidr_p,
        dgrrdr_p,
        dchidt_p,
        dgrrdt_p,
    ) = interpolate_fields_to_particles(
        jnp.stack(
            (
                metric.chi,
                metric.conformal_grr,
                first_derivative(metric.chi, metric.dr, parity=1),
                first_derivative(
                    metric.conformal_grr,
                    metric.dr,
                    parity=1,
                ),
                metric_derivative.chi,
                metric_derivative.conformal_grr,
            )
        ),
        r_particle,
        grid,
        shape_mode=particle_shape,
    )

    total_dchi_dt = dchidt_p + dchidr_p * dr_dt
    total_dgrr_dt = dgrrdt_p + dgrrdr_p * dr_dt

    drs_dt = dr_dt / chi_p**(3.0 / 4.0)
    drs_dt -= (
        3.0
        * r_particle
        * total_dchi_dt
        / (4.0 * chi_p**(7.0 / 4.0))
    )

    dubar_dt = jnp.sqrt(chi_p / conformal_grr_p) * du_r_dt
    dubar_dt += 0.5 * ubar * (
        total_dchi_dt / chi_p - total_dgrr_dt / conformal_grr_p
    )

    return dubar_dt, du_phi_dt, drs_dt, dphi_dt
