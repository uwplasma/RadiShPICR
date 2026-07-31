import jax.numpy as jnp
import jax

from RadiShPICR.particles.particle_shapes import interpolate_field_to_particles
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative


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
    # unpack particle positions, velocities, and shape

    alpha = metric.alpha
    beta = metric.beta
    # get the lapse and shift from the metric
    chi = metric.chi
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    dr = metric.dr
    # unpack metric components
    grid = _radial_grid_from_metric(metric)


    # du_phidt = 0
    # u_theta  = 0
    # dx^rdt = alpha * u^r - beta^r
    # dx^phi dt = alpha * u^phi
    # du_rdt = (- gamma * dalphadr ) + u_r * dbetadr
    # du_rdt -= (alpha / 2 * gamma) * (u^r**2 * ( dgrrdr/chi - grr * dchidr/chi**2 )
    # du_rdt -= (alpha / 2 * gamma) * (u^phi**2 * ( 2 * r * gt/chi + r**2 * (dgtdr/chi - gt * dchidr/chi**2) ) )
    # using covariant velocities


    dalphadr = first_derivative(alpha, dr, parity=1)
    dbetadr = first_derivative(beta, dr, parity=-1)
    dchidr = first_derivative(chi, dr, parity=1)
    dgrrdr = first_derivative(conformal_grr, dr, parity=1)
    dgtdr = first_derivative(conformal_gt, dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods

    dalphadr_p = interpolate_field_to_particles(dalphadr, r_particle, grid, shape_mode=particle_shape)
    dbetadr_p = interpolate_field_to_particles(dbetadr, r_particle, grid, shape_mode=particle_shape)
    dgrrdr_p = interpolate_field_to_particles(dgrrdr, r_particle, grid, shape_mode=particle_shape)
    dgtdr_p = interpolate_field_to_particles(dgtdr, r_particle, grid, shape_mode=particle_shape)
    dchidr_p = interpolate_field_to_particles(dchidr, r_particle, grid, shape_mode=particle_shape)
    # interpolate metric derivatives to particle

    conformal_grr_p = interpolate_field_to_particles(conformal_grr, r_particle, grid, shape_mode=particle_shape)
    conformal_gt_p = interpolate_field_to_particles(conformal_gt, r_particle, grid, shape_mode=particle_shape)
    chi_p = interpolate_field_to_particles(chi, r_particle, grid, shape_mode=particle_shape)
    alpha_p = interpolate_field_to_particles(alpha, r_particle, grid, shape_mode=particle_shape)
    beta_p = interpolate_field_to_particles(beta, r_particle, grid, shape_mode=particle_shape)
    # interpolate metric components to particle

    W = jnp.sqrt(1 + ur**2 / conformal_grr + uphi**2 * r_particle**2 * conformal_gt)
    # compute the Lorentz factor W for the particle based on its velocities and the metric components

    drdt = alpha_p * ur - beta_p
    dphidt = alpha_p * uphi
    # compute the time derivatives of the particle's radial and angular positions using the lapse and shift

    du_rdt = - W * dalphadr_p + ur * dbetadr_p
    du_rdt -= (alpha_p / (2 * W)) * (ur**2 * (dgrrdr_p / chi_p - conformal_grr_p * dchidr_p / chi_p**2))
    du_rdt -= (alpha_p / (2 * W)) * (uphi**2 * (2 * r_particle * conformal_gt_p / chi_p + r_particle**2 *(dgtdr_p / chi_p - conformal_gt_p * dchidr_p / chi_p**2)))
    # compute the time derivative of the particle's radial velocity using the geodesic equation in covariant form
    du_phidt = 0.0
    # compute the time derivative of the particle's angular velocity (zero for spherically symmetric spacetimes)

    return du_rdt, du_phidt, drdt, dphidt