from typing import NamedTuple

import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative
from RadiShPICR.Z4C.geodesic import normal_particle_state
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles.particle_shapes import (
    interpolate_fields_to_particles,
    unbounded_radial_shape_stencil,
)


class MatterTerms(NamedTuple):
    """Eulerian matter sources used by the spherical Z4C equations.

    ``Srr`` is the covariant radial stress ``S_rr`` and ``Stt`` is the
    shell-averaged tangential stress ``S_theta_theta / r**2``.  ``Sr`` is the
    contravariant radial momentum density ``S^r``.  Spherical symmetry leaves
    the tangential momentum density ``St`` equal to zero.
    """

    rho: jnp.ndarray
    # energy density
    Srr: jnp.ndarray
    Stt: jnp.ndarray
    # stress tensor components
    Sr: jnp.ndarray
    St: jnp.ndarray
    # momentum density


def _radial_grid_from_metric(metric: Z4C_Metric):
    return RadialGrid(
        r_full=metric.r,
        r_interior=metric.r,
        dr=metric.dr,
        r_max=metric.r[-1],
    )


def initialize_vacuum_matter_terms(metric):
    zeros = jnp.zeros_like(metric.r)

    return MatterTerms(
        rho=zeros,
        Srr=zeros,
        Stt=zeros,
        Sr=zeros,
        St=zeros,
    )


def _radial_matter_deposition_data(particles, metric):
    r_particle, ur = normal_particle_state(particles, metric)
    _, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()

    chi = metric.chi
    grid = _radial_grid_from_metric(metric)
    scaling_factor = jnp.sqrt(1.0 / chi**3)
    scaling_factor_p, grr_p, gt_p = interpolate_fields_to_particles(
        jnp.stack(
            (
                scaling_factor,
                metric.conformal_grr / chi,
                metric.conformal_gt / chi,
            )
        ),
        r_particle,
        grid,
        shape_mode=particle_shape,
    )
    gamma_rr_inv_p = 1.0 / grr_p

    def angular_matter_terms(uphi_p, r_p, gt_at_p):
        return jax.lax.cond(
            uphi_p == 0.0,
            lambda: (jnp.zeros_like(uphi_p), jnp.zeros_like(uphi_p)),
            lambda: (
                uphi_p**2 / (r_p**2 * gt_at_p),
                uphi_p**2 / r_p**2,
            ),
        )

    angular_lorentz, angular_stress = jax.vmap(angular_matter_terms)(
        uphi,
        r_particle,
        gt_p,
    )

    particle_volume_element = (
        4.0 * jnp.pi * r_particle**2 * scaling_factor_p
    )
    lorentz_factor = jnp.sqrt(
        1.0
        + gamma_rr_inv_p * ur**2
        + angular_lorentz
    )
    particle_mass = particles.get_mass()

    indices, weights = unbounded_radial_shape_stencil(
        r_particle,
        metric.r,
        metric.dr,
        shape_mode=particle_shape,
    )
    rho_contribution = particle_mass * lorentz_factor / particle_volume_element
    Srr_contribution = (
        particle_mass * ur**2 / (particle_volume_element * lorentz_factor)
    )
    Stt_contribution = particle_mass * angular_stress / (
        2.0 * particle_volume_element * lorentz_factor
    )
    Sr_contribution = (
        particle_mass
        * gamma_rr_inv_p
        * ur
        / particle_volume_element
    )

    return (
        indices,
        weights,
        rho_contribution,
        Srr_contribution,
        Stt_contribution,
        Sr_contribution,
    )


def _deposit_radial_particle_quantity(indices, weights, contribution, r):
    return jnp.zeros_like(r).at[indices].add(
        weights * contribution[jnp.newaxis, :]
    )


def compute_radial_matter_terms(particles, metric: Z4C_Metric):
    """Deposit all nonzero radial matter terms with one compact stencil."""

    (
        indices,
        weights,
        rho_contribution,
        Srr_contribution,
        Stt_contribution,
        Sr_contribution,
    ) = _radial_matter_deposition_data(particles, metric)

    rho = _deposit_radial_particle_quantity(
        indices,
        weights,
        rho_contribution,
        metric.r,
    )
    Srr = _deposit_radial_particle_quantity(
        indices,
        weights,
        Srr_contribution,
        metric.r,
    )
    Stt = _deposit_radial_particle_quantity(
        indices,
        weights,
        Stt_contribution,
        metric.r,
    )
    Sr = _deposit_radial_particle_quantity(
        indices,
        weights,
        Sr_contribution,
        metric.r,
    )

    return MatterTerms(
        rho=rho,
        Srr=Srr,
        Stt=Stt,
        Sr=Sr,
        St=jnp.zeros_like(rho),
    )


def relativistic_mass_energy_density(particles, metric: Z4C_Metric):
    indices, weights, rho_contribution, _, _, _ = (
        _radial_matter_deposition_data(particles, metric)
    )
    return _deposit_radial_particle_quantity(
        indices,
        weights,
        rho_contribution,
        metric.r,
    )


def compute_radial_momentum_density(particles, metric: Z4C_Metric):
    indices, weights, _, _, _, Sr_contribution = (
        _radial_matter_deposition_data(particles, metric)
    )
    return _deposit_radial_particle_quantity(
        indices,
        weights,
        Sr_contribution,
        metric.r,
    )


def compute_radial_stress_tensor_component(particles, metric: Z4C_Metric):
    indices, weights, _, Srr_contribution, _, _ = (
        _radial_matter_deposition_data(particles, metric)
    )
    return _deposit_radial_particle_quantity(
        indices,
        weights,
        Srr_contribution,
        metric.r,
    )


def compute_hamiltonian_constraint(metric: Z4C_Metric):

    # ASSUMES VACUUM and IGNORES THETA FOR NOW. NEEDS TO BE FIXED FOR NON-VACUUM CASES
    
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Arr = metric.Arr
    At = metric.At
    K  = metric.Kh
    r  = metric.r

    dchidr = first_derivative(chi, metric.dr, parity=1)
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    d2gtdr     = second_derivative(gt, metric.dr, parity=1 )
    d2chidr    = second_derivative(chi, metric.dr, parity=1)


    constraint =  -(Arr**2/grr**2) + (2*d2chidr)/grr - (5*dchidr**2)/(2*(jnp.maximum(chi, 1e-10))*grr) - (2*At**2)/gt**2 + (2*K**2)/3 + \
        dchidr*(-(dgrrdr/grr**2) + (2*dgtdr)/(grr*gt) + 4/(grr*r)) + \
        chi*(dgtdr**2/(2*grr*gt**2) + (dgrrdr*dgtdr)/(grr**2*gt) - (2*d2gtdr)/(grr*gt) - 2/(grr*r**2) + 2/(gt*r**2) + 
        (2*dgrrdr)/(grr**2*r) - (6*dgtdr)/(grr*gt*r))


    return constraint
