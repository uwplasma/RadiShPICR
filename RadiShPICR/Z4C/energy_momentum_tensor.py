from typing import NamedTuple

import jax
import jax.numpy as jnp

from RadiShPICR.particles.shape_factors.common import (
    proper_radial_shell_volume,
)
from RadiShPICR.particles.shape_factors import (
    particle_deposition_stencil,
)
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    _interpolate_cell_centered_fields_to_particles,
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


def _radial_matter_deposition_data(particles, metric, inner_open=False):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()

    chi = metric.chi
    grid = _radial_grid_from_metric(metric)
    grr_p, gt_p = _interpolate_cell_centered_fields_to_particles(
        jnp.stack(
            (
                metric.conformal_grr / chi,
                metric.conformal_gt / chi,
            )
        ),
        r_particle,
        grid,
        shape_mode=particle_shape,
        field_parities=jnp.asarray((1, 1)),
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

    lorentz_factor = jnp.sqrt(
        1.0
        + gamma_rr_inv_p * ur**2
        + angular_lorentz
    )
    particle_mass = particles.get_mass()

    indices, even_weights, odd_weights = particle_deposition_stencil(
        particles,
        metric,
        inner_open,
    )
    rho_numerator = particle_mass * lorentz_factor
    Srr_numerator = particle_mass * ur**2 / lorentz_factor
    Stt_numerator = particle_mass * angular_stress / (2.0 * lorentz_factor)
    Sr_numerator = particle_mass * gamma_rr_inv_p * ur

    return (
        indices,
        even_weights,
        odd_weights,
        rho_numerator,
        Srr_numerator,
        Stt_numerator,
        Sr_numerator,
    )


def _deposit_radial_particle_quantity(
    indices,
    weights,
    numerator,
    proper_shell_volume,
):
    deposited_numerator = jnp.zeros_like(proper_shell_volume).at[indices].add(
        weights * numerator[jnp.newaxis, :]
    )

    return deposited_numerator / proper_shell_volume


def compute_radial_matter_terms(
    particles,
    metric: Z4C_Metric,
    inner_open=False,
):
    """Deposit all nonzero radial matter terms with one compact stencil.

    ``inner_open`` discards the share of each raw particle shape below the
    instantaneous areal-radius minimum.  The default retains regular origin
    parity for simulations without the absorbing particle boundary.
    """

    (
        indices,
        even_weights,
        odd_weights,
        rho_numerator,
        Srr_numerator,
        Stt_numerator,
        Sr_numerator,
    ) = _radial_matter_deposition_data(particles, metric, inner_open)
    proper_shell_volume = proper_radial_shell_volume(metric)

    rho = _deposit_radial_particle_quantity(
        indices,
        even_weights,
        rho_numerator,
        proper_shell_volume,
    )
    Srr = _deposit_radial_particle_quantity(
        indices,
        even_weights,
        Srr_numerator,
        proper_shell_volume,
    )
    Stt = _deposit_radial_particle_quantity(
        indices,
        even_weights,
        Stt_numerator,
        proper_shell_volume,
    )
    Sr = _deposit_radial_particle_quantity(
        indices,
        odd_weights,
        Sr_numerator,
        proper_shell_volume,
    )

    return MatterTerms(
        rho=rho,
        Srr=Srr,
        Stt=Stt,
        Sr=Sr,
        St=jnp.zeros_like(rho),
    )


def relativistic_mass_energy_density(
    particles,
    metric: Z4C_Metric,
    inner_open=False,
):
    indices, even_weights, _, rho_numerator, _, _, _ = (
        _radial_matter_deposition_data(particles, metric, inner_open)
    )
    return _deposit_radial_particle_quantity(
        indices,
        even_weights,
        rho_numerator,
        proper_radial_shell_volume(metric),
    )


def compute_radial_momentum_density(
    particles,
    metric: Z4C_Metric,
    inner_open=False,
):
    indices, _, odd_weights, _, _, _, Sr_numerator = (
        _radial_matter_deposition_data(particles, metric, inner_open)
    )
    return _deposit_radial_particle_quantity(
        indices,
        odd_weights,
        Sr_numerator,
        proper_radial_shell_volume(metric),
    )


def compute_radial_stress_tensor_component(
    particles,
    metric: Z4C_Metric,
    inner_open=False,
):
    indices, even_weights, _, _, Srr_numerator, _, _ = (
        _radial_matter_deposition_data(particles, metric, inner_open)
    )
    return _deposit_radial_particle_quantity(
        indices,
        even_weights,
        Srr_numerator,
        proper_radial_shell_volume(metric),
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
