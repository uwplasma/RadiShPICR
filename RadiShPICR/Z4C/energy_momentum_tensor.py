from typing import NamedTuple

import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles.particle_shapes import (
    _cell_centered_radial_shape_stencil,
    _cell_centered_open_inner_shape_stencil,
    _interpolate_cell_centered_fields_to_particles,
    _unbounded_raw_radial_shape_stencil,
)
from RadiShPICR.Z4C.particle_boundaries import _inner_areal_radius_index


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


def _proper_radial_shell_volume(metric):
    inner_radius = jnp.maximum(metric.r - 0.5 * metric.dr, 0.0)
    outer_radius = metric.r + 0.5 * metric.dr
    coordinate_volume = (4.0 * jnp.pi / 3.0) * (
        outer_radius**3 - inner_radius**3
    )

    proper_volume_factor = (
        jnp.sqrt(metric.conformal_grr)
        * metric.conformal_gt
        / metric.chi**1.5
    )

    return coordinate_volume * proper_volume_factor


def _proper_radial_shell_quadrature(metric):
    """Return proper-volume quadrature on both halves of every shell."""

    gauss_abscissae = jnp.asarray(
        (-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)),
        dtype=metric.r.dtype,
    )
    gauss_weights = jnp.asarray(
        (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0),
        dtype=metric.r.dtype,
    )

    inner_radius = jnp.maximum(metric.r - 0.5 * metric.dr, 0.0)
    outer_radius = metric.r + 0.5 * metric.dr

    # The Ruyten transfer changes intervals at a grid center.  Integrate the
    # two polynomial pieces separately with the same cell-centered metric
    # factor used by the production finite-volume shell.
    segment_inner = jnp.stack((inner_radius, metric.r), axis=1)
    segment_outer = jnp.stack((metric.r, outer_radius), axis=1)
    segment_center = 0.5 * (segment_inner + segment_outer)
    segment_half_width = 0.5 * (segment_outer - segment_inner)

    quadrature_radius = (
        segment_center[:, :, jnp.newaxis]
        + segment_half_width[:, :, jnp.newaxis]
        * gauss_abscissae[jnp.newaxis, jnp.newaxis, :]
    )
    proper_volume_factor = (
        jnp.sqrt(metric.conformal_grr)
        * metric.conformal_gt
        / metric.chi**1.5
    )
    quadrature_volume = (
        4.0
        * jnp.pi
        * proper_volume_factor[:, jnp.newaxis, jnp.newaxis]
        * quadrature_radius**2
        * segment_half_width[:, :, jnp.newaxis]
        * gauss_weights[jnp.newaxis, jnp.newaxis, :]
    )

    return quadrature_radius.reshape(-1), quadrature_volume.reshape(-1)


def _density_conserving_quadratic_coefficients(metric, inner_open):
    """Solve the radial Ruyten transfer recurrence for quadratic TSC."""

    quadrature_radius, quadrature_volume = _proper_radial_shell_quadrature(
        metric
    )
    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        quadrature_radius,
        metric.r,
        metric.dr,
        shape_mode="quadratic",
    )

    def open_inner_weights(_):
        valid = jnp.logical_and(
            raw_indices >= 0,
            raw_indices < metric.r.shape[0],
        )
        indices = jnp.clip(raw_indices, 0, metric.r.shape[0] - 1)
        return indices, jnp.where(valid, raw_weights, 0.0)

    def parity_origin_weights(_):
        reflected_indices = jnp.where(
            raw_indices < 0,
            -raw_indices - 1,
            raw_indices,
        )
        valid = reflected_indices < metric.r.shape[0]
        indices = jnp.clip(
            reflected_indices,
            0,
            metric.r.shape[0] - 1,
        )
        return indices, jnp.where(valid, raw_weights, 0.0)

    indices, retained_weights = jax.lax.cond(
        inner_open,
        open_inner_weights,
        parity_origin_weights,
        operand=None,
    )
    uncorrected_shell_volume = jnp.zeros_like(metric.r).at[indices].add(
        retained_weights * quadrature_volume[jnp.newaxis, :]
    )

    floating_index = (quadrature_radius - metric.r[0]) / metric.dr
    lower_index = jnp.floor(floating_index).astype(jnp.int32)
    upper_fraction = floating_index - lower_index.astype(metric.r.dtype)
    valid_interval = jnp.logical_and(
        lower_index >= 0,
        lower_index < metric.r.shape[0] - 1,
    )
    interval_index = jnp.clip(lower_index, 0, metric.r.shape[0] - 2)
    interval_moment = jnp.zeros_like(metric.r[:-1]).at[interval_index].add(
        jnp.where(
            valid_interval,
            quadrature_volume * upper_fraction * (1.0 - upper_fraction),
            0.0,
        )
    )

    shell_volume_error = (
        _proper_radial_shell_volume(metric)[:-1]
        - uncorrected_shell_volume[:-1]
    )
    beta_limit = 1.5 * (
        1.0 - 8.0 * jnp.finfo(metric.r.dtype).eps
    )

    def solve_one_interval(previous_transfer, interval_data):
        volume_error, volume_moment = interval_data
        unconstrained_beta = (
            volume_error + previous_transfer
        ) / volume_moment
        # |beta| < 3/2 is the sharp bound that keeps all three quadratic
        # weights positive throughout the interval between grid centers.
        beta = jnp.clip(
            unconstrained_beta,
            -beta_limit,
            beta_limit,
        )
        transfer = beta * volume_moment
        return transfer, beta

    _, beta = jax.lax.scan(
        solve_one_interval,
        jnp.asarray(0.0, dtype=metric.r.dtype),
        (shell_volume_error, interval_moment),
    )

    return beta


def _density_conserving_quadratic_stencil(
    radial_positions,
    metric,
    inner_open,
):
    """Return raw quadratic weights corrected for spherical shell volume."""

    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        radial_positions,
        metric.r,
        metric.dr,
        shape_mode="quadratic",
    )
    beta = _density_conserving_quadratic_coefficients(metric, inner_open)

    floating_index = (radial_positions - metric.r[0]) / metric.dr
    lower_index = jnp.floor(floating_index).astype(jnp.int32)
    upper_fraction = floating_index - lower_index.astype(metric.r.dtype)

    positive_interval = jnp.logical_and(
        lower_index >= 0,
        lower_index < beta.shape[0],
    )
    mirrored_interval = jnp.logical_and(
        lower_index <= -2,
        -lower_index - 2 < beta.shape[0],
    )
    positive_beta_index = jnp.clip(lower_index, 0, beta.shape[0] - 1)
    mirrored_beta_index = jnp.clip(
        -lower_index - 2,
        0,
        beta.shape[0] - 1,
    )
    interval_beta = jnp.where(
        positive_interval,
        beta[positive_beta_index],
        jnp.where(
            mirrored_interval,
            -beta[mirrored_beta_index],
            0.0,
        ),
    )
    transfer = interval_beta * upper_fraction * (1.0 - upper_fraction)

    corrected_weights = raw_weights
    corrected_weights = corrected_weights + jnp.where(
        raw_indices == lower_index[jnp.newaxis, :],
        transfer[jnp.newaxis, :],
        0.0,
    )
    corrected_weights = corrected_weights - jnp.where(
        raw_indices == lower_index[jnp.newaxis, :] + 1,
        transfer[jnp.newaxis, :],
        0.0,
    )

    return raw_indices, corrected_weights


def _radial_particle_deposition_stencil(particles, metric, inner_open):
    if particles.get_shape() != "quadratic":
        def open_inner_stencil(_):
            return _cell_centered_open_inner_shape_stencil(
                particles.r,
                metric.r,
                metric.dr,
                particles.get_shape(),
                _inner_areal_radius_index(metric),
            )

        def parity_origin_stencil(_):
            indices, even_weights, odd_weights, _ = (
                _cell_centered_radial_shape_stencil(
                    particles.r,
                    metric.r,
                    metric.dr,
                    particles.get_shape(),
                )
            )
            return indices, even_weights, odd_weights

        return jax.lax.cond(
            inner_open,
            open_inner_stencil,
            parity_origin_stencil,
            operand=None,
        )

    raw_indices, corrected_weights = _density_conserving_quadratic_stencil(
        particles.r,
        metric,
        inner_open,
    )

    def open_inner_stencil(_):
        inner_boundary_index = _inner_areal_radius_index(metric)
        last_grid_index = metric.r.shape[0] - 1
        physical = jnp.logical_and(
            raw_indices >= inner_boundary_index,
            raw_indices <= last_grid_index,
        )
        indices = jnp.clip(raw_indices, 0, last_grid_index)
        physical_weights = jnp.where(physical, corrected_weights, 0.0)
        return indices, physical_weights, physical_weights

    def parity_origin_stencil(_):
        reflected_indices = jnp.where(
            raw_indices < 0,
            -raw_indices - 1,
            raw_indices,
        )
        valid = reflected_indices < metric.r.shape[0]
        indices = jnp.clip(reflected_indices, 0, metric.r.shape[0] - 1)
        even_weights = jnp.where(valid, corrected_weights, 0.0)
        reflection_sign = jnp.where(raw_indices < 0, -1.0, 1.0)
        odd_weights = even_weights * reflection_sign
        return indices, even_weights, odd_weights

    return jax.lax.cond(
        inner_open,
        open_inner_stencil,
        parity_origin_stencil,
        operand=None,
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

    indices, even_weights, odd_weights = _radial_particle_deposition_stencil(
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
    proper_shell_volume = _proper_radial_shell_volume(metric)

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
        _proper_radial_shell_volume(metric),
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
        _proper_radial_shell_volume(metric),
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
        _proper_radial_shell_volume(metric),
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
