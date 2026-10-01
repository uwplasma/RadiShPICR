"""Proper-volume quadrature, conservative transfers, and stencil boundaries.

These routines read metric arrays directly and do not import Z4C modules.
"""

import jax
import jax.numpy as jnp

from .cartesian_shapes import (
    _unbounded_raw_radial_shape_stencil,
    apply_stencil_boundaries,
)


def inner_areal_radius_index(metric):
    """Locate the minimum of the sampled areal radius."""

    areal_radius = metric.r * jnp.sqrt(metric.conformal_gt / metric.chi)
    return jnp.argmin(areal_radius)


def proper_radial_shell_volume(metric):
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


def proper_radial_shell_quadrature(metric):
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


def metric_correction_coefficients(metric, inner_open, shape_mode="quadratic"):
    """Match deposited proper volume with a limited Ruyten transfer.

    CIC uses |beta| < 1; quadratic TSC uses |beta| < 3/2. Limiting keeps
    weights nonnegative but can leave a shell-volume residual. The final
    shell is excluded from the recurrence, preserving outer truncation.
    """

    quadrature_radius, quadrature_volume = proper_radial_shell_quadrature(
        metric
    )
    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        quadrature_radius,
        metric.r,
        metric.dr,
        shape_mode=shape_mode,
    )

    # Calibrate on the full metric grid. The moving absorbing boundary is
    # applied only to the final particle stencil, not to this recurrence.
    indices, retained_weights, _ = apply_stencil_boundaries(
        raw_indices, raw_weights, metric.r.size, inner_open,
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
        proper_radial_shell_volume(metric)[:-1]
        - uncorrected_shell_volume[:-1]
    )
    beta_limit = (1.0 if shape_mode == "linear" else 1.5) * (
        1.0 - 8.0 * jnp.finfo(metric.r.dtype).eps
    )

    def solve_one_interval(previous_transfer, interval_data):
        volume_error, volume_moment = interval_data
        unconstrained_beta = (
            volume_error + previous_transfer
        ) / volume_moment
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


def apply_metric_transfer(radial_positions, metric, beta, raw_indices, raw_weights):
    """Transfer beta*f*(1-f) from the upper center to the lower center.

    The equal and opposite changes preserve each raw stencil's total weight.
    Mirror beta with odd parity on negative-radius intervals; leave the
    interval straddling the origin and exterior intervals uncorrected.
    """

    floating_index = (radial_positions - metric.r[0]) / metric.dr
    lower_index = jnp.floor(floating_index).astype(jnp.int32)
    upper_fraction = floating_index - lower_index.astype(metric.r.dtype)

    # Reflection maps interval i to -i-2 on a half-cell-centered grid.
    # The origin interval maps to -1 and has no correction coefficient.
    beta_index = jnp.where(lower_index >= 0, lower_index, -lower_index - 2)
    valid_interval = (beta_index >= 0) & (beta_index < beta.size)
    reflection_sign = jnp.where(lower_index >= 0, 1.0, -1.0)
    interval_beta = jnp.where(
        valid_interval,
        reflection_sign * beta[jnp.clip(beta_index, 0, beta.size - 1)],
        0.0,
    )
    transfer = interval_beta * upper_fraction * (1.0 - upper_fraction)

    corrected_weights = raw_weights + jnp.where(
        raw_indices == lower_index[jnp.newaxis, :],
        transfer[jnp.newaxis, :],
        0.0,
    )
    corrected_weights = corrected_weights - jnp.where(
        raw_indices == lower_index[jnp.newaxis, :] + 1,
        transfer[jnp.newaxis, :],
        0.0,
    )

    return corrected_weights
