"""Original coordinate-space shapes and radial interpolation conventions."""

from functools import partial

import jax
import jax.numpy as jnp


def _linear_shape_weight(delta):
    """Evaluate the one-dimensional linear CIC shape function."""

    return jnp.maximum(1.0 - jnp.abs(delta), 0.0)


def _quadratic_shape_weight(delta):
    """Evaluate the one-dimensional quadratic TSC shape function."""

    absolute_delta = jnp.abs(delta)
    center_weight = 0.75 - delta**2
    outer_weight = 0.5 * (1.5 - absolute_delta) ** 2

    return jnp.where(
        absolute_delta <= 0.5,
        center_weight,
        jnp.where(absolute_delta <= 1.5, outer_weight, 0.0),
    )


def nearest_interior_index(radial_positions, grid):
    """Map particles to the nearest radial point with origin reflection."""

    floating_index = (radial_positions - grid.r_full[0]) / grid.dr
    nearest = jnp.rint(floating_index).astype(jnp.int32)
    reflected_nearest = jnp.abs(nearest)

    return jnp.clip(reflected_nearest, 0, grid.r_full.shape[0] - 2)


@partial(jax.jit, static_argnames=("shape_mode",))
def radial_shape_stencil(radial_positions, grid, shape_mode="nearest", parity=1):
    """Return radial indices and weights with parity reflection at the origin."""

    if shape_mode == "nearest":
        floating_index = (radial_positions - grid.r_full[0]) / grid.dr
        anchor = jnp.rint(floating_index).astype(jnp.int32)
        raw_indices = anchor[jnp.newaxis, :]
        raw_weights = jnp.ones_like(raw_indices, dtype=radial_positions.dtype)
    elif shape_mode == "linear":
        floating_index = (radial_positions - grid.r_full[0]) / grid.dr
        anchor = jnp.floor(floating_index).astype(jnp.int32)
        delta = floating_index - anchor.astype(radial_positions.dtype)
        offsets = jnp.asarray([0, 1], dtype=anchor.dtype)

        stencil_delta = (
            delta[jnp.newaxis, :]
            - offsets[:, jnp.newaxis].astype(radial_positions.dtype)
        )
        raw_weights = _linear_shape_weight(stencil_delta)
    else:
        floating_index = (radial_positions - grid.r_full[0]) / grid.dr
        anchor = jnp.rint(floating_index).astype(jnp.int32)
        delta = floating_index - anchor.astype(radial_positions.dtype)
        offsets = jnp.asarray([-1, 0, 1], dtype=anchor.dtype)

        stencil_delta = (
            delta[jnp.newaxis, :]
            - offsets[:, jnp.newaxis].astype(radial_positions.dtype)
        )
        raw_weights = _quadratic_shape_weight(stencil_delta)

    if shape_mode != "nearest":
        raw_indices = anchor[jnp.newaxis, :] + offsets[:, jnp.newaxis]

    reflected_indices = jnp.abs(raw_indices)
    last_interior = jnp.asarray(grid.r_full.shape[0] - 2, dtype=raw_indices.dtype)
    valid_source_cell = reflected_indices <= last_interior

    retained_weights = jnp.where(valid_source_cell, raw_weights, 0.0)
    weight_sum = jnp.sum(retained_weights, axis=0, keepdims=True)

    parity_value = jnp.asarray(parity, dtype=raw_weights.dtype)
    origin_value = jnp.where(parity_value < 0.0, 0.0, 1.0)
    reflection_sign = jnp.where(
        raw_indices < 0,
        parity_value,
        jnp.where(raw_indices == 0, origin_value, 1.0),
    )
    parity_weights = retained_weights * reflection_sign

    fallback_index = nearest_interior_index(radial_positions, grid)[jnp.newaxis, :]
    fallback_indices = jnp.broadcast_to(fallback_index, raw_indices.shape)
    clipped_indices = jnp.clip(reflected_indices, 0, last_interior)
    has_valid_weight = weight_sum > 0.0

    fallback_raw_index = jnp.rint(
        (radial_positions - grid.r_full[0]) / grid.dr
    )
    fallback_sign = jnp.where(
        fallback_raw_index < 0,
        parity_value,
        jnp.where(fallback_raw_index == 0, origin_value, 1.0),
    )
    fallback_weights = jnp.zeros_like(retained_weights)
    fallback_weights = fallback_weights.at[0, :].set(fallback_sign)

    indices = jnp.where(has_valid_weight, clipped_indices, fallback_indices)
    weights = jnp.where(
        has_valid_weight,
        parity_weights / jnp.where(has_valid_weight, weight_sum, 1.0),
        fallback_weights,
    )

    return indices, weights


def _unbounded_raw_radial_shape_stencil(
    radial_positions,
    radial_grid,
    dr,
    shape_mode="nearest",
):
    """Return raw compact radial indices and weights before boundary handling."""

    floating_index = (radial_positions - radial_grid[0]) / dr

    if shape_mode == "nearest":
        anchor = jnp.floor(floating_index + 0.5).astype(jnp.int32)
        raw_indices = anchor[jnp.newaxis, :]
        # The anchor already makes a deterministic right-sided choice at an
        # exact half-cell tie, so every particle carries one full NGP weight.
        raw_weights = jnp.ones_like(raw_indices, dtype=radial_positions.dtype)
    elif shape_mode == "linear":
        anchor = jnp.floor(floating_index).astype(jnp.int32)
        offsets = jnp.asarray([0, 1], dtype=anchor.dtype)
        raw_indices = anchor[jnp.newaxis, :] + offsets[:, jnp.newaxis]
        delta = (
            floating_index[jnp.newaxis, :]
            - raw_indices.astype(radial_positions.dtype)
        )
        raw_weights = _linear_shape_weight(delta)
    else:
        anchor = jnp.rint(floating_index).astype(jnp.int32)
        offsets = jnp.asarray([-1, 0, 1], dtype=anchor.dtype)
        raw_indices = anchor[jnp.newaxis, :] + offsets[:, jnp.newaxis]
        delta = (
            floating_index[jnp.newaxis, :]
            - raw_indices.astype(radial_positions.dtype)
        )
        raw_weights = _quadratic_shape_weight(delta)

    return raw_indices, raw_weights


def _cell_centered_radial_shape_stencil(
    radial_positions,
    radial_grid,
    dr,
    shape_mode,
):
    """Fold compact weights across a half-cell-centered parity origin."""

    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        radial_positions,
        radial_grid,
        dr,
        shape_mode=shape_mode,
    )

    reflected_indices = jnp.where(
        raw_indices < 0,
        -raw_indices - 1,
        raw_indices,
    )
    valid = reflected_indices < radial_grid.shape[0]
    indices = jnp.clip(reflected_indices, 0, radial_grid.shape[0] - 1)

    even_weights = jnp.where(valid, raw_weights, 0.0)
    reflection_sign = jnp.where(raw_indices < 0, -1.0, 1.0)
    odd_weights = even_weights * reflection_sign
    if shape_mode == "nearest":
        # At the parity fixed point the two equally near half cells represent
        # opposite sides of the origin.  Even quantities retain unit weight;
        # an odd radial field vanishes there.
        odd_weights = jnp.where(
            radial_positions[jnp.newaxis, :] == 0.0,
            0.0,
            odd_weights,
        )
    origin_stencil = jnp.any(raw_indices < 0, axis=0)

    return indices, even_weights, odd_weights, origin_stencil


def _cell_centered_open_inner_shape_stencil(
    radial_positions,
    radial_grid,
    dr,
    shape_mode,
    inner_boundary_index,
):
    """Return the physical share of shapes crossing an open inner boundary.

    Raw compact weights below ``inner_boundary_index`` belong to the logical
    inner ghost cells.  They are neither reflected nor renormalized onto the
    physical grid, so a particle's deposited source decreases continuously as
    its finite-width shape leaves the domain.
    """

    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        radial_positions,
        radial_grid,
        dr,
        shape_mode=shape_mode,
    )

    last_grid_index = radial_grid.shape[0] - 1
    physical = jnp.logical_and(
        raw_indices >= inner_boundary_index,
        raw_indices <= last_grid_index,
    )
    indices = jnp.clip(raw_indices, 0, last_grid_index)
    physical_weights = jnp.where(physical, raw_weights, 0.0)

    return indices, physical_weights, physical_weights


@partial(jax.jit, static_argnames=("shape_mode",))
def unbounded_radial_shape_stencil(
    radial_positions,
    radial_grid,
    dr,
    shape_mode="nearest",
):
    """Return compact weights without boundary clipping or renormalization."""

    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        radial_positions,
        radial_grid,
        dr,
        shape_mode=shape_mode,
    )
    valid = jnp.logical_and(
        raw_indices >= 0,
        raw_indices < radial_grid.shape[0],
    )
    indices = jnp.clip(raw_indices, 0, radial_grid.shape[0] - 1)
    weights = jnp.where(valid, raw_weights, 0.0)

    return indices, weights


@partial(jax.jit, static_argnames=("shape_mode",))
def interpolate_field_to_particles(
    field,
    radial_positions,
    grid,
    shape_mode="nearest",
    parity=1,
):
    """Interpolate a radial grid field to particle positions."""

    if shape_mode == "nearest":
        reflected_positions = jnp.abs(radial_positions)
        interpolated_field = jnp.interp(reflected_positions, grid.r_full, field)
        origin_value = jnp.where(parity < 0, 0.0, 1.0)
        reflection_sign = jnp.where(
            radial_positions < 0.0,
            parity,
            jnp.where(radial_positions == 0.0, origin_value, 1.0),
        )
        return reflection_sign * interpolated_field

    indices, weights = radial_shape_stencil(
        radial_positions,
        grid,
        shape_mode=shape_mode,
        parity=parity,
    )
    return jnp.sum(field[indices] * weights, axis=0)


@partial(jax.jit, static_argnames=("shape_mode",))
def interpolate_fields_to_particles(
    fields,
    radial_positions,
    grid,
    shape_mode="nearest",
    field_parities=None,
):
    """Interpolate several radial fields using one particle stencil."""

    fields = jnp.asarray(fields)
    if field_parities is None:
        field_parities = jnp.ones(fields.shape[0], dtype=fields.dtype)
    else:
        field_parities = jnp.asarray(field_parities, dtype=fields.dtype)

    if shape_mode == "nearest":
        reflected_positions = jnp.abs(radial_positions)
        interpolated_fields = jax.vmap(
            lambda field: jnp.interp(reflected_positions, grid.r_full, field)
        )(fields)
        reflection_sign = jnp.where(
            radial_positions[jnp.newaxis, :] < 0.0,
            field_parities[:, jnp.newaxis],
            jnp.where(
                jnp.logical_and(
                    radial_positions[jnp.newaxis, :] == 0.0,
                    field_parities[:, jnp.newaxis] < 0.0,
                ),
                0.0,
                1.0,
            ),
        )
        return reflection_sign * interpolated_fields

    def interpolate_one_field(field, parity):
        indices, weights = radial_shape_stencil(
            radial_positions,
            grid,
            shape_mode=shape_mode,
            parity=parity,
        )
        return jnp.sum(field[indices] * weights, axis=0)

    return jax.vmap(interpolate_one_field)(fields, field_parities)


@partial(jax.jit, static_argnames=("shape_mode",))
def _interpolate_cell_centered_fields_to_particles(
    fields,
    radial_positions,
    grid,
    shape_mode="nearest",
    field_parities=None,
):
    """Interpolate fields across a half-cell-centered parity origin."""

    fields = jnp.asarray(fields)
    if field_parities is None:
        field_parities = jnp.ones(fields.shape[0], dtype=fields.dtype)
    else:
        field_parities = jnp.asarray(field_parities, dtype=fields.dtype)

    if shape_mode == "nearest":
        reflected_positions = jnp.abs(radial_positions)
        interpolated_fields = jax.vmap(
            lambda field: jnp.interp(reflected_positions, grid.r_full, field)
        )(fields)

        first_radius = grid.r_full[0]
        center_fraction = reflected_positions / first_radius
        center_values = jnp.where(
            field_parities[:, jnp.newaxis] < 0.0,
            fields[:, :1] * center_fraction[jnp.newaxis, :],
            fields[:, :1],
        )
        interpolated_fields = jnp.where(
            reflected_positions[jnp.newaxis, :] < first_radius,
            center_values,
            interpolated_fields,
        )

        reflection_sign = jnp.where(
            radial_positions[jnp.newaxis, :] < 0.0,
            field_parities[:, jnp.newaxis],
            1.0,
        )

        return reflection_sign * interpolated_fields

    # Keep the existing clipping and outer-boundary behavior when the raw
    # stencil never crosses the half-cell origin.
    existing_interpolation = interpolate_fields_to_particles(
        fields,
        radial_positions,
        grid,
        shape_mode=shape_mode,
    )
    indices, even_weights, odd_weights, origin_stencil = (
        _cell_centered_radial_shape_stencil(
            radial_positions,
            grid.r_full,
            grid.dr,
            shape_mode,
        )
    )
    field_weights = jnp.where(
        field_parities[:, jnp.newaxis, jnp.newaxis] < 0.0,
        odd_weights[jnp.newaxis, :, :],
        even_weights[jnp.newaxis, :, :],
    )
    origin_interpolation = jnp.sum(
        fields[:, indices] * field_weights,
        axis=1,
    )

    return jnp.where(
        origin_stencil[jnp.newaxis, :],
        origin_interpolation,
        existing_interpolation,
    )


def shape_weights_at_point(
    radial_positions,
    radial_coordinate,
    dr,
    shape_mode="nearest",
    grid=None,
    parity=1,
):
    """Evaluate particle weights at one radial coordinate.

    When ``grid`` is supplied, the weights use the same clipped and normalized
    interior stencil as deposition and interpolation. The no-grid path uses
    the original unbounded coordinate-space shape, without metric correction.
    """

    if grid is not None:
        indices, stencil_weights = radial_shape_stencil(
            radial_positions,
            grid,
            shape_mode=shape_mode,
            parity=parity,
        )
        floating_index = (radial_coordinate - grid.r_full[0]) / grid.dr
        grid_index = jnp.rint(floating_index).astype(indices.dtype)

        weights_at_point = jnp.where(
            indices == grid_index,
            stencil_weights,
            0.0,
        )
        return jnp.sum(weights_at_point, axis=0)

    if shape_mode == "nearest":
        offset = radial_positions - radial_coordinate
        return jnp.where(
            jnp.logical_and(offset >= -0.5 * dr, offset < 0.5 * dr),
            1.0,
            0.0,
        )

    delta = (radial_positions - radial_coordinate) / dr
    if shape_mode == "linear":
        return _linear_shape_weight(delta)

    return _quadratic_shape_weight(delta)
