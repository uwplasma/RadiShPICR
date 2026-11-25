"""Spherical apparent horizons from the outgoing null expansion.

The equation follows Thornburg, Living Reviews in Relativity 10, 3 (2007),
Section 8.1, equation (17): https://doi.org/10.12942/lrr-2007-3.
"""

import jax
import jax.numpy as jnp

from RadiShPICR.Z4C.derivatives import first_derivative


@jax.jit
def outgoing_expansion(metric):
    """Return the outgoing null expansion on the cell-centered radial grid."""

    physical_grr = metric.conformal_grr / metric.chi
    areal_radius = metric.r * jnp.sqrt(metric.conformal_gt / metric.chi)
    dareal_radius_dr = first_derivative(areal_radius, metric.dr, parity=-1)

    # Z4C evolves Kh = K - 2 theta, while the horizon equation uses physical K.
    K = metric.Kh + 2.0 * metric.theta
    angular_curvature = metric.At / metric.conformal_gt + K / 3.0

    return (
        2.0 * dareal_radius_dr / (areal_radius * jnp.sqrt(physical_grr))
        - 2.0 * angular_curvature
    )


@jax.jit
def find_apparent_horizon(metric, tolerance=1.0e-10, max_iterations=64):
    """Return the outermost horizon's coordinate and areal radii as a tuple.

    Search for exact grid-point zeros and sign changes of the outgoing
    expansion, then bisect its linear interpolant in the outermost bracket.
    Outermost means largest coordinate r, even when areal radius folds back.
    No extrapolation beyond the sampled radial domain is performed.

    Return (nan, nan) if no root is resolved, the geometry is invalid, or the
    iteration limit is reached before the bracket width meets ``tolerance``.
    The tolerance is in coordinate-radius units; use float64 for the default.
    It controls the root solve, not the finite-difference/interpolation error.
    Tangential zeros between samples and unresolved root pairs may be missed.
    """

    r = metric.r
    areal_radius = r * jnp.sqrt(metric.conformal_gt / metric.chi)
    expansion = outgoing_expansion(metric)

    valid = jnp.all(
        jnp.isfinite(expansion) & jnp.isfinite(areal_radius)
        & jnp.isfinite(metric.conformal_grr / metric.chi)
        & (areal_radius > 0.0) & (metric.chi > 0.0)
        & (metric.conformal_grr > 0.0) & (metric.conformal_gt > 0.0)
    )
    sign_change = (
        ((expansion[:-1] < 0.0) & (expansion[1:] > 0.0))
        | ((expansion[:-1] > 0.0) & (expansion[1:] < 0.0))
    )
    indices = jnp.arange(r.size)
    bracket_index = jnp.max(jnp.where(sign_change, indices[:-1], -1))
    exact_index = jnp.max(jnp.where(expansion == 0.0, indices, -1))
    found = (bracket_index >= 0) | (exact_index >= 0)

    # An exact root needs a zero-width bracket, including at the final cell.
    use_exact = exact_index > bracket_index
    left_index = jnp.maximum(jnp.where(use_exact, exact_index, bracket_index), 0)
    right_index = jnp.where(use_exact, left_index, left_index + 1)
    left, right = r[left_index], r[right_index]
    expansion_left = expansion[left_index]

    def bisect(_, interval):
        left, right, expansion_left = interval
        middle = 0.5 * (left + right)
        expansion_middle = jnp.interp(middle, r, expansion)

        move_left = jnp.signbit(expansion_middle) == jnp.signbit(expansion_left)
        exact = expansion_middle == 0.0
        active = right - left > tolerance
        next_left = jnp.where(active & (move_left | exact), middle, left)
        next_right = jnp.where(active & (~move_left | exact), middle, right)
        next_expansion_left = jnp.where(
            active & (move_left | exact), expansion_middle, expansion_left,
        )
        return next_left, next_right, next_expansion_left

    left, right, _ = jax.lax.fori_loop(
        0, max_iterations, bisect, (left, right, expansion_left),
    )
    converged = valid & found & (right - left <= tolerance)
    horizon_r = jnp.where(converged, 0.5 * (left + right), jnp.nan)
    horizon_R = jnp.interp(horizon_r, r, areal_radius)

    return horizon_r, horizon_R
