"""Coordinate-space shapes and metric-corrected particle deposition.

Existing linear deposition remains ordinary CIC. Call
metric_corrected_cic_stencil explicitly to use the new CIC correction.
"""

import jax.numpy as jnp

from .cartesian_shapes import (
    _unbounded_raw_radial_shape_stencil,
    apply_stencil_boundaries,
)
from .common import inner_areal_radius_index
from .metric_corrected_cic import metric_corrected_cic_stencil
from .metric_correct_quadratic import metric_corrected_quadratic_stencil

__all__ = [
    "particle_deposition_stencil",
    "metric_corrected_cic_stencil",
    "metric_corrected_quadratic_stencil",
]


def particle_deposition_stencil(particles, metric, inner_open=False):
    """Return the production stencil: corrected TSC or ordinary NGP/CIC.

    The tuple contains indices, even weights, and odd weights, with the
    stencil axis first and the particle axis second.
    """

    shape_mode = particles.get_shape()
    if shape_mode == "quadratic":
        return metric_corrected_quadratic_stencil(particles, metric, inner_open)

    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        particles.r, metric.r, metric.dr, shape_mode,
    )
    indices, even_weights, odd_weights = apply_stencil_boundaries(
        raw_indices, raw_weights, metric.r.size, inner_open,
        inner_areal_radius_index(metric),
    )
    if shape_mode == "nearest":
        # An odd field vanishes at the parity fixed point, but an open
        # boundary must retain the sign of the incoming particle momentum.
        odd_weights = jnp.where(
            jnp.logical_and(jnp.logical_not(inner_open), particles.r == 0.0),
            0.0, odd_weights,
        )

    return indices, even_weights, odd_weights
