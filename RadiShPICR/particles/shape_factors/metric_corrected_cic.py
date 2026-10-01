"""Metric-corrected CIC deposition on the spherical cell-centered grid."""

from .cartesian_shapes import (
    _unbounded_raw_radial_shape_stencil,
    apply_stencil_boundaries,
)
from .common import (
    apply_metric_transfer,
    inner_areal_radius_index,
    metric_correction_coefficients,
)


def raw_cic_stencil(radial_positions, metric, inner_open=False):
    """Return corrected weights before origin folding or boundary truncation.

    Between centers, w_lower=(1-f)*(1+beta*f) and
    w_upper=f*(1-beta*(1-f)). Thus |beta| <= 1 preserves positivity.
    """

    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        radial_positions, metric.r, metric.dr, shape_mode="linear",
    )
    beta = metric_correction_coefficients(metric, inner_open, shape_mode="linear")
    corrected_weights = apply_metric_transfer(
        radial_positions, metric, beta, raw_indices, raw_weights,
    )

    return raw_indices, corrected_weights


def metric_corrected_cic_stencil(particles, metric, inner_open=False):
    """Return (indices, even_weights, odd_weights), each shaped (2, N).

    Reads particle positions and the Z4C metric arrays. Weights are
    dimensionless: mass, charge, particle weight, and shell-volume division
    belong to deposition. This function selects CIC explicitly,
    independently of particles.get_shape(). It can be called under jax.jit.
    """

    raw_indices, corrected_weights = raw_cic_stencil(
        particles.r, metric, inner_open,
    )
    return apply_stencil_boundaries(
        raw_indices, corrected_weights, metric.r.size, inner_open,
        inner_areal_radius_index(metric),
    )
