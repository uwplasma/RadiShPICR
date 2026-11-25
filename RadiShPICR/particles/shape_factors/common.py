"""Proper shell volumes and the moving inner areal boundary."""

import jax.numpy as jnp


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
