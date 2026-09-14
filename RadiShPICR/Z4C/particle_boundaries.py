import jax.numpy as jnp

from RadiShPICR.particles.shape_factors.common import (
    inner_areal_radius_index,
)
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    _cell_centered_open_inner_shape_stencil,
)


INNER_AREAL_GHOST_CELLS = 2


def _delete_particles(particles, delete):
    weight = jnp.where(delete, 0.0, particles.weight)
    inactive = weight == 0.0

    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=weight,
        r=jnp.where(inactive, 0.0, particles.r),
        ur=jnp.where(inactive, 0.0, particles.ur),
        phi=jnp.where(inactive, 0.0, particles.phi),
        uphi=jnp.where(inactive, 0.0, particles.uphi),
        shape_mode=particles.shape_mode,
    )


def deleting_particle_boundary(particles, metric=None):
    """Delete particles that crossed the isotropic radial origin.

    Deleted particles remain in the fixed-size JAX arrays with zero weight and
    inert phase-space coordinates, so they no longer deposit matter.
    """

    return _delete_particles(particles, particles.r < 0.0)


def deleting_inner_areal_radius_boundary(particles, metric):
    """Irreversibly absorb particles whose inner-open shape has left the grid."""

    inner_boundary_index = inner_areal_radius_index(metric)
    _, physical_weights, _ = _cell_centered_open_inner_shape_stencil(
        particles.r,
        metric.r,
        metric.dr,
        particles.get_shape(),
        inner_boundary_index,
    )
    physical_overlap = jnp.sum(physical_weights, axis=0)
    no_inner_overlap = jnp.logical_and(
        particles.r < metric.r[inner_boundary_index],
        physical_overlap == 0.0,
    )

    inner_ghost_edge = (
        metric.r[inner_boundary_index]
        - (INNER_AREAL_GHOST_CELLS + 0.5) * metric.dr
    )
    beyond_inner_ghosts = particles.r < inner_ghost_edge

    return _delete_particles(
        particles,
        jnp.logical_or(no_inner_overlap, beyond_inner_ghosts),
    )
