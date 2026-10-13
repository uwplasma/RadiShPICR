"""Constrained particle sources from one completed areal-coordinate map."""

import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.utils import (
    angular_lorentz_term,
    radial_shell_volume,
)
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    interpolate_fields_to_particles,
    radial_shape_stencil,
)


def particle_coordinate_radius(particles, A_values, r_grid):
    """Invert the completed monotone map R = r A(r)."""
    radius = jnp.interp(
        particles.r, A_values * r_grid, r_grid,
        left=jnp.nan, right=jnp.nan,
    )
    valid_map = jnp.all(jnp.diff(A_values * r_grid) > 0.0) & jnp.all(A_values > 0.0)
    return jnp.where(valid_map, radius, jnp.nan)


def deposit_source_numerators(particles, A_values, grid, shape_mode=None):
    """Deposit (m W, q, m u_r^2 / W, m u_r) with covariant momenta."""
    shape = particles.get_shape() if shape_mode is None else shape_mode
    r_particle = particle_coordinate_radius(particles, A_values, grid.r_full)
    active = particles.weight != 0.0
    r_particle = jnp.where(active, r_particle, 0.0)
    A_particle = interpolate_fields_to_particles(
        A_values[jnp.newaxis, :], r_particle, grid, shape_mode=shape,
    )[0]
    # Stored u_r/A is the orthonormal momentum; angular momentum uses areal R.
    ur = A_particle * particles.ur
    W = jnp.sqrt(
        1.0 + particles.ur**2 + angular_lorentz_term(particles.uphi, 1.0, particles.r)
    )
    even_indices, even_weights = radial_shape_stencil(r_particle, grid, shape, parity=1)
    odd_indices, odd_weights = radial_shape_stencil(r_particle, grid, shape, parity=-1)
    mass = particles.get_mass()
    moments = (mass * W, particles.get_charge(), mass * ur**2 / W, mass * ur)
    sources = []
    for i, moment in enumerate(moments):
        indices, weights = (odd_indices, odd_weights) if i == 3 else (even_indices, even_weights)
        values = jnp.zeros_like(grid.r_full).at[indices].add(
            weights * jnp.where(active, moment, 0.0)
        )
        sources.append(jnp.where(jnp.all(jnp.isfinite(r_particle)), values, jnp.nan))
    return tuple(sources)


def source_values_at_point(numerators, A_at_point, radial_coordinate, grid):
    index = jnp.rint((radial_coordinate - grid.r_full[0]) / grid.dr).astype(jnp.int32)
    volume = radial_shell_volume(A_at_point, radial_coordinate, grid.dr)
    return tuple(values[index] / volume for values in numerators)


def source_terms_at_point(
    particles, A_at_point, radial_coordinate, grid, shape_mode=None, *, A_values,
):
    """Pointwise API requires the completed metric, not a scalar mapping guess."""
    numerators = deposit_source_numerators(particles, A_values, grid, shape_mode)
    return source_values_at_point(numerators, A_at_point, radial_coordinate, grid)
