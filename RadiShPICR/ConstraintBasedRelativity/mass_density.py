import jax.numpy as jnp

from RadiShPICR.particles.particle_shapes import shape_weights_at_point
from RadiShPICR.ConstraintBasedRelativity.utils import radial_shell_volume


def mass_density_at_point(
    particles,
    A_at_point,
    radial_coordinate,
    grid,
    shape_mode=None,
):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape() if shape_mode is None else shape_mode
    dr = grid.dr

    weights = shape_weights_at_point(
        r_particle,
        radial_coordinate,
        dr,
        particle_shape,
        grid=grid,
    )


    lorentz_factors = jnp.sqrt(
        1.0
        + ur**2 / A_at_point**2
        + uphi**2 / (A_at_point**2 * radial_coordinate**2)
    )

    cell_volume = radial_shell_volume(
        A_at_point,
        radial_coordinate,
        dr,
    )

    conformal_mass_density = jnp.sum(particles.get_mass() * weights * lorentz_factors)
    mass_density = conformal_mass_density / cell_volume

    return mass_density