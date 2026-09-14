import jax.numpy as jnp
from RadiShPICR.particles.shape_factors.cartesian_shapes import shape_weights_at_point
from RadiShPICR.ConstraintBasedRelativity.utils import radial_shell_volume


def charge_density_at_point(
    particles,
    A_at_point,
    radial_coordinate,
    grid,
    shape_mode=None,
):
    rs, _ = particles.get_positions()
    particle_shape = particles.get_shape() if shape_mode is None else shape_mode
    dr = grid.dr
    r_particle = rs / A_at_point

    weights = shape_weights_at_point(
        r_particle,
        radial_coordinate,
        dr,
        particle_shape,
        grid=grid,
    )

    cell_volume = radial_shell_volume(
        A_at_point,
        radial_coordinate,
        dr,
    )

    conformal_charge_density = jnp.sum(particles.get_charge() * weights)
    charge_density = conformal_charge_density / cell_volume

    return charge_density
