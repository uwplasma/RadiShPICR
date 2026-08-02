import jax.numpy as jnp

from RadiShPICR.particles.particle_shapes import shape_weights_at_point
from RadiShPICR.ConstraintBasedRelativity.utils import (
    angular_lorentz_term,
    radial_shell_volume,
)


def Sr_at_point(
    particles,
    A_at_point,
    radial_coordinate,
    grid,
    shape_mode=None,
):
    r_particle, _ = particles.get_positions()
    ur, _ = particles.get_velocities()
    particle_shape = particles.get_shape() if shape_mode is None else shape_mode
    dr = grid.dr

    weights = shape_weights_at_point(
        r_particle,
        radial_coordinate,
        dr,
        particle_shape,
        grid=grid,
        parity=-1,
    )

    cell_volume = radial_shell_volume(
        A_at_point,
        radial_coordinate,
        dr,
    )

    conformal_Sr = jnp.sum(particles.get_mass() * weights * ur)
    Sr = conformal_Sr / cell_volume

    return Sr


def Srr_at_point(
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

    lorentz_factor = jnp.sqrt(
        1.0
        + ur**2 / A_at_point**2
        + angular_lorentz_term(uphi, A_at_point, radial_coordinate)
    )
    cell_volume = radial_shell_volume(
        A_at_point,
        radial_coordinate,
        dr,
    )

    conformal_Srr = jnp.sum(particles.get_mass() * weights * ur**2 / lorentz_factor)
    Srr = conformal_Srr / cell_volume

    return Srr
