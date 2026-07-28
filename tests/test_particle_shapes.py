import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.charge_density import charge_density_at_point
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.mass_density import mass_density_at_point
from RadiShPICR.ConstraintBasedRelativity.utils import radial_shell_volume
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import (
    interpolate_field_to_particles,
    radial_shape_stencil,
    shape_weights_at_point,
)


def make_grid(r_max=8.0, dr=1.0):
    r_grid = jnp.arange(0.0, r_max + dr, dr)
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr,
        r_max=r_max,
    )


def test_linear_shape_has_expected_cic_weights():
    radial_grid_points = jnp.asarray([3.0, 4.0, 5.0])
    particle_positions = (
        jnp.asarray([4.0]),
        jnp.asarray([4.25]),
        jnp.asarray([4.5]),
    )
    expected_weights = (
        jnp.asarray([0.0, 1.0, 0.0]),
        jnp.asarray([0.0, 0.75, 0.25]),
        jnp.asarray([0.0, 0.5, 0.5]),
    )

    for radial_position, expected in zip(particle_positions, expected_weights):
        weights = shape_weights_at_point(
            radial_position,
            radial_grid_points[:, jnp.newaxis],
            dr=1.0,
            shape_mode="linear",
        )

        assert jnp.allclose(weights[:, 0], expected)
        assert jnp.all(weights >= 0.0)
        assert jnp.allclose(jnp.sum(weights), 1.0)


def test_linear_shape_exactly_interpolates_an_affine_field():
    grid = make_grid()
    radial_positions = jnp.asarray([1.25, 3.5, 6.75])
    field = 2.0 + 3.0 * grid.r_full

    interpolated_field = interpolate_field_to_particles(
        field,
        radial_positions,
        grid,
        shape_mode="linear",
    )

    assert jnp.allclose(interpolated_field, 2.0 + 3.0 * radial_positions)


def test_quadratic_shape_has_expected_tsc_weights():
    radial_grid_points = jnp.asarray([3.0, 4.0, 5.0])
    particle_positions = (
        jnp.asarray([4.0]),
        jnp.asarray([4.25]),
        jnp.asarray([4.5]),
    )
    expected_weights = (
        jnp.asarray([0.125, 0.75, 0.125]),
        jnp.asarray([0.03125, 0.6875, 0.28125]),
        jnp.asarray([0.0, 0.5, 0.5]),
    )

    for radial_position, expected in zip(particle_positions, expected_weights):
        weights = shape_weights_at_point(
            radial_position,
            radial_grid_points[:, jnp.newaxis],
            dr=1.0,
            shape_mode="quadratic",
        )

        assert jnp.allclose(weights[:, 0], expected)
        assert jnp.all(weights >= 0.0)
        assert jnp.allclose(jnp.sum(weights), 1.0)


def test_grid_aware_pointwise_weights_match_indexed_stencil():
    grid = make_grid()
    boundary_indices = jnp.asarray([0, grid.r_full.shape[0] - 1])
    radial_positions = jnp.asarray(
        [-0.75, 0.0, 0.25, 0.5, 0.75, 3.25, 7.25, 7.5, 7.75, 8.0, 8.75]
    )

    for shape_mode in ("nearest", "linear", "quadratic"):
        indices, stencil_weights = radial_shape_stencil(
            radial_positions,
            grid,
            shape_mode=shape_mode,
        )
        particle_columns = jnp.broadcast_to(
            jnp.arange(radial_positions.shape[0])[jnp.newaxis, :],
            indices.shape,
        )
        indexed_weights = jnp.zeros(
            (grid.r_full.shape[0], radial_positions.shape[0])
        )
        indexed_weights = indexed_weights.at[indices, particle_columns].add(
            stencil_weights
        )

        pointwise_weights = jax.vmap(
            lambda radial_coordinate: shape_weights_at_point(
                radial_positions,
                radial_coordinate,
                grid.dr,
                shape_mode=shape_mode,
                grid=grid,
            )
        )(grid.r_full)

        assert jnp.allclose(pointwise_weights, indexed_weights)
        assert jnp.allclose(pointwise_weights[boundary_indices], 0.0)
        assert jnp.allclose(jnp.sum(pointwise_weights, axis=0), 1.0)


def test_boundary_source_deposition_conserves_particle_mass_and_charge():
    grid = make_grid()
    boundary_indices = jnp.asarray([0, grid.r_full.shape[0] - 1])
    cell_volume = jax.vmap(
        lambda r: radial_shell_volume(jnp.asarray(1.0), r, grid.dr)
    )(grid.r_full)

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = particle_species(
            name="test",
            charge=3.0,
            mass=2.0,
            weight=0.25,
            r=jnp.asarray([0.25, 7.75]),
            ur=jnp.asarray([0.0, 0.0]),
            phi=jnp.asarray([0.0, 0.0]),
            uphi=jnp.asarray([0.0, 0.0]),
            shape_mode=shape_mode,
        )

        mass_density = jax.vmap(
            lambda r: mass_density_at_point(
                particles,
                jnp.asarray(1.0),
                r,
                grid,
            )
        )(grid.r_full)
        charge_density = jax.vmap(
            lambda r: charge_density_at_point(
                particles,
                jnp.asarray(1.0),
                r,
                grid,
            )
        )(grid.r_full)

        deposited_mass = jnp.sum(mass_density * cell_volume)
        deposited_charge = jnp.sum(charge_density * cell_volume)
        particle_count = particles.r.shape[0]

        assert jnp.allclose(mass_density[boundary_indices], 0.0)
        assert jnp.allclose(charge_density[boundary_indices], 0.0)
        assert jnp.allclose(
            deposited_mass,
            particle_count * particles.get_mass(),
        )
        assert jnp.allclose(
            deposited_charge,
            particle_count * particles.get_charge(),
        )


def test_charge_density_is_independent_of_particle_momentum():
    grid = make_grid()
    A = 1.0 + 0.05 * grid.r_full
    cell_volume = jax.vmap(radial_shell_volume, in_axes=(0, 0, None))(
        A,
        grid.r_full,
        grid.dr,
    )

    for shape_mode in ("nearest", "linear", "quadratic"):
        stationary_particles = particle_species(
            name="stationary",
            charge=3.0,
            mass=2.0,
            weight=0.25,
            r=jnp.asarray([0.25, 3.25, 7.75]),
            ur=jnp.zeros(3),
            phi=jnp.zeros(3),
            uphi=jnp.zeros(3),
            shape_mode=shape_mode,
        )
        moving_particles = particle_species(
            name="moving",
            charge=3.0,
            mass=2.0,
            weight=0.25,
            r=stationary_particles.r,
            ur=jnp.asarray([1.2, -0.7, 0.9]),
            phi=stationary_particles.phi,
            uphi=jnp.asarray([0.4, 1.1, -0.5]),
            shape_mode=shape_mode,
        )

        def deposit_charge(particles):
            return jax.vmap(
                lambda A_at_point, radial_coordinate: charge_density_at_point(
                    particles,
                    A_at_point,
                    radial_coordinate,
                    grid,
                )
            )(A, grid.r_full)

        stationary_charge_density = deposit_charge(stationary_particles)
        moving_charge_density = deposit_charge(moving_particles)
        jitted_charge_density = jax.jit(deposit_charge)(moving_particles)

        deposited_charge = jnp.sum(moving_charge_density * cell_volume)
        expected_charge = moving_particles.r.shape[0] * moving_particles.get_charge()

        assert jnp.allclose(moving_charge_density, stationary_charge_density)
        assert jnp.allclose(jitted_charge_density, moving_charge_density)
        assert jnp.allclose(deposited_charge, expected_charge)
        assert jnp.allclose(
            moving_charge_density[jnp.asarray([0, -1])],
            0.0,
        )


def test_shape_jit_matches_eager_and_preserves_boundary_stencil():
    grid = make_grid()
    radial_positions = jnp.asarray([0.25, 4.25, 7.75])
    radial_coordinate = jnp.asarray(4.0)

    jitted_shape_weights = jax.jit(
        shape_weights_at_point,
        static_argnames=("shape_mode",),
    )

    for shape_mode in ("linear", "quadratic"):
        eager_weights = shape_weights_at_point(
            radial_positions,
            radial_coordinate,
            grid.dr,
            shape_mode=shape_mode,
            grid=grid,
        )
        jitted_weights = jitted_shape_weights(
            radial_positions,
            radial_coordinate,
            grid.dr,
            shape_mode=shape_mode,
            grid=grid,
        )
        indices, stencil_weights = radial_shape_stencil(
            radial_positions,
            grid,
            shape_mode=shape_mode,
        )

        assert jnp.allclose(jitted_weights, eager_weights)
        assert jnp.all(indices >= 1)
        assert jnp.all(indices <= grid.r_full.shape[0] - 2)
        assert jnp.allclose(jnp.sum(stencil_weights, axis=0), 1.0)
