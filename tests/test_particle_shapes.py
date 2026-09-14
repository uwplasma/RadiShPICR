import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.charge_density import charge_density_at_point
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.mass_density import mass_density_at_point
from RadiShPICR.ConstraintBasedRelativity.utils import (
    angular_lorentz_term,
    radial_shell_volume,
)
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import (
    _cell_centered_open_inner_shape_stencil,
    _interpolate_cell_centered_fields_to_particles,
    interpolate_field_to_particles,
    interpolate_fields_to_particles,
    radial_shape_stencil,
    shape_weights_at_point,
    unbounded_radial_shape_stencil,
)
from RadiShPICR.ConstraintBasedRelativity.solve_metric import (
    _source_terms_at_point,
)
from RadiShPICR.ConstraintBasedRelativity.energy_momentum_tensor import (
    Srr_at_point,
    Sr_at_point,
)


def make_grid(r_max=8.0, dr=1.0):
    r_grid = jnp.arange(0.0, r_max + dr, dr)
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr,
        r_max=r_max,
    )


def test_radial_shell_volume_uses_spherical_cell_faces():
    A = jnp.asarray(1.4)
    dr = jnp.asarray(0.25)
    radial_coordinates = jnp.arange(4, dtype=A.dtype) * dr

    inner_radius = jnp.maximum(radial_coordinates - 0.5 * dr, 0.0)
    outer_radius = radial_coordinates + 0.5 * dr
    expected = A**3 * (4.0 * jnp.pi / 3.0) * (
        outer_radius**3 - inner_radius**3
    )

    actual = radial_shell_volume(A, radial_coordinates, dr)
    compiled = jax.jit(radial_shell_volume)(A, radial_coordinates, dr)

    assert jnp.allclose(actual, expected)
    assert jnp.allclose(compiled, expected)
    assert jnp.allclose(actual[0], A**3 * jnp.pi * dr**3 / 6.0)
    assert jnp.allclose(actual[1], A**3 * 13.0 * jnp.pi * dr**3 / 3.0)


def test_nearest_deposition_recovers_uniform_density_at_origin():
    grid = make_grid(r_max=6.0, dr=1.0)
    A = jnp.asarray(1.4)
    expected_density = jnp.asarray(2.5)
    particle_coordinates = grid.r_full[:5]
    particle_volumes = radial_shell_volume(
        A,
        particle_coordinates,
        grid.dr,
    )

    particles = particle_species(
        name="uniform",
        charge=0.0,
        mass=1.0,
        weight=expected_density * particle_volumes,
        # Constrained relativity stores r_s = A r in the particle r field.
        r=A * particle_coordinates,
        ur=jnp.zeros_like(particle_coordinates),
        phi=jnp.zeros_like(particle_coordinates),
        uphi=jnp.zeros_like(particle_coordinates),
        shape_mode="nearest",
    )

    mass_density = jax.vmap(
        lambda radial_coordinate: mass_density_at_point(
            particles,
            A,
            radial_coordinate,
            grid,
        )
    )(particle_coordinates)

    assert jnp.allclose(mass_density, expected_density)


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


def test_batched_field_interpolation_matches_individual_fields():
    grid = make_grid()
    radial_positions = jnp.asarray([0.25, 1.25, 3.5, 7.75])
    fields = jnp.stack(
        (
            2.0 + 3.0 * grid.r_full,
            1.0 + grid.r_full**2,
            jnp.sin(grid.r_full),
        )
    )

    for shape_mode in ("nearest", "linear", "quadratic"):
        expected = jnp.stack(
            tuple(
                interpolate_field_to_particles(
                    field,
                    radial_positions,
                    grid,
                    shape_mode=shape_mode,
                )
                for field in fields
            )
        )
        actual = interpolate_fields_to_particles(
            fields,
            radial_positions,
            grid,
            shape_mode=shape_mode,
        )

        assert jnp.allclose(actual, expected)


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
    outer_boundary_index = grid.r_full.shape[0] - 1
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
        assert jnp.allclose(pointwise_weights[outer_boundary_index], 0.0)
        assert jnp.allclose(jnp.sum(pointwise_weights, axis=0), 1.0)


def test_origin_stencil_folds_even_and_odd_quadratic_weights_by_parity():
    grid = make_grid()
    radial_positions = jnp.asarray([0.0, 0.25, -0.25])

    def assembled_stencil(parity):
        indices, weights = radial_shape_stencil(
            radial_positions,
            grid,
            shape_mode="quadratic",
            parity=parity,
        )
        particle_columns = jnp.broadcast_to(
            jnp.arange(radial_positions.shape[0])[jnp.newaxis, :],
            indices.shape,
        )
        assembled = jnp.zeros((grid.r_full.shape[0], radial_positions.shape[0]))
        return assembled.at[indices, particle_columns].add(weights)

    even_weights = assembled_stencil(parity=1)
    odd_weights = assembled_stencil(parity=-1)

    assert jnp.allclose(
        even_weights[:2],
        jnp.asarray(
            [
                [0.75, 0.6875, 0.6875],
                [0.25, 0.3125, 0.3125],
            ]
        ),
    )
    assert jnp.allclose(
        odd_weights[:2],
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.25, -0.25],
            ]
        ),
    )


def test_origin_parity_interpolation_matches_even_and_odd_fields():
    grid = make_grid()
    radial_positions = jnp.asarray([-0.25, 0.0, 0.25])

    for shape_mode in ("nearest", "linear", "quadratic"):
        even_values = interpolate_field_to_particles(
            grid.r_full**2,
            radial_positions,
            grid,
            shape_mode=shape_mode,
            parity=1,
        )
        odd_values = interpolate_field_to_particles(
            grid.r_full,
            radial_positions,
            grid,
            shape_mode=shape_mode,
            parity=-1,
        )

        assert jnp.allclose(even_values[0], even_values[2])
        assert jnp.allclose(odd_values[0], -odd_values[2])
        assert odd_values[1] == 0.0


def test_cell_centered_interpolation_respects_origin_parity_and_jit():
    radial_grid = 0.5 + jnp.arange(8.0)
    grid = RadialGrid(
        r_full=radial_grid,
        r_interior=radial_grid,
        dr=1.0,
        r_max=radial_grid[-1],
    )
    radial_positions = jnp.asarray(
        [-1.25, -0.49, -0.1, 0.0, 0.1, 0.49, 1.25]
    )
    fields = jnp.stack((jnp.ones_like(radial_grid), radial_grid))
    parities = jnp.asarray((1, -1))

    for shape_mode in ("nearest", "linear", "quadratic"):
        eager = _interpolate_cell_centered_fields_to_particles(
            fields,
            radial_positions,
            grid,
            shape_mode=shape_mode,
            field_parities=parities,
        )
        compiled = jax.jit(
            _interpolate_cell_centered_fields_to_particles,
            static_argnames=("shape_mode",),
        )(
            fields,
            radial_positions,
            grid,
            shape_mode=shape_mode,
            field_parities=parities,
        )

        assert jnp.allclose(eager[0], 1.0)
        assert jnp.allclose(eager[1], radial_positions)
        assert eager[1, 3] == 0.0
        assert jnp.allclose(compiled, eager)


def test_cell_centered_interpolation_preserves_non_origin_gather():
    radial_grid = 0.5 + jnp.arange(8.0)
    grid = RadialGrid(
        r_full=radial_grid,
        r_interior=radial_grid,
        dr=1.0,
        r_max=radial_grid[-1],
    )
    radial_positions = jnp.asarray([2.25, 4.75, 7.25])
    fields = jnp.stack(
        (
            1.0 + 0.2 * radial_grid,
            jnp.sin(radial_grid),
        )
    )

    for shape_mode in ("nearest", "linear", "quadratic"):
        expected = interpolate_fields_to_particles(
            fields,
            radial_positions,
            grid,
            shape_mode=shape_mode,
        )
        actual = _interpolate_cell_centered_fields_to_particles(
            fields,
            radial_positions,
            grid,
            shape_mode=shape_mode,
            field_parities=jnp.asarray((1, -1)),
        )

        assert jnp.allclose(actual, expected)


def test_open_inner_quadratic_shape_loses_physical_overlap_without_renormalizing():
    radial_grid = 0.5 + jnp.arange(6.0)
    radial_positions = jnp.asarray(
        [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
    )

    indices, even_weights, odd_weights = (
        _cell_centered_open_inner_shape_stencil(
            radial_positions,
            radial_grid,
            dr=1.0,
            shape_mode="quadratic",
            inner_boundary_index=jnp.asarray(0),
        )
    )
    physical_overlap = jnp.sum(even_weights, axis=0)

    assert indices.shape == even_weights.shape
    assert jnp.allclose(odd_weights, even_weights)
    assert jnp.allclose(
        physical_overlap,
        jnp.asarray(
            [1.0, 0.96875, 0.875, 0.71875, 0.5, 0.28125, 0.125, 0.03125, 0.0]
        ),
    )
    assert jnp.all(jnp.diff(physical_overlap) <= 0.0)


def test_open_inner_shape_tracks_a_nonzero_boundary_index():
    radial_grid = 0.5 + jnp.arange(6.0)
    radial_positions = jnp.asarray([1.5, 2.0, 2.5, 3.0])

    indices, weights, _ = _cell_centered_open_inner_shape_stencil(
        radial_positions,
        radial_grid,
        dr=1.0,
        shape_mode="quadratic",
        inner_boundary_index=jnp.asarray(2),
    )

    particle_columns = jnp.broadcast_to(
        jnp.arange(radial_positions.size)[jnp.newaxis, :],
        indices.shape,
    )
    deposited = jnp.zeros((radial_grid.size, radial_positions.size))
    deposited = deposited.at[indices, particle_columns].add(weights)

    assert jnp.allclose(deposited[:2], 0.0)
    assert jnp.allclose(
        jnp.sum(deposited, axis=0),
        jnp.asarray([0.125, 0.5, 0.875, 1.0]),
    )


def test_unbounded_compact_stencil_matches_pointwise_shape_weights():
    radial_grid = jnp.arange(0.5, 6.0, 1.0)
    radial_positions = jnp.asarray([-0.25, 0.5, 1.0, 2.25, 5.5, 6.25])

    for shape_mode in ("nearest", "linear", "quadratic"):
        indices, stencil_weights = unbounded_radial_shape_stencil(
            radial_positions,
            radial_grid,
            dr=1.0,
            shape_mode=shape_mode,
        )
        particle_columns = jnp.broadcast_to(
            jnp.arange(radial_positions.shape[0])[jnp.newaxis, :],
            indices.shape,
        )
        compact_weights = jnp.zeros(
            (radial_grid.shape[0], radial_positions.shape[0])
        )
        compact_weights = compact_weights.at[indices, particle_columns].add(
            stencil_weights
        )
        pointwise_weights = shape_weights_at_point(
            radial_positions[jnp.newaxis, :],
            radial_grid[:, jnp.newaxis],
            1.0,
            shape_mode=shape_mode,
        )

        assert jnp.allclose(compact_weights, pointwise_weights)


def test_fused_constraint_sources_match_individual_source_functions():
    grid = make_grid()
    radial_coordinates = grid.r_full
    A = 1.0 + 0.03 * radial_coordinates

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = particle_species(
            name="moving",
            charge=3.0,
            mass=2.0,
            weight=jnp.asarray([0.2, 0.3, 0.5]),
            r=jnp.asarray([0.25, 3.25, 7.75]),
            ur=jnp.asarray([0.4, -0.2, 0.7]),
            phi=jnp.zeros(3),
            uphi=jnp.zeros(3),
            shape_mode=shape_mode,
        )

        for A_at_point, radial_coordinate in zip(A, radial_coordinates):
            fused = _source_terms_at_point(
                particles,
                A_at_point,
                radial_coordinate,
                grid,
            )
            expected = (
                mass_density_at_point(
                    particles,
                    A_at_point,
                    radial_coordinate,
                    grid,
                ),
                charge_density_at_point(
                    particles,
                    A_at_point,
                    radial_coordinate,
                    grid,
                ),
                Srr_at_point(
                    particles,
                    A_at_point,
                    radial_coordinate,
                    grid,
                ),
                Sr_at_point(
                    particles,
                    A_at_point,
                    radial_coordinate,
                    grid,
                ),
            )

            for fused_source, expected_source in zip(fused, expected):
                assert jnp.allclose(fused_source, expected_source)


def test_constraint_sources_convert_rs_and_ur_over_A_with_local_A():
    grid = make_grid()
    A_at_point = jnp.asarray(2.0)
    radial_coordinate = jnp.asarray(2.0)
    floating_index = radial_coordinate / grid.dr

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = particle_species(
            name="moving",
            charge=3.0,
            mass=2.0,
            weight=jnp.asarray([0.25, 0.75]),
            r=jnp.asarray([3.5, 4.5]),
            ur=jnp.asarray([0.4, -0.2]),
            phi=jnp.zeros(2),
            uphi=jnp.asarray([0.3, -0.1]),
            shape_mode=shape_mode,
        )

        r_particle = particles.r / A_at_point
        covariant_ur = A_at_point * particles.ur
        even_indices, even_stencil = radial_shape_stencil(
            r_particle,
            grid,
            shape_mode=shape_mode,
            parity=1,
        )
        odd_indices, odd_stencil = radial_shape_stencil(
            r_particle,
            grid,
            shape_mode=shape_mode,
            parity=-1,
        )
        grid_index = jnp.rint(floating_index).astype(even_indices.dtype)
        even_weights = jnp.sum(
            jnp.where(even_indices == grid_index, even_stencil, 0.0),
            axis=0,
        )
        odd_weights = jnp.sum(
            jnp.where(odd_indices == grid_index, odd_stencil, 0.0),
            axis=0,
        )

        W = jnp.sqrt(
            1.0
            + covariant_ur**2 / A_at_point**2
            + angular_lorentz_term(
                particles.uphi,
                A_at_point,
                r_particle,
            )
        )
        cell_volume = radial_shell_volume(
            A_at_point,
            radial_coordinate,
            grid.dr,
        )
        weighted_mass = particles.get_mass() * even_weights
        expected = (
            jnp.sum(weighted_mass * W) / cell_volume,
            jnp.sum(particles.get_charge() * even_weights) / cell_volume,
            jnp.sum(weighted_mass * covariant_ur**2 / W) / cell_volume,
            jnp.sum(particles.get_mass() * odd_weights * covariant_ur)
            / cell_volume,
        )

        actual = _source_terms_at_point(
            particles,
            A_at_point,
            radial_coordinate,
            grid,
        )
        compiled = jax.jit(_source_terms_at_point)(
            particles,
            A_at_point,
            radial_coordinate,
            grid,
        )

        for actual_source, compiled_source, expected_source in zip(
            actual,
            compiled,
            expected,
        ):
            assert jnp.allclose(actual_source, expected_source)
            assert jnp.allclose(compiled_source, expected_source)


def test_nonzero_uphi_source_keeps_origin_singularity_visible():
    grid = make_grid()
    particles = particle_species(
        name="angular",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([0.0]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([1.0]),
        shape_mode="quadratic",
    )

    mass_density, _, Srr, _ = _source_terms_at_point(
        particles,
        jnp.asarray(1.0),
        grid.r_full[0],
        grid,
    )

    assert not jnp.isfinite(mass_density)
    assert jnp.isfinite(Srr)


def test_boundary_source_deposition_conserves_particle_mass_and_charge():
    grid = make_grid()
    outer_boundary_index = grid.r_full.shape[0] - 1
    cell_volume = jax.vmap(
        lambda r: radial_shell_volume(jnp.asarray(1.0), r, grid.dr)
    )(grid.r_full)

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = particle_species(
            name="test",
            charge=3.0,
            mass=2.0,
            weight=jnp.asarray([0.25, 0.75]),
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

        assert mass_density[0] > 0.0
        assert charge_density[0] > 0.0
        assert mass_density[outer_boundary_index] == 0.0
        assert charge_density[outer_boundary_index] == 0.0
        assert jnp.allclose(
            deposited_mass,
            jnp.sum(particles.get_mass()),
        )
        assert jnp.allclose(
            deposited_charge,
            jnp.sum(particles.get_charge()),
        )


def test_charge_density_is_independent_of_particle_momentum():
    grid = make_grid()
    A = jnp.full_like(grid.r_full, 1.4)
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
        expected_charge = jnp.sum(moving_particles.get_charge())

        assert jnp.allclose(moving_charge_density, stationary_charge_density)
        assert jnp.allclose(jitted_charge_density, moving_charge_density)
        assert jnp.allclose(deposited_charge, expected_charge)
        assert moving_charge_density[0] > 0.0
        assert moving_charge_density[-1] == 0.0


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
        assert jnp.all(indices >= 0)
        assert jnp.all(indices <= grid.r_full.shape[0] - 2)
        assert jnp.allclose(jnp.sum(stencil_weights, axis=0), 1.0)
