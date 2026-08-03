import jax
import jax.numpy as jnp

import RadiShPICR.ConstraintBasedRelativity.solve_metric as solve_metric
from RadiShPICR.particles import particle_species


def test_calculate_metric_uses_rescaled_origin_for_second_heun_shot(monkeypatch):
    particles = particle_species(
        name="charged",
        charge=1.5,
        mass=2.0,
        weight=jnp.asarray([0.25, 0.50, 0.75]),
        r=jnp.asarray([0.25, 0.50, 0.75]),
        ur=jnp.zeros(3),
        phi=jnp.zeros(3),
        uphi=jnp.zeros(3),
        shape_mode="nearest",
    )
    r_grid = jnp.asarray([0.0, 1.0, 2.0])
    shot_centers = []
    rescale_arguments = []

    def fake_heun_shot(particles, grid, center_A, center_alpha):
        shot_centers.append((center_A, center_alpha))

        A = jnp.asarray([center_A, 1.4, 1.6])
        alpha = jnp.asarray([center_alpha, 0.75, 0.5])
        zeros = jnp.zeros_like(grid.r_full)
        source_terms = (zeros, zeros, zeros, zeros)

        return (
            A,
            zeros,
            alpha,
            zeros,
            zeros,
            zeros,
            source_terms,
            grid.r_full,
        )

    def fake_rescale(A_outer, alpha_outer, r_outer, mass, charge):
        rescale_arguments.append(
            (A_outer, alpha_outer, r_outer, mass, charge)
        )
        return jnp.asarray(2.0), jnp.asarray(4.0)

    monkeypatch.setattr(
        solve_metric,
        "_integrate_metric_from_origin",
        fake_heun_shot,
    )
    monkeypatch.setattr(
        solve_metric,
        "vacuum_rescale_factors",
        fake_rescale,
    )

    U_state = solve_metric.calculate_metric(
        particles,
        r_grid,
        r_grid[1] - r_grid[0],
    )

    assert len(shot_centers) == 2
    assert jnp.allclose(shot_centers[0][0], 1.0)
    assert jnp.allclose(shot_centers[0][1], 1.0)
    assert jnp.allclose(shot_centers[1][0], 0.5)
    assert jnp.allclose(shot_centers[1][1], 0.25)

    assert len(rescale_arguments) == 1
    A_outer, alpha_outer, r_outer, mass, charge = rescale_arguments[0]
    assert jnp.allclose(A_outer, 1.6)
    assert jnp.allclose(alpha_outer, 0.5)
    assert jnp.allclose(r_outer, 2.0)
    assert jnp.allclose(mass, 3.0)
    assert jnp.allclose(charge, 2.25)

    assert jnp.allclose(U_state[0][0], 0.5)
    assert jnp.allclose(U_state[2][0], 0.25)


def test_charged_two_shot_metric_is_finite_and_jit_compatible():
    particles = particle_species(
        name="charged",
        charge=0.5,
        mass=1.0,
        weight=jnp.asarray([0.01, 0.01, 0.01]),
        r=jnp.asarray([0.5, 1.0, 1.5]),
        ur=jnp.zeros(3),
        phi=jnp.zeros(3),
        uphi=jnp.zeros(3),
        shape_mode="nearest",
    )
    r_grid = jnp.linspace(0.0, 4.0, 17)
    dr = r_grid[1] - r_grid[0]

    eager_U_state = solve_metric.calculate_metric(particles, r_grid, dr)
    jitted_U_state = jax.jit(solve_metric.calculate_metric)(
        particles,
        r_grid,
        dr,
    )

    for eager_values, jitted_values in zip(
        jax.tree_util.tree_leaves(eager_U_state),
        jax.tree_util.tree_leaves(jitted_U_state),
    ):
        assert jnp.all(jnp.isfinite(eager_values))
        assert jnp.allclose(jitted_values, eager_values)

    assert not jnp.allclose(eager_U_state[0][0], 1.0)
    assert not jnp.allclose(eager_U_state[2][0], 1.0)
