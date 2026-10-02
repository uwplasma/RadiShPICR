import jax
import jax.numpy as jnp
import pytest

import RadiShPICR.ConstraintBasedRelativity.evolve as evolve
from RadiShPICR.ConstraintBasedRelativity.solve_metric import calculate_metric
from RadiShPICR.particles import particle_species


def flat_metric(r):
    zero = jnp.zeros_like(r)
    one = jnp.ones_like(r)
    return one, zero, one, zero, zero, zero, (zero, zero, zero, zero), r


@pytest.mark.parametrize("shape_mode", ["nearest", "linear", "quadratic"])
@pytest.mark.parametrize("method", ["step", "step_rk4", "step_rk4_with_metric"])
def test_flat_crossings_follow_reflected_ballistic_trajectory(monkeypatch, shape_mode, method):
    r_grid = jnp.linspace(0.0, 2.0, 65)
    U_state = flat_metric(r_grid)
    monkeypatch.setattr(evolve, "calculate_metric", lambda *args, **kwargs: U_state)
    # Cross before, at, and after the half-step; include moving and stationary
    # particles at the center, and an ordinary outward-moving particle.
    rs0 = jnp.asarray([0.02, 0.075, 0.12, 0.0, 0.0, 0.6])
    ur0 = jnp.asarray([-0.75, -0.75, -0.75, -0.75, 0.0, 0.75])
    phi0 = jnp.linspace(0.1, 0.6, rs0.size)
    weight = jnp.linspace(0.2, 0.7, rs0.size)
    charges = jnp.asarray([1.0, -1.0, 0.0, 2.0, 0.0, -2.0])
    masses = jnp.linspace(1.0, 2.0, rs0.size)

    def advance(particles):
        if method == "step_rk4_with_metric":
            return evolve.step_rk4_with_metric(particles, U_state, r_grid, r_grid[1], 0.25)
        return getattr(evolve, method)(particles, r_grid, r_grid[1], 0.25), U_state

    compiled = jax.jit(advance)
    for advance_method in (advance, compiled):
        particles = particle_species(
            name="ballistic", charge=charges, mass=masses, weight=weight,
            r=rs0, ur=ur0, phi=phi0, uphi=jnp.zeros_like(rs0), shape_mode=shape_mode,
        )
        for step_number in (1, 2):
            previous_particles = particles
            particles, metric = advance_method(particles)
            if advance_method is advance:
                assert particles is previous_particles

            signed_radius = rs0 + step_number * 0.25 * ur0 / jnp.sqrt(1.0 + ur0**2)
            expected_ur = jnp.where(signed_radius < 0.0, -ur0, ur0)
            assert jnp.allclose(particles.r, jnp.abs(signed_radius), rtol=0.0, atol=1.0e-14)
            assert jnp.allclose(particles.ur, expected_ur, rtol=0.0, atol=1.0e-14)
            assert jnp.array_equal(particles.phi, phi0)
            assert jnp.array_equal(particles.uphi, jnp.zeros_like(rs0))
            assert jnp.array_equal(particles.weight, weight)
            assert jnp.array_equal(particles.get_mass(), masses * weight)
            assert jnp.array_equal(particles.get_charge(), charges * weight)
            for actual, expected in zip(jax.tree.leaves(metric), jax.tree.leaves(U_state)):
                assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize("method", ["step", "step_rk4", "step_rk4_with_metric"])
def test_exact_center_endpoint_has_outgoing_momentum(monkeypatch, method):
    r_grid = jnp.linspace(0.0, 2.0, 17)
    U_state = flat_metric(r_grid)
    monkeypatch.setattr(evolve, "calculate_metric", lambda *args, **kwargs: U_state)
    # Unit-speed free motion makes the exact-center endpoint representable.
    monkeypatch.setattr(evolve, "compute_geodesic_terms", lambda p, U, dur_dt_EM=None: (
        p.ur, jnp.zeros_like(p.phi), jnp.zeros_like(p.ur),
    ))
    particles = particle_species(
        name="center", charge=0.0, mass=1.0, weight=1.0,
        r=jnp.asarray([0.25]), ur=jnp.asarray([-1.0]),
        phi=jnp.asarray([0.3]), uphi=jnp.asarray([0.0]), shape_mode="nearest",
    )

    if method == "step_rk4_with_metric":
        particles, _ = evolve.step_rk4_with_metric(particles, U_state, r_grid, r_grid[1], 0.25)
    else:
        particles = getattr(evolve, method)(particles, r_grid, r_grid[1], 0.25)

    assert particles.r[0] == 0.0
    assert particles.ur[0] == 1.0
    assert particles.weight[0] == 1.0
    assert particles.phi[0] == 0.3


@pytest.mark.parametrize("cached_metric", [False, True])
def test_crossing_force_returns_to_signed_rk_coordinates(monkeypatch, cached_metric):
    r_grid = jnp.linspace(0.0, 2.0, 17)
    U_state = flat_metric(r_grid)
    metric_particles = []

    def metric_at_stage(particles, *args, **kwargs):
        assert jnp.all(particles.r >= 0.0)
        metric_particles.append(particles)
        return U_state

    def radial_force(particles, U_state, dur_dt_EM=None):
        assert jnp.all(particles.r >= 0.0)
        # Smooth signed oscillator: x' = p, p' = -x. Both radial RHS
        # components change orientation when a physical stage is reflected.
        return particles.ur, jnp.zeros_like(particles.phi), -particles.r

    monkeypatch.setattr(evolve, "calculate_metric", metric_at_stage)
    monkeypatch.setattr(evolve, "compute_geodesic_terms", radial_force)
    rs0, ur0, dt = 0.05, -1.0, 0.2
    particles = particle_species(
        name="oscillator", charge=0.0, mass=1.0, weight=1.0,
        r=jnp.asarray([rs0]), ur=jnp.asarray([ur0]),
        phi=jnp.asarray([0.3]), uphi=jnp.asarray([0.2]), shape_mode="nearest",
    )
    if cached_metric:
        particles, _ = evolve.step_rk4_with_metric(particles, U_state, r_grid, r_grid[1], dt)
        assert jnp.array_equal(metric_particles[-1].r, particles.r)
        assert jnp.array_equal(metric_particles[-1].ur, particles.ur)
    else:
        particles = evolve.step_rk4(particles, r_grid, r_grid[1], dt)

    cosine_rk4 = 1.0 - dt**2 / 2.0 + dt**4 / 24.0
    sine_rk4 = dt - dt**3 / 6.0
    signed_r = cosine_rk4 * rs0 + sine_rk4 * ur0
    signed_ur = -sine_rk4 * rs0 + cosine_rk4 * ur0
    assert jnp.allclose(particles.r, -signed_r, rtol=0.0, atol=1.0e-14)
    assert jnp.allclose(particles.ur, -signed_ur, rtol=0.0, atol=1.0e-14)
    assert jnp.array_equal(particles.uphi, jnp.asarray([0.2]))
    assert jnp.array_equal(particles.phi, jnp.asarray([0.3]))
    assert len(metric_particles) == 4


@pytest.mark.parametrize("shape_mode", ["nearest", "linear", "quadratic"])
def test_weak_self_gravitating_crossing_stays_finite(shape_mode):
    r_grid = jnp.linspace(0.0, 2.0, 65)
    dr = r_grid[1]
    results = []
    for advance in (evolve.step_rk4_with_metric, jax.jit(evolve.step_rk4_with_metric)):
        particles = particle_species(
            name="infall", charge=0.0, mass=1.0, weight=1.0e-9,
            r=jnp.asarray([0.05]), ur=jnp.asarray([-1.0]),
            phi=jnp.zeros(1), uphi=jnp.zeros(1), shape_mode=shape_mode,
        )
        U_state = calculate_metric(particles, r_grid, dr)
        particles, U_state = advance(particles, U_state, r_grid, dr, 0.2)
        for values in jax.tree.leaves((particles, U_state)):
            assert jnp.all(jnp.isfinite(values))
        assert particles.r[0] > 0.0
        assert particles.ur[0] > 0.0
        assert particles.weight[0] == 1.0e-9
        results.append((particles, U_state))

    for eager, compiled in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
        assert jnp.allclose(eager, compiled, rtol=1.0e-12, atol=1.0e-14)
