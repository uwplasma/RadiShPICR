"""Conservation and nonlinear-solve regressions for constrained particle maps."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.ConstraintBasedRelativity import build_radial_grid, calculate_metric
from RadiShPICR.ConstraintBasedRelativity.particle_sources import (
    deposit_source_numerators, particle_coordinate_radius, source_terms_at_point,
)
from RadiShPICR.ConstraintBasedRelativity.utils import radial_shell_volume
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors.cartesian_shapes import interpolate_fields_to_particles
from RadiShPICR.initial_data import initialize_oppenheimer_snyder_particles, solve_constrained_star_initial_data


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', previous)


@pytest.mark.parametrize('shape', ('nearest', 'linear', 'quadratic'))
@pytest.mark.parametrize('charge', (0., .3))
def test_completed_variable_map_conserves_source_moments(shape, charge):
    grid = build_radial_grid(10., 101)
    A = 1. + .15 * grid.r_full
    r = jnp.array([.025, 2.37, 6.41, 9.96])
    R = jnp.interp(r, grid.r_full, grid.r_full * A)
    p = particle_species('moving', charge, 2., jnp.array([.2, .3, .1, .4]),
                         R, jnp.array([0., .2, -.3, .1]), jnp.zeros(4),
                         jnp.array([0., .4, .2, .7]), shape)
    assert jnp.allclose(particle_coordinate_radius(p, A, grid.r_full), r, atol=1e-14)
    numerators = deposit_source_numerators(p, A, grid)
    compiled = jax.jit(deposit_source_numerators)(p, A, grid)
    A_p = interpolate_fields_to_particles(A[None, :], r, grid, shape_mode=shape)[0]
    ur = A_p * p.ur
    W = jnp.sqrt(1. + p.ur**2 + p.uphi**2 / R**2)
    expected = (jnp.sum(p.get_mass() * W), jnp.sum(p.get_charge()),
                jnp.sum(p.get_mass() * ur**2 / W), jnp.sum(p.get_mass() * ur))
    density = jax.vmap(lambda a, x: source_terms_at_point(p, a, x, grid, A_values=A))(A, grid.r_full)
    volume = radial_shell_volume(A, grid.r_full, grid.dr)
    for values, jit_values, rho, integral in zip(numerators, compiled, density, expected):
        assert jnp.allclose(values, jit_values, atol=1e-14)
        assert jnp.sum(values) == pytest.approx(float(integral), abs=1e-13)
        assert jnp.sum(rho * volume) == pytest.approx(float(integral), abs=1e-13)
        assert values[-1] == 0.  # Existing vacuum endpoint and renormalization.
    assert numerators[3][0] == 0.  # Odd momentum parity.


def test_point_sources_require_completed_metric():
    with pytest.raises(TypeError, match='A_values'):
        source_terms_at_point(None, 1., 0., None)


def test_folded_map_is_rejected():
    grid = build_radial_grid(2., 5)
    p = particle_species('test', 0., 1., 1., jnp.array([.5]), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), 'nearest')
    A = jnp.array([1., 1., .1, .1, .1])
    assert jnp.all(jnp.isnan(particle_coordinate_radius(p, A, grid.r_full)))


def test_nearest_discontinuous_nonconverged_shot_is_rejected():
    # The second particle alternates across a cell face under this iteration.
    # Reject the unconverged coarse source instead of accepting the final sweep.
    p = particle_species('test', 0., 1., jnp.array([.5, 1.5]), jnp.array([2.5, 7.5]),
                         jnp.array([.01, -.01]), jnp.zeros(2), jnp.zeros(2), 'nearest')
    grid = build_radial_grid(10., 9)
    state = jax.jit(calculate_metric)(p, grid.r_full, grid.dr)
    assert all(jnp.all(jnp.isnan(x)) for x in jax.tree.leaves(state))


@pytest.mark.parametrize('charge', (0., .2))
def test_constrained_star_calibrates_mass_without_changing_charge_or_positions(charge):
    grid = build_radial_grid(30., 300)
    p, _, _ = initialize_oppenheimer_snyder_particles(
        grid, 3./(4*jnp.pi*10**3), 1., 10., 3000, 'quadratic',
    )
    p.charges = jnp.asarray(charge)
    state, calibrated = solve_constrained_star_initial_data(p, grid, 1.)
    A, phi, _, _, _, _, _, r = state
    Q = float(jnp.sum(p.get_charge()))
    R = A * r
    F = (1. + 2*r*phi/jnp.sqrt(A))**2
    mass = R[-1]*(1-F[-1])/2 + Q**2/(8*jnp.pi*R[-1])
    assert float(mass) == pytest.approx(1., abs=1e-9)
    for name in ('r', 'weight', 'charges', 'ur', 'uphi'):
        assert np.array_equal(getattr(p, name), getattr(calibrated, name))


def test_empty_species_has_finite_vacuum_metric():
    empty = jnp.empty(0)
    p = particle_species('empty', 0., 1., empty, empty, empty, empty, empty, 'quadratic')
    grid = build_radial_grid(2., 17)
    state = jax.jit(calculate_metric)(p, grid.r_full, grid.dr)
    assert jnp.allclose(state[0], 1.)
    assert jnp.allclose(state[2], 1.)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(state))
    assert all(jnp.all(x == 0.) for x in state[6])
