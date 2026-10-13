"""Constraint and invariant-mass checks for charged-star initial data."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.ConstraintBasedRelativity import build_radial_grid
from RadiShPICR.initial_data import (
    constrained_state_to_z4c,
    initialize_oppenheimer_snyder_particles,
    solve_charged_star_initial_data,
)
from RadiShPICR.initial_data.charged_star import _radial_laplacian
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative
from RadiShPICR.Z4C.electric_field import (
    compute_radial_charge_density,
    solve_radial_electric_field,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    compute_hamiltonian_constraint,
    compute_radial_matter_terms,
)


@pytest.fixture(autouse=True)
def use_double_precision():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def initial_guess(num_cells, charge_to_mass=0.2):
    grid = build_radial_grid(100.0, num_cells + 1)
    particles, _, _ = initialize_oppenheimer_snyder_particles(
        grid, 3.0 / (4.0 * np.pi * 10.0**3), 1.0, 10.0,
        12000, "quadratic",
    )
    particles.charges = jnp.asarray(charge_to_mass * np.sqrt(4.0 * np.pi))
    ones = jnp.ones_like(grid.r_full)
    alpha = 1.0 / (1.0 + 0.1 * jnp.exp(-grid.r_full**2 / 100.0))
    state = (ones, None, alpha, None, None, None, None, grid.r_full)
    r = 0.5 * (grid.r_full[:-1] + grid.r_full[1:])
    metric = constrained_state_to_z4c(state, r, kappa=0.5, eta=1.5, nu=0.01)

    return particles, metric


def test_sparse_laplacian_matches_evolution_parity_and_boundary_stencils():
    r = (jnp.arange(40) + 0.5) * 0.1
    field = jnp.exp(-r**2) + 0.02 * jnp.cos(3.0 * r)
    expected = (
        second_derivative(field, 0.1, parity=1)
        + 2.0 / r * first_derivative(field, 0.1, parity=1)
    )
    actual = _radial_laplacian(np.asarray(r), 0.1) @ np.asarray(field)

    assert np.allclose(actual, np.asarray(expected[:-1]), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("num_cells", (400, 800, 1600))
@pytest.mark.parametrize("charge_to_mass", (0.0, 0.2))
def test_charged_star_matches_production_sources_and_requested_rn_mass(num_cells, charge_to_mass):
    initial_particles, initial_metric = initial_guess(num_cells, charge_to_mass)
    metric, particles, E_r = solve_charged_star_initial_data(
        initial_particles, initial_metric, 1.0, tolerance=1e-11,
    )
    rho = compute_radial_matter_terms(particles, metric, inner_open=True).rho
    rho_EM = 0.5 * metric.chi * E_r**2
    H = compute_hamiltonian_constraint(metric) - 16.0 * jnp.pi * (rho + rho_EM)
    R = metric.r / jnp.sqrt(metric.chi)
    dR_dr = first_derivative(R, metric.dr, parity=-1)
    Q = float(jnp.sum(particles.get_charge()))
    q_squared = Q**2 / (4.0 * np.pi)
    mass = 0.5 * R * (1.0 - metric.chi * dR_dr**2) + q_squared / (2.0 * R)
    exterior = (R > 12.0) & (R < 90.0)
    RN_psi = jnp.sqrt(
        (1.0 + 1.0 / (2.0 * metric.r))**2
        - q_squared / (4.0 * metric.r**2)
    )

    # Check the actual evolution density, including finite-volume Gauss
    # data, rather than only the elliptic solver's algebraic residual.
    assert float(jnp.max(jnp.abs(H))) < (3e-7 if num_cells == 400 else 1e-7)
    assert float(jnp.max(jnp.abs(mass[exterior] - 1.0))) < 1e-6
    assert jnp.allclose(metric.chi[exterior]**-0.25, RN_psi[exterior], atol=1e-7)
    assert np.array_equal(particles.charges, initial_particles.charges)
    assert np.array_equal(particles.weight, initial_particles.weight)
    assert Q == pytest.approx(float(jnp.sum(initial_particles.get_charge())), abs=1e-14)
    assert 0.99 < float(jnp.sum(particles.get_mass()) / jnp.sum(initial_particles.get_mass())) < 1.01

    reconstructed_R = jnp.interp(
        particles.r, jnp.concatenate((jnp.zeros(1), metric.r)),
        jnp.concatenate((jnp.zeros(1), R)),
    )
    assert jnp.allclose(reconstructed_R, initial_particles.r, rtol=1e-12, atol=1e-12)
    charge_density = compute_radial_charge_density(particles, metric, inner_open=True)
    assert jnp.allclose(E_r, solve_radial_electric_field(metric, charge_density), atol=1e-15)
    assert np.array_equal(metric.alpha, initial_metric.alpha)
    for name in ("kappa", "eta", "nu"):
        assert getattr(metric, name) == getattr(initial_metric, name)
    for name in ("beta", "Kh", "Arr", "At", "theta", "Gamma"):
        assert jnp.all(getattr(metric, name) == 0.0)


def test_time_symmetric_solve_rejects_nonzero_particle_momentum():
    particles, metric = initial_guess(40)
    particles.ur = particles.ur.at[0].set(0.1)
    with pytest.raises(ValueError, match="zero particle momenta"):
        solve_charged_star_initial_data(particles, metric, 1.0)
