import jax
import jax.numpy as jnp

import RadiShPICR.ConstraintBasedRelativity.evolve as constraint_evolve
from RadiShPICR.ConstraintBasedRelativity.charge_density import charge_density_at_point
from RadiShPICR.ConstraintBasedRelativity.mass_density import mass_density_at_point
from RadiShPICR.ConstraintBasedRelativity.evolve import (
    step,
    step_rk4,
    step_rk4_with_metric,
)
from RadiShPICR.particles.particle_shapes import interpolate_field_to_particles
from RadiShPICR.ConstraintBasedRelativity import build_radial_grid, calculate_metric
from RadiShPICR.ConstraintBasedRelativity.energy_momentum_tensor import Srr_at_point, Sr_at_point
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.geodesic import compute_geodesic_terms
from RadiShPICR.ConstraintBasedRelativity.lorentz_force import compute_lorentz_terms
from RadiShPICR.ConstraintBasedRelativity.solve_metric import (
    beta_over_r_from_integral,
    dr_alpha,
    dr_Er,
    dr_sqrt_phi,
)
from RadiShPICR.ConstraintBasedRelativity.utils import pad_value
from RadiShPICR.particles import particle_species


def make_species(charge=0.0, mass=1.0, weight=1.0):
    return particle_species(
        name="test",
        charge=charge,
        mass=mass,
        weight=weight,
        r=jnp.asarray([0.25, 0.75]),
        ur=jnp.asarray([0.10, -0.20]),
        phi=jnp.asarray([0.0, 0.2]),
        uphi=jnp.asarray([0.0, 0.0]),
        shape_mode="nearest",
    )


def make_interpolation_grid(r_grid):
    dr = r_grid[1] - r_grid[0]
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr,
        r_max=r_grid[-1],
    )


def test_build_radial_grid_has_no_public_epsilon_parameter():
    grid = build_radial_grid(r_max=2.0, num_interior_points=5)

    assert grid._fields == ("r_full", "r_interior", "dr", "r_max")
    assert jnp.allclose(grid.r_full, jnp.linspace(0.0, 2.0, 5))
    assert jnp.allclose(grid.r_interior, grid.r_full)
    assert jnp.allclose(grid.dr, 0.5)
    assert jnp.allclose(grid.r_max, 2.0)


def make_metric_result(r_grid, Er=None):
    zeros = jnp.zeros_like(r_grid)
    if Er is None:
        Er = zeros
    source_terms = (zeros, zeros, zeros, zeros)
    U_state = (
        jnp.ones_like(r_grid),
        zeros,
        jnp.ones_like(r_grid),
        zeros,
        zeros,
        Er,
        source_terms,
        r_grid,
    )

    return U_state


def test_particle_species_current_api():
    species = make_species(charge=2.0, mass=4.0, weight=0.5)

    r, phi = species.get_positions()
    ur, uphi = species.get_velocities()

    assert species.name == "test"
    assert jnp.allclose(r, species.r)
    assert jnp.allclose(phi, species.phi)
    assert jnp.allclose(ur, species.ur)
    assert jnp.allclose(uphi, species.uphi)
    assert species.weight.shape == species.r.shape
    assert jnp.allclose(species.weight, jnp.asarray([0.5, 0.5]))
    assert jnp.allclose(species.get_charge(), jnp.asarray([1.0, 1.0]))
    assert jnp.allclose(species.get_mass(), jnp.asarray([2.0, 2.0]))
    assert species.get_shape() == "nearest"


def test_particle_species_preserves_per_particle_weights_under_jit():
    species = make_species(
        charge=2.0,
        mass=4.0,
        weight=jnp.asarray([0.25, 0.75]),
    )

    def macroparticle_properties(particles):
        return particles.weight, particles.get_charge(), particles.get_mass()

    eager_weight, eager_charge, eager_mass = macroparticle_properties(species)
    jitted_weight, jitted_charge, jitted_mass = jax.jit(
        macroparticle_properties
    )(species)

    assert jnp.allclose(eager_weight, jnp.asarray([0.25, 0.75]))
    assert jnp.allclose(eager_charge, jnp.asarray([0.5, 1.5]))
    assert jnp.allclose(eager_mass, jnp.asarray([1.0, 3.0]))
    assert jnp.allclose(jitted_weight, eager_weight)
    assert jnp.allclose(jitted_charge, eager_charge)
    assert jnp.allclose(jitted_mass, eager_mass)


def test_particle_species_supports_explicit_lower_and_compile():
    species = make_species(
        charge=2.0,
        mass=4.0,
        weight=jnp.asarray([0.25, 0.75]),
    )
    compiled_properties = jax.jit(
        lambda particles: (
            particles.weight,
            particles.get_charge(),
            particles.get_mass(),
        )
    ).lower(species).compile()

    weight, charge, mass = compiled_properties(species)

    assert jnp.allclose(weight, jnp.asarray([0.25, 0.75]))
    assert jnp.allclose(charge, jnp.asarray([0.5, 1.5]))
    assert jnp.allclose(mass, jnp.asarray([1.0, 3.0]))


def test_pad_value_adds_small_denominator_offset_with_input_dtype():
    values = jnp.asarray([0.0, 2.0], dtype=jnp.float32)

    padded = pad_value(values)

    assert padded.dtype == values.dtype
    assert padded[0] == jnp.asarray(1.0e-15, dtype=values.dtype)
    assert padded[1] == values[1] + jnp.asarray(1.0e-15, dtype=values.dtype)


def test_calculate_metric_returns_grid_level_ustate_for_zero_source():
    particles = make_species(charge=0.0, mass=0.0)
    r_grid = jnp.linspace(0.0, 1.0, 5)
    U_state = calculate_metric(particles, r_grid, r_grid[1] - r_grid[0])

    A, phi, alpha, Krr, beta_over_r, Er, source_terms, returned_grid = U_state
    mass_density, charge_density, Srr, Sr = source_terms

    assert jnp.allclose(returned_grid, r_grid)
    assert A.shape == r_grid.shape
    assert phi.shape == r_grid.shape
    assert alpha.shape == r_grid.shape
    assert Krr.shape == r_grid.shape
    assert beta_over_r.shape == r_grid.shape
    assert Er.shape == r_grid.shape
    assert mass_density.shape == r_grid.shape
    assert charge_density.shape == r_grid.shape
    assert Srr.shape == r_grid.shape
    assert Sr.shape == r_grid.shape
    assert jnp.allclose(A, 1.0)
    assert jnp.allclose(phi, 0.0)
    assert jnp.allclose(alpha, 1.0)
    assert jnp.allclose(Krr, 0.0)
    assert jnp.allclose(beta_over_r, 0.0)
    assert jnp.allclose(Er, 0.0)


def test_calculate_metric_returns_source_terms_for_nonzero_particles():
    particles = make_species(charge=1.0, mass=2.0, weight=1.0)
    r_grid = jnp.linspace(0.0, 1.0, 5)
    U_state = calculate_metric(particles, r_grid, r_grid[1] - r_grid[0])

    _, _, _, _, _, _, source_terms, _ = U_state
    mass_density, charge_density, Srr, Sr = source_terms

    assert mass_density.shape == r_grid.shape
    assert charge_density.shape == r_grid.shape
    assert Srr.shape == r_grid.shape
    assert Sr.shape == r_grid.shape
    assert jnp.any(mass_density > 0.0)
    assert jnp.any(charge_density > 0.0)


def test_beta_over_r_uses_trapezoidal_tail_integral():
    r_grid = jnp.asarray([0.0, 0.10, 0.25, 0.55, 1.0])
    dr = r_grid[1] - r_grid[0]
    alpha = jnp.asarray([1.0, 1.1, 1.2, 1.3, 1.4])
    Krr = r_grid * jnp.asarray([0.0, 0.3, 0.5, 0.7, 0.9])

    beta_over_r = beta_over_r_from_integral(alpha, Krr, r_grid, dr)

    integrand = jnp.where(r_grid == 0.0, 0.0, alpha * Krr / r_grid)
    trapezoid_segments = 0.5 * (integrand[:-1] + integrand[1:]) * (
        r_grid[1:] - r_grid[:-1]
    )
    expected_tail_integral = jnp.concatenate(
        (
            jnp.cumsum(trapezoid_segments[::-1])[::-1],
            jnp.zeros_like(integrand[-1:]),
        )
    )
    expected_beta_over_r = -expected_tail_integral

    assert jnp.all(jnp.isfinite(beta_over_r))
    assert beta_over_r[-1] == 0.0
    assert jnp.allclose(beta_over_r, expected_beta_over_r)


def test_hamiltonian_constraint_uses_regular_center_limit():
    r_grid = jnp.asarray([0.0, 0.5])
    A = jnp.asarray([1.2, 1.2])
    rho = jnp.asarray([0.4, 0.4])
    zeros = jnp.zeros_like(r_grid)
    source_terms = (rho, zeros, zeros, zeros)
    U_state = (A, zeros, jnp.ones_like(A), zeros, zeros, zeros, source_terms, r_grid)

    dphi_dr = dr_sqrt_phi(U_state, r_grid[1] - r_grid[0])
    expected_center = -(2.0 * jnp.pi / 3.0) * A[0] ** (5.0 / 2.0) * rho[0]

    assert jnp.allclose(dphi_dr[0], expected_center)


def test_electromagnetic_sources_use_covariant_radial_field():
    A = jnp.asarray(2.0)
    phi = jnp.asarray(0.15)
    alpha = jnp.asarray(0.8)
    Er = jnp.asarray(0.4)
    rho = jnp.asarray(0.3)
    charge_density = jnp.asarray(0.25)
    particle_Srr = jnp.asarray(0.2)
    r = jnp.asarray(0.75)
    dr = jnp.asarray(0.25)
    zeros = jnp.asarray(0.0)
    source_terms = (rho, charge_density, particle_Srr, zeros)
    U_state = (A, phi, alpha, zeros, zeros, Er, source_terms, r)

    A_for_denominators = pad_value(A)
    total_rho = rho + 0.5 * Er**2 / A_for_denominators**2
    expected_dphi_dr = (
        -2.0 * jnp.pi * jnp.sqrt(A) ** 5 * total_rho
        - 2.0 * phi / r
    )

    total_Srr = particle_Srr - 0.5 * Er**2
    expected_dalpha_dr = (
        4.0 * jnp.pi * alpha * total_Srr * r * A
        - 2.0 * alpha * phi * jnp.sqrt(A)
        - 2.0 * alpha * phi**2 * r
    ) / (
        A_for_denominators
        * (1.0 + 2.0 * r * phi / jnp.sqrt(A_for_denominators))
    )

    expected_dEr_dr = (
        A**2 * charge_density
        - 2.0 * Er / r
        - 2.0 * phi * Er / jnp.sqrt(A_for_denominators)
    )

    assert jnp.allclose(dr_sqrt_phi(U_state, dr), expected_dphi_dr)
    assert jnp.allclose(dr_alpha(U_state, dr), expected_dalpha_dr)
    assert jnp.allclose(dr_Er(U_state, dr), expected_dEr_dr)


def test_gauss_law_uses_regular_covariant_center_limit():
    A = jnp.asarray(1.7)
    charge_density = jnp.asarray(0.6)
    zeros = jnp.asarray(0.0)
    source_terms = (zeros, charge_density, zeros, zeros)
    U_state = (A, zeros, jnp.asarray(1.0), zeros, zeros, zeros, source_terms, zeros)

    expected_center = A**2 * charge_density / 3.0

    assert jnp.allclose(dr_Er(U_state, jnp.asarray(0.25)), expected_center)


def test_force_terms_consume_ustate_directly():
    particles = make_species(charge=1.0, mass=2.0)
    r_grid = jnp.linspace(0.0, 1.0, 5)
    U_state = calculate_metric(particles, r_grid, r_grid[1] - r_grid[0])

    dr_dt, dur_dt_GR = compute_geodesic_terms(particles, U_state)
    dur_dt_EM = compute_lorentz_terms(particles, U_state)

    assert dr_dt.shape == particles.r.shape
    assert dur_dt_GR.shape == particles.r.shape
    assert dur_dt_EM.shape == particles.r.shape


def test_force_terms_jit_match_eager_outputs():
    particles = make_species(charge=1.0, mass=2.0)
    r_grid = jnp.linspace(0.0, 1.0, 5)
    source_terms = tuple(jnp.zeros_like(r_grid) for _ in range(4))
    U_state = (
        jnp.ones_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.ones_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.linspace(0.0, 0.2, r_grid.shape[0]),
        source_terms,
        r_grid,
    )

    eager_geodesic = compute_geodesic_terms(particles, U_state)
    eager_lorentz = compute_lorentz_terms(particles, U_state)
    jitted_geodesic = jax.jit(compute_geodesic_terms)(particles, U_state)
    jitted_lorentz = jax.jit(compute_lorentz_terms)(particles, U_state)

    assert jnp.allclose(jitted_geodesic[0], eager_geodesic[0])
    assert jnp.allclose(jitted_geodesic[1], eager_geodesic[1])
    assert jnp.allclose(jitted_lorentz, eager_lorentz)


def test_calculate_metric_jit_matches_eager_output():
    particles = make_species(charge=0.0, mass=1.0, weight=0.0)
    r_grid = jnp.linspace(0.0, 1.0, 5)
    dr = r_grid[1] - r_grid[0]

    eager_U_state = calculate_metric(particles, r_grid, dr)
    jitted_U_state = jax.jit(calculate_metric)(particles, r_grid, dr)

    assert jnp.allclose(jitted_U_state[0], eager_U_state[0])
    assert jnp.allclose(jitted_U_state[1], eager_U_state[1])
    assert jnp.allclose(jitted_U_state[2], eager_U_state[2])
    assert jnp.allclose(jitted_U_state[3], eager_U_state[3])
    assert jnp.allclose(jitted_U_state[4], eager_U_state[4])
    assert jnp.allclose(jitted_U_state[5], eager_U_state[5])
    for jitted_source, eager_source in zip(jitted_U_state[6], eager_U_state[6]):
        assert jnp.allclose(jitted_source, eager_source)
    assert jnp.allclose(jitted_U_state[7], eager_U_state[7])


def test_calculate_metric_keeps_particle_state_unchanged():
    eager_particles = make_species(charge=0.0, mass=1.0, weight=0.0)
    jitted_particles = make_species(charge=0.0, mass=1.0, weight=0.0)
    eager_r = eager_particles.r.copy()
    eager_ur = eager_particles.ur.copy()
    jitted_r = jitted_particles.r.copy()
    jitted_ur = jitted_particles.ur.copy()
    r_grid = jnp.linspace(0.0, 1.0, 5)
    dr = r_grid[1] - r_grid[0]

    calculate_metric(eager_particles, r_grid, dr)
    jax.jit(calculate_metric)(jitted_particles, r_grid, dr)

    assert jnp.allclose(eager_particles.r, eager_r)
    assert jnp.allclose(eager_particles.ur, eager_ur)
    assert jnp.allclose(jitted_particles.r, jitted_r)
    assert jnp.allclose(jitted_particles.ur, jitted_ur)


def test_step_rk4_jit_matches_eager_output():
    eager_particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([0.5, 1.5]),
        r=jnp.asarray([2.5, 7.5]),
        ur=jnp.asarray([0.01, -0.01]),
        phi=jnp.asarray([0.0, 0.2]),
        uphi=jnp.asarray([0.0, 0.0]),
        shape_mode="nearest",
    )
    jitted_particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([0.5, 1.5]),
        r=jnp.asarray([2.5, 7.5]),
        ur=jnp.asarray([0.01, -0.01]),
        phi=jnp.asarray([0.0, 0.2]),
        uphi=jnp.asarray([0.0, 0.0]),
        shape_mode="nearest",
    )
    r_grid = jnp.linspace(0.0, 10.0, 9)
    dr = r_grid[1] - r_grid[0]
    dt = 1.0e-4

    eager_result = step_rk4(eager_particles, r_grid, dr, dt)
    jitted_result = jax.jit(step_rk4)(jitted_particles, r_grid, dr, dt)

    assert jnp.all(jnp.isfinite(jitted_result.r))
    assert jnp.allclose(jitted_result.r, eager_result.r)
    assert jnp.allclose(jitted_result.phi, eager_result.phi)
    assert jnp.allclose(jitted_result.ur, eager_result.ur)
    assert jnp.allclose(jitted_result.uphi, eager_result.uphi)
    assert jnp.allclose(jitted_result.weight, eager_particles.weight)
    assert jnp.allclose(eager_result.weight, eager_particles.weight)


def test_particle_derivatives_use_raw_particle_and_metric_coordinates(monkeypatch):
    metric_particles = []
    force_particles = []

    def fake_calculate_metric(stage_particles, r_grid, dr):
        metric_particles.append(stage_particles)
        return make_metric_result(r_grid)

    def fake_geodesic_terms(stage_particles, U_state):
        force_particles.append(stage_particles)
        return (
            jnp.full_like(stage_particles.r, 4.0),
            jnp.full_like(stage_particles.ur, 5.0),
        )

    def fake_lorentz_terms(stage_particles, U_state):
        return jnp.full_like(stage_particles.ur, 1.0)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(
        constraint_evolve,
        "compute_geodesic_terms",
        fake_geodesic_terms,
    )
    monkeypatch.setattr(
        constraint_evolve,
        "compute_lorentz_terms",
        fake_lorentz_terms,
    )

    particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([1.0]),
        ur=jnp.asarray([0.2]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.8]),
        shape_mode="nearest",
    )
    original_r = particles.r.copy()
    original_ur = particles.ur.copy()
    r_grid = jnp.linspace(0.0, 2.0, 5)

    dr_dt, dphi_dt, dur_dt = constraint_evolve._particle_derivatives(
        particles,
        r_grid,
        r_grid[1] - r_grid[0],
    )

    expected_dr_dt = 4.0
    expected_dphi_dt = particles.uphi / particles.r
    expected_dur_dt = 6.0

    assert jnp.allclose(dr_dt, expected_dr_dt)
    assert jnp.allclose(dphi_dt, expected_dphi_dt)
    assert jnp.allclose(dur_dt, expected_dur_dt)
    assert metric_particles == [particles]
    assert force_particles == [particles]
    assert jnp.allclose(particles.r, original_r)
    assert jnp.allclose(particles.ur, original_ur)


def test_source_terms_pad_zero_A_denominators():
    particles = particle_species(
        name="test",
        charge=1.0,
        mass=1.0,
        weight=0.0,
        r=jnp.asarray([0.25]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )
    A_at_point = jnp.asarray(0.0)
    radial_coordinate = jnp.asarray(0.25)
    dr = jnp.asarray(0.25)
    grid = make_interpolation_grid(jnp.asarray([0.0, 0.25, 0.50]))

    source_terms = jnp.asarray(
        [
            mass_density_at_point(particles, A_at_point, radial_coordinate, grid),
            charge_density_at_point(particles, A_at_point, radial_coordinate, grid),
            Srr_at_point(particles, A_at_point, radial_coordinate, grid),
            Sr_at_point(particles, A_at_point, radial_coordinate, grid),
        ]
    )

    assert jnp.all(jnp.isfinite(source_terms))
    assert jnp.allclose(source_terms, 0.0)


def test_lorentz_force_uses_particle_shape_interpolation_for_fields():
    particles = particle_species(
        name="test",
        charge=3.0,
        mass=2.0,
        weight=1.0,
        r=jnp.asarray([1.25]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )
    r_grid = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    grid = make_interpolation_grid(r_grid)
    alpha_values = jnp.asarray([1.0, 1.5, 2.5, 4.0, 6.0])
    Er_values = jnp.asarray([0.0, 1.0, 4.0, 9.0, 16.0])
    source_terms = tuple(jnp.zeros_like(r_grid) for _ in range(4))
    U_state = (
        jnp.ones_like(r_grid),
        jnp.zeros_like(r_grid),
        alpha_values,
        jnp.zeros_like(r_grid),
        jnp.zeros_like(r_grid),
        Er_values,
        source_terms,
        r_grid,
    )

    lapse_at_particle = interpolate_field_to_particles(
        alpha_values,
        particles.r,
        grid,
        shape_mode=particles.get_shape(),
    )
    electric_field_at_particle = interpolate_field_to_particles(
        Er_values,
        particles.r,
        grid,
        shape_mode=particles.get_shape(),
    )
    expected = (
        lapse_at_particle
        * particles.get_charge()
        * electric_field_at_particle
        / particles.get_mass()
    )
    linear_expected = (
        jnp.interp(particles.r, r_grid, alpha_values)
        * particles.get_charge()
        * jnp.interp(particles.r, r_grid, Er_values)
        / particles.get_mass()
    )

    assert not jnp.allclose(expected, linear_expected)
    assert jnp.allclose(compute_lorentz_terms(particles, U_state), expected)


def test_geodesic_terms_use_particle_shape_interpolation_for_metric_fields():
    particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([1.25]),
        ur=jnp.asarray([0.2]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )
    r_grid = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    grid = make_interpolation_grid(r_grid)
    A_values = jnp.asarray([1.0, 2.0, 5.0, 10.0, 17.0])
    source_terms = tuple(jnp.zeros_like(r_grid) for _ in range(4))
    U_state = (
        A_values,
        jnp.zeros_like(r_grid),
        jnp.ones_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.zeros_like(r_grid),
        source_terms,
        r_grid,
    )

    A_at_particle = interpolate_field_to_particles(
        A_values,
        particles.r,
        grid,
        shape_mode=particles.get_shape(),
    )
    W = jnp.sqrt(1.0 + particles.ur**2 / A_at_particle**2)
    expected_dr_dt = particles.ur / (A_at_particle**2 * W)

    A_linear = jnp.interp(particles.r, r_grid, A_values)
    W_linear = jnp.sqrt(1.0 + particles.ur**2 / A_linear**2)
    linear_dr_dt = particles.ur / (A_linear**2 * W_linear)

    dr_dt, du_r_dt = compute_geodesic_terms(particles, U_state)

    assert not jnp.allclose(expected_dr_dt, linear_dr_dt)
    assert jnp.allclose(dr_dt, expected_dr_dt)
    assert jnp.allclose(du_r_dt, 0.0)


def test_geodesic_terms_pad_zero_A_denominators():
    particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([0.25]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )
    r_grid = jnp.asarray([0.0, 0.25, 0.50])
    source_terms = tuple(jnp.zeros_like(r_grid) for _ in range(4))
    U_state = (
        jnp.zeros_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.ones_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.zeros_like(r_grid),
        jnp.zeros_like(r_grid),
        source_terms,
        r_grid,
    )

    dr_dt, du_r_dt = compute_geodesic_terms(particles, U_state)

    assert jnp.all(jnp.isfinite(dr_dt))
    assert jnp.all(jnp.isfinite(du_r_dt))


def test_step_updates_current_particle_class_in_place_and_preserves_uphi():
    particles = make_species(charge=1.0, mass=2.0)
    initial_r = particles.r.copy()
    initial_ur = particles.ur.copy()
    initial_uphi = particles.uphi.copy()
    r_grid = jnp.linspace(0.0, 1.0, 5)

    updated = step(particles, r_grid, r_grid[1] - r_grid[0], dt=0.05)

    assert updated is particles
    assert not jnp.allclose(updated.r, initial_r)
    assert updated.ur.shape == initial_ur.shape
    assert jnp.allclose(updated.uphi, initial_uphi)


def test_step_freezes_particle_that_crosses_center(monkeypatch):
    def fake_calculate_metric(stage_particles, r_grid, dr):
        return make_metric_result(r_grid)

    def fake_geodesic_terms(stage_particles, U_state):
        return -jnp.ones_like(stage_particles.r), jnp.zeros_like(stage_particles.ur)

    def fake_lorentz_terms(stage_particles, U_state):
        return jnp.zeros_like(stage_particles.ur)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(constraint_evolve, "compute_geodesic_terms", fake_geodesic_terms)
    monkeypatch.setattr(constraint_evolve, "compute_lorentz_terms", fake_lorentz_terms)

    particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([0.05]),
        ur=jnp.asarray([-1.0]),
        phi=jnp.asarray([0.3]),
        uphi=jnp.asarray([0.4]),
        shape_mode="nearest",
    )
    r_grid = jnp.linspace(0.0, 1.0, 5)

    updated = step(particles, r_grid, r_grid[1] - r_grid[0], dt=0.1)

    assert updated is particles
    assert jnp.allclose(updated.r, 0.0)
    assert jnp.allclose(updated.ur, 0.0)
    assert jnp.allclose(updated.uphi, 0.0)


def test_step_rk4_imports_as_additional_timestep_option():
    assert callable(step_rk4)
    assert callable(step_rk4_with_metric)


def test_step_rk4_with_metric_reuses_initial_metric_and_returns_final_metric(
    monkeypatch,
):
    metric_stage_positions = []

    def fake_calculate_metric(stage_particles, r_grid, dr):
        metric_stage_positions.append(stage_particles.r.copy())
        return make_metric_result(r_grid)

    def fake_geodesic_terms(stage_particles, U_state):
        return (
            jnp.ones_like(stage_particles.r),
            jnp.zeros_like(stage_particles.ur),
        )

    def fake_lorentz_terms(stage_particles, U_state):
        return jnp.zeros_like(stage_particles.ur)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(
        constraint_evolve,
        "compute_geodesic_terms",
        fake_geodesic_terms,
    )
    monkeypatch.setattr(
        constraint_evolve,
        "compute_lorentz_terms",
        fake_lorentz_terms,
    )

    particles = make_species()
    r_grid = jnp.linspace(0.0, 1.0, 5)
    initial_metric = make_metric_result(r_grid)

    updated_particles, final_metric = step_rk4_with_metric(
        particles,
        initial_metric,
        r_grid,
        r_grid[1] - r_grid[0],
        dt=0.1,
    )

    assert updated_particles is particles
    assert len(metric_stage_positions) == 4
    assert jnp.allclose(metric_stage_positions[-1], updated_particles.r)
    assert jnp.allclose(final_metric[-1], r_grid)


def test_step_rk4_with_metric_matches_existing_multistep_path():
    def make_collapse_particles():
        return particle_species(
            name="test",
            charge=0.2,
            mass=1.0,
            weight=jnp.asarray([0.5, 1.5]),
            r=jnp.asarray([2.5, 7.5]),
            ur=jnp.asarray([0.01, -0.01]),
            phi=jnp.asarray([0.0, 0.2]),
            uphi=jnp.asarray([0.0, 0.0]),
            shape_mode="linear",
        )

    reference_particles = make_collapse_particles()
    cached_particles = make_collapse_particles()
    r_grid = jnp.linspace(0.0, 10.0, 9)
    dr = r_grid[1] - r_grid[0]
    dt = 1.0e-4
    reference_metric = calculate_metric(reference_particles, r_grid, dr)
    cached_metric = calculate_metric(cached_particles, r_grid, dr)

    for _ in range(2):
        reference_particles = step_rk4(
            reference_particles,
            r_grid,
            dr,
            dt,
        )
        reference_metric = calculate_metric(
            reference_particles,
            r_grid,
            dr,
        )
        cached_particles, cached_metric = step_rk4_with_metric(
            cached_particles,
            cached_metric,
            r_grid,
            dr,
            dt,
        )

    assert jnp.allclose(cached_particles.r, reference_particles.r)
    assert jnp.allclose(cached_particles.phi, reference_particles.phi)
    assert jnp.allclose(cached_particles.ur, reference_particles.ur)
    assert jnp.allclose(cached_particles.uphi, reference_particles.uphi)

    for cached_field, reference_field in zip(
        cached_metric[:6],
        reference_metric[:6],
    ):
        assert jnp.allclose(cached_field, reference_field)
    for cached_source, reference_source in zip(
        cached_metric[6],
        reference_metric[6],
    ):
        assert jnp.allclose(cached_source, reference_source)
    assert jnp.allclose(cached_metric[7], reference_metric[7])


def test_step_rk4_updates_current_particle_class_in_place_and_preserves_uphi(monkeypatch):
    calls = []

    def fake_calculate_metric(stage_particles, r_grid, dr):
        calls.append(stage_particles.r.copy())
        return make_metric_result(r_grid)

    def fake_geodesic_terms(stage_particles, U_state):
        return jnp.ones_like(stage_particles.r), jnp.full_like(stage_particles.ur, 0.5)

    def fake_lorentz_terms(stage_particles, U_state):
        return jnp.full_like(stage_particles.ur, 0.25)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(constraint_evolve, "compute_geodesic_terms", fake_geodesic_terms)
    monkeypatch.setattr(constraint_evolve, "compute_lorentz_terms", fake_lorentz_terms)

    particles = make_species(charge=1.0, mass=2.0)
    initial_r = particles.r.copy()
    initial_uphi = particles.uphi.copy()
    r_grid = jnp.linspace(0.0, 1.0, 5)

    updated = step_rk4(particles, r_grid, r_grid[1] - r_grid[0], dt=0.1)

    assert updated is particles
    assert len(calls) == 4
    assert updated.r.shape == initial_r.shape
    assert updated.ur.shape == initial_r.shape
    assert jnp.allclose(updated.r, initial_r + 0.1)
    assert jnp.allclose(updated.uphi, initial_uphi)


def test_step_rk4_recomputes_stage_specific_metric_and_em_field(monkeypatch):
    metric_stage_positions = []
    geodesic_Er_values = []
    lorentz_Er_values = []

    def fake_calculate_metric(stage_particles, r_grid, dr):
        stage_number = len(metric_stage_positions) + 1
        metric_stage_positions.append(stage_particles.r.copy())
        Er = jnp.full_like(r_grid, float(stage_number))
        return make_metric_result(r_grid, Er=Er)

    def fake_geodesic_terms(stage_particles, U_state):
        geodesic_Er_values.append(U_state[5][0])
        stage_number = U_state[5][0]
        return jnp.full_like(stage_particles.r, stage_number), jnp.zeros_like(stage_particles.ur)

    def fake_lorentz_terms(stage_particles, U_state):
        lorentz_Er_values.append(U_state[5][0])
        return jnp.zeros_like(stage_particles.ur)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(constraint_evolve, "compute_geodesic_terms", fake_geodesic_terms)
    monkeypatch.setattr(constraint_evolve, "compute_lorentz_terms", fake_lorentz_terms)

    particles = make_species(charge=1.0, mass=2.0)
    r_grid = jnp.linspace(0.0, 1.0, 5)
    dt = 0.1

    step_rk4(particles, r_grid, r_grid[1] - r_grid[0], dt)


    assert len(metric_stage_positions) == 4
    assert jnp.allclose( jnp.array(geodesic_Er_values), jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    assert jnp.allclose(jnp.array(lorentz_Er_values), jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    assert jnp.allclose(metric_stage_positions[0], jnp.asarray([0.25, 0.75]))
    assert jnp.allclose(metric_stage_positions[1], jnp.asarray([0.30, 0.80]))
    assert jnp.allclose(metric_stage_positions[2], jnp.asarray([0.35, 0.85]))
    assert jnp.allclose(metric_stage_positions[3], jnp.asarray([0.55, 1.05]))


def test_step_rk4_freezes_center_crossing_before_stage_metric_solves(monkeypatch):
    metric_stage_positions = []

    def fake_calculate_metric(stage_particles, r_grid, dr):
        metric_stage_positions.append(stage_particles.r.copy())
        return make_metric_result(r_grid)

    def fake_geodesic_terms(stage_particles, U_state):
        return -jnp.ones_like(stage_particles.r), jnp.zeros_like(stage_particles.ur)

    def fake_lorentz_terms(stage_particles, U_state):
        return jnp.zeros_like(stage_particles.ur)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(constraint_evolve, "compute_geodesic_terms", fake_geodesic_terms)
    monkeypatch.setattr(constraint_evolve, "compute_lorentz_terms", fake_lorentz_terms)

    particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([0.05]),
        ur=jnp.asarray([-1.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.2]),
        shape_mode="nearest",
    )
    r_grid = jnp.linspace(0.0, 1.0, 5)

    step_rk4(particles, r_grid, r_grid[1] - r_grid[0], dt=0.2)

    assert len(metric_stage_positions) == 4
    assert all(jnp.all(stage_r >= 0.0) for stage_r in metric_stage_positions)
    assert jnp.allclose(particles.r, 0.0)
    assert jnp.allclose(particles.ur, 0.0)
    assert jnp.allclose(particles.uphi, 0.0)


def test_step_rk4_keeps_center_frozen_against_force_terms(monkeypatch):
    metric_stage_positions = []

    def fake_calculate_metric(stage_particles, r_grid, dr):
        metric_stage_positions.append(stage_particles.r.copy())
        return make_metric_result(r_grid)

    def fake_geodesic_terms(stage_particles, U_state):
        return jnp.ones_like(stage_particles.r), jnp.ones_like(stage_particles.ur)

    def fake_lorentz_terms(stage_particles, U_state):
        return jnp.ones_like(stage_particles.ur)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(constraint_evolve, "compute_geodesic_terms", fake_geodesic_terms)
    monkeypatch.setattr(constraint_evolve, "compute_lorentz_terms", fake_lorentz_terms)

    particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([0.0]),
        ur=jnp.asarray([-1.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.5]),
        shape_mode="nearest",
    )
    r_grid = jnp.linspace(0.0, 1.0, 5)

    step_rk4(particles, r_grid, r_grid[1] - r_grid[0], dt=0.1)

    assert len(metric_stage_positions) == 4
    assert all(jnp.allclose(stage_r, 0.0) for stage_r in metric_stage_positions)
    assert jnp.allclose(particles.r, 0.0)
    assert jnp.allclose(particles.ur, 0.0)
    assert jnp.allclose(particles.uphi, 0.0)


def test_step_rk4_uses_classic_weighted_derivative_combination(monkeypatch):
    derivative_calls = []

    def fake_calculate_metric(stage_particles, r_grid, dr):
        return make_metric_result(r_grid)

    def fake_geodesic_terms(stage_particles, U_state):
        derivative_calls.append(
            (
                stage_particles.r.copy(),
                stage_particles.phi.copy(),
                stage_particles.ur.copy(),
            )
        )
        dr_dt = stage_particles.r
        dur_dt_GR = 2.0 * stage_particles.ur
        return dr_dt, dur_dt_GR

    def fake_lorentz_terms(stage_particles, U_state):
        return jnp.zeros_like(stage_particles.ur)

    monkeypatch.setattr(
        constraint_evolve,
        "calculate_metric",
        fake_calculate_metric,
    )
    monkeypatch.setattr(constraint_evolve, "compute_geodesic_terms", fake_geodesic_terms)
    monkeypatch.setattr(constraint_evolve, "compute_lorentz_terms", fake_lorentz_terms)

    particles = particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([1.0]),
        ur=jnp.asarray([2.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )
    r_grid = jnp.linspace(0.0, 2.0, 5)
    dt = 0.1

    step_rk4(particles, r_grid, r_grid[1] - r_grid[0], dt)

    r1, _, ur1 = derivative_calls[0]
    r2, _, ur2 = derivative_calls[1]
    r3, _, ur3 = derivative_calls[2]
    r4, _, ur4 = derivative_calls[3]

    expected_r = 1.0 + (dt / 6.0) * (r1 + 2.0 * r2 + 2.0 * r3 + r4)
    expected_ur = 2.0 + (dt / 6.0) * (
        2.0 * ur1 + 2.0 * (2.0 * ur2) + 2.0 * (2.0 * ur3) + 2.0 * ur4
    )

    assert len(derivative_calls) == 4
    assert jnp.allclose(particles.r, expected_r)
    assert jnp.allclose(particles.ur, expected_ur)
