import jax
import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.ConstraintBasedRelativity.solve_metric import (
    dr_A,
    dr_alpha,
    dr_Er,
    dr_sqrt_phi,
)
from RadiShPICR.ConstraintBasedRelativity.vacuum_conditions import (
    reissner_nordstrom_A,
    reissner_nordstrom_lapse,
    rescale_to_schwarzschild_coordinates,
    schwarzschild_rescale_factors,
    total_particle_charge,
    total_particle_mass,
    vacuum_rescale_factors,
)


def make_species(charge=1.5, mass=2.0, weight=0.5):
    return particle_species(
        name="test",
        charge=charge,
        mass=mass,
        weight=weight,
        r=jnp.asarray([0.25, 0.50, 0.75]),
        ur=jnp.asarray([0.0, 0.0, 0.0]),
        phi=jnp.asarray([0.0, 0.0, 0.0]),
        uphi=jnp.asarray([0.0, 0.0, 0.0]),
        shape_mode="nearest",
    )


def test_reissner_nordstrom_helpers_match_rationalized_charge_convention():
    r = jnp.asarray(4.0)
    mass = jnp.asarray(1.5)
    charge = jnp.asarray(2.0)

    charge_radius_squared = charge**2 / (4.0 * jnp.pi)
    expected_A = (
        (1.0 + mass / (2.0 * r))**2
        - charge_radius_squared / (4.0 * r**2)
    )
    expected_lapse = (1.0 - mass / (2.0 * r)) * (1.0 + mass / (2.0 * r))
    expected_lapse = expected_lapse + charge_radius_squared / (4.0 * r**2)
    expected_lapse = expected_lapse / expected_A

    assert jnp.allclose(reissner_nordstrom_A(r, mass, charge), expected_A)
    assert jnp.allclose(
        reissner_nordstrom_lapse(r, mass, charge),
        expected_lapse,
    )


def test_reissner_nordstrom_fields_satisfy_electrovac_radial_equations():
    mass = jnp.asarray(1.0)
    charge = jnp.asarray(0.8)
    radii = jnp.asarray([2.0, 3.0, 5.0, 8.0])
    zeros = jnp.asarray(0.0)
    vacuum_sources = (zeros, zeros, zeros, zeros)

    def exact_A(r):
        return reissner_nordstrom_A(r, mass, charge)

    def exact_alpha(r):
        return reissner_nordstrom_lapse(r, mass, charge)

    def exact_Er(r):
        return charge / (4.0 * jnp.pi * exact_A(r) * r**2)

    def equation_residuals(r):
        A = exact_A(r)
        phi = jax.grad(lambda x: jnp.sqrt(exact_A(x)))(r)
        alpha = exact_alpha(r)
        Er = exact_Er(r)
        U_state = (
            A,
            phi,
            alpha,
            zeros,
            zeros,
            Er,
            vacuum_sources,
            r,
        )

        return jnp.asarray(
            [
                dr_A(U_state) - jax.grad(exact_A)(r),
                dr_sqrt_phi(U_state) - jax.grad(
                    lambda x: jax.grad(lambda y: jnp.sqrt(exact_A(y)))(x)
                )(r),
                dr_alpha(U_state) - jax.grad(exact_alpha)(r),
                dr_Er(U_state) - jax.grad(exact_Er)(r),
            ]
        )

    residuals = jax.vmap(equation_residuals)(radii)

    assert jnp.allclose(residuals, 0.0, rtol=2.0e-4, atol=2.0e-6)


def test_reissner_nordstrom_helpers_reduce_to_schwarzschild_without_charge():
    r = jnp.asarray(5.0)
    mass = jnp.asarray(1.25)
    charge = jnp.asarray(0.0)

    expected_A = (1.0 + mass / (2.0 * r))**2
    expected_lapse = (1.0 - mass / (2.0 * r)) / (1.0 + mass / (2.0 * r))

    assert jnp.allclose(reissner_nordstrom_A(r, mass, charge), expected_A)
    assert jnp.allclose(
        reissner_nordstrom_lapse(r, mass, charge),
        expected_lapse,
    )


def test_charged_vacuum_rescale_recovers_known_coordinate_factors():
    mass = jnp.asarray(1.2)
    charge = jnp.asarray(0.7)
    r_outer = jnp.asarray(5.0)
    expected_X_r = jnp.asarray(1.3)
    expected_X_t = jnp.asarray(0.8)
    rescaled_outer_radius = expected_X_r * r_outer

    A_outer = expected_X_r * reissner_nordstrom_A(
        rescaled_outer_radius,
        mass,
        charge,
    )
    alpha_outer = expected_X_t * reissner_nordstrom_lapse(
        rescaled_outer_radius,
        mass,
        charge,
    )

    X_r, X_t = vacuum_rescale_factors(
        A_outer,
        alpha_outer,
        r_outer,
        mass,
        charge,
    )

    assert jnp.allclose(X_r, expected_X_r)
    assert jnp.allclose(X_t, expected_X_t)


def test_total_particle_mass_and_charge_sum_getter_values_over_particles():
    particles = make_species(
        charge=1.5,
        mass=2.0,
        weight=jnp.asarray([0.25, 0.50, 0.75]),
    )

    assert jnp.allclose(total_particle_mass(particles), 3.0)
    assert jnp.allclose(total_particle_charge(particles), 2.25)


def test_schwarzschild_rescale_matches_outer_boundary_cell():
    particles = make_species(charge=0.0, mass=1.0, weight=0.25)
    r_grid = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    A = jnp.asarray([1.0, 1.1, 1.2, 1.3])
    phi = jnp.asarray([0.0, -0.01, -0.02, -0.03])
    alpha = jnp.asarray([1.0, 0.95, 0.90, 0.85])
    Krr = jnp.asarray([0.0, 0.01, 0.02, 0.03])
    beta_over_r = jnp.asarray([0.06, 0.04, 0.02, 0.0])
    Er = jnp.asarray([0.0, 0.1, 0.2, 0.3])
    source_terms = (
        jnp.asarray([0.0, 1.0, 2.0, 0.0]),
        jnp.asarray([0.0, 0.5, 1.0, 0.0]),
        jnp.asarray([0.0, 0.2, 0.4, 0.0]),
        jnp.asarray([0.0, 0.3, 0.6, 0.0]),
    )
    U_state = (A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid)

    exterior_mass = total_particle_mass(particles)
    matched, rescaled_particles, r_matched = (
        rescale_to_schwarzschild_coordinates(
            U_state,
            particles,
            exterior_mass,
        )
    )
    (
        A_matched,
        phi_matched,
        alpha_matched,
        Krr_matched,
        beta_matched,
        Er_matched,
        matched_sources,
        r_matched,
    ) = matched
    mass_density, charge_density, Srr_matched, Sr_matched = matched_sources

    expected_A_outer = reissner_nordstrom_A(
        r_matched[-1],
        total_particle_mass(particles),
        0.0,
    )
    expected_alpha_outer = reissner_nordstrom_lapse(
        r_matched[-1],
        total_particle_mass(particles),
        0.0,
    )

    assert jnp.allclose(A_matched[-1], expected_A_outer)
    assert jnp.allclose(alpha_matched[-1], expected_alpha_outer)
    assert jnp.allclose(mass_density, source_terms[0])
    assert jnp.allclose(charge_density, source_terms[1])

    expected_X_r, expected_X_t = schwarzschild_rescale_factors(
        U_state,
        exterior_mass,
    )
    assert jnp.allclose(phi_matched, phi / expected_X_r ** (3.0 / 2.0))
    assert jnp.allclose(Krr_matched, Krr)
    assert jnp.allclose(beta_matched, beta_over_r / expected_X_t)
    assert jnp.allclose(Er_matched, Er / expected_X_r)
    assert jnp.allclose(Srr_matched, source_terms[2] / expected_X_r**2)
    assert jnp.allclose(Sr_matched, source_terms[3] / expected_X_r)
    assert jnp.allclose(rescaled_particles.r, expected_X_r * particles.r)
    assert jnp.allclose(rescaled_particles.ur, particles.ur / expected_X_r)
    assert jnp.allclose(rescaled_particles.phi, particles.phi)
    assert jnp.allclose(rescaled_particles.uphi, particles.uphi)
    assert jnp.allclose(rescaled_particles.weight, particles.weight)


def test_vacuum_rescale_pads_zero_lapse_rescaling_denominator():
    particles = make_species(charge=0.0, mass=0.0, weight=0.0)
    r_grid = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    A = jnp.ones_like(r_grid)
    phi = jnp.zeros_like(r_grid)
    alpha = jnp.asarray([1.0, 0.8, 0.4, 0.0])
    Krr = jnp.zeros_like(r_grid)
    beta_over_r = jnp.zeros_like(r_grid)
    Er = jnp.zeros_like(r_grid)
    source_terms = tuple(jnp.zeros_like(r_grid) for _ in range(4))
    U_state = (A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid)

    matched, rescaled_particles, r_matched = (
        rescale_to_schwarzschild_coordinates(U_state, particles, 0.0)
    )
    X_r, X_t = schwarzschild_rescale_factors(U_state, 0.0)
    A_matched, phi_matched, alpha_matched, Krr_matched, beta_matched, Er_matched, matched_sources, r_matched = matched
    mass_density, charge_density, Srr_matched, Sr_matched = matched_sources

    assert jnp.all(jnp.isfinite(A_matched))
    assert jnp.all(jnp.isfinite(phi_matched))
    assert jnp.all(jnp.isfinite(alpha_matched))
    assert jnp.all(jnp.isfinite(Krr_matched))
    assert jnp.all(jnp.isfinite(beta_matched))
    assert jnp.all(jnp.isfinite(Er_matched))
    assert jnp.all(jnp.isfinite(mass_density))
    assert jnp.all(jnp.isfinite(charge_density))
    assert jnp.all(jnp.isfinite(Srr_matched))
    assert jnp.all(jnp.isfinite(Sr_matched))
    assert jnp.all(jnp.isfinite(r_matched))
    assert jnp.all(jnp.isfinite(rescaled_particles.r))
    assert jnp.all(jnp.isfinite(rescaled_particles.ur))
    assert jnp.isfinite(X_r)
    assert jnp.isfinite(X_t)


def test_covariant_particle_and_source_rescaling_preserves_lorentz_factor():
    particles = particle_species(
        name="test",
        charge=0.2,
        mass=0.5,
        weight=1.0,
        r=jnp.asarray([0.75, 1.25]),
        ur=jnp.asarray([0.3, -0.4]),
        phi=jnp.asarray([0.1, 0.2]),
        uphi=jnp.asarray([0.2, -0.1]),
        shape_mode="nearest",
    )
    original_r = particles.r.copy()
    original_ur = particles.ur.copy()

    r_grid = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    A = jnp.full_like(r_grid, 1.4)
    alpha = jnp.asarray([1.0, 0.95, 0.90, 0.85])
    zeros = jnp.zeros_like(r_grid)
    source_terms = (
        jnp.asarray([0.0, 1.0, 2.0, 0.0]),
        jnp.asarray([0.0, 0.5, 1.0, 0.0]),
        jnp.asarray([0.0, 0.2, 0.4, 0.0]),
        jnp.asarray([0.0, 0.3, 0.6, 0.0]),
    )
    U_state = (
        A,
        zeros,
        alpha,
        zeros,
        zeros,
        zeros,
        source_terms,
        r_grid,
    )

    matched, rescaled_particles, _ = (
        rescale_to_schwarzschild_coordinates(
            U_state,
            particles,
            total_particle_mass(particles),
        )
    )
    X_r, X_t = schwarzschild_rescale_factors(
        U_state,
        total_particle_mass(particles),
    )
    A_matched, _, _, _, _, _, matched_sources, _ = matched
    _, _, Srr_matched, Sr_matched = matched_sources

    W = jnp.sqrt(
        1.0
        + particles.ur**2 / A[0] ** 2
        + particles.uphi**2 / (A[0] ** 2 * particles.r**2)
    )
    W_rescaled = jnp.sqrt(
        1.0
        + rescaled_particles.ur**2 / A_matched[0] ** 2
        + rescaled_particles.uphi**2
        / (A_matched[0] ** 2 * rescaled_particles.r**2)
    )

    assert jnp.allclose(W_rescaled, W)
    assert jnp.allclose(Sr_matched, source_terms[3] / X_r)
    assert jnp.allclose(Srr_matched, source_terms[2] / X_r**2)
    assert jnp.allclose(particles.r, original_r)
    assert jnp.allclose(particles.ur, original_ur)

    U_state_different_time_scale = (
        A,
        zeros,
        0.7 * alpha,
        zeros,
        zeros,
        zeros,
        source_terms,
        r_grid,
    )
    matched_2, rescaled_particles_2, _ = (
        rescale_to_schwarzschild_coordinates(
            U_state_different_time_scale,
            particles,
            total_particle_mass(particles),
        )
    )
    X_r_2, X_t_2 = schwarzschild_rescale_factors(
        U_state_different_time_scale,
        total_particle_mass(particles),
    )

    assert jnp.allclose(X_r_2, X_r)
    assert not jnp.allclose(X_t_2, X_t)
    assert jnp.allclose(rescaled_particles_2.ur, rescaled_particles.ur)
    assert jnp.allclose(matched_2[6][2], Srr_matched)
    assert jnp.allclose(matched_2[6][3], Sr_matched)
