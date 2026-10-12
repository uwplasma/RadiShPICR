"""The shared initial data lifted out of the demos.

These pin the physics that used to live in
``demos/os_collapse_Z4C_GR/run_oppenheimer_snyder_z4c.py`` and the three
``initialize_puncture`` copies, against closed-form values rather than against
the demo code, so the move is checkable without importing a demo.
"""

import math

import numpy as np
import pytest
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity import build_radial_grid
from RadiShPICR.initial_data import (
    constrained_state_to_z4c,
    homogeneous_rest_mass_profile,
    initial_center_A,
    initialize_oppenheimer_snyder_particles,
    isotropic_radius_from_areal,
    puncture_electric_field,
    puncture_metric,
    surface_isotropic_radius,
)
from RadiShPICR.Z4C.utils import freefall_collapse_time_step, generate_r_grid

MASS = 1.0
SURFACE_AREAL_RADIUS = 10.0


def test_surface_isotropic_radius_matches_the_schwarzschild_relation():
    isotropic = surface_isotropic_radius(SURFACE_AREAL_RADIUS, MASS)

    # Inverting r_areal = r (1 + M/2r)**2 must return the surface.
    areal = isotropic * (1.0 + MASS / (2.0 * isotropic)) ** 2
    assert areal == pytest.approx(SURFACE_AREAL_RADIUS)


def test_rest_mass_profile_is_monotonic_and_exceeds_gravitational_mass():
    density = 3.0 * MASS / (4.0 * math.pi * SURFACE_AREAL_RADIUS**3)
    areal, cumulative, total = homogeneous_rest_mass_profile(
        density, MASS, SURFACE_AREAL_RADIUS, num_quadrature_points=4001
    )

    assert areal[0] == 0.0 and areal[-1] == pytest.approx(SURFACE_AREAL_RADIUS)
    assert np.all(np.diff(cumulative) >= 0.0)
    assert cumulative[0] == 0.0
    # Rest mass exceeds gravitational mass by the binding energy.
    assert total > MASS
    assert total == pytest.approx(float(cumulative[-1]))


def test_isotropic_radius_from_areal_matches_the_surface_and_origin():
    areal = np.array([0.0, 5.0, SURFACE_AREAL_RADIUS])
    isotropic = isotropic_radius_from_areal(areal, MASS, SURFACE_AREAL_RADIUS)

    assert isotropic[0] == pytest.approx(0.0)
    assert np.all(np.diff(isotropic) > 0.0)
    assert isotropic[-1] == pytest.approx(
        surface_isotropic_radius(SURFACE_AREAL_RADIUS, MASS)
    )


def test_initial_center_A_is_the_analytic_os_seed():
    surface = surface_isotropic_radius(SURFACE_AREAL_RADIUS, MASS)
    surface_root = math.sqrt(1.0 - 2.0 * MASS / SURFACE_AREAL_RADIUS)

    assert initial_center_A(MASS, SURFACE_AREAL_RADIUS) == pytest.approx(
        2.0 * SURFACE_AREAL_RADIUS / (surface * (1.0 + surface_root))
    )


def test_particles_carry_the_total_rest_mass_and_fill_the_star():
    density = 3.0 * MASS / (4.0 * math.pi * SURFACE_AREAL_RADIUS**3)
    grid = build_radial_grid(20.0, 64)

    particles, areal_radius, total_rest_mass = (
        initialize_oppenheimer_snyder_particles(
            grid, density, MASS, SURFACE_AREAL_RADIUS,
            total_particles=256, shape_mode="quadratic",
        )
    )

    assert particles.name == "oppenheimer_snyder_dust"
    assert particles.get_shape() == "quadratic"
    assert np.asarray(particles.r).size == 256
    # Equal-rest-mass stratification: the weights sum to the profile total.
    assert float(np.sum(np.asarray(particles.weight))) == pytest.approx(
        total_rest_mass, rel=1e-6
    )
    assert np.all(areal_radius >= 0.0)
    assert areal_radius.max() <= SURFACE_AREAL_RADIUS
    # Time-symmetric data start at rest.
    assert np.all(np.asarray(particles.ur) == 0.0)


def test_solver_chart_scale_shrinks_the_populated_grid_range():
    density = 3.0 * MASS / (4.0 * math.pi * SURFACE_AREAL_RADIUS**3)
    grid = build_radial_grid(20.0, 64)

    _, unit_chart, _ = initialize_oppenheimer_snyder_particles(
        grid, density, MASS, SURFACE_AREAL_RADIUS, 256, "quadratic",
    )
    _, scaled_chart, _ = initialize_oppenheimer_snyder_particles(
        grid, density, MASS, SURFACE_AREAL_RADIUS, 256, "quadratic",
        initial_X_r=1.5,
    )

    # initial_X_r defaults to the Z4C chart, where the shoot normalizes it.
    assert not np.allclose(unit_chart, scaled_chart)


def test_puncture_reduces_to_schwarzschild_at_zero_charge():
    r = generate_r_grid(0.0, 20.0, 64)
    psi = 1.0 + MASS / (2.0 * r)

    metric = puncture_metric(r, MASS)

    assert np.allclose(np.asarray(metric.alpha), np.asarray(psi**-2))
    assert np.allclose(np.asarray(metric.chi), np.asarray(psi**-4))
    assert np.allclose(np.asarray(metric.conformal_grr), 1.0)
    assert np.allclose(np.asarray(metric.conformal_gt), 1.0)
    # Time symmetric: no curvature, no shift.
    for name in ("beta", "Kh", "Arr", "At", "theta", "Gamma"):
        assert np.allclose(np.asarray(getattr(metric, name)), 0.0)
    assert float(metric.dr) == pytest.approx(float(r[1] - r[0]))


def test_charged_puncture_matches_the_reissner_nordstrom_conformal_factor():
    r = generate_r_grid(0.0, 20.0, 64)
    charge = 0.5

    metric = puncture_metric(r, MASS, charge=charge)

    q_squared = charge**2 / (4.0 * np.pi)
    psi_squared = (1.0 + MASS / (2.0 * r)) ** 2 - q_squared / (4.0 * r**2)
    assert np.allclose(np.asarray(metric.alpha), np.asarray(psi_squared**-1))
    assert np.allclose(np.asarray(metric.chi), np.asarray(psi_squared**-2))


def test_puncture_electric_field_carries_the_requested_flux():
    r = generate_r_grid(0.0, 20.0, 200)
    charge = 0.5
    metric = puncture_metric(r, MASS, charge=charge)

    E_r = puncture_electric_field(metric, charge)

    # The conserved quantity is the flux, which must reproduce the charge.
    grr = metric.conformal_grr / metric.chi
    gT = metric.conformal_gt / metric.chi
    flux = 4.0 * np.pi * r**2 * gT * E_r / np.sqrt(metric.chi * grr)
    assert np.allclose(np.asarray(flux), charge)
    assert np.allclose(np.asarray(puncture_electric_field(metric, 0.0)), 0.0)


def test_constrained_state_to_z4c_uses_the_isotropic_ansatz():
    nodes = jnp.linspace(0.0, 20.0, 65)
    A = 1.0 + 0.5 / (1.0 + nodes)
    alpha = 1.0 / A
    U_state = (A, None, alpha, None, None, None, None, nodes)
    z4c_r = generate_r_grid(0.0, 20.0, 32)

    metric = constrained_state_to_z4c(U_state, z4c_r)

    interpolated_A = jnp.interp(z4c_r, nodes, A)
    # Isotropic ansatz: gamma_rr = gamma_T = A**2, so chi = A**-2 and both
    # conformal components are one.
    assert np.allclose(np.asarray(metric.chi), np.asarray(1.0 / interpolated_A**2))
    assert np.allclose(np.asarray(metric.conformal_grr), 1.0)
    assert np.allclose(np.asarray(metric.conformal_gt), 1.0)
    assert np.allclose(np.asarray(metric.Kh), 0.0)


def test_freefall_step_falls_back_to_infinity_without_matter():
    r = generate_r_grid(0.0, 20.0, 64)
    metric = puncture_metric(r, MASS)

    from RadiShPICR.particles import particle_species

    empty = particle_species(
        name="dust", charge=0.0, mass=jnp.ones(4), weight=jnp.zeros(4),
        r=jnp.asarray([1.0, 2.0, 3.0, 4.0]), ur=jnp.zeros(4),
        phi=jnp.zeros(4), uphi=jnp.zeros(4), shape_mode="quadratic",
    )

    assert freefall_collapse_time_step(empty, metric) == math.inf

    heavy = particle_species(
        name="dust", charge=0.0, mass=jnp.ones(4), weight=jnp.ones(4),
        r=jnp.asarray([1.0, 2.0, 3.0, 4.0]), ur=jnp.zeros(4),
        phi=jnp.zeros(4), uphi=jnp.zeros(4), shape_mode="quadratic",
    )
    step = freefall_collapse_time_step(heavy, metric)
    assert math.isfinite(step) and step > 0.0
