import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import interpolate_fields_to_particles
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.geodesic import (
    _areal_radius_grid,
    _explicit_lapse_freezing_rhs,
    compute_geodesic_terms,
    isotropic_particle_state,
    lapse_freezing_particle_state,
)
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _metric_from_fields(r, alpha, beta, chi, conformal_grr, conformal_gt):
    zeros = jnp.zeros_like(r)

    return Z4C_Metric(
        alpha=alpha,
        beta=beta,
        conformal_grr=conformal_grr,
        conformal_gt=conformal_gt,
        chi=chi,
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Gamma=zeros,
        kappa=zeros,
        eta=zeros,
        nu=zeros,
        r=r,
        dr=r[1] - r[0],
    )


def _metric_derivative(metric, dchi_dt=0.0, dgrr_dt=0.0):
    zeros = jnp.zeros_like(metric.r)

    return Z4C_Metric(
        alpha=zeros,
        beta=zeros,
        conformal_grr=jnp.full_like(metric.r, dgrr_dt),
        conformal_gt=zeros,
        chi=jnp.full_like(metric.r, dchi_dt),
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Gamma=zeros,
        kappa=zeros,
        eta=zeros,
        nu=zeros,
        r=zeros,
        dr=jnp.asarray(0.0, dtype=metric.dr.dtype),
    )


def _lapse_freezing_particles(metric, r_particle, ur, uphi, shape_mode="nearest"):
    rs, ubar = lapse_freezing_particle_state(
        r_particle,
        ur,
        metric,
        shape_mode,
    )

    return particle_species(
        name="geodesic",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=rs,
        ur=ubar,
        phi=jnp.linspace(0.1, 0.7, r_particle.size),
        uphi=uphi,
        shape_mode=shape_mode,
    )


def test_lapse_freezing_particle_state_round_trip_and_jit():
    r = 0.25 + 0.25 * jnp.arange(32)
    conformal_grr = 1.1 + 0.02 * r
    metric = _metric_from_fields(
        r,
        alpha=jnp.ones_like(r),
        beta=jnp.zeros_like(r),
        chi=0.8 + 0.01 * r,
        conformal_grr=conformal_grr,
        conformal_gt=1.0 / jnp.sqrt(conformal_grr),
    )
    r_particle = jnp.asarray([r[5], r[17]])
    ur = jnp.asarray([0.4, -0.2])
    particles = _lapse_freezing_particles(
        metric,
        r_particle,
        ur,
        jnp.zeros_like(ur),
        shape_mode="linear",
    )

    eager_r, eager_ur = isotropic_particle_state(particles, metric)
    compiled_r, compiled_ur = jax.jit(isotropic_particle_state)(particles, metric)
    determinant_reduced_radius = _areal_radius_grid(metric)
    geometric_areal_radius = metric.r * jnp.sqrt(
        metric.conformal_gt / metric.chi
    )

    assert jnp.allclose(determinant_reduced_radius, geometric_areal_radius)
    assert jnp.allclose(eager_r, r_particle)
    assert jnp.allclose(eager_ur, ur)
    assert jnp.allclose(compiled_r, r_particle)
    assert jnp.allclose(compiled_ur, ur)


def test_invalid_areal_radius_maps_and_particle_positions_are_nonfinite():
    r = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    ones = jnp.ones_like(r)
    nonmonotonic_metric = _metric_from_fields(
        r,
        alpha=ones,
        beta=jnp.zeros_like(r),
        chi=jnp.asarray([0.25, 4.0, 0.5625, 0.25]),
        conformal_grr=ones,
        conformal_gt=ones,
    )
    particles = particle_species(
        name="geodesic",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([1.25]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )

    mapped_r, mapped_ur = isotropic_particle_state(
        particles,
        nonmonotonic_metric,
    )
    assert jnp.all(jnp.isnan(mapped_r))
    assert jnp.all(jnp.isnan(mapped_ur))

    mapped_rs, mapped_ubar = lapse_freezing_particle_state(
        jnp.asarray([1.0]),
        jnp.asarray([0.2]),
        nonmonotonic_metric,
        "nearest",
    )
    assert jnp.all(jnp.isnan(mapped_rs))
    assert jnp.all(jnp.isnan(mapped_ubar))

    flat_metric = nonmonotonic_metric._replace(chi=ones)
    particles.r = jnp.asarray([-0.1, 3.0])
    particles.ur = jnp.zeros(2)
    particles.phi = jnp.zeros(2)
    particles.uphi = jnp.zeros(2)
    mapped_r, mapped_ur = isotropic_particle_state(particles, flat_metric)
    assert jnp.all(jnp.isnan(mapped_r))
    assert jnp.all(jnp.isnan(mapped_ur))

    compiled_r, compiled_ur = jax.jit(isotropic_particle_state)(
        particles,
        flat_metric,
    )
    assert jnp.all(jnp.isnan(compiled_r))
    assert jnp.all(jnp.isnan(compiled_ur))

    invalid_rs, invalid_ubar = lapse_freezing_particle_state(
        jnp.asarray([-0.1, 2.5]),
        jnp.asarray([0.2, 0.2]),
        flat_metric,
        "nearest",
    )
    assert jnp.all(jnp.isnan(invalid_rs))
    assert jnp.all(jnp.isnan(invalid_ubar))

    center_rs, center_ubar = lapse_freezing_particle_state(
        jnp.asarray([0.25]),
        jnp.asarray([0.2]),
        flat_metric,
        "nearest",
    )
    center_particles = particle_species(
        name="geodesic",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=center_rs,
        ur=center_ubar,
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )
    center_r, center_ur = isotropic_particle_state(center_particles, flat_metric)
    assert jnp.allclose(center_rs, 0.25)
    assert jnp.allclose(center_r, 0.25)
    assert jnp.allclose(center_ur, 0.2)


def test_flat_radial_lapse_freezing_rhs_is_exact():
    r = jnp.arange(0.5, 8.5, 1.0)
    metric = _metric_from_fields(
        r,
        alpha=jnp.ones_like(r),
        beta=jnp.zeros_like(r),
        chi=jnp.ones_like(r),
        conformal_grr=jnp.ones_like(r),
        conformal_gt=jnp.ones_like(r),
    )
    r_particle = jnp.asarray([2.5, 5.5])
    ur = jnp.asarray([0.4, -0.2])
    uphi = jnp.zeros_like(ur)
    particles = _lapse_freezing_particles(metric, r_particle, ur, uphi)

    dubar_dt, du_phi_dt, drs_dt, dphi_dt = compute_geodesic_terms(
        particles,
        metric,
        _metric_derivative(metric),
    )
    lorentz_factor = jnp.sqrt(1.0 + ur**2)

    assert jnp.allclose(dubar_dt, 0.0)
    assert jnp.allclose(du_phi_dt, 0.0)
    assert jnp.allclose(drs_dt, ur / lorentz_factor)
    assert jnp.allclose(dphi_dt, 0.0)


def test_even_quadratic_lapse_has_regular_inner_cell_acceleration():
    dr = 0.04
    r = 0.5 * dr + dr * jnp.arange(32)
    c = 0.1
    metric = _metric_from_fields(
        r,
        alpha=1.0 + c * r**2,
        beta=jnp.zeros_like(r),
        chi=jnp.ones_like(r),
        conformal_grr=jnp.ones_like(r),
        conformal_gt=jnp.ones_like(r),
    )
    metric_derivative = _metric_derivative(metric)
    r_particle = jnp.asarray([0.0, 0.001, 0.005, 0.01, 0.019, 0.02, 0.03])
    expected_dubar_dt = -2.0 * c * r_particle

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = _lapse_freezing_particles(
            metric,
            r_particle,
            jnp.zeros_like(r_particle),
            jnp.zeros_like(r_particle),
            shape_mode=shape_mode,
        )
        eager_terms = compute_geodesic_terms(
            particles,
            metric,
            metric_derivative,
        )
        compiled_terms = jax.jit(compute_geodesic_terms)(
            particles,
            metric,
            metric_derivative,
        )

        assert jnp.allclose(eager_terms[0], expected_dubar_dt, atol=1.0e-7)
        assert jnp.allclose(eager_terms[1], 0.0)
        assert jnp.allclose(eager_terms[2], 0.0)
        assert jnp.allclose(eager_terms[3], 0.0)
        for eager_term, compiled_term in zip(eager_terms, compiled_terms):
            assert jnp.allclose(compiled_term, eager_term)


def test_nonzero_angular_momentum_is_rejected_eager_and_jit():
    r = jnp.arange(0.5, 8.5, 1.0)
    metric = _metric_from_fields(
        r,
        alpha=jnp.ones_like(r),
        beta=jnp.zeros_like(r),
        chi=jnp.ones_like(r),
        conformal_grr=jnp.ones_like(r),
        conformal_gt=jnp.ones_like(r),
    )
    particles = _lapse_freezing_particles(
        metric,
        jnp.asarray([2.5]),
        jnp.asarray([0.4]),
        jnp.asarray([0.7]),
    )
    metric_derivative = _metric_derivative(metric)

    eager_terms = compute_geodesic_terms(particles, metric, metric_derivative)
    compiled_terms = jax.jit(compute_geodesic_terms)(
        particles,
        metric,
        metric_derivative,
    )

    for eager_term, compiled_term in zip(eager_terms, compiled_terms):
        assert jnp.all(jnp.isnan(eager_term))
        assert jnp.all(jnp.isnan(compiled_term))


def test_shift_aware_lapse_freezing_rhs_matches_notebook_expression():
    r = 0.25 + 0.25 * jnp.arange(32)
    conformal_grr = 1.1 + 0.01 * r
    metric = _metric_from_fields(
        r,
        alpha=1.0 + 0.01 * r,
        beta=0.02 * r,
        chi=0.9 + 0.015 * r,
        conformal_grr=conformal_grr,
        conformal_gt=1.0 / jnp.sqrt(conformal_grr),
    )
    metric_derivative = _metric_derivative(
        metric,
        dchi_dt=0.03,
        dgrr_dt=-0.02,
    )
    r_particle = jnp.asarray([r[7], r[20]])
    ur = jnp.asarray([0.3, -0.4])
    particles = _lapse_freezing_particles(
        metric,
        r_particle,
        ur,
        jnp.zeros_like(ur),
    )

    dubar_dt, _, drs_dt, _ = compute_geodesic_terms(
        particles,
        metric,
        metric_derivative,
    )
    grid = RadialGrid(r, r, metric.dr, r[-1])
    (
        alpha_p,
        beta_p,
        chi_p,
        grr_p,
        dalpha_dr_p,
        dbeta_dr_p,
        dchi_dr_p,
        dgrr_dr_p,
    ) = interpolate_fields_to_particles(
        jnp.stack(
            (
                metric.alpha,
                metric.beta,
                metric.chi,
                metric.conformal_grr,
                first_derivative(metric.alpha, metric.dr, parity=1),
                first_derivative(metric.beta, metric.dr, parity=-1),
                first_derivative(metric.chi, metric.dr, parity=1),
                first_derivative(metric.conformal_grr, metric.dr, parity=1),
            )
        ),
        r_particle,
        grid,
        shape_mode=particles.shape_mode,
    )
    rs = particles.r
    ubar = particles.ur
    lorentz_factor = jnp.sqrt(1.0 + ubar**2)

    # LapseFreezingZ4C.nb: eqRsFinal after applying detgRule.
    expected_drs_dt = (
        4.0 * chi_p * grr_p**1.25 * alpha_p * ubar
        - dgrr_dr_p
        * jnp.sqrt(chi_p**3 * grr_p)
        * alpha_p
        * rs
        * ubar
        - 2.0
        * dchi_dr_p
        * jnp.sqrt(chi_p * grr_p**3)
        * alpha_p
        * rs
        * ubar
        - 4.0
        * beta_p
        * jnp.sqrt(chi_p)
        * grr_p**1.75
        * lorentz_factor
        + grr_p
        * (
            beta_p * chi_p * dgrr_dr_p
            + 0.02 * chi_p
            + 2.0 * beta_p * dchi_dr_p * grr_p
            - 0.06 * grr_p
        )
        * rs
        * lorentz_factor
    ) / (4.0 * chi_p * grr_p**2 * lorentz_factor)

    # LapseFreezingZ4C.nb: eqUbarFinal.
    expected_dubar_dt = (
        (
            beta_p * chi_p * dgrr_dr_p
            + 0.02 * chi_p
            + 2.0 * chi_p * dbeta_dr_p * grr_p
            - beta_p * dchi_dr_p * grr_p
            + 0.03 * grr_p
        )
        * ubar
        - 2.0
        * dalpha_dr_p
        * jnp.sqrt(chi_p**3 * grr_p)
        * lorentz_factor
    ) / (2.0 * chi_p * grr_p)

    assert jnp.allclose(drs_dt, expected_drs_dt)
    assert jnp.allclose(dubar_dt, expected_dubar_dt)

    compiled_terms = jax.jit(compute_geodesic_terms)(
        particles,
        metric,
        metric_derivative,
    )
    eager_terms = compute_geodesic_terms(particles, metric, metric_derivative)
    for compiled_term, eager_term in zip(compiled_terms, eager_terms):
        assert jnp.allclose(compiled_term, eager_term)


def test_fixed_schwarzschild_areal_radius_and_notebook_horizon_rhs():
    mass = 1.0
    r_particle = jnp.asarray(
        [1.10, 1.00, 0.90, 0.75, 0.60, 0.55, 0.51, 0.501, 0.5001, 0.50001]
    )

    psi = 1.0 + mass / (2.0 * r_particle)
    alpha = (1.0 - mass / (2.0 * r_particle)) / psi
    chi = psi**-4
    d_alpha_dr = mass / (r_particle**2 * psi**2)
    d_chi_dr = 2.0 * mass / (r_particle**2 * psi**5)

    ubar = jnp.full_like(r_particle, -0.4)
    lorentz_factor = jnp.sqrt(1.0 + ubar**2)
    rs = r_particle / jnp.sqrt(chi)

    dubar_dt, drs_dt = _explicit_lapse_freezing_rhs(
        rs=rs,
        ubar=ubar,
        alpha_p=alpha,
        beta_p=jnp.zeros_like(r_particle),
        chi_p=chi,
        conformal_grr_p=jnp.ones_like(r_particle),
        dalphadr_p=d_alpha_dr,
        dbetadr_p=jnp.zeros_like(r_particle),
        dchidr_p=d_chi_dr,
        dgrrdr_p=jnp.zeros_like(r_particle),
        dchidt_p=jnp.zeros_like(r_particle),
        dgrrdt_p=jnp.zeros_like(r_particle),
    )

    # R = r (1 + M / (2 r))**2 is strictly monotonic outside the horizon.
    assert jnp.all(jnp.diff(rs[::-1]) > 0.0)
    assert jnp.all(drs_dt < 0.0)
    assert jnp.all(jnp.diff(jnp.abs(drs_dt)) < 0.0)
    assert jnp.abs(drs_dt[-1]) < 1.0e-7
    assert jnp.allclose(rs[-1], 2.0 * mass, rtol=1.0e-6)

    # The notebook freezes R, but not ubar, at r_H = M / 2.
    expected_horizon_dubar_dt = -lorentz_factor[-1] / (4.0 * mass)
    assert jnp.allclose(
        dubar_dt[-1],
        expected_horizon_dubar_dt,
        rtol=1.0e-4,
        atol=1.0e-6,
    )
