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


def _metric_derivative(
    metric,
    dalpha_dt=0.0,
    dchi_dt=0.0,
    dgrr_dt=0.0,
):
    zeros = jnp.zeros_like(metric.r)

    return Z4C_Metric(
        alpha=jnp.full_like(metric.r, dalpha_dt),
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
        alpha=0.7 + 0.03 * r,
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
    expected_ubar = (
        metric.alpha[jnp.asarray([5, 17])]
        * ur
        * jnp.sqrt(
            metric.chi[jnp.asarray([5, 17])]
            / metric.conformal_grr[jnp.asarray([5, 17])]
        )
    )

    assert jnp.allclose(determinant_reduced_radius, geometric_areal_radius)
    assert jnp.allclose(particles.ur, expected_ubar)
    assert jnp.allclose(eager_r, r_particle)
    assert jnp.allclose(eager_ur, ur)
    assert jnp.allclose(compiled_r, r_particle)
    assert jnp.allclose(compiled_ur, ur)


def test_monotonic_areal_lookup_matches_previous_interpolation_for_all_shapes():
    r = 0.25 + 0.25 * jnp.arange(32)
    conformal_grr = 1.1 + 0.02 * r
    metric = _metric_from_fields(
        r,
        alpha=0.7 + 0.03 * r,
        beta=jnp.zeros_like(r),
        chi=0.8 + 0.01 * r,
        conformal_grr=conformal_grr,
        conformal_gt=1.0 / jnp.sqrt(conformal_grr),
    )
    r_particle = jnp.asarray([0.0, 0.05, 0.25, 0.875, 4.13])
    ur = jnp.asarray([0.0, 0.1, -0.2, 0.4, -0.3])
    r_grid = jnp.concatenate((jnp.asarray([0.0]), metric.r))
    rs_grid = jnp.concatenate((jnp.asarray([0.0]), _areal_radius_grid(metric)))

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = _lapse_freezing_particles(
            metric,
            r_particle,
            ur,
            jnp.zeros_like(ur),
            shape_mode=shape_mode,
        )
        expected_r = jnp.interp(particles.r, rs_grid, r_grid)

        eager_r, eager_ur = isotropic_particle_state(particles, metric)
        compiled_r, compiled_ur = jax.jit(isotropic_particle_state)(
            particles,
            metric,
        )

        assert jnp.allclose(eager_r, expected_r, rtol=1.0e-14, atol=1.0e-14)
        assert jnp.allclose(eager_r, r_particle, rtol=1.0e-14, atol=1.0e-14)
        assert jnp.allclose(eager_ur, ur, rtol=1.0e-14, atol=1.0e-14)
        assert jnp.allclose(compiled_r, eager_r, rtol=1.0e-14, atol=1.0e-14)
        assert jnp.allclose(compiled_ur, eager_ur, rtol=1.0e-14, atol=1.0e-14)


def test_nonmonotonic_areal_map_uses_only_the_origin_branch():
    r = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    ones = jnp.ones_like(r)
    areal_radius = jnp.asarray([0.5, 1.0, 1.5, 1.499999])
    nonmonotonic_metric = _metric_from_fields(
        r,
        alpha=ones,
        beta=jnp.zeros_like(r),
        chi=(r / areal_radius) ** 2,
        conformal_grr=ones,
        conformal_gt=ones,
    )
    particles = particle_species(
        name="geodesic",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([0.25, 1.25, 1.5001]),
        ur=jnp.zeros(3),
        phi=jnp.zeros(3),
        uphi=jnp.zeros(3),
        shape_mode="nearest",
    )

    mapped_r, mapped_ur = isotropic_particle_state(
        particles,
        nonmonotonic_metric,
    )
    compiled_r, compiled_ur = jax.jit(isotropic_particle_state)(
        particles,
        nonmonotonic_metric,
    )
    assert jnp.allclose(mapped_r[:2], jnp.asarray([0.25, 1.25]))
    assert jnp.allclose(mapped_ur[:2], 0.0)
    assert jnp.isnan(mapped_r[-1])
    assert jnp.isnan(mapped_ur[-1])
    assert jnp.allclose(compiled_r[:2], mapped_r[:2])
    assert jnp.allclose(compiled_ur[:2], mapped_ur[:2])
    assert jnp.isnan(compiled_r[-1])
    assert jnp.isnan(compiled_ur[-1])

    mapped_rs, mapped_ubar = lapse_freezing_particle_state(
        jnp.asarray([1.25, 1.75]),
        jnp.asarray([0.2, -0.1]),
        nonmonotonic_metric,
        "nearest",
    )
    compiled_rs, compiled_ubar = jax.jit(
        lapse_freezing_particle_state,
        static_argnames=("shape_mode",),
    )(
        jnp.asarray([1.25, 1.75]),
        jnp.asarray([0.2, -0.1]),
        nonmonotonic_metric,
        "nearest",
    )
    expected_rs = jnp.interp(
        jnp.asarray([1.25, 1.75]),
        jnp.concatenate((jnp.asarray([0.0]), r)),
        jnp.concatenate((jnp.asarray([0.0]), areal_radius)),
    )
    assert jnp.all(jnp.isfinite(mapped_rs))
    assert jnp.all(jnp.isfinite(mapped_ubar))
    assert jnp.allclose(mapped_rs, expected_rs)
    assert jnp.allclose(compiled_rs, mapped_rs)
    assert jnp.allclose(compiled_ubar, mapped_ubar)

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
    expected_dubar_dt = (
        -(1.0 + c * r_particle**2) * 2.0 * c * r_particle
    )

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

        assert jnp.allclose(eager_terms[0], expected_dubar_dt, atol=3.0e-7)
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
        dalpha_dt=0.04,
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
    lapse_magnitude = jnp.abs(alpha_p)
    coordinate_energy = jnp.sqrt(alpha_p**2 + ubar**2)

    # LapseFreezingZ4C.nb: eqRsFinal after applying detgRule.
    metric_advection = grr_p * (
        4.0 * beta_p * jnp.sqrt(chi_p) * grr_p**0.75
        - beta_p * chi_p * dgrr_dr_p * rs
        - 0.02 * chi_p * rs
        - 2.0 * beta_p * dchi_dr_p * grr_p * rs
        + 0.06 * grr_p * rs
    )
    radial_motion = (
        -4.0 * chi_p * grr_p**1.25
        + dgrr_dr_p * jnp.sqrt(chi_p**3 * grr_p) * rs
        + 2.0 * dchi_dr_p * jnp.sqrt(chi_p * grr_p**3) * rs
    ) * ubar * lapse_magnitude / coordinate_energy
    expected_drs_dt = -(
        metric_advection + radial_motion
    ) / (4.0 * chi_p * grr_p**2)

    # LapseFreezingZ4C.nb: eqUbarFinal.
    expected_dubar_dt = (
        2.0
        * chi_p
        * (-beta_p * dalpha_dr_p + 0.04)
        * grr_p
        * ubar
        + (
            beta_p * chi_p * dgrr_dr_p
            + 0.02 * chi_p
            + 2.0 * chi_p * dbeta_dr_p * grr_p
            - beta_p * dchi_dr_p * grr_p
            + 0.03 * grr_p
        )
        * alpha_p
        * ubar
        - 2.0
        * dalpha_dr_p
        * jnp.sqrt(chi_p**3 * grr_p / coordinate_energy**2)
        * lapse_magnitude**3
    ) / (2.0 * chi_p * grr_p * alpha_p)

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
    rs = r_particle / jnp.sqrt(chi)

    dubar_dt, drs_dt = _explicit_lapse_freezing_rhs(
        rs=rs,
        ubar=ubar,
        alpha_p=alpha,
        beta_p=jnp.zeros_like(r_particle),
        chi_p=chi,
        conformal_grr_p=jnp.ones_like(r_particle),
        dalphadr_p=d_alpha_dr,
        dalphadt_p=jnp.zeros_like(r_particle),
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
    assert jnp.all(dubar_dt < 0.0)
    assert jnp.all(jnp.diff(jnp.abs(dubar_dt)) < 0.0)
    assert jnp.abs(drs_dt[-1]) < 1.0e-10
    assert jnp.abs(dubar_dt[-1]) < 1.0e-10
    assert jnp.allclose(rs[-1], 2.0 * mass, rtol=1.0e-6)

    # The revised notebook freezes both R and ubar at r_H = M / 2.


def test_fixed_schwarzschild_particle_freezes_at_event_horizon():
    mass = 1.0
    dt = 0.05
    num_steps = 1000

    def schwarzschild_rhs(rs, ubar):
        isotropic_r = 0.5 * (
            rs
            - mass
            + jnp.sqrt(rs * (rs - 2.0 * mass))
        )
        psi = 1.0 + mass / (2.0 * isotropic_r)
        alpha = (1.0 - mass / (2.0 * isotropic_r)) / psi
        chi = psi**-4
        dalpha_dr = mass / (isotropic_r**2 * psi**2)
        dchi_dr = 2.0 * mass / (isotropic_r**2 * psi**5)
        zeros = jnp.zeros_like(rs)
        ones = jnp.ones_like(rs)

        dubar_dt, drs_dt = _explicit_lapse_freezing_rhs(
            rs=rs,
            ubar=ubar,
            alpha_p=alpha,
            beta_p=zeros,
            chi_p=chi,
            conformal_grr_p=ones,
            dalphadr_p=dalpha_dr,
            dalphadt_p=zeros,
            dbetadr_p=zeros,
            dchidr_p=dchi_dr,
            dgrrdr_p=zeros,
            dchidt_p=zeros,
            dgrrdt_p=zeros,
        )

        return drs_dt, dubar_dt

    def advance_particle(carry, _):
        rs, ubar = carry

        k1_rs, k1_ubar = schwarzschild_rhs(rs, ubar)
        k2_rs, k2_ubar = schwarzschild_rhs(
            rs + 0.5 * dt * k1_rs,
            ubar + 0.5 * dt * k1_ubar,
        )
        k3_rs, k3_ubar = schwarzschild_rhs(
            rs + 0.5 * dt * k2_rs,
            ubar + 0.5 * dt * k2_ubar,
        )
        k4_rs, k4_ubar = schwarzschild_rhs(
            rs + dt * k3_rs,
            ubar + dt * k3_ubar,
        )

        rs = rs + (dt / 6.0) * (
            k1_rs + 2.0 * k2_rs + 2.0 * k3_rs + k4_rs
        )
        ubar = ubar + (dt / 6.0) * (
            k1_ubar + 2.0 * k2_ubar + 2.0 * k3_ubar + k4_ubar
        )

        return (rs, ubar), (rs, ubar)

    initial_state = (jnp.asarray(2.25), jnp.asarray(-0.4))
    (final_rs, final_ubar), (rs_history, ubar_history) = jax.lax.scan(
        advance_particle,
        initial_state,
        xs=None,
        length=num_steps,
    )
    final_drs_dt, final_dubar_dt = schwarzschild_rhs(
        final_rs,
        final_ubar,
    )

    assert jnp.all(jnp.diff(rs_history) <= 0.0)
    assert jnp.all(rs_history >= 2.0 * mass)
    assert jnp.all(jnp.isfinite(ubar_history))
    assert jnp.abs(final_rs - 2.0 * mass) < 1.0e-8 * mass
    assert jnp.abs(final_drs_dt) < 1.0e-10
    assert jnp.abs(final_dubar_dt) < 1.0e-10
