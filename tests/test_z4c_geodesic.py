import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import interpolate_fields_to_particles
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.geodesic import (
    _explicit_lapse_freezing_rhs,
    compute_geodesic_terms,
    compute_normal_geodesic_terms,
    lapse_freezing_particle_state,
    normal_particle_state,
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
    metric = _metric_from_fields(
        r,
        alpha=jnp.ones_like(r),
        beta=jnp.zeros_like(r),
        chi=0.8 + 0.01 * r,
        conformal_grr=1.1 + 0.02 * r,
        conformal_gt=jnp.ones_like(r),
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

    eager_r, eager_ur = normal_particle_state(particles, metric)
    compiled_r, compiled_ur = jax.jit(normal_particle_state)(particles, metric)

    assert jnp.allclose(eager_r, r_particle)
    assert jnp.allclose(eager_ur, ur)
    assert jnp.allclose(compiled_r, r_particle)
    assert jnp.allclose(compiled_ur, ur)


def test_flat_polar_lapse_freezing_rhs_matches_ordinary_geodesic():
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
    uphi = jnp.asarray([0.7, 0.9])
    particles = _lapse_freezing_particles(metric, r_particle, ur, uphi)

    dubar_dt, du_phi_dt, drs_dt, dphi_dt = compute_geodesic_terms(
        particles,
        metric,
        _metric_derivative(metric),
    )
    lorentz_factor = jnp.sqrt(1.0 + ur**2 + uphi**2 / r_particle**2)

    assert jnp.allclose(
        dubar_dt,
        uphi**2 / (lorentz_factor * r_particle**3),
    )
    assert jnp.allclose(du_phi_dt, 0.0)
    assert jnp.allclose(drs_dt, ur / lorentz_factor)
    assert jnp.allclose(
        dphi_dt,
        uphi / (lorentz_factor * r_particle**2),
    )


def test_normal_geodesic_rhs_matches_inverse_metric_reference():
    r = 0.25 + 0.25 * jnp.arange(32)
    alpha = 1.0 + 0.03 * r + 0.002 * r**2
    beta = 0.01 * r
    chi = 1.0 + 0.02 * r**2
    conformal_grr = 1.0 + 0.015 * r**2
    conformal_gt = 1.0 + 0.01 * r**2
    metric = _metric_from_fields(
        r,
        alpha,
        beta,
        chi,
        conformal_grr,
        conformal_gt,
    )
    rp = jnp.asarray([r[8], r[18]])
    ur = jnp.asarray([0.45, -0.35])
    uphi = jnp.asarray([0.6, 1.1])
    particles = _lapse_freezing_particles(metric, rp, ur, uphi)

    du_r_dt, du_phi_dt, dr_dt, dphi_dt = compute_normal_geodesic_terms(
        particles,
        metric,
    )

    alpha_p = 1.0 + 0.03 * rp + 0.002 * rp**2
    beta_p = 0.01 * rp
    chi_p = 1.0 + 0.02 * rp**2
    conformal_grr_p = 1.0 + 0.015 * rp**2
    conformal_gt_p = 1.0 + 0.01 * rp**2
    dalpha_dr_p = 0.03 + 0.004 * rp
    dbeta_dr_p = jnp.full_like(rp, 0.01)
    dchi_dr_p = 0.04 * rp
    dgrr_dr_p = 0.03 * rp
    dgt_dr_p = 0.02 * rp

    gamma_rr_inv_p = chi_p / conformal_grr_p
    gamma_phiphi_inv_p = chi_p / (rp**2 * conformal_gt_p)
    dgamma_rr_inv_dr_p = (
        dchi_dr_p / conformal_grr_p
        - chi_p * dgrr_dr_p / conformal_grr_p**2
    )
    dgamma_phiphi_inv_dr_p = (
        dchi_dr_p / (rp**2 * conformal_gt_p)
        - 2.0 * chi_p / (rp**3 * conformal_gt_p)
        - chi_p * dgt_dr_p / (rp**2 * conformal_gt_p**2)
    )
    lorentz_factor = jnp.sqrt(
        1.0 + gamma_rr_inv_p * ur**2 + gamma_phiphi_inv_p * uphi**2
    )
    expected_du_r_dt = (
        -lorentz_factor * dalpha_dr_p
        + ur * dbeta_dr_p
        - alpha_p
        / (2.0 * lorentz_factor)
        * (
            ur**2 * dgamma_rr_inv_dr_p
            + uphi**2 * dgamma_phiphi_inv_dr_p
        )
    )
    expected_dr_dt = alpha_p * gamma_rr_inv_p * ur / lorentz_factor - beta_p
    expected_dphi_dt = alpha_p * gamma_phiphi_inv_p * uphi / lorentz_factor

    assert jnp.allclose(du_r_dt, expected_du_r_dt, rtol=1.0e-6, atol=1.0e-7)
    assert jnp.allclose(du_phi_dt, 0.0)
    assert jnp.allclose(dr_dt, expected_dr_dt, rtol=1.0e-6, atol=1.0e-7)
    assert jnp.allclose(dphi_dt, expected_dphi_dt, rtol=1.0e-6, atol=1.0e-7)


def test_shift_aware_lapse_freezing_rhs_matches_notebook_expression():
    r = 0.25 + 0.25 * jnp.arange(32)
    metric = _metric_from_fields(
        r,
        alpha=1.0 + 0.01 * r,
        beta=0.02 * r,
        chi=0.9 + 0.015 * r,
        conformal_grr=1.1 + 0.01 * r,
        conformal_gt=1.0 + 0.005 * r,
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
        jnp.asarray([0.2, 0.5]),
    )

    dubar_dt, _, drs_dt, _ = compute_geodesic_terms(
        particles,
        metric,
        metric_derivative,
    )
    du_r_dt, _, dr_dt, _ = compute_normal_geodesic_terms(particles, metric)

    grid = RadialGrid(r, r, metric.dr, r[-1])
    chi_p, grr_p, dchi_dr_p, dgrr_dr_p = interpolate_fields_to_particles(
        jnp.stack(
            (
                metric.chi,
                metric.conformal_grr,
                first_derivative(metric.chi, metric.dr, parity=1),
                first_derivative(metric.conformal_grr, metric.dr, parity=1),
            )
        ),
        r_particle,
        grid,
        shape_mode=particles.shape_mode,
    )
    alpha_p = 1.0 + 0.01 * r_particle
    beta_p = 0.02 * r_particle
    gt_p = 1.0 + 0.005 * r_particle
    rs = particles.r
    ubar = particles.ur
    uphi = particles.uphi
    lorentz_factor = jnp.sqrt(
        1.0
        + chi_p * ur**2 / grr_p
        + chi_p * uphi**2 / (r_particle**2 * gt_p)
    )
    sqrt_grr_over_chi = jnp.sqrt(grr_p / chi_p)
    coordinate_factor = (
        4.0 * chi_p - 3.0 * chi_p**(3.0 / 4.0) * dchi_dr_p * rs
    )
    expected_drs_dt = (
        alpha_p * coordinate_factor * ubar
        + sqrt_grr_over_chi
        * (
            -3.0 * chi_p**(3.0 / 4.0) * 0.03 * rs
            - beta_p * coordinate_factor
        )
        * lorentz_factor
    ) / (
        4.0
        * chi_p**(7.0 / 4.0)
        * sqrt_grr_over_chi
        * lorentz_factor
    )

    conformal_metric_term = ubar * (
        -4.0 * chi_p * dgrr_dr_p * alpha_p * ubar
        + (
            4.0 * beta_p * chi_p * dgrr_dr_p
            + 4.0 * chi_p * 0.02
        )
        * sqrt_grr_over_chi
        * lorentz_factor
    )
    momentum_term = (
        4.0 * 0.03 * sqrt_grr_over_chi * ubar * lorentz_factor
        + 8.0 * chi_p * du_r_dt * lorentz_factor
        + 6.0
        * chi_p**(3.0 / 4.0)
        * dchi_dr_p
        * rs
        * du_r_dt
        * lorentz_factor
        + dchi_dr_p
        * (
            4.0 * alpha_p * ubar**2
            - 4.0
            * beta_p
            * sqrt_grr_over_chi
            * ubar
            * lorentz_factor
            - 6.0
            * chi_p**(3.0 / 4.0)
            * rs
            * du_r_dt
            * lorentz_factor
        )
    )
    expected_dubar_dt = (
        conformal_metric_term + grr_p * momentum_term
    ) / (
        8.0
        * chi_p
        * grr_p
        * sqrt_grr_over_chi
        * lorentz_factor
    )

    total_dchi_dt = 0.03 + dchi_dr_p * dr_dt
    total_dgrr_dt = -0.02 + dgrr_dr_p * dr_dt
    chain_rule_drs_dt = (
        dr_dt / chi_p**(3.0 / 4.0)
        - 3.0 * rs * total_dchi_dt / (4.0 * chi_p)
    )
    chain_rule_dubar_dt = (
        jnp.sqrt(chi_p / grr_p) * du_r_dt
        + 0.5
        * ubar
        * (total_dchi_dt / chi_p - total_dgrr_dt / grr_p)
    )

    assert jnp.allclose(drs_dt, expected_drs_dt)
    assert jnp.allclose(dubar_dt, expected_dubar_dt)
    assert jnp.allclose(drs_dt, chain_rule_drs_dt)
    assert jnp.allclose(dubar_dt, chain_rule_dubar_dt)

    compiled_terms = jax.jit(compute_geodesic_terms)(
        particles,
        metric,
        metric_derivative,
    )
    eager_terms = compute_geodesic_terms(particles, metric, metric_derivative)
    for compiled_term, eager_term in zip(compiled_terms, eager_terms):
        assert jnp.allclose(compiled_term, eager_term)


def test_fixed_schwarzschild_radial_infall_freezes_position_rhs():
    mass = 1.0
    r_initial = 5.0
    ubar_initial = -0.1
    # Sample one conserved-energy trajectory as it approaches r_H = M / 2.
    r_particle = jnp.asarray(
        [1.10, 1.00, 0.90, 0.75, 0.60, 0.55, 0.51, 0.501, 0.5001, 0.50001]
    )

    psi = 1.0 + mass / (2.0 * r_particle)
    alpha = (1.0 - mass / (2.0 * r_particle)) / psi
    chi = psi**-4
    d_alpha_dr = mass / (r_particle**2 * psi**2)
    d_chi_dr = 2.0 * mass / (r_particle**2 * psi**5)

    psi_initial = 1.0 + mass / (2.0 * r_initial)
    alpha_initial = (1.0 - mass / (2.0 * r_initial)) / psi_initial
    conserved_energy = alpha_initial * jnp.sqrt(1.0 + ubar_initial**2)
    lorentz_factor = conserved_energy / alpha
    ubar = -jnp.sqrt(lorentz_factor**2 - 1.0)
    ur = ubar / jnp.sqrt(chi)
    rs = r_particle / chi**(3.0 / 4.0)

    du_r_dt = -lorentz_factor * d_alpha_dr
    du_r_dt -= alpha * ur**2 * d_chi_dr / (2.0 * lorentz_factor)

    dubar_dt, drs_dt = _explicit_lapse_freezing_rhs(
        rs=rs,
        ubar=ubar,
        uphi=jnp.zeros_like(r_particle),
        r_particle=r_particle,
        ur=ur,
        du_r_dt=du_r_dt,
        alpha_p=alpha,
        beta_p=jnp.zeros_like(r_particle),
        chi_p=chi,
        conformal_grr_p=jnp.ones_like(r_particle),
        conformal_gt_p=jnp.ones_like(r_particle),
        dchidr_p=d_chi_dr,
        dgrrdr_p=jnp.zeros_like(r_particle),
        dchidt_p=jnp.zeros_like(r_particle),
        dgrrdt_p=jnp.zeros_like(r_particle),
    )

    assert drs_dt[0] < 0.0
    assert jnp.abs(drs_dt[1]) < 1.0e-12
    assert jnp.all(drs_dt[2:] > 0.0)
    assert jnp.all(jnp.diff(jnp.abs(drs_dt[4:])) < 0.0)
    assert jnp.abs(drs_dt[-1]) < 1.1e-5

    # The momentum relative to static Eulerian observers does not freeze.
    assert jnp.all(jnp.diff(jnp.abs(dubar_dt[4:])) > 0.0)

    horizon_radius = 0.5 * mass
    exterior_collision_radius = (1.0 + jnp.sqrt(5.0) / 2.0) * mass

    def schwarzschild_rs(radius):
        psi_at_radius = 1.0 + mass / (2.0 * radius)
        chi_at_radius = psi_at_radius**-4
        return radius / chi_at_radius**(3.0 / 4.0)

    assert jnp.allclose(
        schwarzschild_rs(horizon_radius),
        schwarzschild_rs(exterior_collision_radius),
    )
    assert jnp.allclose(schwarzschild_rs(horizon_radius), 4.0 * mass)
