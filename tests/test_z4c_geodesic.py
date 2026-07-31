import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.geodesic import compute_geodesic_terms
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


def test_flat_polar_covariant_momentum_rhs_with_unequal_particle_grid_counts():
    r = jnp.arange(0.5, 8.5, 1.0)
    metric = _metric_from_fields(
        r,
        alpha=jnp.ones_like(r),
        beta=jnp.zeros_like(r),
        chi=jnp.ones_like(r),
        conformal_grr=jnp.ones_like(r),
        conformal_gt=jnp.ones_like(r),
    )
    particles = particle_species(
        name="geodesic",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([2.5, 5.5]),
        ur=jnp.asarray([0.4, -0.2]),
        phi=jnp.asarray([0.1, 0.7]),
        uphi=jnp.asarray([0.7, 0.9]),
        shape_mode="nearest",
    )

    du_r_dt, du_phi_dt, dr_dt, dphi_dt = compute_geodesic_terms(
        particles,
        metric,
    )

    lorentz_factor = jnp.sqrt(
        1.0
        + particles.ur**2
        + particles.uphi**2 / particles.r**2
    )

    assert du_r_dt.shape == particles.r.shape
    assert jnp.allclose(
        du_r_dt,
        particles.uphi**2 / (lorentz_factor * particles.r**3),
    )
    assert jnp.allclose(du_phi_dt, 0.0)
    assert jnp.allclose(dr_dt, particles.ur / lorentz_factor)
    assert jnp.allclose(
        dphi_dt,
        particles.uphi / (lorentz_factor * particles.r**2),
    )


def test_covariant_momentum_rhs_matches_inverse_metric_fpic_reference():
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
    particles = particle_species(
        name="geodesic",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([r[8], r[18]]),
        ur=jnp.asarray([0.45, -0.35]),
        phi=jnp.asarray([0.2, 0.8]),
        uphi=jnp.asarray([0.6, 1.1]),
        shape_mode="nearest",
    )

    du_r_dt, du_phi_dt, dr_dt, dphi_dt = compute_geodesic_terms(
        particles,
        metric,
    )

    rp = particles.r
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
        1.0
        + gamma_rr_inv_p * particles.ur**2
        + gamma_phiphi_inv_p * particles.uphi**2
    )

    expected_du_r_dt = (
        -lorentz_factor * dalpha_dr_p
        + particles.ur * dbeta_dr_p
        - alpha_p
        / (2.0 * lorentz_factor)
        * (
            particles.ur**2 * dgamma_rr_inv_dr_p
            + particles.uphi**2 * dgamma_phiphi_inv_dr_p
        )
    )
    expected_dr_dt = (
        alpha_p * gamma_rr_inv_p * particles.ur / lorentz_factor
        - beta_p
    )
    expected_dphi_dt = (
        alpha_p
        * gamma_phiphi_inv_p
        * particles.uphi
        / lorentz_factor
    )

    assert jnp.allclose(du_r_dt, expected_du_r_dt, rtol=1.0e-6, atol=1.0e-7)
    assert jnp.allclose(du_phi_dt, 0.0)
    assert jnp.allclose(dr_dt, expected_dr_dt, rtol=1.0e-6, atol=1.0e-7)
    assert jnp.allclose(
        dphi_dt,
        expected_dphi_dt,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
