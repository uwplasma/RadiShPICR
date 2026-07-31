import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import (
    interpolate_field_to_particles,
    shape_weights_at_point,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    MatterTerms,
    compute_radial_momentum_density,
    compute_radial_stress_tensor_component,
    compute_radial_matter_terms,
    initialize_vacuum_matter_terms,
    relativistic_mass_energy_density,
)
from RadiShPICR.Z4C.constraint_terms import dGammadt
from RadiShPICR.Z4C.extrinsic_curvature import dArrdt, dAtdt, dKhdt
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def test_initialize_vacuum_matter_terms_matches_metric_grid():
    r = jnp.linspace(0.1, 1.0, 8)
    zeros = jnp.zeros_like(r)

    metric = Z4C_Metric(
        alpha=jnp.ones_like(r),
        beta=zeros,
        conformal_grr=jnp.ones_like(r),
        conformal_gt=jnp.ones_like(r),
        chi=jnp.ones_like(r),
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

    matter_terms = initialize_vacuum_matter_terms(metric)

    assert matter_terms.rho.shape == r.shape
    assert matter_terms.Srr.shape == r.shape
    assert matter_terms.Stt.shape == r.shape
    assert matter_terms.Sr.shape == r.shape
    assert matter_terms.St.shape == r.shape

    assert jnp.allclose(matter_terms.rho, 0.0)
    assert jnp.allclose(matter_terms.Srr, 0.0)
    assert jnp.allclose(matter_terms.Stt, 0.0)
    assert jnp.allclose(matter_terms.Sr, 0.0)
    assert jnp.allclose(matter_terms.St, 0.0)


def test_sparse_matter_deposition_matches_dense_reference():
    r = jnp.arange(0.5, 6.0, 1.0)
    zeros = jnp.zeros_like(r)
    metric = Z4C_Metric(
        alpha=jnp.ones_like(r),
        beta=zeros,
        conformal_grr=1.0 + 0.02 * r,
        conformal_gt=1.0 + 0.01 * r,
        chi=1.0 / (1.0 + 0.03 * r),
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
    grid = RadialGrid(
        r_full=r,
        r_interior=r,
        dr=metric.dr,
        r_max=r[-1],
    )

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = particle_species(
            name="matter",
            charge=0.0,
            mass=2.0,
            weight=jnp.asarray([0.2, 0.3, 0.5]),
            r=jnp.asarray([0.75, 2.25, 5.25]),
            ur=jnp.asarray([0.4, -0.2, 0.7]),
            phi=jnp.zeros(3),
            uphi=jnp.asarray([0.1, 0.3, -0.2]),
            shape_mode=shape_mode,
        )

        scaling_factor = jnp.sqrt(1.0 / metric.chi**3)
        scaling_factor_p = interpolate_field_to_particles(
            scaling_factor,
            particles.r,
            grid,
            shape_mode=shape_mode,
        )
        grr_p = interpolate_field_to_particles(
            metric.conformal_grr / metric.chi,
            particles.r,
            grid,
            shape_mode=shape_mode,
        )
        gt_p = interpolate_field_to_particles(
            metric.conformal_gt / metric.chi,
            particles.r,
            grid,
            shape_mode=shape_mode,
        )
        safe_r_particle = jnp.maximum(particles.r, 0.5 * metric.dr)
        particle_volume = 4.0 * jnp.pi * particles.r**2 * scaling_factor_p
        gamma_rr_inv_p = 1.0 / grr_p
        lorentz_factor = jnp.sqrt(
            1.0
            + gamma_rr_inv_p * particles.ur**2
            + particles.uphi**2 / (safe_r_particle**2 * gt_p)
        )
        weights = shape_weights_at_point(
            particles.r[jnp.newaxis, :],
            r[:, jnp.newaxis],
            metric.dr,
            shape_mode=shape_mode,
        )
        particle_mass = particles.get_mass()
        expected_rho = jnp.sum(
            weights
            * particle_mass
            * lorentz_factor
            / particle_volume,
            axis=1,
        )
        expected_Srr = jnp.sum(
            weights
            * particle_mass
            * particles.ur**2
            / (particle_volume * lorentz_factor),
            axis=1,
        )
        expected_Stt = jnp.sum(
            weights
            * particle_mass
            * particles.uphi**2
            / (
                2.0
                * safe_r_particle**2
                * particle_volume
                * lorentz_factor
            ),
            axis=1,
        )
        expected_Sr = jnp.sum(
            weights
            * particle_mass
            * gamma_rr_inv_p
            * particles.ur
            / particle_volume,
            axis=1,
        )

        matter_terms = compute_radial_matter_terms(particles, metric)

        assert jnp.allclose(matter_terms.rho, expected_rho)
        assert jnp.allclose(matter_terms.Srr, expected_Srr)
        assert jnp.allclose(matter_terms.Stt, expected_Stt)
        assert jnp.allclose(matter_terms.Sr, expected_Sr)
        assert jnp.allclose(matter_terms.St, 0.0)
        assert jnp.allclose(
            relativistic_mass_energy_density(particles, metric),
            expected_rho,
        )
        assert jnp.allclose(
            compute_radial_stress_tensor_component(particles, metric),
            expected_Srr,
        )
        assert jnp.allclose(
            compute_radial_momentum_density(particles, metric),
            expected_Sr,
        )


def test_matter_stress_trace_and_zero_angular_momentum_limit():
    r = jnp.arange(0.5, 6.5, 1.0)
    zeros = jnp.zeros_like(r)
    conformal_grr = jnp.full_like(r, 1.25)
    conformal_gt = jnp.full_like(r, 0.9)
    chi = jnp.full_like(r, 0.8)
    metric = Z4C_Metric(
        alpha=jnp.ones_like(r),
        beta=zeros,
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
    particle_index = 2
    particles = particle_species(
        name="matter",
        charge=0.0,
        mass=2.0,
        weight=jnp.asarray([0.7]),
        r=jnp.asarray([r[particle_index]]),
        ur=jnp.asarray([-0.4]),
        phi=jnp.asarray([0.3]),
        uphi=jnp.asarray([0.8]),
        shape_mode="nearest",
    )

    matter_terms = compute_radial_matter_terms(particles, metric)

    rp = particles.r[0]
    gamma_rr_inv = chi[particle_index] / conformal_grr[particle_index]
    gamma_t_inv = chi[particle_index] / conformal_gt[particle_index]
    lorentz_factor = jnp.sqrt(
        1.0
        + gamma_rr_inv * particles.ur[0] ** 2
        + gamma_t_inv * particles.uphi[0] ** 2 / rp**2
    )
    particle_volume = 4.0 * jnp.pi * rp**2 / chi[particle_index] ** 1.5
    particle_mass = particles.get_mass()[0]
    stress_trace = (
        gamma_rr_inv * matter_terms.Srr[particle_index]
        + 2.0 * gamma_t_inv * matter_terms.Stt[particle_index]
    )
    expected_trace = (
        particle_mass
        * (lorentz_factor**2 - 1.0)
        / (particle_volume * lorentz_factor)
    )

    assert jnp.allclose(stress_trace, expected_trace)
    assert jnp.allclose(matter_terms.St, 0.0)

    radial_particles = particle_species(
        name="matter",
        charge=0.0,
        mass=particles.masses,
        weight=particles.weight,
        r=particles.r,
        ur=particles.ur,
        phi=particles.phi,
        uphi=jnp.zeros_like(particles.uphi),
        shape_mode=particles.shape_mode,
    )
    radial_matter_terms = compute_radial_matter_terms(radial_particles, metric)

    assert jnp.allclose(radial_matter_terms.Stt, 0.0)
    assert jnp.allclose(radial_matter_terms.St, 0.0)


def test_tangential_stress_drives_extrinsic_curvature_sources():
    r = jnp.arange(0.5, 8.5, 1.0)
    zeros = jnp.zeros_like(r)
    metric = Z4C_Metric(
        alpha=1.0 + 0.01 * r,
        beta=zeros,
        conformal_grr=1.0 + 0.02 * r,
        conformal_gt=1.0 + 0.03 * r,
        chi=0.9 + 0.01 * r,
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
    Stt = 0.1 + 0.02 * r
    legacy_St = 3.0 + r
    without_Stt = MatterTerms(
        rho=zeros,
        Srr=zeros,
        Stt=zeros,
        Sr=zeros,
        St=legacy_St,
    )
    with_Stt = without_Stt._replace(Stt=Stt)

    dKh_source = dKhdt(metric, with_Stt) - dKhdt(metric, without_Stt)
    dArr_source = dArrdt(metric, with_Stt) - dArrdt(metric, without_Stt)
    dAt_source = dAtdt(metric, with_Stt) - dAtdt(metric, without_Stt)

    assert jnp.allclose(
        dKh_source[:-1],
        (
            8.0
            * jnp.pi
            * metric.alpha
            * Stt
            * metric.chi
            / metric.conformal_gt
        )[:-1],
    )
    assert jnp.allclose(
        dArr_source[:-1],
        (
            16.0
            * jnp.pi
            * metric.conformal_grr
            * metric.alpha
            * Stt
            * metric.chi
            / (3.0 * metric.conformal_gt)
        )[:-1],
    )
    assert jnp.allclose(
        dAt_source[:-1],
        (-(8.0 / 3.0) * jnp.pi * metric.alpha * Stt * metric.chi)[:-1],
    )


def test_gamma_constraint_source_consumes_contravariant_radial_momentum():
    r = jnp.arange(0.5, 8.5, 1.0)
    zeros = jnp.zeros_like(r)
    metric = Z4C_Metric(
        alpha=1.0 + 0.01 * r,
        beta=zeros,
        conformal_grr=jnp.ones_like(r),
        conformal_gt=jnp.ones_like(r),
        chi=0.9 + 0.01 * r,
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
    Sr = -0.2 + 0.03 * r
    vacuum = MatterTerms(
        rho=zeros,
        Srr=zeros,
        Stt=zeros,
        Sr=zeros,
        St=zeros,
    )
    with_radial_momentum = vacuum._replace(Sr=Sr)

    source = (
        dGammadt(metric, with_radial_momentum)
        - dGammadt(metric, vacuum)
    )

    assert jnp.allclose(
        source[:-1],
        (-16.0 * jnp.pi * metric.alpha * Sr / metric.chi)[:-1],
    )
