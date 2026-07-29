import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import (
    interpolate_field_to_particles,
    shape_weights_at_point,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    compute_radial_momentum_density,
    compute_radial_stress_tensor_component,
    compute_radial_matter_terms,
    initialize_vacuum_matter_terms,
    relativistic_mass_energy_density,
)
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
        particle_volume = (
            4.0 * jnp.pi * particles.r**2 * scaling_factor_p
        )
        lorentz_factor = jnp.sqrt(
            1.0
            + particles.ur**2 / grr_p
            + particles.uphi**2 / (particles.r**2 * gt_p)
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
        expected_Sr = jnp.sum(
            weights
            * particle_mass
            * particles.ur
            / particle_volume,
            axis=1,
        )

        matter_terms = compute_radial_matter_terms(particles, metric)

        assert jnp.allclose(matter_terms.rho, expected_rho)
        assert jnp.allclose(matter_terms.Srr, expected_Srr)
        assert jnp.allclose(matter_terms.Sr, expected_Sr)
        assert jnp.allclose(matter_terms.Stt, 0.0)
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
