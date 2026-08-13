import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import (
    _cell_centered_radial_shape_stencil,
    interpolate_field_to_particles,
    shape_weights_at_point,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    MatterTerms,
    _proper_radial_shell_volume,
    compute_radial_momentum_density,
    compute_radial_stress_tensor_component,
    compute_radial_matter_terms,
    initialize_vacuum_matter_terms,
    relativistic_mass_energy_density,
)
from RadiShPICR.Z4C.constraint_terms import dGammadt
from RadiShPICR.Z4C.extrinsic_curvature import dArrdt, dAtdt, dKhdt
from RadiShPICR.Z4C.geodesic import (
    compute_geodesic_terms,
    isotropic_particle_state,
    lapse_freezing_particle_state,
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
        alpha=0.7 + 0.02 * r,
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
        r_particle = jnp.asarray([1.75, 2.25, 4.25])
        ur = jnp.asarray([0.4, -0.2, 0.7])
        rs, ubar = lapse_freezing_particle_state(
            r_particle,
            ur,
            metric,
            shape_mode,
        )
        particles = particle_species(
            name="matter",
            charge=0.0,
            mass=2.0,
            weight=jnp.asarray([0.2, 0.3, 0.5]),
            r=rs,
            ur=ubar,
            phi=jnp.zeros(3),
            uphi=jnp.asarray([0.1, 0.3, -0.2]),
            shape_mode=shape_mode,
        )

        grr_p = interpolate_field_to_particles(
            metric.conformal_grr / metric.chi,
            r_particle,
            grid,
            shape_mode=shape_mode,
        )
        gt_p = interpolate_field_to_particles(
            metric.conformal_gt / metric.chi,
            r_particle,
            grid,
            shape_mode=shape_mode,
        )
        proper_shell_volume = _proper_radial_shell_volume(metric)
        gamma_rr_inv_p = 1.0 / grr_p
        lorentz_factor = jnp.sqrt(
            1.0
            + gamma_rr_inv_p * ur**2
            + particles.uphi**2 / (r_particle**2 * gt_p)
        )
        weights = shape_weights_at_point(
            r_particle[jnp.newaxis, :],
            r[:, jnp.newaxis],
            metric.dr,
            shape_mode=shape_mode,
        )
        particle_mass = particles.get_mass()
        expected_rho = (
            jnp.sum(weights * particle_mass * lorentz_factor, axis=1)
            / proper_shell_volume
        )
        expected_Srr = (
            jnp.sum(
                weights * particle_mass * ur**2 / lorentz_factor,
                axis=1,
            )
            / proper_shell_volume
        )
        expected_Stt = (
            jnp.sum(
                weights
                * particle_mass
                * particles.uphi**2
                / (2.0 * r_particle**2 * lorentz_factor),
                axis=1,
            )
            / proper_shell_volume
        )
        expected_Sr = (
            jnp.sum(
                weights * particle_mass * gamma_rr_inv_p * ur,
                axis=1,
            )
            / proper_shell_volume
        )

        matter_terms = compute_radial_matter_terms(particles, metric)
        compiled_matter_terms = jax.jit(compute_radial_matter_terms)(
            particles,
            metric,
        )

        assert jnp.allclose(matter_terms.rho, expected_rho)
        assert jnp.allclose(matter_terms.Srr, expected_Srr)
        assert jnp.allclose(matter_terms.Stt, expected_Stt)
        assert jnp.allclose(matter_terms.Sr, expected_Sr)
        assert jnp.allclose(matter_terms.St, 0.0)
        for actual, compiled in zip(matter_terms, compiled_matter_terms):
            assert jnp.allclose(actual, compiled)

        assert jnp.allclose(
            jnp.sum(matter_terms.rho * proper_shell_volume),
            jnp.sum(particle_mass * lorentz_factor),
        )
        assert jnp.allclose(
            jnp.sum(matter_terms.Srr * proper_shell_volume),
            jnp.sum(particle_mass * ur**2 / lorentz_factor),
        )
        assert jnp.allclose(
            jnp.sum(matter_terms.Stt * proper_shell_volume),
            jnp.sum(
                particle_mass
                * particles.uphi**2
                / (2.0 * r_particle**2 * lorentz_factor)
            ),
        )
        assert jnp.allclose(
            jnp.sum(matter_terms.Sr * proper_shell_volume),
            jnp.sum(particle_mass * gamma_rr_inv_p * ur),
        )
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


def test_turnover_outside_particle_support_preserves_sources_and_geodesics():
    r = 0.5 + 0.5 * jnp.arange(8)
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)
    areal_radius = r.at[-2].set(r[-3] - 1.0e-8)
    metric = Z4C_Metric(
        alpha=0.8 + 0.01 * r,
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=(r / areal_radius) ** 2,
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
    metric_derivative = Z4C_Metric(
        alpha=zeros,
        beta=zeros,
        conformal_grr=zeros,
        conformal_gt=zeros,
        chi=zeros,
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
    r_particle = jnp.asarray([0.2, 0.7, 1.2, 2.2])
    ur = jnp.asarray([0.0, 0.2, -0.1, 0.3])
    particle_mass = jnp.asarray([0.2, 0.3, 0.1, 0.4])

    assert jnp.min(jnp.diff(areal_radius)) < 0.0
    assert jnp.max(r_particle) < r[-3]

    for shape_mode in ("nearest", "linear", "quadratic"):
        rs, ubar = lapse_freezing_particle_state(
            r_particle,
            ur,
            metric,
            shape_mode,
        )
        particles = particle_species(
            name="turnover",
            charge=0.0,
            mass=1.0,
            weight=particle_mass,
            r=rs,
            ur=ubar,
            phi=jnp.zeros_like(rs),
            uphi=jnp.zeros_like(rs),
            shape_mode=shape_mode,
        )

        mapped_r, mapped_ur = isotropic_particle_state(particles, metric)
        _, even_weights, _, _ = _cell_centered_radial_shape_stencil(
            mapped_r,
            metric.r,
            metric.dr,
            shape_mode,
        )
        matter_terms = compute_radial_matter_terms(particles, metric)
        compiled_matter_terms = jax.jit(compute_radial_matter_terms)(
            particles,
            metric,
        )
        geodesic_terms = compute_geodesic_terms(
            particles,
            metric,
            metric_derivative,
        )

        particle_energy = jnp.sum(
            particles.get_mass() * jnp.sqrt(1.0 + mapped_ur**2)
        )
        deposited_energy = jnp.sum(
            matter_terms.rho * _proper_radial_shell_volume(metric)
        )

        assert jnp.allclose(mapped_r, r_particle)
        assert jnp.allclose(mapped_ur, ur)
        assert jnp.allclose(jnp.sum(even_weights, axis=0), 1.0)
        assert jnp.all(jnp.isfinite(jnp.stack(geodesic_terms)))
        assert jnp.allclose(deposited_energy, particle_energy)
        for eager_term, compiled_term in zip(matter_terms, compiled_matter_terms):
            assert jnp.all(jnp.isfinite(eager_term))
            assert jnp.allclose(compiled_term, eager_term)


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
    r_particle = jnp.asarray([r[particle_index]])
    ur = jnp.asarray([-0.4])
    rs, ubar = lapse_freezing_particle_state(
        r_particle,
        ur,
        metric,
        "nearest",
    )
    particles = particle_species(
        name="matter",
        charge=0.0,
        mass=2.0,
        weight=jnp.asarray([0.7]),
        r=rs,
        ur=ubar,
        phi=jnp.asarray([0.3]),
        uphi=jnp.asarray([0.8]),
        shape_mode="nearest",
    )

    matter_terms = compute_radial_matter_terms(particles, metric)

    rp = r_particle[0]
    gamma_rr_inv = chi[particle_index] / conformal_grr[particle_index]
    gamma_t_inv = chi[particle_index] / conformal_gt[particle_index]
    lorentz_factor = jnp.sqrt(
        1.0
        + gamma_rr_inv * ur[0] ** 2
        + gamma_t_inv * particles.uphi[0] ** 2 / rp**2
    )
    proper_shell_volume = _proper_radial_shell_volume(metric)[particle_index]
    particle_mass = particles.get_mass()[0]
    stress_trace = (
        gamma_rr_inv * matter_terms.Srr[particle_index]
        + 2.0 * gamma_t_inv * matter_terms.Stt[particle_index]
    )
    expected_trace = (
        particle_mass
        * (lorentz_factor**2 - 1.0)
        / (proper_shell_volume * lorentz_factor)
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


def test_cell_centered_origin_deposition_uses_even_and_odd_parity():
    r = jnp.arange(0.5, 5.5, 1.0)
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)
    metric = Z4C_Metric(
        alpha=ones,
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=ones,
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
    r_particle = r[:1]
    ur = jnp.ones_like(r_particle)
    lorentz_factor = jnp.sqrt(2.0)
    proper_shell_volume = _proper_radial_shell_volume(metric)

    expected_even_numerators = {
        "nearest": jnp.asarray([1.0, 0.0]),
        "linear": jnp.asarray([1.0, 0.0]),
        "quadratic": jnp.asarray([0.875, 0.125]),
    }
    expected_odd_numerators = {
        "nearest": jnp.asarray([1.0, 0.0]),
        "linear": jnp.asarray([1.0, 0.0]),
        "quadratic": jnp.asarray([0.625, 0.125]),
    }

    for shape_mode in ("nearest", "linear", "quadratic"):
        rs, ubar = lapse_freezing_particle_state(
            r_particle,
            ur,
            metric,
            shape_mode,
        )
        particles = particle_species(
            name="origin",
            charge=0.0,
            mass=1.0,
            weight=1.0,
            r=rs,
            ur=ubar,
            phi=zeros[:1],
            uphi=zeros[:1],
            shape_mode=shape_mode,
        )
        initial_rs = particles.r
        initial_ubar = particles.ur

        matter_terms = compute_radial_matter_terms(particles, metric)
        deposited_energy = matter_terms.rho * proper_shell_volume
        deposited_Srr = matter_terms.Srr * proper_shell_volume
        deposited_Sr = matter_terms.Sr * proper_shell_volume

        assert jnp.allclose(
            deposited_energy[:2],
            lorentz_factor * expected_even_numerators[shape_mode],
        )
        assert jnp.allclose(
            deposited_Srr[:2],
            expected_even_numerators[shape_mode] / lorentz_factor,
        )
        assert jnp.allclose(
            deposited_Sr[:2],
            expected_odd_numerators[shape_mode],
        )
        assert jnp.allclose(jnp.sum(deposited_energy), lorentz_factor)
        assert jnp.allclose(particles.r, initial_rs)
        assert jnp.allclose(particles.ur, initial_ubar)


def test_uniform_density_is_grid_independent_including_inner_shell():
    expected_density = 2.5

    for dr in (0.5, 0.25):
        r = jnp.arange(0.5 * dr, 4.0, dr)
        zeros = jnp.zeros_like(r)
        ones = jnp.ones_like(r)
        metric = Z4C_Metric(
            alpha=ones,
            beta=zeros,
            conformal_grr=ones,
            conformal_gt=ones,
            chi=ones,
            Kh=zeros,
            Arr=zeros,
            At=zeros,
            theta=zeros,
            Gamma=zeros,
            kappa=zeros,
            eta=zeros,
            nu=zeros,
            r=r,
            dr=jnp.asarray(dr),
        )
        proper_shell_volume = _proper_radial_shell_volume(metric)
        rs, ubar = lapse_freezing_particle_state(
            r,
            zeros,
            metric,
            "nearest",
        )
        particles = particle_species(
            name="uniform",
            charge=0.0,
            mass=1.0,
            weight=expected_density * proper_shell_volume,
            r=rs,
            ur=ubar,
            phi=zeros,
            uphi=zeros,
            shape_mode="nearest",
        )

        matter_terms = compute_radial_matter_terms(particles, metric)
        compiled_rho = jax.jit(relativistic_mass_energy_density)(
            particles,
            metric,
        )

        assert jnp.allclose(matter_terms.rho, expected_density)
        assert jnp.allclose(compiled_rho, expected_density)
        assert jnp.allclose(
            jnp.sum(matter_terms.rho * proper_shell_volume),
            jnp.sum(particles.get_mass()),
        )


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
