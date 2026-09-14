import jax
import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _flat_metric(num_cells=32, dr=0.25):
    r = (jnp.arange(num_cells) + 0.5) * dr
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)

    return Z4C_Metric(
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
        kappa=jnp.asarray(0.0),
        eta=jnp.asarray(0.0),
        nu=jnp.asarray(0.0),
        r=r,
        dr=jnp.asarray(dr),
    )


def _particles(r=3.0, ur=0.2, charge=-1.0):
    return particle_species(
        name="electrons",
        charge=charge,
        mass=1.0,
        weight=jnp.asarray([1.0]),
        r=jnp.asarray([r]),
        ur=jnp.asarray([ur]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )


def test_electrostatic_step_recomputes_field_at_every_rk4_stage(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    metric = _flat_metric(num_cells=8, dr=0.5)
    particles = _particles(r=1.0, ur=0.0)
    deposited_positions = []
    field_sources = []
    gathered_positions = []

    def record_charge(stage_particles, stage_metric):
        deposited_positions.append(stage_particles.r.copy())
        return jnp.full_like(stage_metric.r, stage_particles.r[0])

    def record_field(stage_metric, charge_density, epsilon_0=1.0):
        field_sources.append(charge_density.copy())
        return jnp.zeros_like(stage_metric.r)

    def record_lorentz(stage_particles, stage_metric, E_r):
        gathered_positions.append(stage_particles.r.copy())
        return jnp.zeros_like(stage_particles.r)

    def constant_velocity(stage_particles, stage_metric):
        zeros = jnp.zeros_like(stage_particles.r)
        return zeros, zeros, jnp.ones_like(stage_particles.r), zeros

    monkeypatch.setattr(time_evolve, "compute_radial_charge_density", record_charge)
    monkeypatch.setattr(time_evolve, "solve_radial_electric_field", record_field)
    monkeypatch.setattr(time_evolve, "compute_radial_lorentz_force", record_lorentz)
    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", constant_velocity)

    with jax.disable_jit():
        updated, updated_metric, charge_density, E_r = time_evolve.rk4_step(
            particles,
            metric,
            dt=0.2,
            EM_on=True,
            GR_on=False,
        )

    assert len(deposited_positions) == 5
    assert len(field_sources) == 5
    assert len(gathered_positions) == 4
    assert jnp.allclose(
        jnp.asarray(deposited_positions)[:, 0],
        jnp.asarray([1.0, 1.1, 1.1, 1.2, 1.2]),
    )
    assert jnp.allclose(
        jnp.asarray(gathered_positions)[:, 0],
        jnp.asarray([1.0, 1.1, 1.1, 1.2]),
    )
    assert jnp.allclose(updated.r, 1.2)
    for updated_field, initial_field in zip(updated_metric, metric):
        assert jnp.array_equal(updated_field, initial_field)
    assert jnp.allclose(charge_density, 1.2)
    assert jnp.allclose(E_r, 0.0)


def test_jitted_neutral_electrostatic_step_preserves_flat_free_motion():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    metric = _flat_metric()
    particles = _particles(charge=0.0)
    initial_metric = tuple(field.copy() for field in metric)
    dt = 0.4
    exact_r = particles.r + dt * particles.ur / jnp.sqrt(1.0 + particles.ur**2)

    updated, updated_metric, charge_density, E_r = jax.jit(rk4_step)(
        particles,
        metric,
        dt,
        EM_on=jnp.asarray(True),
        GR_on=jnp.asarray(False),
    )

    assert jnp.allclose(updated.r, exact_r)
    assert jnp.allclose(updated.ur, 0.2)
    assert jnp.allclose(charge_density, 0.0, atol=1.0e-12)
    assert jnp.allclose(E_r, 0.0, atol=1.0e-12)
    for metric_field, initial_field in zip(updated_metric, initial_metric):
        assert jnp.array_equal(metric_field, initial_field)


def test_one_jitted_step_accepts_all_runtime_em_and_gr_modes():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    metric = _flat_metric(num_cells=8, dr=0.5)
    particles = _particles(charge=0.0)
    step_jit = jax.jit(rk4_step)

    for EM_on in (False, True):
        for GR_on in (False, True):
            updated = step_jit(
                particles,
                metric,
                1.0e-3,
                EM_on=jnp.asarray(EM_on),
                GR_on=jnp.asarray(GR_on),
            )
            updated_particles, updated_metric, charge_density, E_r = updated

            assert updated_particles.r.shape == particles.r.shape
            assert updated_metric.r.shape == metric.r.shape
            assert charge_density.shape == metric.r.shape
            assert E_r.shape == metric.r.shape
            assert jnp.all(jnp.isfinite(updated_particles.r))
            assert jnp.all(jnp.isfinite(updated_metric.alpha))


def test_gr_off_keeps_unprojected_static_metric_exactly():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    metric = _flat_metric(num_cells=8, dr=0.5)._replace(
        conformal_grr=2.0 * jnp.ones(8),
        conformal_gt=3.0 * jnp.ones(8),
    )
    particles = _particles(charge=0.0)
    updated_particles, updated_metric, charge_density, E_r = jax.jit(rk4_step)(
        particles,
        metric,
        0.1,
        EM_on=jnp.asarray(False),
        GR_on=jnp.asarray(False),
    )

    assert not jnp.array_equal(updated_particles.r, particles.r)
    for updated_field, initial_field in zip(updated_metric, metric):
        assert jnp.array_equal(updated_field, initial_field)
    assert jnp.allclose(charge_density, 0.0)
    assert jnp.allclose(E_r, 0.0)


def test_em_and_gr_add_field_stress_energy_at_matching_rk_stages(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    metric = _flat_metric(num_cells=8, dr=0.5)
    particles = _particles(r=1.0, ur=0.0, charge=1.0)
    particles.masses = jnp.asarray(0.0)

    def stationary_particles(stage_particles, stage_metric):
        zeros = jnp.zeros_like(stage_particles.r)
        return zeros, zeros, zeros, zeros

    def derivative_from_energy(stage_metric, matter_terms, metric_boundary=0):
        zeros = jnp.zeros_like(stage_metric.r)
        return Z4C_Metric(
            alpha=matter_terms.rho,
            beta=zeros,
            conformal_grr=zeros,
            conformal_gt=zeros,
            chi=zeros,
            Kh=zeros,
            Arr=zeros,
            At=zeros,
            theta=zeros,
            Gamma=zeros,
            kappa=jnp.zeros_like(stage_metric.kappa),
            eta=jnp.zeros_like(stage_metric.eta),
            nu=jnp.zeros_like(stage_metric.nu),
            r=zeros,
            dr=jnp.zeros_like(stage_metric.dr),
        )

    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", stationary_particles)
    monkeypatch.setattr(time_evolve, "metric_time_derivatives", derivative_from_energy)

    charge_density = time_evolve.compute_radial_charge_density(particles, metric)
    E_r = time_evolve.solve_radial_electric_field(metric, charge_density)
    field_matter = time_evolve.compute_electrostatic_matter_terms(metric, E_r)
    _, updated_metric, final_charge_density, final_E_r = time_evolve.rk4_step(
        particles,
        metric,
        0.1,
        EM_on=True,
        GR_on=True,
    )

    assert jnp.allclose(updated_metric.alpha, metric.alpha + 0.1 * field_matter.rho)
    assert jnp.allclose(final_charge_density, charge_density)
    assert jnp.allclose(final_E_r, E_r)
