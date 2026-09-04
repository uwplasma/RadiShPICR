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
    ion_charge_density = jnp.zeros_like(metric.r)
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

    updated, charge_density, E_r = time_evolve.electrostatic_particles_rk4_step(
        particles,
        metric,
        ion_charge_density,
        dt=0.2,
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
    assert jnp.allclose(charge_density, 1.2)
    assert jnp.allclose(E_r, 0.0)


def test_jitted_neutral_electrostatic_step_preserves_flat_free_motion():
    from RadiShPICR.Z4C.time_evolve import electrostatic_particles_rk4_step

    metric = _flat_metric()
    particles = _particles(charge=0.0)
    ion_charge_density = jnp.zeros_like(metric.r)
    initial_ion_density = ion_charge_density.copy()
    initial_metric = tuple(field.copy() for field in metric)
    dt = 0.4
    exact_r = particles.r + dt * particles.ur / jnp.sqrt(1.0 + particles.ur**2)

    updated, charge_density, E_r = jax.jit(electrostatic_particles_rk4_step)(
        particles,
        metric,
        ion_charge_density,
        dt,
    )

    assert jnp.allclose(updated.r, exact_r)
    assert jnp.allclose(updated.ur, 0.2)
    assert jnp.allclose(charge_density, 0.0, atol=1.0e-12)
    assert jnp.allclose(E_r, 0.0, atol=1.0e-12)
    assert jnp.array_equal(ion_charge_density, initial_ion_density)
    for metric_field, initial_field in zip(metric, initial_metric):
        assert jnp.array_equal(metric_field, initial_field)
