import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


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


@pytest.mark.parametrize("inner_open", [False, True])
def test_current_and_field_match_every_rk4_stage(monkeypatch, inner_open):
    import RadiShPICR.Z4C.time_evolve as time_evolve
    import RadiShPICR.Z4C.electric_field as electric_field

    metric = _flat_metric(num_cells=8, dr=0.5)
    particles = _particles(r=1.0, ur=0.0)
    particles.masses = jnp.asarray(0.0)
    deposited_positions, gathered_fields, stress_fields, final_depositions = [], [], [], []
    field_matter = electric_field.compute_electrostatic_matter_terms

    def record_current(stage_particles, stage_metric, dr_dt, inner_open=False):
        deposited_positions.append((stage_particles.r.copy(), inner_open))
        assert jnp.all(dr_dt == 1.0)
        return jnp.full_like(stage_metric.r, stage_particles.r[0])

    def record_charge(stage_particles, stage_metric, inner_open=False):
        final_depositions.append(stage_particles.r.copy())
        return jnp.full_like(stage_metric.r, stage_particles.r[0])

    def forbidden_solve(*args, **kwargs):
        raise AssertionError("Gauss law must not be solved during evolution")

    def record_lorentz(stage_particles, stage_metric, E_r):
        gathered_fields.append(E_r.copy())
        return jnp.zeros_like(stage_particles.r)

    def record_stress(stage_metric, E_r, epsilon_0=1.0):
        stress_fields.append(E_r.copy())
        return field_matter(stage_metric, E_r, epsilon_0)

    def constant_velocity(stage_particles, stage_metric):
        zeros = jnp.zeros_like(stage_particles.r)
        return zeros, zeros, jnp.ones_like(stage_particles.r), zeros

    def metric_from_energy(stage_metric, matter, metric_boundary=0, zero_shift=0):
        return jax.tree.map(jnp.zeros_like, stage_metric)._replace(alpha=matter.rho)

    monkeypatch.setattr(time_evolve, "compute_radial_current_density", record_current)
    monkeypatch.setattr(time_evolve, "compute_radial_charge_density", record_charge)
    monkeypatch.setattr(electric_field, "solve_radial_electric_field", forbidden_solve)
    assert not hasattr(time_evolve, "solve_radial_electric_field")
    monkeypatch.setattr(time_evolve, "compute_radial_lorentz_force", record_lorentz)
    monkeypatch.setattr(time_evolve, "compute_electrostatic_matter_terms", record_stress)
    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", constant_velocity)
    monkeypatch.setattr(time_evolve, "metric_time_derivatives", metric_from_energy)

    with jax.disable_jit():
        updated, updated_metric, charge_density, E_r = time_evolve.rk4_step(
            particles, metric, dt=0.2, E_r=jnp.full_like(metric.r, 2.0),
            EM_on=True, GR_on=True, inner_open=inner_open,
        )

    assert len(final_depositions) == 1
    assert len(deposited_positions) == len(gathered_fields) == len(stress_fields) == 4
    assert jnp.allclose(jnp.asarray([p for p, _ in deposited_positions])[:, 0],
                        jnp.asarray([1.0, 1.1, 1.1, 1.2]))
    assert [value for _, value in deposited_positions] == [inner_open] * 4
    expected_fields = jnp.asarray([2.0, 1.9, 1.89, 1.78])
    assert jnp.allclose(jnp.asarray(gathered_fields), expected_fields[:, None])
    assert jnp.allclose(jnp.asarray(stress_fields), expected_fields[:, None])
    expected_alpha = 1.0 + 0.2 / 6.0 * jnp.dot(
        jnp.asarray([1., 2., 2., 1.]), 0.5 * expected_fields**2,
    )
    assert jnp.allclose(updated_metric.alpha, expected_alpha)
    assert jnp.allclose(updated.r, 1.2)
    assert jnp.allclose(charge_density, 1.2)
    assert jnp.allclose(E_r, 1.78)


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
        E_r=jnp.zeros_like(metric.r), EM_on=jnp.asarray(True),
        GR_on=jnp.asarray(False),
    )

    assert jnp.allclose(updated.r, exact_r)
    assert jnp.allclose(updated.ur, 0.2)
    assert jnp.allclose(charge_density, 0.0, atol=1.0e-12)
    assert jnp.allclose(E_r, 0.0, atol=1.0e-12)
    for metric_field, initial_field in zip(updated_metric, initial_metric):
        assert jnp.array_equal(metric_field, initial_field)


@pytest.mark.parametrize("zero_shift", [0, 1])
def test_one_jitted_step_accepts_all_runtime_em_and_gr_modes(zero_shift):
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
                E_r=jnp.zeros_like(metric.r), EM_on=jnp.asarray(EM_on),
                GR_on=jnp.asarray(GR_on),
                zero_shift=jnp.asarray(zero_shift),
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
        E_r=jnp.zeros_like(metric.r), EM_on=jnp.asarray(False),
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

    def derivative_from_energy(stage_metric, matter_terms, metric_boundary=0, zero_shift=0):
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
    from RadiShPICR.Z4C import solve_radial_electric_field
    E_r = solve_radial_electric_field(metric, charge_density)
    field_matter = time_evolve.compute_electrostatic_matter_terms(metric, E_r)
    _, updated_metric, final_charge_density, final_E_r = time_evolve.rk4_step(
        particles,
        metric,
        0.1,
        E_r=E_r, EM_on=True,
        GR_on=True,
    )

    assert jnp.allclose(updated_metric.alpha, metric.alpha + 0.1 * field_matter.rho)
    assert jnp.allclose(final_charge_density, charge_density)
    assert jnp.allclose(final_E_r, E_r)


@pytest.mark.parametrize("GR_on", [False, True])
def test_wrapped_absorbing_boundary_preserves_explicit_deposition_eager_and_jit(GR_on):
    from RadiShPICR.Z4C import rk4_step, deleting_inner_areal_radius_boundary
    from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume

    metric = _flat_metric(num_cells=12, dr=1.0)
    particles = _particles(r=0.1, ur=0.0, charge=1.0)

    def wrapped_boundary(particles, metric):
        return deleting_inner_areal_radius_boundary(particles, metric)

    step = jax.jit(rk4_step, static_argnames=("particle_boundary",))
    for inner_open in (False, True):
        results = []
        for callback in (deleting_inner_areal_radius_boundary, wrapped_boundary):
            kwargs = dict(
                EM_on=jnp.asarray(True), GR_on=jnp.asarray(GR_on),
                inner_open=jnp.asarray(inner_open), particle_boundary=callback,
            )
            with jax.disable_jit():
                eager = rk4_step(particles, metric, 0.0, E_r=jnp.zeros_like(metric.r), **kwargs)
            compiled = step(particles, metric, 0.0, E_r=jnp.zeros_like(metric.r), **kwargs)
            for actual, expected in zip(jax.tree.leaves(compiled), jax.tree.leaves(eager)):
                assert jnp.allclose(actual, expected, rtol=1.e-12, atol=1.e-14)
            results.append(compiled)

        for actual, expected in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
            assert jnp.array_equal(actual, expected)
        deposited_charge = jnp.sum(results[0][2] * proper_radial_shell_volume(metric))
        assert jnp.allclose(deposited_charge, 0.595 if inner_open else 1.0)


def test_coupled_particle_field_manufactured_fourth_order(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    # Smooth coupled oscillator: u' = E, E' = -u, r' = u.
    # This tests field carry and stage coupling without shape-knot crossings.
    def motion(particles, metric):
        zeros = jnp.zeros_like(particles.r)
        return zeros, zeros, particles.ur, zeros

    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", motion)
    monkeypatch.setattr(time_evolve, "compute_radial_lorentz_force",
                        lambda particles, metric, E: jnp.full_like(particles.ur, E[0]))
    monkeypatch.setattr(time_evolve, "compute_radial_current_density",
                        lambda particles, metric, dr_dt, inner_open=False:
                        jnp.full_like(metric.r, dr_dt[0]))
    metric = _flat_metric(num_cells=8, dr=1.)
    # A fresh callable keeps monkeypatched equations out of other JIT tests.
    step = jax.jit(lambda particles, metric, dt, **kwargs:
                   time_evolve.rk4_step(particles, metric, dt, **kwargs))
    errors = []
    for steps in (5, 10, 20):
        particles = _particles(r=3., ur=0.)
        E = jnp.ones_like(metric.r)
        for _ in range(steps):
            particles, _, _, E = step(particles, metric, 1. / steps, E_r=E,
                                       EM_on=True, GR_on=False)
        result = np.asarray([particles.ur[0], E[0], particles.r[0]])
        exact = np.asarray([np.sin(1.), np.cos(1.), 4. - np.cos(1.)])
        errors.append(np.max(np.abs(result - exact)))
    ratios = np.asarray(errors[:-1]) / errors[1:]
    print("manufactured RK4 errors:", errors, "ratios:", ratios)
    assert np.all((ratios > 14.) & (ratios < 18.))


def test_geometric_field_evolution_uses_stage_metric_rhs(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    def expanding_metric(metric, matter, metric_boundary=0, zero_shift=0):
        return jax.tree.map(jnp.zeros_like, metric)._replace(chi=2. * metric.chi)

    monkeypatch.setattr(time_evolve, "metric_time_derivatives", expanding_metric)
    metric = _flat_metric(num_cells=8, dr=1.)
    particles = _particles(charge=0., ur=0.)
    E = .1 * metric.r
    # A fresh callable keeps monkeypatched equations out of other JIT tests.
    step = jax.jit(lambda particles, metric, dt, **kwargs:
                   time_evolve.rk4_step(particles, metric, dt, **kwargs))
    _, static, _, static_E = step(particles, metric, .1, E_r=E, EM_on=True, GR_on=False)
    np.testing.assert_array_equal(static_E, E)
    np.testing.assert_array_equal(static.chi, metric.chi)
    _, evolved, _, evolved_E = step(particles, metric, .1, E_r=E, EM_on=True, GR_on=True)
    # chi'=2 chi and E'=E, each using its own classical RK4 polynomial.
    np.testing.assert_allclose(evolved.chi, 1. + .2 + .2**2/2 + .2**3/6 + .2**4/24)
    np.testing.assert_allclose(evolved_E, E * (1. + .1 + .1**2/2 + .1**3/6 + .1**4/24))


def test_nonzero_field_persists_at_zero_dt_and_after_absorption():
    from RadiShPICR.Z4C import rk4_step, deleting_particle_boundary

    metric = _flat_metric(num_cells=8, dr=1.)
    particles = _particles(r=-.1, ur=-.2, charge=1.)
    E = .03 * metric.r
    step = jax.jit(rk4_step, static_argnames=("particle_boundary",))
    for dt in (0., .02, .02):
        particles, _, rho, actual_E = step(
            particles, metric, dt, E_r=E, EM_on=True, GR_on=False,
            particle_boundary=deleting_particle_boundary,
        )
        np.testing.assert_array_equal(particles.weight, 0.)
        np.testing.assert_array_equal(rho, 0.)
        np.testing.assert_array_equal(actual_E, E)
    _, _, rho, off_E = step(particles, metric, .02, E_r=E, EM_on=False, GR_on=False)
    np.testing.assert_array_equal(off_E, 0.)
    np.testing.assert_array_equal(rho, 0.)


def test_short_charged_run_measures_gauss_drift():
    from RadiShPICR.Z4C import (
        rk4_step, solve_radial_electric_field, compute_radial_charge_density,
        radial_gauss_residual,
    )

    metric = _flat_metric(num_cells=32, dr=.25)
    particles = _particles(r=3.03, ur=.2, charge=.1)
    rho = compute_radial_charge_density(particles, metric)
    initial_E = solve_radial_electric_field(metric, rho)
    initial_residual = radial_gauss_residual(metric, initial_E, rho)
    E = initial_E
    step = jax.jit(rk4_step)
    for _ in range(10):
        particles, _, rho, E = step(particles, metric, .01, E_r=E, EM_on=True, GR_on=False)
    final_residual = radial_gauss_residual(metric, E, rho)
    initial_norm = float(jnp.max(jnp.abs(initial_residual)))
    final_norm = float(jnp.max(jnp.abs(final_residual)))
    drift = float(jnp.max(jnp.abs(final_residual - initial_residual)))
    print("charged run Gauss Linf initial/final/drift:", initial_norm, final_norm, drift)
    assert np.all(np.isfinite([initial_norm, final_norm, drift]))
    assert drift > 1.e-10
    assert not np.allclose(E, initial_E, rtol=1.e-10, atol=1.e-14)
    with jax.disable_jit():
        eager = rk4_step(particles, metric, .01, E_r=E, EM_on=True, GR_on=False)
    compiled = step(particles, metric, .01, E_r=E, EM_on=True, GR_on=False)
    for actual, expected in zip(jax.tree.leaves(compiled), jax.tree.leaves(eager)):
        np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=1.e-14)
