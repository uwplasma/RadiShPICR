import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.energy_momentum_tensor import MatterTerms
from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _flat_metric(r):
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
        kappa=zeros,
        eta=zeros,
        nu=zeros,
        r=r,
        dr=r[1] - r[0],
    )


def _metric_derivative(metric, alpha_value):
    zeros = jnp.zeros_like(metric.r)

    return Z4C_Metric(
        alpha=jnp.full_like(metric.alpha, alpha_value),
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


def _make_particles():
    return particle_species(
        name="test",
        charge=0.0,
        mass=1.0,
        weight=1.0,
        r=jnp.asarray([0.25, 0.75]),
        ur=jnp.asarray([0.1, -0.1]),
        phi=jnp.asarray([0.0, 0.5]),
        uphi=jnp.asarray([0.2, 0.4]),
        shape_mode="nearest",
    )


def _assert_algebraic_constraints(metric):
    conformal_determinant = metric.conformal_grr * metric.conformal_gt**2
    curvature_trace = (
        metric.Arr / metric.conformal_grr
        + 2.0 * metric.At / metric.conformal_gt
    )

    assert jnp.allclose(conformal_determinant, 1.0)
    assert jnp.allclose(curvature_trace, 0.0, atol=1.0e-6)


def test_unit_determinant_conformal_metric_is_idempotent():
    from RadiShPICR.Z4C.utils import unit_determinant_conformal_metric

    conformal_grr = jnp.asarray([2.0, 5.0, 0.75])
    conformal_gt = jnp.asarray([3.0, 0.5, 1.25])

    conformal_grr, conformal_gt = unit_determinant_conformal_metric(
        conformal_grr,
        conformal_gt,
    )
    projected_again = unit_determinant_conformal_metric(
        conformal_grr,
        conformal_gt,
    )

    assert jnp.allclose(conformal_grr * conformal_gt**2, 1.0)
    assert jnp.allclose(projected_again[0], conformal_grr)
    assert jnp.allclose(projected_again[1], conformal_gt)


def test_metric_time_derivatives_match_metric_layout():
    from RadiShPICR.Z4C.time_evolve import metric_time_derivatives

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    derivatives = metric_time_derivatives(metric, matter_terms)

    for derivative_field, metric_field in zip(derivatives, metric):
        assert jnp.shape(derivative_field) == jnp.shape(metric_field)

    assert jnp.allclose(derivatives.kappa, 0.0)
    assert jnp.allclose(derivatives.eta, 0.0)
    assert jnp.allclose(derivatives.nu, 0.0)
    assert jnp.allclose(derivatives.r, 0.0)
    assert jnp.allclose(derivatives.dr, 0.0)


def test_rk4_step_keeps_grid_and_damping_parameters_fixed():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    updated = rk4_step(metric, matter_terms, dt=1.0e-3)

    assert jnp.allclose(updated.kappa, metric.kappa)
    assert jnp.allclose(updated.eta, metric.eta)
    assert jnp.allclose(updated.nu, metric.nu)
    assert jnp.allclose(updated.r, metric.r)
    assert jnp.allclose(updated.dr, metric.dr)


def test_rk4_step_preserves_flat_vacuum_metric():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    updated = rk4_step(metric, matter_terms, dt=1.0e-3)

    for updated_field, metric_field in zip(updated, metric):
        assert jnp.allclose(updated_field, metric_field)


def test_compiled_vacuum_scan_matches_repeated_rk4_steps():
    import jax

    from RadiShPICR.Z4C.time_evolve import advance_vacuum_steps, rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)
    dt = 1.0e-3
    expected = metric
    for _ in range(3):
        expected = rk4_step(expected, matter_terms, dt)

    actual, first_nonfinite_step = jax.jit(
        advance_vacuum_steps,
        static_argnames=("num_steps",),
    )(metric, dt, num_steps=3)

    for actual_field, expected_field in zip(actual, expected):
        assert jnp.allclose(actual_field, expected_field)
    assert first_nonfinite_step == -1


def test_rk4_step_projects_every_metric_stage(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    ones = jnp.ones_like(r)
    metric = _flat_metric(r)._replace(
        conformal_grr=2.0 * ones,
        conformal_gt=3.0 * ones,
        Arr=0.5 * ones,
        At=-0.25 * ones,
    )
    matter_terms = initialize_vacuum_matter_terms(metric)
    stage_metrics = []

    def fake_metric_time_derivatives(stage_metric, stage_matter_terms):
        stage_metrics.append(stage_metric)
        derivative = _metric_derivative(stage_metric, alpha_value=0.0)

        return derivative._replace(
            conformal_grr=0.5 * ones,
            conformal_gt=0.25 * ones,
            Arr=0.4 * ones,
            At=-0.3 * ones,
        )

    monkeypatch.setattr(
        time_evolve,
        "metric_time_derivatives",
        fake_metric_time_derivatives,
    )

    updated = time_evolve.rk4_step(metric, matter_terms, dt=0.1)

    assert len(stage_metrics) == 4
    for stage_metric in stage_metrics:
        _assert_algebraic_constraints(stage_metric)
    _assert_algebraic_constraints(updated)


def test_rk4_step_uses_classic_stage_weights(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)
    stage_values = [1.0, 2.0, 3.0, 4.0]

    def fake_metric_time_derivatives(stage_metric, stage_matter_terms):
        stage_value = stage_values.pop(0)
        zeros = jnp.zeros_like(stage_metric.r)
        derivative = jnp.full_like(stage_metric.alpha, stage_value)

        return Z4C_Metric(
            alpha=derivative,
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
            dr=jnp.asarray(0.0, dtype=stage_metric.dr.dtype),
        )

    monkeypatch.setattr(
        time_evolve,
        "metric_time_derivatives",
        fake_metric_time_derivatives,
    )

    updated = time_evolve.rk4_step(metric, matter_terms, dt=0.6)

    expected_alpha = metric.alpha + 0.6 * (1.0 + 2.0 * 2.0 + 2.0 * 3.0 + 4.0) / 6.0
    assert jnp.allclose(updated.alpha, expected_alpha)
    assert jnp.allclose(updated.beta, metric.beta)
    assert jnp.allclose(updated.r, metric.r)
    assert stage_values == []


def test_particles_rk4_step_projects_every_metric_stage(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    ones = jnp.ones_like(r)
    metric = _flat_metric(r)._replace(
        conformal_grr=2.0 * ones,
        conformal_gt=3.0 * ones,
        Arr=0.5 * ones,
        At=-0.25 * ones,
    )
    particles = _make_particles()
    stage_metrics = []

    def fake_compute_geodesic_terms(stage_particles, stage_metric):
        stage_metrics.append(stage_metric)
        particle_zeros = jnp.zeros_like(stage_particles.r)

        return particle_zeros, particle_zeros, particle_zeros, particle_zeros

    def fake_compute_radial_matter_terms(stage_particles, stage_metric):
        metric_zeros = jnp.zeros_like(stage_metric.r)

        return MatterTerms(
            rho=metric_zeros,
            Srr=metric_zeros,
            Stt=metric_zeros,
            Sr=metric_zeros,
            St=metric_zeros,
        )

    def fake_metric_time_derivatives(stage_metric, stage_matter_terms):
        derivative = _metric_derivative(stage_metric, alpha_value=0.0)

        return derivative._replace(
            conformal_grr=0.5 * ones,
            conformal_gt=0.25 * ones,
            Arr=0.4 * ones,
            At=-0.3 * ones,
        )

    monkeypatch.setattr(
        time_evolve,
        "compute_geodesic_terms",
        fake_compute_geodesic_terms,
    )
    monkeypatch.setattr(
        time_evolve,
        "compute_radial_matter_terms",
        fake_compute_radial_matter_terms,
    )
    monkeypatch.setattr(
        time_evolve,
        "metric_time_derivatives",
        fake_metric_time_derivatives,
    )

    _, updated_metric = time_evolve.particles_rk4_step(
        particles,
        metric,
        dt=0.1,
    )

    assert len(stage_metrics) == 4
    for stage_metric in stage_metrics:
        _assert_algebraic_constraints(stage_metric)
    _assert_algebraic_constraints(updated_metric)


def test_particles_rk4_step_updates_source_backed_particle_class(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    particles = _make_particles()
    dt = 0.1

    def fake_compute_geodesic_terms(stage_particles, stage_metric):
        dvr_dt = jnp.ones_like(stage_particles.ur)
        duphi_dt = jnp.full_like(stage_particles.uphi, 10.0)
        drdt = jnp.full_like(stage_particles.r, 2.0)
        dphidt = -jnp.ones_like(stage_particles.phi)

        return dvr_dt, duphi_dt, drdt, dphidt

    def fake_compute_radial_matter_terms(stage_particles, stage_metric):
        return MatterTerms(
            rho=jnp.zeros_like(stage_metric.r),
            Srr=jnp.zeros_like(stage_metric.r),
            Stt=jnp.zeros_like(stage_metric.r),
            Sr=jnp.zeros_like(stage_metric.r),
            St=jnp.zeros_like(stage_metric.r),
        )

    def fake_metric_time_derivatives(stage_metric, stage_matter_terms):
        return _metric_derivative(stage_metric, alpha_value=1.0)

    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(time_evolve, "compute_radial_matter_terms", fake_compute_radial_matter_terms)
    monkeypatch.setattr(time_evolve, "metric_time_derivatives", fake_metric_time_derivatives)

    r0 = particles.r.copy()
    phi0 = particles.phi.copy()
    ur0 = particles.ur.copy()
    uphi0 = particles.uphi.copy()

    updated_particles, updated_metric = time_evolve.particles_rk4_step(particles, metric, dt)

    assert updated_particles is particles
    assert jnp.allclose(updated_particles.r, r0 + 2.0 * dt)
    assert jnp.allclose(updated_particles.phi, phi0 - dt)
    assert jnp.allclose(updated_particles.ur, ur0 + dt)
    assert jnp.allclose(updated_particles.uphi, uphi0 + 10.0 * dt)
    assert jnp.allclose(updated_metric.alpha, metric.alpha + dt)


def test_particles_rk4_step_recomputes_matter_from_each_particle_stage(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    particles = _make_particles()
    dt = 0.2
    matter_stage_positions = []
    derivative_stage_rho = []

    def fake_compute_geodesic_terms(stage_particles, stage_metric):
        stage_number = len(matter_stage_positions) + 1.0
        dvr_dt = jnp.full_like(stage_particles.ur, stage_number)
        duphi_dt = jnp.zeros_like(stage_particles.uphi)
        drdt = jnp.full_like(stage_particles.r, stage_number)
        dphidt = jnp.zeros_like(stage_particles.phi)

        return dvr_dt, duphi_dt, drdt, dphidt

    def fake_compute_radial_matter_terms(stage_particles, stage_metric):
        matter_stage_positions.append(stage_particles.r.copy())
        rho = jnp.full_like(stage_metric.r, stage_particles.r[0])

        return MatterTerms(
            rho=rho,
            Srr=jnp.zeros_like(stage_metric.r),
            Stt=jnp.zeros_like(stage_metric.r),
            Sr=jnp.zeros_like(stage_metric.r),
            St=jnp.zeros_like(stage_metric.r),
        )

    def fake_metric_time_derivatives(stage_metric, stage_matter_terms):
        derivative_stage_rho.append(stage_matter_terms.rho[0])
        return _metric_derivative(stage_metric, alpha_value=1.0)

    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(time_evolve, "compute_radial_matter_terms", fake_compute_radial_matter_terms)
    monkeypatch.setattr(time_evolve, "metric_time_derivatives", fake_metric_time_derivatives)

    r0 = particles.r.copy()

    time_evolve.particles_rk4_step(particles, metric, dt)

    assert len(matter_stage_positions) == 4
    assert jnp.allclose(matter_stage_positions[0], r0)
    assert jnp.allclose(matter_stage_positions[1], r0 + 0.5 * dt)
    assert jnp.allclose(matter_stage_positions[2], r0 + dt)
    assert jnp.allclose(matter_stage_positions[3], r0 + 3.0 * dt)
    assert jnp.allclose(
        jnp.asarray(derivative_stage_rho),
        jnp.asarray([r0[0], r0[0] + 0.5 * dt, r0[0] + dt, r0[0] + 3.0 * dt]),
    )
