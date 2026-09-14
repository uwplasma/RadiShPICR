import jax
import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.energy_momentum_tensor import MatterTerms
from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
from RadiShPICR.Z4C.particle_boundaries import (
    deleting_inner_areal_radius_boundary,
    deleting_particle_boundary,
)
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


def _empty_particles():
    empty = jnp.asarray([], dtype=jnp.float64)
    return particle_species(
        name="vacuum",
        charge=0.0,
        mass=0.0,
        weight=empty,
        r=empty,
        ur=empty,
        phi=empty,
        uphi=empty,
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


def test_metric_rk4_step_keeps_grid_and_damping_parameters_fixed():
    from RadiShPICR.Z4C.time_evolve import metric_rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    updated = metric_rk4_step(metric, matter_terms, dt=1.0e-3)

    assert jnp.allclose(updated.kappa, metric.kappa)
    assert jnp.allclose(updated.eta, metric.eta)
    assert jnp.allclose(updated.nu, metric.nu)
    assert jnp.allclose(updated.r, metric.r)
    assert jnp.allclose(updated.dr, metric.dr)


def test_metric_rk4_step_preserves_flat_vacuum_metric():
    from RadiShPICR.Z4C.time_evolve import metric_rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    updated = metric_rk4_step(metric, matter_terms, dt=1.0e-3)

    for updated_field, metric_field in zip(updated, metric):
        assert jnp.allclose(updated_field, metric_field)


def test_metric_rk4_step_uses_fixed_matter_and_classic_stage_weights(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    matter_terms = MatterTerms(
        rho=jnp.linspace(0.1, 0.8, 8),
        Srr=jnp.linspace(0.2, 0.9, 8),
        Stt=jnp.linspace(0.3, 1.0, 8),
        Sr=jnp.linspace(0.4, 1.1, 8),
        St=jnp.zeros_like(r),
    )
    stage_values = [1.0, 2.0, 3.0, 4.0]
    stage_matter_terms = []

    def fake_metric_time_derivatives(
        stage_metric,
        stage_matter,
        metric_boundary=0,
    ):
        stage_matter_terms.append(stage_matter)
        stage_value = stage_values.pop(0)
        return _metric_derivative(stage_metric, alpha_value=stage_value)

    monkeypatch.setattr(
        time_evolve,
        "metric_time_derivatives",
        fake_metric_time_derivatives,
    )

    updated = time_evolve.metric_rk4_step(metric, matter_terms, dt=0.6)

    expected_alpha = metric.alpha + 0.6 * (
        1.0 + 2.0 * 2.0 + 2.0 * 3.0 + 4.0
    ) / 6.0
    assert jnp.allclose(updated.alpha, expected_alpha)
    assert all(stage_matter is matter_terms for stage_matter in stage_matter_terms)
    assert stage_values == []


def test_metric_rk4_step_projects_every_metric_stage(monkeypatch):
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

    def fake_metric_time_derivatives(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
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

    updated = time_evolve.metric_rk4_step(metric, matter_terms, dt=0.1)

    assert len(stage_metrics) == 4
    for stage_metric in stage_metrics:
        _assert_algebraic_constraints(stage_metric)
    _assert_algebraic_constraints(updated)


def test_rk4_step_keeps_grid_and_damping_parameters_fixed():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    _, updated, charge_density, E_r = rk4_step(
        _empty_particles(),
        metric,
        dt=1.0e-3,
        EM_on=False,
        GR_on=True,
    )

    assert jnp.allclose(updated.kappa, metric.kappa)
    assert jnp.allclose(updated.eta, metric.eta)
    assert jnp.allclose(updated.nu, metric.nu)
    assert jnp.allclose(updated.r, metric.r)
    assert jnp.allclose(updated.dr, metric.dr)
    assert jnp.allclose(charge_density, 0.0)
    assert jnp.allclose(E_r, 0.0)


def test_rk4_step_preserves_flat_vacuum_metric():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    _, updated, _, _ = rk4_step(
        _empty_particles(),
        metric,
        dt=1.0e-3,
        EM_on=False,
        GR_on=True,
    )

    for updated_field, metric_field in zip(updated, metric):
        assert jnp.allclose(updated_field, metric_field)


def test_compiled_vacuum_scan_matches_repeated_rk4_steps():
    import jax

    from RadiShPICR.Z4C.time_evolve import (
        advance_vacuum_steps,
        metric_rk4_step,
    )

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    dt = 1.0e-3
    expected = metric
    for _ in range(3):
        expected = metric_rk4_step(
            expected,
            initialize_vacuum_matter_terms(expected),
            dt,
        )

    actual, first_nonfinite_step = jax.jit(
        advance_vacuum_steps,
        static_argnames=("num_steps",),
    )(metric, dt, num_steps=3)

    for actual_field, expected_field in zip(actual, expected):
        assert jnp.allclose(actual_field, expected_field)
    assert first_nonfinite_step == -1


def test_compiled_vacuum_scan_crosses_puncture_regression_step():
    from RadiShPICR.Z4C.time_evolve import advance_vacuum_steps
    from RadiShPICR.Z4C.utils import generate_r_grid

    r = generate_r_grid(0.0, 100.0, 8000)
    dr = r[1] - r[0]
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)
    psi = 1.0 + 1.0 / (2.0 * r)
    metric = Z4C_Metric(
        alpha=psi**(-2),
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=psi**(-4),
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Gamma=zeros,
        kappa=jnp.asarray(0.02, dtype=r.dtype),
        eta=jnp.asarray(2.0, dtype=r.dtype),
        nu=jnp.asarray(0.02, dtype=r.dtype),
        r=r,
        dr=dr,
    )

    metric, first_nonfinite_step = jax.jit(
        advance_vacuum_steps,
        static_argnames=("num_steps",),
    )(metric, 0.2 * dr, num_steps=143)

    for field_name in Z4C_Metric._fields[:10]:
        assert jnp.all(jnp.isfinite(getattr(metric, field_name)))
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
    particles = _empty_particles()
    stage_metrics = []

    def fake_metric_time_derivatives(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
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

    with jax.disable_jit():
        _, updated, _, _ = time_evolve.rk4_step(
            particles,
            metric,
            dt=0.1,
            EM_on=False,
            GR_on=True,
        )

    assert len(stage_metrics) == 4
    for stage_metric in stage_metrics:
        _assert_algebraic_constraints(stage_metric)
    _assert_algebraic_constraints(updated)


def test_rk4_step_uses_classic_stage_weights(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    particles = _empty_particles()
    stage_values = [1.0, 2.0, 3.0, 4.0]

    def fake_metric_time_derivatives(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
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

    with jax.disable_jit():
        _, updated, _, _ = time_evolve.rk4_step(
            particles,
            metric,
            dt=0.6,
            EM_on=False,
            GR_on=True,
        )

    expected_alpha = metric.alpha + 0.6 * (1.0 + 2.0 * 2.0 + 2.0 * 3.0 + 4.0) / 6.0
    assert jnp.allclose(updated.alpha, expected_alpha)
    assert jnp.allclose(updated.beta, metric.beta)
    assert jnp.allclose(updated.r, metric.r)
    assert stage_values == []


def test_rk4_step_projects_every_gr_metric_stage(monkeypatch):
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

    def fake_compute_geodesic_terms(
        stage_particles,
        stage_metric,
    ):
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

    def fake_metric_time_derivatives(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
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

    with jax.disable_jit():
        _, updated_metric, _, _ = time_evolve.rk4_step(
            particles,
            metric,
            dt=0.1,
            EM_on=False,
            GR_on=True,
        )

    assert len(stage_metrics) == 4
    for stage_metric in stage_metrics:
        _assert_algebraic_constraints(stage_metric)
    _assert_algebraic_constraints(updated_metric)


def test_rk4_step_keeps_unrestricted_standard_particle_state(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    particles = _make_particles()
    dt = 0.1

    def fake_compute_geodesic_terms(
        stage_particles,
        stage_metric,
    ):
        du_r_dt = jnp.ones_like(stage_particles.ur)
        du_phi_dt = jnp.full_like(stage_particles.uphi, 10.0)
        dr_dt = jnp.asarray([-4.0, 2.0])
        dphi_dt = -jnp.ones_like(stage_particles.phi)

        return du_r_dt, du_phi_dt, dr_dt, dphi_dt

    def fake_compute_radial_matter_terms(stage_particles, stage_metric):
        return MatterTerms(
            rho=jnp.zeros_like(stage_metric.r),
            Srr=jnp.zeros_like(stage_metric.r),
            Stt=jnp.zeros_like(stage_metric.r),
            Sr=jnp.zeros_like(stage_metric.r),
            St=jnp.zeros_like(stage_metric.r),
        )

    def fake_metric_time_derivatives(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
        return _metric_derivative(stage_metric, alpha_value=1.0)

    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(time_evolve, "compute_radial_matter_terms", fake_compute_radial_matter_terms)
    monkeypatch.setattr(time_evolve, "metric_time_derivatives", fake_metric_time_derivatives)

    r0 = particles.r.copy()
    phi0 = particles.phi.copy()
    ur0 = particles.ur.copy()
    uphi0 = particles.uphi.copy()

    updated_particles, updated_metric, _, _ = time_evolve.rk4_step(
        particles,
        metric,
        dt,
        EM_on=False,
        GR_on=True,
    )

    expected_r = r0 + dt * jnp.asarray([-4.0, 2.0])
    expected_ur = ur0 + dt

    assert updated_particles is not particles
    assert jnp.allclose(particles.r, r0)
    assert jnp.allclose(particles.phi, phi0)
    assert jnp.allclose(particles.ur, ur0)
    assert jnp.allclose(particles.uphi, uphi0)
    assert jnp.allclose(updated_particles.r, expected_r)
    assert updated_particles.r[0] < 0.0
    assert jnp.allclose(updated_particles.phi, phi0 - dt)
    assert jnp.allclose(updated_particles.ur, expected_ur)
    assert jnp.allclose(updated_particles.uphi, uphi0 + 10.0 * dt)
    assert jnp.allclose(updated_metric.alpha, metric.alpha + dt)


def test_rk4_step_recomputes_matter_from_each_particle_stage(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    particles = _make_particles()
    dt = 0.2
    matter_stage_positions = []
    derivative_stage_rho = []
    geodesic_stage = []

    def fake_compute_geodesic_terms(
        stage_particles,
        stage_metric,
    ):
        stage_number = len(geodesic_stage) + 1.0
        geodesic_stage.append(stage_number)
        du_r_dt = jnp.full_like(stage_particles.ur, stage_number)
        du_phi_dt = jnp.zeros_like(stage_particles.uphi)
        dr_dt = jnp.full_like(stage_particles.r, stage_number)
        dphi_dt = jnp.zeros_like(stage_particles.phi)

        return du_r_dt, du_phi_dt, dr_dt, dphi_dt

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

    def fake_metric_time_derivatives(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
        derivative_stage_rho.append(stage_matter_terms.rho[0])
        return _metric_derivative(stage_metric, alpha_value=1.0)

    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(time_evolve, "compute_radial_matter_terms", fake_compute_radial_matter_terms)
    monkeypatch.setattr(time_evolve, "metric_time_derivatives", fake_metric_time_derivatives)

    r0 = particles.r.copy()

    with jax.disable_jit():
        time_evolve.rk4_step(
            particles,
            metric,
            dt,
            EM_on=False,
            GR_on=True,
        )

    assert len(matter_stage_positions) == 4
    assert jnp.allclose(matter_stage_positions[0], r0)
    assert jnp.allclose(matter_stage_positions[1], r0 + 0.5 * dt)
    assert jnp.allclose(matter_stage_positions[2], r0 + dt)
    assert jnp.allclose(matter_stage_positions[3], r0 + 3.0 * dt)
    assert jnp.allclose(
        jnp.asarray(derivative_stage_rho),
        jnp.asarray([r0[0], r0[0] + 0.5 * dt, r0[0] + dt, r0[0] + 3.0 * dt]),
    )


def test_flat_space_radial_particle_trajectory_is_exact():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    grid_r = jnp.arange(0.5, 20.5, 0.5)
    initial_r = 5.0
    initial_phi = 0.3
    initial_ur = 0.2
    final_time = 0.8
    lorentz_factor = jnp.sqrt(1.0 + initial_ur**2)
    exact_r = initial_r + final_time * initial_ur / lorentz_factor

    metric = _flat_metric(grid_r)
    particles = particle_species(
        name="test",
        charge=0.0,
        mass=0.0,
        weight=1.0,
        r=jnp.asarray([initial_r]),
        ur=jnp.asarray([initial_ur]),
        phi=jnp.asarray([initial_phi]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )

    for _ in range(4):
        particles, metric, _, _ = rk4_step(
            particles,
            metric,
            dt=0.2,
            EM_on=False,
            GR_on=True,
        )

    assert jnp.allclose(particles.r, exact_r)
    assert jnp.allclose(particles.phi, initial_phi)
    assert jnp.allclose(particles.ur, initial_ur)
    assert jnp.allclose(particles.uphi, 0.0)


def test_deleting_particle_boundary_removes_center_crossing(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    particles = particle_species(
        name="deleting",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([1.0]),
        r=jnp.asarray([0.1]),
        ur=jnp.asarray([-1.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )

    def constant_infall(stage_particles, stage_metric):
        zeros = jnp.zeros_like(stage_particles.r)
        return zeros, zeros, -jnp.ones_like(stage_particles.r), zeros

    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", constant_infall)
    particles, _, _, _ = time_evolve.rk4_step(
        particles,
        metric,
        dt=0.2,
        EM_on=False,
        GR_on=True,
        particle_boundary=deleting_particle_boundary,
    )

    assert jnp.allclose(particles.r, 0.0)
    assert jnp.allclose(particles.ur, 0.0)
    assert jnp.allclose(particles.weight, 0.0)


def test_deleting_particle_boundary_is_irreversible_across_rk_stages(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    particles = particle_species(
        name="deleting",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([1.0]),
        r=jnp.asarray([0.1]),
        ur=jnp.asarray([-1.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="nearest",
    )
    stage_weights = []

    def record_matter_weights(stage_particles, stage_metric):
        stage_weights.append(stage_particles.weight.copy())
        return initialize_vacuum_matter_terms(stage_metric)

    def constant_infall(stage_particles, stage_metric):
        zeros = jnp.zeros_like(stage_particles.r)
        return zeros, zeros, -jnp.ones_like(stage_particles.r), zeros

    monkeypatch.setattr(time_evolve, "compute_radial_matter_terms", record_matter_weights)
    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", constant_infall)
    with jax.disable_jit():
        particles, _, _, _ = time_evolve.rk4_step(
            particles,
            metric,
            dt=0.4,
            EM_on=False,
            GR_on=True,
            particle_boundary=deleting_particle_boundary,
        )

    assert jnp.allclose(
        jnp.asarray(stage_weights)[:, 0],
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    assert jnp.allclose(particles.weight, 0.0)


def test_zero_overlap_absorption_is_irreversible_across_rk_stages(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    particles = particle_species(
        name="areal-open",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([1.0]),
        r=jnp.asarray([0.0]),
        ur=jnp.asarray([-1.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )
    stage_weights = []

    def record_matter_weights(
        stage_particles,
        stage_metric,
        inner_open=False,
    ):
        assert inner_open
        stage_weights.append(stage_particles.weight.copy())
        return initialize_vacuum_matter_terms(stage_metric)

    def constant_infall(stage_particles, stage_metric):
        zeros = jnp.zeros_like(stage_particles.r)
        return zeros, zeros, -jnp.ones_like(stage_particles.r), zeros

    monkeypatch.setattr(
        time_evolve,
        "compute_radial_matter_terms",
        record_matter_weights,
    )
    monkeypatch.setattr(time_evolve, "compute_geodesic_terms", constant_infall)
    with jax.disable_jit():
        particles, _, _, _ = time_evolve.rk4_step(
            particles,
            metric,
            dt=0.6,
            EM_on=False,
            GR_on=True,
            particle_boundary=deleting_inner_areal_radius_boundary,
        )

    assert jnp.allclose(
        jnp.asarray(stage_weights)[:, 0],
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    assert jnp.allclose(particles.weight, 0.0)
    assert jnp.allclose(particles.r, 0.0)


def test_deleting_particle_boundary_preserves_active_particles():
    import RadiShPICR.Z4C as z4c
    from RadiShPICR.Z4C import deleting_particle_boundary as public_boundary

    particles = particle_species(
        name="deleting",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([0.4, 0.6]),
        r=jnp.asarray([-0.1, 0.2]),
        ur=jnp.asarray([-1.0, 0.3]),
        phi=jnp.asarray([0.1, 0.2]),
        uphi=jnp.asarray([0.0, 0.4]),
        shape_mode="nearest",
    )

    particles = public_boundary(particles)

    assert public_boundary is deleting_particle_boundary
    assert not hasattr(z4c, "reflecting_particle_boundary")
    assert jnp.allclose(particles.weight, jnp.asarray([0.0, 0.6]))
    assert jnp.allclose(particles.r, jnp.asarray([0.0, 0.2]))
    assert jnp.allclose(particles.ur, jnp.asarray([0.0, 0.3]))
    assert jnp.allclose(particles.phi, jnp.asarray([0.0, 0.2]))
    assert jnp.allclose(particles.uphi, jnp.asarray([0.0, 0.4]))


def test_deleting_inner_areal_radius_boundary_absorbs_zero_overlap_particles():
    import RadiShPICR.Z4C as z4c

    metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    particles = particle_species(
        name="areal-open",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 0.0]),
        r=jnp.asarray([-0.751, -0.75, -0.25, -0.249, 6.0, -1.0]),
        ur=jnp.asarray([-1.0, -0.5, 0.5, 0.25, -1.0, 2.0]),
        phi=jnp.asarray([0.1, 0.2, 0.3, 0.35, 0.4, 0.5]),
        uphi=jnp.asarray([0.0, 0.1, 0.2, 0.25, 0.3, 0.4]),
        shape_mode="quadratic",
    )

    eager_particles = deleting_inner_areal_radius_boundary(
        particles,
        metric,
    )
    compiled_particles = jax.jit(
        deleting_inner_areal_radius_boundary
    )(particles, metric)

    # The quadratic shape has zero physical overlap at r=-0.25 and a small
    # positive overlap immediately to its right.  The outer particle is not
    # part of the inner absorbing boundary.
    expected_weight = jnp.asarray([0.0, 0.0, 0.0, 4.0, 5.0, 0.0])
    expected_r = jnp.asarray([0.0, 0.0, 0.0, -0.249, 6.0, 0.0])
    expected_ur = jnp.asarray([0.0, 0.0, 0.0, 0.25, -1.0, 0.0])
    expected_phi = jnp.asarray([0.0, 0.0, 0.0, 0.35, 0.4, 0.0])
    expected_uphi = jnp.asarray([0.0, 0.0, 0.0, 0.25, 0.3, 0.0])

    assert z4c.deleting_inner_areal_radius_boundary is (
        deleting_inner_areal_radius_boundary
    )
    for result in (eager_particles, compiled_particles):
        assert jnp.allclose(result.weight, expected_weight)
        assert jnp.allclose(result.r, expected_r)
        assert jnp.allclose(result.ur, expected_ur)
        assert jnp.allclose(result.phi, expected_phi)
        assert jnp.allclose(result.uphi, expected_uphi)


def test_deleting_inner_areal_radius_boundary_uses_global_grid_minimum():
    metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    metric = metric._replace(
        conformal_gt=metric.conformal_gt.at[0].set(9.0),
    )
    particles = particle_species(
        name="areal-open",
        charge=0.0,
        mass=1.0,
        weight=jnp.ones(5),
        r=jnp.asarray([-0.251, 0.25, 0.251, 0.3, 6.0]),
        ur=jnp.zeros(5),
        phi=jnp.zeros(5),
        uphi=jnp.zeros(5),
        shape_mode="quadratic",
    )

    particles = deleting_inner_areal_radius_boundary(particles, metric)

    # R_grid = [1.5, 1.0, ...].  Relative to boundary index 1, the TSC shape
    # has zero physical overlap at r=0.25 and positive overlap just above it.
    assert jnp.allclose(
        particles.weight,
        jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    assert jnp.allclose(
        particles.r,
        jnp.asarray([0.0, 0.0, 0.251, 0.3, 6.0]),
    )


def test_areal_inner_absorption_is_irreversible_when_grid_minimum_returns():
    from RadiShPICR.Z4C.electric_field import compute_radial_charge_density
    from RadiShPICR.Z4C.energy_momentum_tensor import (
        compute_radial_matter_terms,
    )

    inner_zero_metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    inner_one_metric = inner_zero_metric._replace(
        conformal_gt=inner_zero_metric.conformal_gt.at[0].set(9.0),
    )
    particles = particle_species(
        name="areal-open",
        charge=2.0,
        mass=1.0,
        weight=jnp.asarray([1.0]),
        r=jnp.asarray([0.0]),
        ur=jnp.asarray([-0.4]),
        phi=jnp.asarray([0.2]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )

    def move_boundary_out_and_back(stage_particles):
        stage_particles = deleting_inner_areal_radius_boundary(
            stage_particles,
            inner_zero_metric,
        )
        stage_particles = deleting_inner_areal_radius_boundary(
            stage_particles,
            inner_one_metric,
        )
        return deleting_inner_areal_radius_boundary(
            stage_particles,
            inner_zero_metric,
        )

    eager_particles = move_boundary_out_and_back(particles)
    compiled_particles = jax.jit(move_boundary_out_and_back)(particles)

    for result in (eager_particles, compiled_particles):
        assert jnp.allclose(result.weight, 0.0)
        assert jnp.allclose(result.r, 0.0)
        assert jnp.allclose(result.ur, 0.0)
        assert jnp.allclose(result.phi, 0.0)
        assert jnp.allclose(result.uphi, 0.0)

        matter_terms = compute_radial_matter_terms(
            result,
            inner_zero_metric,
            True,
        )
        charge_density = compute_radial_charge_density(
            result,
            inner_zero_metric,
            True,
        )
        for source in (*matter_terms, charge_density):
            assert jnp.allclose(source, 0.0)


def test_particle_boundary_receives_matching_rk_stage_metric(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    particles = _empty_particles()
    boundary_metric_values = []

    def constant_metric_derivative(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
        return _metric_derivative(stage_metric, alpha_value=1.0)

    def record_boundary(stage_particles, stage_metric):
        boundary_metric_values.append(float(stage_metric.alpha[0]))
        return stage_particles

    monkeypatch.setattr(
        time_evolve,
        "metric_time_derivatives",
        constant_metric_derivative,
    )
    with jax.disable_jit():
        time_evolve.rk4_step(
            particles,
            metric,
            dt=0.2,
            EM_on=False,
            GR_on=True,
            particle_boundary=record_boundary,
        )

    assert jnp.allclose(
        jnp.asarray(boundary_metric_values),
        jnp.asarray([1.0, 1.1, 1.1, 1.2, 1.2]),
    )


def test_areal_inner_boundary_uses_open_matter_deposition_at_every_stage(
    monkeypatch,
):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    metric = _flat_metric(jnp.arange(0.5, 5.5, 0.5))
    particles = particle_species(
        name="areal-open",
        charge=0.0,
        mass=1.0,
        weight=jnp.asarray([1.0]),
        r=jnp.asarray([1.0]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )
    inner_open_values = []

    def record_matter_boundary(
        stage_particles,
        stage_metric,
        inner_open=False,
    ):
        inner_open_values.append(inner_open)
        return initialize_vacuum_matter_terms(stage_metric)

    def zero_metric_derivative(
        stage_metric,
        stage_matter_terms,
        metric_boundary=0,
    ):
        return _metric_derivative(stage_metric, alpha_value=0.0)

    monkeypatch.setattr(
        time_evolve,
        "compute_radial_matter_terms",
        record_matter_boundary,
    )
    monkeypatch.setattr(
        time_evolve,
        "metric_time_derivatives",
        zero_metric_derivative,
    )
    with jax.disable_jit():
        time_evolve.rk4_step(
            particles,
            metric,
            dt=0.2,
            EM_on=False,
            GR_on=True,
            particle_boundary=deleting_inner_areal_radius_boundary,
        )

    assert inner_open_values == [True, True, True, True]
