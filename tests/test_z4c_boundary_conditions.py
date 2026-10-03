import jax
import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.boundary_conditions import (
    METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    METRIC_BOUNDARY_SOMMERFELD,
    conformal_connection_constraint,
    constraint_preserving_characteristic_speeds,
)
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative
from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
from RadiShPICR.Z4C.shift_and_lapse import dalphadt, dbetadt
from RadiShPICR.Z4C.spatial_metric import dchidt, dgrrdt, dgtdt
from RadiShPICR.Z4C.time_evolve import (
    advance_vacuum_steps,
    metric_rk4_step,
    metric_time_derivatives,
    rk4_step,
)
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _flat_metric(num_cells=64):
    r = (
        jnp.arange(num_cells, dtype=jnp.float64) + 0.5
    ) * 8.0 / num_cells
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
        dr=r[1] - r[0],
    )


def _smooth_constrained_metric():
    metric = _flat_metric()
    r = metric.r
    pulse = jnp.exp(-((r - 6.0) / 0.8) ** 2)
    conformal_grr = 1.0 + 0.02 * pulse
    conformal_gt = conformal_grr**-0.5
    Arr = 0.003 * pulse
    At = -0.5 * conformal_gt * Arr / conformal_grr

    return metric._replace(
        alpha=1.0 + 0.01 * pulse,
        beta=0.002 * r * pulse,
        conformal_grr=conformal_grr,
        conformal_gt=conformal_gt,
        chi=1.0 - 0.015 * pulse,
        Kh=0.004 * pulse,
        Arr=Arr,
        At=At,
        theta=0.002 * pulse,
        Gamma=0.003 * r * pulse,
        kappa=jnp.asarray(0.02),
        eta=jnp.asarray(0.1),
    )


def test_standard_sommerfeld_is_the_default_boundary():
    metric = _smooth_constrained_metric()
    matter_terms = initialize_vacuum_matter_terms(metric)

    default_rhs = metric_time_derivatives(metric, matter_terms)
    explicit_rhs = metric_time_derivatives(
        metric,
        matter_terms,
        METRIC_BOUNDARY_SOMMERFELD,
    )

    for default_field, explicit_field in zip(default_rhs, explicit_rhs):
        assert jnp.array_equal(default_field, explicit_field)


def test_constraint_preserving_boundary_keeps_bulk_metric_equations():
    metric = _smooth_constrained_metric()
    matter_terms = initialize_vacuum_matter_terms(metric)
    rhs = metric_time_derivatives(
        metric,
        matter_terms,
        METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    )

    assert rhs.alpha[-1] == dalphadt(
        metric,
        matter_terms,
        apply_sommerfeld_boundary=False,
    )[-1]
    assert rhs.beta[-1] == dbetadt(
        metric,
        matter_terms,
        apply_sommerfeld_boundary=False,
    )[-1]
    assert rhs.chi[-1] == dchidt(
        metric,
        matter_terms,
        apply_sommerfeld_boundary=False,
    )[-1]
    assert rhs.conformal_grr[-1] == dgrrdt(
        metric,
        matter_terms,
        apply_sommerfeld_boundary=False,
    )[-1]
    assert rhs.conformal_gt[-1] == dgtdt(
        metric,
        matter_terms,
        apply_sommerfeld_boundary=False,
    )[-1]


def test_constraint_preserving_curvature_rhs_is_trace_tangent():
    metric = _smooth_constrained_metric()
    matter_terms = initialize_vacuum_matter_terms(metric)
    rhs = metric_time_derivatives(
        metric,
        matter_terms,
        METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    )

    trace_rhs = (
        rhs.Arr / metric.conformal_grr
        - metric.Arr * rhs.conformal_grr / metric.conformal_grr**2
        + 2.0 * rhs.At / metric.conformal_gt
        - 2.0 * metric.At * rhs.conformal_gt / metric.conformal_gt**2
    )

    assert jnp.abs(trace_rhs[-1]) < 1.0e-14


def test_flat_constraint_preserving_boundary_is_stationary_and_jittable():
    metric = _flat_metric()
    matter_terms = initialize_vacuum_matter_terms(metric)
    jitted_step = jax.jit(metric_rk4_step)

    updated = jitted_step(
        metric,
        matter_terms,
        1.0e-3,
        METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    )

    for updated_field, initial_field in zip(updated[:10], metric[:10]):
        assert jnp.allclose(updated_field, initial_field, atol=2.0e-15)


def test_constraint_preserving_selector_compiles_through_public_steppers():
    metric = _flat_metric()
    empty = jnp.asarray([], dtype=metric.r.dtype)
    particles = particle_species(
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

    _, coupled_metric, _, _ = jax.jit(rk4_step)(
        particles,
        metric,
        1.0e-3,
        E_r=jnp.zeros_like(metric.r), EM_on=False,
        GR_on=True,
        metric_boundary=METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    )
    vacuum_step = jax.jit(
        advance_vacuum_steps,
        static_argnames=("num_steps",),
    )
    vacuum_metric, first_nonfinite_step = vacuum_step(
        metric,
        1.0e-3,
        num_steps=2,
        metric_boundary=METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    )

    assert first_nonfinite_step == -1
    for updated_metric in (coupled_metric, vacuum_metric):
        for updated_field, initial_field in zip(updated_metric[:10], metric[:10]):
            assert jnp.allclose(updated_field, initial_field, atol=2.0e-15)


def test_flat_connection_constraint_and_characteristic_speeds():
    metric = _flat_metric()

    connection_constraint = conformal_connection_constraint(metric)
    physical_speed, lapse_speed, shift_speed = (
        constraint_preserving_characteristic_speeds(metric)
    )

    assert jnp.max(jnp.abs(connection_constraint)) < 1.0e-13
    assert jnp.allclose(physical_speed, 1.0)
    assert jnp.allclose(lapse_speed, jnp.sqrt(2.0))
    assert jnp.allclose(shift_speed, jnp.sqrt(10.0 / 3.0))
    assert jnp.all(jnp.isfinite(jnp.stack((physical_speed, lapse_speed, shift_speed))))


def test_nonlinear_theta_and_gamma_boundary_equations():
    metric = _smooth_constrained_metric()
    matter_terms = initialize_vacuum_matter_terms(metric)
    rhs = metric_time_derivatives(
        metric,
        matter_terms,
        METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    )

    dtheta_dr = first_derivative(metric.theta, metric.dr, parity=1)
    normal_speed = jnp.sqrt(metric.chi / metric.conformal_grr)
    expected_theta = (
        -metric.alpha * normal_speed + metric.beta
    ) * dtheta_dr

    dGamma_dr = first_derivative(metric.Gamma, metric.dr, parity=-1)
    dbeta_dr = first_derivative(metric.beta, metric.dr, parity=-1)
    d2beta_dr2 = second_derivative(metric.beta, metric.dr, parity=-1)
    expected_Gamma = (
        metric.eta * metric.beta * dGamma_dr
        + metric.eta * metric.beta**2 * d2beta_dr2 / 2.5
        + metric.eta * metric.beta * dbeta_dr**2 / 2.5
        - metric.eta**2 * metric.beta * dbeta_dr / 2.5
    )

    assert jnp.allclose(rhs.theta[-1], expected_theta[-1])
    assert jnp.allclose(rhs.Gamma[-1], expected_Gamma[-1])


def test_manufactured_outgoing_constraint_pulse_converges_at_boundary():
    cp_errors = []
    sommerfeld_errors = []

    for num_cells in (80, 160, 320):
        metric = _flat_metric(num_cells)
        pulse = 1.0e-3 * jnp.exp(-((metric.r - 7.7) / 0.5) ** 2)
        metric = metric._replace(theta=pulse)
        matter_terms = initialize_vacuum_matter_terms(metric)
        exact_dt_theta = (
            2.0 * (metric.r[-1] - 7.7) * pulse[-1] / 0.5**2
        )

        sommerfeld_rhs = metric_time_derivatives(
            metric,
            matter_terms,
            METRIC_BOUNDARY_SOMMERFELD,
        )
        cp_rhs = metric_time_derivatives(
            metric,
            matter_terms,
            METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
        )
        sommerfeld_errors.append(
            jnp.abs(sommerfeld_rhs.theta[-1] - exact_dt_theta)
        )
        cp_errors.append(jnp.abs(cp_rhs.theta[-1] - exact_dt_theta))

    assert cp_errors[1] < cp_errors[0] / 8.0
    assert cp_errors[2] < cp_errors[1] / 8.0
    assert cp_errors[-1] < sommerfeld_errors[-1] / 100.0
