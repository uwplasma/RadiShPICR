import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
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
        nu=jnp.asarray(0.02),
    )


def test_sommerfeld_endpoints_use_existing_speeds_and_asymptotic_values():
    metric = _flat_metric()
    # Polynomial profiles give exact analytic derivatives for the endpoint
    # stencil. Nonzero shift and chi != 1 distinguish the existing speeds.
    asymptotic_values = (1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    fields = {}
    radial_derivatives = {}
    for name, background in zip(metric._fields[:10], asymptotic_values):
        if name in ("beta", "Gamma"):
            fields[name] = 0.01 * metric.r / 8.0
            radial_derivatives[name] = 0.01 / 8.0
        else:
            fields[name] = background + 0.02 * (metric.r / 8.0)**2
            radial_derivatives[name] = 0.04 * metric.r[-1] / 64.0
    metric = metric._replace(**fields, nu=jnp.asarray(0.02))
    rhs = metric_time_derivatives(metric, initialize_vacuum_matter_terms(metric))

    physical_speed = -metric.beta[-1] + metric.alpha[-1] / jnp.sqrt(metric.conformal_grr[-1])
    lapse_speed = -metric.beta[-1] + jnp.sqrt(2.0 * metric.alpha[-1]) / jnp.sqrt(metric.conformal_grr[-1])
    shift_speed = -metric.beta[-1] * jnp.sqrt(5.0 / 2.0)
    for name, background in zip(metric._fields[:10], asymptotic_values):
        speed = physical_speed
        if name in ("alpha", "Kh"):
            speed = lapse_speed
        elif name in ("beta", "Gamma"):
            speed = shift_speed
        expected = -speed * (
            radial_derivatives[name] + (fields[name][-1] - background) / metric.r[-1]
        )
        np.testing.assert_allclose(getattr(rhs, name)[-1], expected, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("flat", [True, False])
def test_sommerfeld_public_steppers_agree_eager_and_jit(flat):
    metric = _flat_metric() if flat else _smooth_constrained_metric()
    matter = initialize_vacuum_matter_terms(metric)
    empty = jnp.asarray([], dtype=metric.r.dtype)
    particles = particle_species(
        name="vacuum", charge=0.0, mass=0.0, weight=empty,
        r=empty, ur=empty, phi=empty, uphi=empty, shape_mode="nearest",
    )
    eager_rhs = metric_time_derivatives(metric, matter)
    jitted_rhs = jax.jit(metric_time_derivatives)(metric, matter)
    for actual, expected in zip(jitted_rhs, eager_rhs):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    eager_step = metric_rk4_step(metric, matter, 1e-3)
    jitted_step = jax.jit(metric_rk4_step)(metric, matter, 1e-3)
    eager_vacuum, eager_failed = advance_vacuum_steps(metric, 1e-3, num_steps=2)
    jitted_vacuum, jitted_failed = jax.jit(
        advance_vacuum_steps, static_argnames=("num_steps",),
    )(metric, 1e-3, num_steps=2)
    _, eager_coupled, _, _ = rk4_step(
        particles, metric, 1e-3, E_r=jnp.zeros_like(metric.r), EM_on=False, GR_on=True,
    )
    _, jitted_coupled, _, _ = jax.jit(rk4_step)(
        particles, metric, 1e-3, E_r=jnp.zeros_like(metric.r), EM_on=False, GR_on=True,
    )
    assert eager_failed == jitted_failed == -1
    for eager, compiled in (
        (eager_step, jitted_step),
        (eager_vacuum, jitted_vacuum),
        (eager_coupled, jitted_coupled),
    ):
        for actual, expected, initial in zip(compiled, eager, metric):
            assert jnp.all(jnp.isfinite(actual))
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
            if flat:
                np.testing.assert_allclose(actual, initial, rtol=0.0, atol=2e-15)


def test_spherical_outgoing_pulse_converges_at_sommerfeld_boundary():
    errors = []
    for num_cells in (80, 160, 320):
        metric = _flat_metric(num_cells)
        # Theta = f(r-t)/r obeys dt Theta = -dr Theta - Theta/r.
        pulse = 1e-3 * jnp.exp(-((metric.r - 7.7) / 0.5)**2) / metric.r
        metric = metric._replace(theta=pulse)
        rhs = metric_time_derivatives(metric, initialize_vacuum_matter_terms(metric))
        exact_dt_theta = 2.0 * (metric.r[-1] - 7.7) * pulse[-1] / 0.5**2
        errors.append(jnp.abs(rhs.theta[-1] - exact_dt_theta))

    assert errors[1] < errors[0] / 8.0
    assert errors[2] < errors[1] / 8.0


def test_shift_control_cannot_be_set_by_an_old_positional_selector():
    metric = _flat_metric()
    matter = initialize_vacuum_matter_terms(metric)
    for function, args in (
        (metric_time_derivatives, (metric, matter)),
        (metric_rk4_step, (metric, matter, 1e-3)),
        (advance_vacuum_steps, (metric, 1e-3, 2)),
    ):
        with pytest.raises(TypeError):
            function(*args, 1)
