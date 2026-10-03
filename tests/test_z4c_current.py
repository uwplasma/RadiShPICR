import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.Z4C import (
    compute_radial_current_density,
    radial_electric_field_time_derivative,
    radial_gauss_residual,
)
from RadiShPICR.Z4C.geodesic import compute_geodesic_terms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def metric_grid():
    r = jnp.arange(0.5, 12.0, 1.0)
    zeros, ones = jnp.zeros_like(r), jnp.ones_like(r)
    return Z4C_Metric(ones, zeros, ones, ones, ones, zeros, zeros, zeros,
                      zeros, zeros, jnp.asarray(0.), jnp.asarray(0.),
                      jnp.asarray(0.), r, jnp.asarray(1.))


def particle(r, shape):
    return particle_species(
        name="charge", charge=-2., mass=1., weight=jnp.asarray([3.]),
        r=jnp.asarray([r]), ur=jnp.asarray([.8]), phi=jnp.zeros(1),
        uphi=jnp.asarray([.3]), shape_mode=shape,
    )


@pytest.mark.parametrize("shape", ["nearest", "linear", "quadratic"])
def test_transport_current_uses_relativistic_coordinate_velocity(shape):
    metric = metric_grid()._replace(
        alpha=jnp.full(12, .7), beta=jnp.full(12, .1),
        conformal_grr=jnp.full(12, 2.), conformal_gt=jnp.full(12, 3.),
        chi=jnp.full(12, .5),
    )
    particles = particle(4.2, shape)
    _, _, dr_dt, _ = compute_geodesic_terms(particles, metric)
    W = np.sqrt(1. + .8**2 / 4. + .3**2 / (6. * 4.2**2))
    expected_velocity = .7 * .8 / (4. * W) - .1
    np.testing.assert_allclose(dr_dt, expected_velocity, rtol=1.e-13)
    current = compute_radial_current_density(particles, metric, dr_dt)
    compiled = jax.jit(compute_radial_current_density)(particles, metric, dr_dt)
    np.testing.assert_allclose(compiled, current, rtol=1.e-13, atol=1.e-15)
    np.testing.assert_allclose(jnp.sum(current * proper_radial_shell_volume(metric)),
                               -6. * expected_velocity, rtol=1.e-13)
    reversed_current = compute_radial_current_density(particles, metric, -dr_dt)
    np.testing.assert_allclose(reversed_current, -current, rtol=1.e-13)


@pytest.mark.parametrize("shape", ["nearest", "linear", "quadratic"])
def test_current_origin_parity_neutrality_and_zero_weight(shape):
    metric = metric_grid()
    positive = particle(.1, shape)
    negative = particle(-.1, shape)
    velocity = jnp.asarray([.7])
    current = compute_radial_current_density(positive, metric, velocity)
    reflected = compute_radial_current_density(negative, metric, velocity)
    np.testing.assert_allclose(reflected, -current, atol=1.e-14)
    centered = compute_radial_current_density(particle(0., shape), metric, velocity)
    np.testing.assert_allclose(centered, 0., atol=1.e-14)

    # Co-moving opposite charges cancel, including unequal macro weights.
    pair = particle_species(
        name="neutral", charge=jnp.asarray([-2., 3.]), mass=1.,
        weight=jnp.asarray([3., 2.]), r=jnp.asarray([4.2, 4.2]),
        ur=jnp.zeros(2), phi=jnp.zeros(2), uphi=jnp.zeros(2), shape_mode=shape,
    )
    np.testing.assert_allclose(
        compute_radial_current_density(pair, metric, jnp.full(2, .7)), 0., atol=1.e-14,
    )
    positive.weight = jnp.zeros(1)
    np.testing.assert_array_equal(compute_radial_current_density(positive, metric, velocity), 0.)


@pytest.mark.parametrize("shape, retained", [("nearest", 1.), ("linear", .6), ("quadratic", .595)])
def test_open_inner_current_discards_shape_without_renormalizing(shape, retained):
    metric = metric_grid()
    particles = particle(.1, shape)
    current = jax.jit(compute_radial_current_density)(
        particles, metric, jnp.asarray([-.4]), inner_open=jnp.asarray(True),
    )
    np.testing.assert_allclose(jnp.sum(current * proper_radial_shell_volume(metric)),
                               retained * (-6.) * (-.4), rtol=1.e-13)
    outside = compute_radial_current_density(particle(14., shape), metric, jnp.ones(1))
    np.testing.assert_array_equal(outside, 0.)


def test_field_rhs_flat_static_curved_and_geometric_limits():
    metric = metric_grid()
    zero_rhs = jax.tree.map(jnp.zeros_like, metric)
    E = .2 * metric.r
    current = jnp.sin(metric.r)
    rhs = jax.jit(radial_electric_field_time_derivative)
    np.testing.assert_allclose(rhs(metric, zero_rhs, E, current, 2.), -current / 2.)

    curved = metric._replace(conformal_grr=2. * metric.alpha, chi=.5 * metric.alpha)
    np.testing.assert_allclose(rhs(curved, zero_rhs, E, current, 2.), -2. * current)
    # a=e^(2t), b=e^(-t), chi=e^(4t): E_r grows as e^(4t) for zero current.
    metric_rhs = zero_rhs._replace(
        conformal_grr=2. * curved.conformal_grr,
        conformal_gt=-curved.conformal_gt, chi=4. * curved.chi,
    )
    np.testing.assert_allclose(rhs(curved, metric_rhs, E, jnp.zeros_like(E)), 4. * E)


def test_differential_gauss_residual_on_curved_metric():
    metric = metric_grid()._replace(conformal_grr=jnp.full(12, 2.),
                                    conformal_gt=jnp.full(12, 3.), chi=jnp.full(12, .5))
    E = .2 * metric.r
    # D_i E^i = 3 * chi/a * 0.2 for constant spatial coefficients.
    rho = jnp.full(12, 2. * 3. * .25 * .2)
    residual = jax.jit(radial_gauss_residual)(metric, E, rho, 2.)
    np.testing.assert_allclose(residual, 0., atol=1.e-13)
