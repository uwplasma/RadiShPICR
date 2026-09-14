import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors import (
    metric_corrected_cic_stencil,
    metric_corrected_quadratic_stencil,
    particle_deposition_stencil,
)
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    _cell_centered_radial_shape_stencil,
    _unbounded_raw_radial_shape_stencil,
)
from RadiShPICR.particles.shape_factors.common import metric_correction_coefficients
from RadiShPICR.particles.shape_factors.metric_corrected_cic import raw_cic_stencil
from RadiShPICR.particles.shape_factors.metric_correct_quadratic import raw_quadratic_stencil
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def metric_grid(num_cells=32, dr=0.25):
    r = (jnp.arange(num_cells) + 0.5) * dr
    ones = jnp.ones_like(r)
    zeros = jnp.zeros_like(r)
    return Z4C_Metric(
        alpha=ones, beta=zeros, conformal_grr=ones, conformal_gt=ones,
        chi=ones, Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        kappa=zeros, eta=zeros, nu=zeros, r=r, dr=jnp.asarray(dr),
    )


def particle_positions(r, shape_mode="linear"):
    return particle_species(
        name="shape test", charge=2.0, mass=3.0,
        weight=jnp.ones_like(r), r=r, ur=jnp.zeros_like(r),
        phi=jnp.zeros_like(r), uphi=jnp.zeros_like(r), shape_mode=shape_mode,
    )


@pytest.mark.parametrize("stencil", [metric_corrected_cic_stencil, metric_corrected_quadratic_stencil])
@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("inner_open", [False, True])
def test_corrected_density_against_independent_midpoint_integration(stencil, curved, inner_open):
    metric = metric_grid()
    if curved:
        metric = metric._replace(
            conformal_grr=1 + 0.04 * metric.r**2,
            conformal_gt=1 + 0.03 * metric.r,
            chi=1 / (1 + 0.02 * metric.r**2),
        )

    # Integrate with uniform subcell samples, independently of the Gaussian
    # quadrature used to construct beta. The metric is constant in each shell,
    # matching the finite-volume representation of the production metric.
    samples_per_cell = 2048
    dr = float(metric.dr)
    r = np.asarray(metric.r)
    x = (r[:, None] - 0.5 * dr
         + (np.arange(samples_per_cell)[None, :] + 0.5) * dr / samples_per_cell)
    volume_factor = np.asarray(
        jnp.sqrt(metric.conformal_grr) * metric.conformal_gt / metric.chi**1.5
    )
    dv = 4 * np.pi * volume_factor[:, None] * x**2 * dr / samples_per_cell
    shell_volume = 4 * np.pi / 3 * volume_factor * ((r + 0.5 * dr)**3 - (r - 0.5 * dr)**3)

    particles = particle_positions(jnp.asarray(x.reshape(-1)))
    indices, weights, _ = stencil(particles, metric, inner_open)
    deposited_volume = np.zeros_like(r)
    np.add.at(deposited_volume, np.asarray(indices), np.asarray(weights) * dv.reshape(1, -1))
    density = deposited_volume / shell_volume

    # Outer truncation is intentionally not a uniform-density boundary.
    np.testing.assert_allclose(density[:-2], 1.0, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("inner_open", [False, True])
@pytest.mark.parametrize("shape_mode,raw_stencil,limit", [
    ("linear", raw_cic_stencil, 1.0),
    ("quadratic", raw_quadratic_stencil, 1.5),
])
def test_corrected_weights_remain_positive_when_limited(shape_mode, raw_stencil, limit, inner_open):
    metric = metric_grid(num_cells=64)
    metric = metric._replace(chi=jnp.exp(2.0 * metric.r))
    x = (metric.r[:-1, None] + metric.dr * jnp.linspace(0.0, 1.0, 101)[None, :]).reshape(-1)
    x = jnp.concatenate((-x[::-1], x))
    indices, weights = raw_stencil(x, metric, inner_open)
    beta = metric_correction_coefficients(metric, inner_open, shape_mode)
    beta_limit = limit * (1 - 8 * jnp.finfo(metric.r.dtype).eps)

    assert indices.shape[0] == (2 if shape_mode == "linear" else 3)
    assert jnp.all(jnp.diff(indices, axis=0) == 1)
    assert jnp.all(jnp.isfinite(weights))
    assert jnp.min(weights) >= 0.0
    np.testing.assert_allclose(jnp.sum(weights, axis=0), 1.0, atol=2e-15)
    assert jnp.all(jnp.abs(beta) <= beta_limit)
    assert jnp.any(jnp.isclose(jnp.abs(beta), beta_limit))

    # The correction vanishes at centers even when beta has been limited.
    indices, weights = raw_stencil(metric.r, metric, inner_open)
    original_indices, original_weights = _unbounded_raw_radial_shape_stencil(
        metric.r, metric.r, metric.dr, shape_mode,
    )
    np.testing.assert_array_equal(indices, original_indices)
    np.testing.assert_allclose(weights, original_weights, atol=2e-15)


@pytest.mark.parametrize("stencil,raw_stencil", [
    (metric_corrected_cic_stencil, raw_cic_stencil),
    (metric_corrected_quadratic_stencil, raw_quadratic_stencil),
])
def test_corrected_stencil_parity_and_moving_open_boundary(stencil, raw_stencil):
    metric = metric_grid()
    x = jnp.asarray([0.0, 0.03, 0.25, 0.37, 1.13])
    indices, even, odd = stencil(particle_positions(x), metric)
    mirrored_indices, mirrored_even, mirrored_odd = stencil(particle_positions(-x), metric)

    def dense(index, weights):
        columns = jnp.broadcast_to(jnp.arange(x.size), index.shape)
        return jnp.zeros((metric.r.size, x.size)).at[index, columns].add(weights)

    np.testing.assert_allclose(dense(indices, even), dense(mirrored_indices, mirrored_even), atol=2e-15)
    np.testing.assert_allclose(dense(indices, odd), -dense(mirrored_indices, mirrored_odd), atol=2e-15)

    # Prescribe an areal-radius minimum at the fourth grid center.
    target = 3
    areal_radius = 1.0 + (metric.r - metric.r[target])**2
    metric = metric._replace(chi=(metric.r / areal_radius)**2)
    x = jnp.asarray([-0.5, 0.0, 0.7, 0.9, 1.1, 7.8, 8.0, 8.5])
    raw_indices, raw_weights = raw_stencil(x, metric, True)
    indices, even, odd = stencil(particle_positions(x), metric, True)
    expected = np.where((np.asarray(raw_indices) >= target) & (np.asarray(raw_indices) < metric.r.size), raw_weights, 0.0)

    np.testing.assert_allclose(even, expected, atol=2e-15)
    np.testing.assert_array_equal(odd, even)
    assert jnp.all((indices >= 0) & (indices < metric.r.size))
    assert jnp.any((jnp.sum(even, axis=0) > 0) & (jnp.sum(even, axis=0) < 1))


@pytest.mark.parametrize("stencil,width", [
    (metric_corrected_cic_stencil, 2), (metric_corrected_quadratic_stencil, 3),
])
def test_corrected_api_is_jittable_pure_and_handles_empty_species(stencil, width):
    metric = metric_grid()
    compiled = jax.jit(stencil)
    for x in (jnp.asarray([-0.2, 0.0, 0.13, 1.2, 8.1]), jnp.asarray([])):
        # Method-specific APIs select their shape explicitly.
        particles = particle_positions(x, shape_mode="nearest")
        before = [np.asarray(value).copy() for value in jax.tree_util.tree_leaves(particles)]
        for inner_open in (False, True):
            eager = stencil(particles, metric, inner_open)
            actual = compiled(particles, metric, jnp.asarray(inner_open))
            for expected, value in zip(eager, actual):
                assert value.shape == (width, x.size)
                np.testing.assert_allclose(value, expected, atol=2e-15)
        for previous, current in zip(before, jax.tree_util.tree_leaves(particles)):
            np.testing.assert_array_equal(previous, current)


def test_corrected_cic_is_explicit_and_default_linear_stencil_is_unchanged():
    metric = metric_grid()
    particles = particle_positions(jnp.asarray([0.0, 0.2, 0.3, 1.13, 7.9]))
    production = particle_deposition_stencil(particles, metric)
    ordinary = _cell_centered_radial_shape_stencil(
        particles.r, metric.r, metric.dr, "linear",
    )[:3]
    for actual, expected in zip(production, ordinary):
        np.testing.assert_array_equal(actual, expected)
    _, corrected, _ = metric_corrected_cic_stencil(particles, metric)
    assert not jnp.allclose(corrected, production[1])
