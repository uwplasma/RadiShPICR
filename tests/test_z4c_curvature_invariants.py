import jax
import jax.numpy as jnp
import pytest

from RadiShPICR.Z4C import kretschmann_scalar, misner_sharp_mass
from RadiShPICR.Z4C.energy_momentum_tensor import MatterTerms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _metric(r, *, chi=None, Kh=None):
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)
    if chi is None:
        chi = ones
    if Kh is None:
        Kh = zeros

    return Z4C_Metric(
        alpha=ones,
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=chi,
        Kh=Kh,
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


def _matter(metric, *, rho=0.0, radial_stress=0.0, angular_stress=0.0):
    zeros = jnp.zeros_like(metric.r)
    return MatterTerms(
        rho=jnp.broadcast_to(jnp.asarray(rho), metric.r.shape),
        Srr=jnp.broadcast_to(jnp.asarray(radial_stress), metric.r.shape),
        Stt=jnp.broadcast_to(jnp.asarray(angular_stress), metric.r.shape),
        Sr=zeros,
        St=zeros,
    )


def test_flat_vacuum_has_zero_mass_and_kretschmann():
    r = (jnp.arange(128) + 0.5) * 0.1
    metric = _metric(r)
    vacuum = _matter(metric)

    mass = jax.jit(misner_sharp_mass)(metric)
    kretschmann = jax.jit(kretschmann_scalar)(metric, vacuum)

    assert jnp.allclose(mass, 0.0, rtol=0.0, atol=5.0e-13)
    assert jnp.allclose(kretschmann, 0.0, rtol=0.0, atol=2.0e-20)


def test_isotropic_schwarzschild_matches_mass_and_kretschmann():
    mass_parameter = 0.7
    dr = 0.002
    r = (jnp.arange(6000) + 0.5) * dr
    psi = 1.0 + mass_parameter / (2.0 * r)
    metric = _metric(r, chi=psi**-4)
    vacuum = _matter(metric)

    mass = misner_sharp_mass(metric)
    kretschmann = kretschmann_scalar(metric, vacuum)
    areal_radius = r * psi**2
    expected_kretschmann = 48.0 * mass_parameter**2 / areal_radius**6
    exterior = (r > 1.5) & (r < 10.0)

    assert jnp.max(jnp.abs(mass[exterior] - mass_parameter)) < 2.0e-7
    assert jnp.allclose(
        kretschmann[exterior],
        expected_kretschmann[exterior],
        rtol=2.0e-4,
        atol=2.0e-10,
    )


def test_flat_dust_flrw_slice_has_expected_matter_curvature():
    hubble = 0.04
    r = (jnp.arange(128) + 0.5) * 0.1
    metric = _metric(r, Kh=-3.0 * hubble * jnp.ones_like(r))
    rho = 3.0 * hubble**2 / (8.0 * jnp.pi)
    dust = _matter(metric, rho=rho)

    kretschmann = kretschmann_scalar(metric, dust)

    assert jnp.allclose(
        kretschmann,
        15.0 * hubble**4,
        rtol=2.0e-10,
        atol=2.0e-18,
    )


def test_curvature_helpers_preserve_grid_shape():
    r = (jnp.arange(32) + 0.5) * 0.2
    metric = _metric(r)
    vacuum = _matter(metric)

    assert misner_sharp_mass(metric).shape == r.shape
    assert kretschmann_scalar(metric, vacuum).shape == r.shape
    assert jnp.all(jnp.isfinite(kretschmann_scalar(metric, vacuum)))
