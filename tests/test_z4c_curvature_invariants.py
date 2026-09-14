import jax
import jax.numpy as jnp

from RadiShPICR.Z4C import misner_sharp_mass
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


def test_flat_space_has_zero_misner_sharp_mass():
    r = (jnp.arange(128) + 0.5) * 0.1
    metric = _metric(r)

    mass = jax.jit(misner_sharp_mass)(metric)

    assert jnp.allclose(mass, 0.0, rtol=0.0, atol=5.0e-13)


def test_isotropic_schwarzschild_matches_mass():
    mass_parameter = 0.7
    dr = 0.002
    r = (jnp.arange(6000) + 0.5) * dr
    psi = 1.0 + mass_parameter / (2.0 * r)
    metric = _metric(r, chi=psi**-4)

    mass = misner_sharp_mass(metric)
    exterior = (r > 1.5) & (r < 10.0)

    assert jnp.max(jnp.abs(mass[exterior] - mass_parameter)) < 2.0e-7


def test_misner_sharp_mass_preserves_grid_shape():
    r = (jnp.arange(32) + 0.5) * 0.2
    metric = _metric(r)

    assert misner_sharp_mass(metric).shape == r.shape
    assert jnp.all(jnp.isfinite(misner_sharp_mass(metric)))
