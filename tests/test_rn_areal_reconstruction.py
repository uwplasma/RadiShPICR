"""Exact metric checks for the stationary RN comparison diagnostic."""

import jax
import numpy as np
import pytest

from demos.charged_star_collapse_Z4C_GR.plot_RN_grr_areal import reconstruct_grr


@pytest.fixture(autouse=True)
def use_double_precision():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("radial_scale", (0.4, 1.0, 2.7))
@pytest.mark.parametrize("shifted", (False, True))
def test_rn_static_and_kerr_schild_with_rescaled_radial_coordinate(radial_scale, shifted):
    R = np.linspace(2.4, 20.0, 600)
    r = R / radial_scale
    f = 1.0 - 2.0 / R + 0.21348057050543048**2 / R**2
    if shifted:
        # Ingoing Kerr-Schild time: gamma_RR=2-f, beta^R=(1-f)/(2-f).
        gamma_RR = 2.0 - f
        alpha = 1.0 / np.sqrt(gamma_RR)
        beta = (1.0 - f) / gamma_RR / radial_scale
    else:
        gamma_RR = 1.0 / f
        alpha = np.sqrt(f)
        beta = np.zeros_like(r)

    gamma_rr = radial_scale**2 * gamma_RR
    gamma_T = (R / r)**2
    chi = (gamma_rr * gamma_T**2)**(-1.0 / 3.0)
    fields = dict(r=r, alpha=alpha, beta=beta, chi=chi,
                  conformal_grr=chi * gamma_rr, conformal_gt=chi * gamma_T)
    reconstructed_R, g_RR, _ = reconstruct_grr(fields, r[1] - r[0])

    np.testing.assert_allclose(reconstructed_R, R, rtol=1e-14)
    np.testing.assert_allclose(g_RR[2:], 1.0 / f[2:], rtol=2e-11)


def test_nonlinear_isotropic_to_areal_jacobian():
    q_squared = 0.21348057050543048**2
    r = np.linspace(0.6, 20.0, 9701)
    R = r + 1.0 + (1.0 - q_squared) / (4.0 * r)
    f = 1.0 - 2.0 / R + q_squared / R**2
    fields = dict(r=r, alpha=np.sqrt(f), beta=np.zeros_like(r),
                  chi=(r / R)**2, conformal_grr=np.ones_like(r),
                  conformal_gt=np.ones_like(r))
    reconstructed_R, g_RR, _ = reconstruct_grr(fields, r[1] - r[0])

    np.testing.assert_allclose(reconstructed_R, R, rtol=1e-14)
    # Stay away from the zero Jacobian at the isotropic throat.
    exterior = r > 1.0
    np.testing.assert_allclose(g_RR[exterior], 1.0 / f[exterior], rtol=1e-9)


def test_exterior_mask_does_not_connect_across_the_pole():
    r = np.linspace(0.51, 6.0, 500)
    # Schwarzschild Painleve-Gullstrand slicing has a flat spatial metric.
    fields = dict(r=r, alpha=np.ones_like(r), beta=np.sqrt(2.0 / r),
                  chi=np.full_like(r, 0.25), conformal_grr=np.full_like(r, 0.25),
                  conformal_gt=np.full_like(r, 0.25))
    R, g_RR, denominator = reconstruct_grr(fields, r[1] - r[0])

    assert np.all(np.isnan(g_RR[R < 2.0]))
    assert np.all(denominator[R < 2.0] < 0.0)
    np.testing.assert_allclose(g_RR[R > 2.0], 1.0 / (1.0 - 2.0 / R[R > 2.0]),
                               rtol=1e-10)
