"""Hydrogenic radial Proca initial data on a Schwarzschild trumpet."""

import jax.numpy as jnp

from RadiShPICR.Z4C.dark_photon_field import dark_field_energy
from RadiShPICR.Z4C.derivatives import first_derivative


def initial_cloud(metric, bohr, mu, cloud_energy):
    """Return (E'_r, A'_r, φ') normalized to the requested Killing energy."""
    gt, chi = metric.conformal_gt, metric.chi
    radius_factor = jnp.sqrt(gt / chi)
    R = metric.r * radius_factor
    dgt_dr = first_derivative(gt, metric.dr, parity=1)
    dchi_dr = first_derivative(chi, metric.dr, parity=1)
    dR_dr = radius_factor * (1.0 + 0.5 * metric.r * (dgt_dr / gt - dchi_dr / chi))

    # A'_R dR = A'_r dr, including the sign of the radial Jacobian.
    A_r = (R / bohr) * jnp.exp(-R / (2.0 * bohr)) * dR_dr
    zeros = jnp.zeros_like(R)
    unit_energy = dark_field_energy(metric, (zeros, A_r, zeros), mu)
    amplitude = jnp.sqrt(cloud_energy / unit_energy)
    return zeros, amplitude * A_r, zeros
