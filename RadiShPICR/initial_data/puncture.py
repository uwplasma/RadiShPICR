"""Time-symmetric puncture black-hole initial data.

One wormhole-topology slice covering the Schwarzschild and Reissner-Nordstrom
cases.  The conformal factor is

    psi**2 = (1 + M / 2r)**2 - q**2 / (4 r**2),    q**2 = Q**2 / (4 pi eps_0)

with the precollapsed lapse ``alpha = psi**-2`` and ``chi = psi**-4``.  Setting
``charge = 0`` reduces this to ``psi = 1 + M / 2r``, the Schwarzschild puncture.
"""

import jax.numpy as jnp

from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

__all__ = [
    "geometric_charge_squared",
    "puncture_electric_field",
    "puncture_metric",
]


def geometric_charge_squared(charge, epsilon_0=1.0):
    """Return ``q**2 = Q**2 / (4 pi eps_0)``, the charge in geometric units."""

    return charge**2 / (4.0 * jnp.pi * epsilon_0)


def puncture_metric(r, mass, charge=0.0, epsilon_0=1.0, kappa=1.0, eta=2.0, nu=0.02):
    """Exact time-symmetric puncture data, with the OS precollapsed lapse choice.

    Args:
        r: Cell-centered radial grid.
        mass: Gravitational mass ``M``.
        charge: Physical charge ``Q``; zero gives the Schwarzschild puncture.
        epsilon_0: Gauss-law constant, one in geometric units.
        kappa, eta, nu: Z4c damping parameters.

    The data are subextremal only for ``M > |Q| / sqrt(4 pi eps_0)``; outside
    that range ``psi**2`` goes negative near the puncture.
    """

    r = jnp.asarray(r)
    zeros, ones = jnp.zeros_like(r), jnp.ones_like(r)
    psi_squared = (
        (1.0 + mass / (2.0 * r)) ** 2
        - geometric_charge_squared(charge, epsilon_0) / (4.0 * r**2)
    )

    return Z4C_Metric(
        alpha=psi_squared**-1, beta=zeros, chi=psi_squared**-2,
        conformal_grr=ones, conformal_gt=ones,
        Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        r=r, dr=r[1] - r[0], kappa=jnp.asarray(kappa),
        eta=jnp.asarray(eta), nu=jnp.asarray(nu),
    )


def puncture_electric_field(metric, charge, epsilon_0=1.0):
    """Covariant ``E_r`` of a puncture carrying constant flux ``charge``.

    The charged puncture evolves no electric field of its own: the conserved
    quantity is the flux, so ``E_r`` is reconstructed from the current metric
    rather than integrated.  Both the runner and anything reading its output
    rebuild it the same way from here.
    """

    grr = metric.conformal_grr / metric.chi
    gT = metric.conformal_gt / metric.chi

    return (
        charge
        * jnp.sqrt(metric.chi * grr)
        / (4.0 * jnp.pi * epsilon_0 * metric.r**2 * gT)
    )
