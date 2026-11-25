"""Black-hole puncture initial data.

The time-symmetric wormhole slice covers the Schwarzschild and Reissner-Nordstrom
cases.  The conformal factor is

    psi**2 = (1 + M / 2r)**2 - q**2 / (4 r**2),    q**2 = Q**2 / (4 pi eps_0)

with the precollapsed lapse ``alpha = psi**-2`` and ``chi = psi**-4``.  Setting
``charge = 0`` reduces this to ``psi = 1 + M / 2r``, the Schwarzschild puncture.
"""

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

__all__ = [
    "geometric_charge_squared",
    "puncture_electric_field",
    "puncture_metric",
    "schwarzschild_trumpet_metric",
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


def schwarzschild_trumpet_metric(r, mass, kappa=1.0, eta=2.0, nu=0.02):
    """Stationary 1+log Schwarzschild trumpet on an isotropic cell-centered grid.

    Evaluate the regular implicit solution of Brügmann (arXiv:0904.4418)
    once on the host. R and r_iso below are radii in units of the mass.
    """
    r_iso = np.asarray(r, dtype=float) / mass
    alpha_c = np.sqrt(10.0) - 3.0
    R_c = 1.0 / (4.0 * alpha_c)
    C = 0.5 * R_c**3 * np.exp(-alpha_c)
    slope_c = 8.0 * (np.sqrt(10.0 + 3.0 * np.sqrt(10.0)) - 2.0) / (
        16.0 + 5.0 * np.sqrt(10.0)
    )

    def lapse(R):
        # The two roots coincide at the critical point. Its positive tangent
        # selects the smooth branch where the implicit equation loses precision.
        if abs(R - R_c) < 1.0e-6:
            return alpha_c + slope_c * (R - R_c)

        def equation(alpha):
            return (alpha - 1.0) * (alpha + 1.0) + 2.0/R - C*np.exp(alpha)/R**4

        bracket = (0.0, alpha_c) if R < R_c else (alpha_c, 1.0)
        return brentq(equation, *bracket, xtol=1e-15)

    # Isotropy gives d(log R)/d(log r) = alpha. At this large starting
    # radius the O(r^-4) relative trumpet correction to the Schwarzschild
    # isotropic relation is below double precision.
    r_start = max(1.0e5, 2.0 * r_iso[-1])
    R_start = r_start * (1.0 + 0.5/r_start)**2
    log_r = np.log(r_iso[::-1])
    radial_map = solve_ivp(
        lambda log_r, log_R: [lapse(np.exp(log_R[0]))],
        (np.log(r_start), log_r[-1]), [np.log(R_start)], t_eval=log_r,
        method="DOP853", rtol=2.0e-13, atol=2.0e-14, max_step=0.25,
    )
    if not radial_map.success:
        raise RuntimeError(radial_map.message)
    R = np.exp(radial_map.y[0, ::-1])
    alpha = np.array([lapse(radius) for radius in R])

    # v = beta^R / alpha on the future black-hole branch. Evaluate the
    # integration constant directly since alpha^2 - 1 + 2/R cancels far out.
    v = np.sqrt(C) * np.exp(0.5 * alpha) / R**2
    dalpha_dR = np.full_like(R, slope_c)
    np.divide(2.0/R**2 - 4.0*v**2/R, 2.0*alpha - v**2, out=dalpha_dR,
              where=np.abs(R - R_c) >= 1.0e-6)
    K = 0.5 * v * dalpha_dR / mass
    At = v / (mass * R) - K / 3.0

    r = jnp.asarray(r)
    zeros, ones = jnp.zeros_like(r), jnp.ones_like(r)
    return Z4C_Metric(
        alpha=jnp.asarray(alpha), beta=jnp.asarray(r_iso * v / R),
        chi=jnp.asarray((r_iso / R)**2), conformal_grr=ones, conformal_gt=ones,
        Kh=jnp.asarray(K), Arr=jnp.asarray(-2.0 * At), At=jnp.asarray(At),
        theta=zeros, Gamma=zeros, r=r, dr=r[1] - r[0],
        kappa=jnp.asarray(kappa), eta=jnp.asarray(eta), nu=jnp.asarray(nu),
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
