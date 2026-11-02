"""Spherical dark-field equations in the mass eigenbasis.

The stored tuple is ``(E_prime_r, A_prime_r, phi_prime)``. Both radial
components are covariant and odd at the origin; the scalar is even. Magnetic
fields vanish in spherical symmetry. The mass signs follow the supplied
equations literally: vacuum modes obey omega**2 = k**2 - mu**2. These signs
do not give the positive-mass Proca dispersion associated with the supplied
Lagrangian, so its positive energy diagnostic is not asserted conserved.
"""

import jax.numpy as jnp

from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.energy_momentum_tensor import dark_energy_momentum_tensor


def dark_photon_time_derivatives(
    metric, metric_rhs, dark_fields, charge_density, current_density,
    mu, sin_chi=0.0, epsilon_0=1.0,
):
    """Return the literal dark-field RHS, converted to covariant storage.

    ``current_density`` is coordinate transport current, as for ordinary
    Maxwell evolution. Thus alpha times the screenshot current is
    ``-(current_density + beta * charge_density) / epsilon_0``. The actual
    stage metric derivative is required even on a prescribed background.
    """

    E_prime_r, A_prime_r, phi_prime = dark_fields
    alpha = metric.alpha
    beta = metric.beta
    a = metric.conformal_grr / metric.chi
    b = metric.conformal_gt / metric.chi
    K = metric.Kh + 2.0 * metric.theta
    E_prime_up = E_prime_r / a
    A_prime_up = A_prime_r / a

    da_dr = first_derivative(a, metric.dr, parity=1)
    db_dr = first_derivative(b, metric.dr, parity=1)
    dbeta_dr = first_derivative(beta, metric.dr, parity=-1)
    dE_prime_up_dr = first_derivative(E_prime_up, metric.dr, parity=-1)
    dA_prime_up_dr = first_derivative(A_prime_up, metric.dr, parity=-1)
    dphi_prime_dr = first_derivative(phi_prime, metric.dr, parity=1)
    dalpha_phi_dr = first_derivative(alpha * phi_prime, metric.dr, parity=1)

    # Separate the regular odd field from the 2/r factor. Differentiating
    # r**2 times a field and then dividing by r**2 loses order near r=0.
    alpha_A_prime_up = alpha * A_prime_up
    divergence_alpha_A = (
        first_derivative(alpha_A_prime_up, metric.dr, parity=-1)
        + (2.0 / metric.r + 0.5 * da_dr / a + db_dr / b)
        * alpha_A_prime_up
    )

    dE_prime_up_dt = (
        beta * dE_prime_up_dr - E_prime_up * dbeta_dr
        + alpha * K * E_prime_up
        - sin_chi * (current_density + beta * charge_density) / epsilon_0
        - alpha * mu**2 * A_prime_up
    )
    dA_prime_up_dt = (
        beta * dA_prime_up_dr - A_prime_up * dbeta_dr
        - alpha * E_prime_up - dalpha_phi_dr / a
    )
    dphi_prime_dt = (
        beta * dphi_prime_dr - divergence_alpha_A
        + alpha * K * phi_prime
    )

    # Lower the evolving vector using the actual RK metric RHS, rather
    # than replacing metric evolution with a constraint or gauge identity.
    da_dt = (
        metric_rhs.conformal_grr / metric.chi
        - a * metric_rhs.chi / metric.chi
    )
    dE_prime_r_dt = a * dE_prime_up_dt + da_dt * E_prime_up
    dA_prime_r_dt = a * dA_prime_up_dt + da_dt * A_prime_up

    return dE_prime_r_dt, dA_prime_r_dt, dphi_prime_dt


def dark_gauss_residual(
    metric, dark_fields, charge_density, mu, sin_chi=0.0, epsilon_0=1.0,
):
    """Return epsilon_0 * (D_i E'^i - mu**2 phi') - sin_chi * rho_q."""

    E_prime_r, _, phi_prime = dark_fields
    a = metric.conformal_grr / metric.chi
    b = metric.conformal_gt / metric.chi
    E_prime_up = E_prime_r / a
    da_dr = first_derivative(a, metric.dr, parity=1)
    db_dr = first_derivative(b, metric.dr, parity=1)
    divergence_E = (
        first_derivative(E_prime_up, metric.dr, parity=-1)
        + (2.0 / metric.r + 0.5 * da_dr / a + db_dr / b) * E_prime_up
    )

    return epsilon_0 * (divergence_E - mu**2 * phi_prime) - sin_chi * charge_density


def dark_field_energy(metric, dark_fields, mu, epsilon_0=1.0):
    """Integrate the supplied Lagrangian's positive dark energy density.

    The literal evolution signs differ from that Lagrangian's mass signs;
    this diagnostic must not be interpreted as a conserved vacuum energy.
    """

    matter = dark_energy_momentum_tensor(metric, dark_fields, mu, epsilon_0)

    return jnp.sum(matter.rho * proper_radial_shell_volume(metric))


def dark_field_outer_boundary(dark_fields, metric, time=0.0):
    """Zero inward principal characteristics at the last radial cell.

    This local condition is approximate for spherical massive waves. It
    leaves outgoing and zero-speed characteristics unchanged. ``time`` is
    accepted to share the custom RK boundary callback interface.
    """

    E_prime_r, A_prime_r, phi_prime = dark_fields
    sqrt_a = jnp.sqrt(metric.conformal_grr[-1] / metric.chi[-1])
    alpha = metric.alpha[-1]
    beta = metric.beta[-1]
    w_plus = phi_prime[-1] + A_prime_r[-1] / sqrt_a
    w_minus = phi_prime[-1] - A_prime_r[-1] / sqrt_a
    w_plus = jnp.where(-beta + alpha / sqrt_a < 0.0, 0.0, w_plus)
    w_minus = jnp.where(-beta - alpha / sqrt_a < 0.0, 0.0, w_minus)

    E_prime_r = E_prime_r.at[-1].set(
        jnp.where(-beta < 0.0, 0.0, E_prime_r[-1])
    )
    A_prime_r = A_prime_r.at[-1].set(0.5 * sqrt_a * (w_plus - w_minus))
    phi_prime = phi_prime.at[-1].set(0.5 * (w_plus + w_minus))

    return E_prime_r, A_prime_r, phi_prime
