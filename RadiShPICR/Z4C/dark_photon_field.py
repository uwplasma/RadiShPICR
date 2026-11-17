"""Spherical dark-field equations in the mass eigenbasis.

The stored tuple is ``(E_prime_r, A_prime_r, phi_prime)``. Both radial
components are covariant and odd at the origin; the scalar is even. Magnetic
fields vanish in spherical symmetry. The mass term follows the Proca
convention of the notes' section 3.2, equations (16)-(19), so vacuum modes
obey omega**2 = k**2 + mu**2 and the conserved constraint is
D_i E'^i + mu**2 phi' = sin_chi * rho_q. Appendix A of those notes carries
the opposite mass sign, which is tachyonic and is not used here.
"""

import jax
import jax.numpy as jnp

from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume
from RadiShPICR.Z4C.derivatives import (
    first_derivative, upwind_first_derivative, sixth_derivative,
)
from RadiShPICR.Z4C.energy_momentum_tensor import dark_energy_momentum_tensor


def dark_photon_time_derivatives(
    metric, metric_rhs, dark_fields, charge_density, current_density,
    mu, sin_chi=0.0, epsilon_0=1.0, *, dark_ko=False,
):
    """Return the dark-field RHS, converted to covariant storage.

    ``current_density`` is coordinate transport current, as for ordinary
    Maxwell evolution, so ``alpha * J^r`` equals ``current_density + beta *
    charge_density`` and enters Ampere's law with a minus sign. The actual
    stage metric derivative is required for lowering the electric-field RHS,
    even on a prescribed background. Shift transport uses the fourth-order
    upwind stencil for velocity -beta. ``dark_ko`` uses metric.nu for
    sixth-derivative dissipation of E'^r, A'^r and phi' at this RK stage.
    This runtime JAX boolean defaults to False.
    """

    E_prime_r, A_prime_r, phi_prime = dark_fields
    alpha = metric.alpha
    beta = metric.beta
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    chi = metric.chi
    a = grr / chi
    K = metric.Kh + 2.0 * metric.theta
    E_prime_up = E_prime_r / a
    A_prime_up = A_prime_r / a

    # Differentiate the conformal factors separately: the physical metric
    # diverges at a puncture and is not smooth across the parity stencil.
    dgrr_dr = first_derivative(grr, metric.dr, parity=1)
    dgt_dr = first_derivative(gt, metric.dr, parity=1)
    dchi_dr = first_derivative(chi, metric.dr, parity=1)
    dln_a_dr = dgrr_dr / grr - dchi_dr / chi
    radial_divergence_factor = (
        2.0 / metric.r + 0.5 * dgrr_dr / grr + dgt_dr / gt
        - 1.5 * dchi_dr / chi
    )
    dbeta_dr = first_derivative(beta, metric.dr, parity=-1)
    # Only transport derivatives are biased; gradients and divergences below
    # retain their centered discretization and the original geometric terms.

    dE_prime_up_dr = upwind_first_derivative(E_prime_up, metric.dr, beta, -1)
    dA_prime_up_dr = upwind_first_derivative(A_prime_up, metric.dr, beta, -1)
    dphi_prime_dr = upwind_first_derivative(phi_prime, metric.dr, beta, 1)
    # use the upwind stencil for the transport derivative of each field, which
    # is the only term that depends on the shift. The radial fields are odd
    # and the scalar is even.

    dalpha_phi_dr = first_derivative(alpha * phi_prime, metric.dr, parity=1)

    # Separate the regular odd field from the 2/r factor. Differentiating
    # r**2 times a field and then dividing by r**2 loses order near r=0.
    alpha_A_prime_up = alpha * A_prime_up
    divergence_alpha_A = (
        first_derivative(alpha_A_prime_up, metric.dr, parity=-1)
        + radial_divergence_factor * alpha_A_prime_up
    )

    dE_prime_up_dt = (
        beta * dE_prime_up_dr - E_prime_up * dbeta_dr
        + alpha * K * E_prime_up
        - sin_chi * (current_density + beta * charge_density) / epsilon_0
        + alpha * mu**2 * A_prime_up
    )
    # Expand the covector Lie derivative through A'^r to retain its transport
    # stencil while including the radial-metric and shift stretching terms.
    dA_prime_r_dt = (
        a * beta * dA_prime_up_dr
        + (beta * dln_a_dr + dbeta_dr) * A_prime_r
        - alpha * E_prime_r - dalpha_phi_dr
    )
    dphi_prime_dt = (
        beta * dphi_prime_dr - divergence_alpha_A
        + alpha * K * phi_prime
    )

    # Lower the electric-field RHS using the actual RK metric RHS, rather
    # than replacing metric evolution with a constraint or gauge identity.
    da_dt = (
        metric_rhs.conformal_grr / metric.chi
        - a * metric_rhs.chi / metric.chi
    )
    dE_prime_r_dt = a * dE_prime_up_dt + da_dt * E_prime_up

    # Dissipate raised radial fields rather than puncture-singular covectors.
    # Lower the artificial RHS without altering the actual metric-rate term.
    ko_E, ko_A, ko_phi = jax.lax.cond(
        dark_ko,
        lambda _: (
            a * metric.nu * metric.dr**5 / 64 * sixth_derivative(E_prime_up, metric.dr, -1),
            a * metric.nu * metric.dr**5 / 64 * sixth_derivative(A_prime_up, metric.dr, -1),
            metric.nu * metric.dr**5 / 64 * sixth_derivative(phi_prime, metric.dr, 1),
        ),
        lambda _: (jnp.zeros_like(E_prime_r), jnp.zeros_like(A_prime_r),
                   jnp.zeros_like(phi_prime)),
        operand=None,
    )
    return dE_prime_r_dt + ko_E, dA_prime_r_dt + ko_A, dphi_prime_dt + ko_phi


def dark_gauss_residual(
    metric, dark_fields, charge_density, mu, sin_chi=0.0, epsilon_0=1.0,
):
    """Return epsilon_0 * (D_i E'^i + mu**2 phi') - sin_chi * rho_q.

    This combination is the Proca constraint that the evolution preserves:
    taking the divergence of the ``E'`` equation and using the ``phi'``
    equation gives d_t (D_i E'^i + mu**2 phi') = 0 in flat-space vacuum.
    """

    E_prime_r, _, phi_prime = dark_fields
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    chi = metric.chi
    a = grr / chi
    E_prime_up = E_prime_r / a
    # Use the same conformal-factor expansion as in the field evolution.
    dgrr_dr = first_derivative(grr, metric.dr, parity=1)
    dgt_dr = first_derivative(gt, metric.dr, parity=1)
    dchi_dr = first_derivative(chi, metric.dr, parity=1)
    radial_divergence_factor = (
        2.0 / metric.r + 0.5 * dgrr_dr / grr + dgt_dr / gt
        - 1.5 * dchi_dr / chi
    )
    divergence_E = (
        first_derivative(E_prime_up, metric.dr, parity=-1)
        + radial_divergence_factor * E_prime_up
    )

    return epsilon_0 * (divergence_E + mu**2 * phi_prime) - sin_chi * charge_density


def dark_field_energy(metric, dark_fields, mu, epsilon_0=1.0):
    """Integrate the Proca energy associated with coordinate time translation.

    When the background is stationary, t^mu = alpha*n^mu + beta^mu is a
    Killing vector and the density is alpha*rho - beta^i*S_i. Without energy
    exchange with particles, this energy is conserved up to boundary flux.
    On an evolving background the same integral need not be conserved.
    """

    matter = dark_energy_momentum_tensor(metric, dark_fields, mu, epsilon_0)
    a = metric.conformal_grr / metric.chi

    # Sr stores S^r, so lower it before contracting with the radial shift.
    energy_density = metric.alpha * matter.rho - metric.beta * a * matter.Sr

    return jnp.sum(energy_density * proper_radial_shell_volume(metric))


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
