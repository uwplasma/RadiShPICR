"""Radial coordinate transport current on the Z4C cell-centered grid."""

import jax.numpy as jnp

from RadiShPICR.particles.shape_factors import particle_deposition_stencil
from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume


def compute_radial_current_density(particles, metric, dr_dt, inner_open=False):
    """Deposit ``J_transport^r = alpha * J_Eulerian^r - beta * rho_q``.

    ``dr_dt`` is the coordinate velocity from the same particle/metric RK
    stage. It already includes lapse, shift, and the relativistic Lorentz
    factor; stored ``u_r`` is not this velocity. ``get_charge()`` includes
    macro-particle weight. Radial current has odd origin parity and uses
    the same unrenormalized open-boundary support as charge deposition.

    This direct current moment does not enforce discrete continuity.
    """

    indices, _, odd_weights = particle_deposition_stencil(
        particles, metric, inner_open,
    )
    particle_current = particles.get_charge() * dr_dt
    deposited_current = jnp.zeros_like(metric.r).at[indices].add(
        odd_weights * particle_current[jnp.newaxis, :]
    )

    return deposited_current / proper_radial_shell_volume(metric)
