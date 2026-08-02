import jax
import jax.numpy as jnp
from jax import lax


def angular_lorentz_term(uphi, A, radial_coordinate):
    """Return the angular contribution to W without evaluating 0 / 0."""

    uphi = jnp.asarray(uphi)
    A = jnp.broadcast_to(A, uphi.shape)
    radial_coordinate = jnp.broadcast_to(radial_coordinate, uphi.shape)

    def angular_term_at_particle(uphi_particle, A_particle, r_particle):
        return lax.cond(
            uphi_particle == 0.0,
            lambda: jnp.zeros_like(uphi_particle),
            lambda: uphi_particle**2 / (A_particle**2 * r_particle**2),
        )

    angular_term = jax.vmap(angular_term_at_particle)(
        uphi.reshape(-1),
        A.reshape(-1),
        radial_coordinate.reshape(-1),
    )

    return angular_term.reshape(uphi.shape)


def radial_shell_volume(A, radial_coordinate, dr):
    """Proper volume of the spherical cell centered at ``radial_coordinate``."""

    coordinate_volume = 4.0 * jnp.pi * radial_coordinate**2 * dr
    # use the coordinate volume based on the papers

    coordinate_volume = jnp.where(radial_coordinate == 0, 4.0 * jnp.pi * (dr/2)**2 * dr, coordinate_volume)
    # parity average across the origin for the innermost cell, which is centered at r=0 produces a flat 
    # volume of the cell with radius dr/2, so the volume is 4*pi*(dr/2)^2 * dr

    return A**3 * coordinate_volume
