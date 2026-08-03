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

    inner_radius = jnp.maximum(radial_coordinate - 0.5 * dr, 0.0)
    outer_radius = radial_coordinate + 0.5 * dr
    coordinate_volume = (
        (4.0 * jnp.pi / 3.0) * (outer_radius**3 - inner_radius**3)
    )
    # Integrate the spherical coordinate volume between cell faces. The metric
    # factor remains centered at the grid point, as in the existing deposition.

    return A**3 * coordinate_volume
