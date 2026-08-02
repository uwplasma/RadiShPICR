import jax
import jax.numpy as jnp


@jax.jit
def safe_radius(radius, epsilon):
    """Keep radial denominators away from the origin."""

    epsilon_value = jnp.asarray(epsilon, dtype=radius.dtype)
    # convert to array to avoid issues with broadcasting when radius is an array
    return jnp.maximum(radius, epsilon_value)


@jax.jit
def pad_value(value, padding=1.0e-15):
    """Add a small offset before using metric values in denominators."""

    padding_value = jnp.asarray(padding, dtype=value.dtype)
    return value + padding_value


def radial_shell_volume(A, radial_coordinate, dr):
    """Proper volume of the spherical cell centered at ``radial_coordinate``."""

    coordinate_volume = 4.0 * jnp.pi * radial_coordinate**2 * dr
    # use the coordinate volume based on the papers

    return A**3 * coordinate_volume


def nearest_interior_index(radial_positions, grid):
    """Map particles to the nearest interior grid point.

    The two edge cells are reserved as vacuum boundary cells, so matter is only
    deposited on indices `1` through `N-2`.
    """

    floating_index = (radial_positions - grid.r_full[0]) / grid.dr
    nearest = jnp.rint(floating_index).astype(jnp.int32)
    return jnp.clip(nearest, 1, grid.r_full.shape[0] - 2)
