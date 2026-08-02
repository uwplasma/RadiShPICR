import jax
import jax.numpy as jnp

def radial_shell_volume(A, radial_coordinate, dr):
    """Proper volume of the spherical cell centered at ``radial_coordinate``."""

    coordinate_volume = 4.0 * jnp.pi * radial_coordinate**2 * dr
    # use the coordinate volume based on the papers

    coordinate_volume = jnp.where(radial_coordinate == 0, 4.0 * jnp.pi * (dr/2)**2 * dr, coordinate_volume)
    # parity average across the origin for the innermost cell, which is centered at r=0 produces a flat 
    # volume of the cell with radius dr/2, so the volume is 4*pi*(dr/2)^2 * dr

    return A**3 * coordinate_volume


def nearest_interior_index(radial_positions, grid):
    """Map particles to the nearest interior grid point.

    The two edge cells are reserved as vacuum boundary cells, so matter is only
    deposited on indices `1` through `N-2`.
    """

    floating_index = (radial_positions - grid.r_full[0]) / grid.dr
    nearest = jnp.rint(floating_index).astype(jnp.int32)
    return jnp.clip(nearest, 1, grid.r_full.shape[0] - 2)
