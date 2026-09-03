import jax.numpy as jnp


def reflecting_particle_boundary(particles):
    """Reflect particles that crossed the areal-radius origin."""

    crossed_origin = particles.r < 0.0
    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=jnp.abs(particles.r),
        ur=jnp.where(crossed_origin, -particles.ur, particles.ur),
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )


def deleting_particle_boundary(particles):
    """Delete particles that crossed the areal-radius origin.

    Deleted particles remain in the fixed-size JAX arrays with zero weight and
    inert phase-space coordinates, so they no longer deposit matter.
    """

    crossed_origin = particles.r < 0.0
    weight = jnp.where(crossed_origin, 0.0, particles.weight)
    inactive = weight == 0.0

    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=weight,
        r=jnp.where(inactive, 0.0, particles.r),
        ur=jnp.where(inactive, 0.0, particles.ur),
        phi=jnp.where(inactive, 0.0, particles.phi),
        uphi=jnp.where(inactive, 0.0, particles.uphi),
        shape_mode=particles.shape_mode,
    )
