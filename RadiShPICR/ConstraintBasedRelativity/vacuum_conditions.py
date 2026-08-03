import jax.numpy as jnp


def total_particle_mass(particles):
    """Total macro-particle mass from the particle getter contract."""

    return jnp.sum(jnp.ones_like(particles.r) * particles.get_mass())


def total_particle_charge(particles):
    """Total macro-particle charge from the particle getter contract."""

    return jnp.sum(jnp.ones_like(particles.r) * particles.get_charge())


def reissner_nordstrom_A(r, mass, charge):
    """Isotropic RN spatial metric coefficient for physical charge ``charge``."""

    charge_radius_squared = charge**2 / (4.0 * jnp.pi)

    return (
        (1.0 + mass / (2.0 * r))**2
        - charge_radius_squared / (4.0 * r**2)
    )


def reissner_nordstrom_lapse(r, mass, charge):
    """Isotropic RN lapse for physical charge ``charge``."""

    charge_radius_squared = charge**2 / (4.0 * jnp.pi)

    numerator = (1.0 - mass / (2.0 * r)) * (1.0 + mass / (2.0 * r))
    numerator = numerator + charge_radius_squared / (4.0 * r**2)

    return numerator / reissner_nordstrom_A(r, mass, charge)


def vacuum_rescale_factors(A_outer, alpha_outer, r_outer, mass, charge):
    """Coordinate rescaling that matches the vacuum solution at the outer cell."""

    charge_radius_squared = charge**2 / (4.0 * jnp.pi)

    linear_coefficient = mass / r_outer - A_outer
    constant_coefficient = (
        mass**2 - charge_radius_squared
    ) / (4.0 * r_outer**2)
    discriminant = linear_coefficient**2 - 4.0 * constant_coefficient
    X_r = 0.5 * (-linear_coefficient + jnp.sqrt(discriminant))

    rescaled_outer_radius = X_r * r_outer
    X_t = alpha_outer / reissner_nordstrom_lapse(
        rescaled_outer_radius,
        mass,
        charge,
    )

    return X_r, X_t


def vacuum_rescale_factors_from_state(
    U_state,
    exterior_mass,
    exterior_charge=0.0,
):
    """Return the spatial and lapse factors for a diagnostic vacuum chart."""

    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state

    return vacuum_rescale_factors(
        A[-1],
        alpha[-1],
        r_grid[-1],
        exterior_mass,
        exterior_charge,
    )


def rescale_to_vacuum_coordinates(
    U_state,
    particles,
    exterior_mass,
    exterior_charge=0.0,
):
    """Map one solver-chart snapshot to the outer static vacuum chart."""

    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state
    mass_density, charge_density, Srr, Sr = source_terms

    X_r, X_t = vacuum_rescale_factors_from_state(
        U_state,
        exterior_mass,
        exterior_charge,
    )


    A = A / X_r
    phi = phi / X_r ** (3.0 / 2.0)
    alpha = alpha / X_t
    beta_over_r = beta_over_r / X_t
    # Er is covariant, so E_r transforms with dr / dr* = 1 / X_r.
    Er = Er / X_r
    rescaled_grid = X_r * r_grid

    Srr = Srr / X_r**2
    Sr = Sr / X_r
    source_terms = (mass_density, charge_density, Srr, Sr)

    rescaled_particles = type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        # r stores r_s = A r and ur stores u_r / A in constrained relativity;
        # both lapse-freezing variables are invariant under r* = X_r r.
        r=particles.r,
        ur=particles.ur,
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )

    rescaled_U_state = (
        A,
        phi,
        alpha,
        Krr,
        beta_over_r,
        Er,
        source_terms,
        rescaled_grid,
    )

    return rescaled_U_state, rescaled_particles, rescaled_grid


def schwarzschild_rescale_factors(U_state, exterior_mass):
    """Diagnostic rescaling factors for an uncharged Schwarzschild exterior."""

    return vacuum_rescale_factors_from_state(
        U_state,
        exterior_mass,
        exterior_charge=0.0,
    )


def rescale_to_schwarzschild_coordinates(U_state, particles, exterior_mass):
    """Return a snapshot in vacuum-normalized isotropic Schwarzschild coordinates."""

    return rescale_to_vacuum_coordinates(
        U_state,
        particles,
        exterior_mass,
        exterior_charge=0.0,
    )
