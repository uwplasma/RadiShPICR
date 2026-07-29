import jax.numpy as jnp
from jax import lax

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.utils import pad_value, radial_shell_volume
from RadiShPICR.particles.particle_shapes import radial_shape_stencil


def _safe_radius(r, dr):
    return jnp.maximum(r, 0.5 * dr)


def dr_A(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    return 2.0 * phi * jnp.sqrt(A)


def dr_sqrt_phi(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    A_for_denominators = pad_value(A)
    # Er is the covariant radial field, so E_i E^i = Er^2 / A^2.
    mass_energy_density = rho + 0.5 * Er**2 / A_for_denominators**2

    if dr is None:
        safe_r = jnp.where(r == 0.0, 1.0, r)
    else:
        safe_r = _safe_radius(r, dr)

    interior_term = -2.0 * jnp.pi * jnp.sqrt(A) ** 5 * mass_energy_density - 2.0 * phi / safe_r
    center_term = (
        -2.0 * jnp.pi * jnp.sqrt(A) ** 5 * mass_energy_density / 3.0
    )

    return jnp.where(r == 0.0, center_term, interior_term)


def dr_alpha(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    A_for_denominators = pad_value(A)
    # A radial electric field contributes tension along the field direction.
    total_Srr = Srr - 0.5 * Er**2

    first_term = 4.0 * jnp.pi * alpha * total_Srr * r * A
    second_term = -2.0 * alpha * phi * jnp.sqrt(A)
    third_term = -2.0 * alpha * phi**2 * r
    denominator = A_for_denominators * (
        1.0 + 2.0 * r * phi / jnp.sqrt(A_for_denominators)
    )

    return (first_term + second_term + third_term) / denominator


def Krr_from_state(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    A_for_denominators = pad_value(A)

    return 4.0 * jnp.pi * r * Sr / (
        1.0 + 2.0 * r * phi / jnp.sqrt(A_for_denominators)
    )


def dr_beta_over_r(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    safe_r = jnp.where(r == 0.0, 1.0, r) if dr is None else _safe_radius(r, dr)
    return jnp.where(r == 0.0, 0.0, alpha * Krr_from_state(U_state) / safe_r)


def beta_over_r_from_integral(alpha, Krr, r, dr):
    """Shift condition solved as a tail integral after the radial Heun solve."""

    safe_r = jnp.where(r == 0.0, 1.0, r)
    integrand = jnp.where(r == 0.0, 0.0, alpha * Krr / safe_r)

    trapezoid_segments = 0.5 * (integrand[:-1] + integrand[1:]) * (r[1:] - r[:-1])
    tail_integral = jnp.concatenate(
        (
            jnp.cumsum(trapezoid_segments[::-1])[::-1],
            jnp.zeros_like(integrand[-1:]),
        )
    )

    return -tail_integral


def dr_Er(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    A_for_denominators = pad_value(A)

    safe_r = jnp.where(r == 0.0, 1.0, r) if dr is None else _safe_radius(r, dr)
    interior_term = (
        A**2 * charge_density
        - 2.0 * Er / safe_r
        - 2.0 * phi * Er / jnp.sqrt(A_for_denominators)
    )
    center_term = A**2 * charge_density / 3.0

    return jnp.where(r == 0.0, center_term, interior_term)


def _source_terms_at_point(
    particles,
    A_at_point,
    radial_coordinate,
    grid,
    particle_stencil=None,
):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    dr = grid.dr

    if particle_stencil is None:
        particle_stencil = radial_shape_stencil(
            r_particle,
            grid,
            shape_mode=particles.get_shape(),
        )

    indices, stencil_weights = particle_stencil
    floating_index = (radial_coordinate - grid.r_full[0]) / dr
    grid_index = jnp.rint(floating_index).astype(indices.dtype)
    weights = jnp.sum(
        jnp.where(indices == grid_index, stencil_weights, 0.0),
        axis=0,
    )

    safe_r = jnp.maximum(
        jnp.asarray(radial_coordinate, dtype=r_particle.dtype),
        0.5 * dr,
    )
    A_for_denominators = pad_value(A_at_point)
    lorentz_factor = jnp.sqrt(
        1.0
        + ur**2 / A_for_denominators**2
        + uphi**2 / (A_for_denominators**2 * safe_r**2)
    )
    cell_volume = radial_shell_volume(
        A_for_denominators,
        radial_coordinate,
        dr,
    )

    weighted_mass = particles.get_mass() * weights
    mass_density = jnp.sum(weighted_mass * lorentz_factor / cell_volume)
    charge_density = jnp.sum(particles.get_charge() * weights / cell_volume)
    Srr = jnp.sum(
        weighted_mass * ur**2 / (cell_volume * lorentz_factor)
    )
    Sr = jnp.sum(weighted_mass * ur / cell_volume)

    return mass_density, charge_density, Srr, Sr


def heuns_method(U_state, dr, particles, grid, particle_stencil=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state

    dA_dr = dr_A(U_state)
    dphi_dr = dr_sqrt_phi(U_state, dr)
    dalpha_dr = dr_alpha(U_state, dr)
    dE_dr = dr_Er(U_state, dr)

    r_predictor = r + dr
    A_predictor = A + dA_dr * dr
    phi_predictor = phi + dphi_dr * dr
    alpha_predictor = alpha + dalpha_dr * dr
    Er_predictor = Er + dE_dr * dr
    source_terms_predictor = _source_terms_at_point(
        particles,
        A_predictor,
        r_predictor,
        grid,
        particle_stencil,
    )
    Krr_predictor = Krr_from_state(
        (A_predictor, phi_predictor, alpha_predictor, Krr, beta_over_r, Er_predictor, source_terms_predictor, r_predictor)
    )

    predictor_state = (
        A_predictor,
        phi_predictor,
        alpha_predictor,
        Krr_predictor,
        beta_over_r,
        Er_predictor,
        source_terms_predictor,
        r_predictor,
    )

    dA_dr_predictor = dr_A(predictor_state)
    dphi_dr_predictor = dr_sqrt_phi(predictor_state, dr)
    dalpha_dr_predictor = dr_alpha(predictor_state, dr)
    dE_dr_predictor = dr_Er(predictor_state, dr)

    A_corrected = A + 0.5 * (dA_dr + dA_dr_predictor) * dr
    phi_corrected = phi + 0.5 * (dphi_dr + dphi_dr_predictor) * dr
    alpha_corrected = alpha + 0.5 * (dalpha_dr + dalpha_dr_predictor) * dr
    Er_corrected = Er + 0.5 * (dE_dr + dE_dr_predictor) * dr
    source_terms_corrected = _source_terms_at_point(
        particles,
        A_corrected,
        r_predictor,
        grid,
        particle_stencil,
    )
    Krr_corrected = Krr_from_state(
        (
            A_corrected,
            phi_corrected,
            alpha_corrected,
            Krr,
            beta_over_r,
            Er_corrected,
            source_terms_corrected,
            r_predictor,
        )
    )

    return (
        A_corrected,
        phi_corrected,
        alpha_corrected,
        Krr_corrected,
        beta_over_r,
        Er_corrected,
        source_terms_corrected,
        r_predictor,
    )


def calculate_metric(particles, r_grid, dr):
    r_grid = jnp.asarray(r_grid)
    dr = jnp.asarray(dr, dtype=r_grid.dtype)
    grid = RadialGrid(
        r_full=r_grid,
        r_interior=r_grid[1:-1],
        # The physical endpoint nodes remain reserved for vacuum boundary data.
        dr=dr,
        r_max=r_grid[-1],
    )
    particle_stencil = radial_shape_stencil(
        particles.r,
        grid,
        shape_mode=particles.get_shape(),
    )

    initial_A = jnp.asarray(1.0, dtype=r_grid.dtype)
    initial_phi = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_alpha = jnp.asarray(1.0, dtype=r_grid.dtype)
    initial_Krr = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_beta_over_r = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_Er = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_r = r_grid[0]
    initial_source_terms = _source_terms_at_point(
        particles,
        initial_A,
        initial_r,
        grid,
        particle_stencil,
    )

    state = (
        initial_A,
        initial_phi,
        initial_alpha,
        initial_Krr,
        initial_beta_over_r,
        initial_Er,
        initial_source_terms,
        initial_r,
    )

    def radial_step(state, local_dr):
        state = heuns_method(
            state,
            local_dr,
            particles,
            grid,
            particle_stencil,
        )
        A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = state
        mass_density, charge_density, Srr, Sr = source_terms

        values = (
            A,
            phi,
            alpha,
            Krr,
            beta_over_r,
            Er,
            mass_density,
            charge_density,
            Srr,
            Sr,
            r,
        )

        return state, values

    local_dr_values = r_grid[1:] - r_grid[:-1]
    _, scanned_values = lax.scan(radial_step, state, local_dr_values)
    (
        scanned_A,
        scanned_phi,
        scanned_alpha,
        scanned_Krr,
        scanned_beta_over_r,
        scanned_Er,
        scanned_mass_density,
        scanned_charge_density,
        scanned_Srr,
        scanned_Sr,
        scanned_r,
    ) = scanned_values

    (
        initial_A,
        initial_phi,
        initial_alpha,
        initial_Krr,
        initial_beta_over_r,
        initial_Er,
        initial_source_terms,
        initial_r,
    ) = state
    initial_mass_density, initial_charge_density, initial_Srr, initial_Sr = (
        initial_source_terms
    )

    A_values = jnp.concatenate((initial_A[jnp.newaxis], scanned_A))
    phi_values = jnp.concatenate((initial_phi[jnp.newaxis], scanned_phi))
    alpha_values = jnp.concatenate((initial_alpha[jnp.newaxis], scanned_alpha))
    Krr_values = jnp.concatenate((initial_Krr[jnp.newaxis], scanned_Krr))
    beta_over_r_values = jnp.concatenate(
        (initial_beta_over_r[jnp.newaxis], scanned_beta_over_r)
    )
    Er_values = jnp.concatenate((initial_Er[jnp.newaxis], scanned_Er))
    mass_density_values = jnp.concatenate(
        (initial_mass_density[jnp.newaxis], scanned_mass_density)
    )
    charge_density_values = jnp.concatenate(
        (initial_charge_density[jnp.newaxis], scanned_charge_density)
    )
    Srr_values = jnp.concatenate((initial_Srr[jnp.newaxis], scanned_Srr))
    Sr_values = jnp.concatenate((initial_Sr[jnp.newaxis], scanned_Sr))

    source_terms = (
        mass_density_values,
        charge_density_values,
        Srr_values,
        Sr_values,
    )
    r_values = jnp.concatenate((initial_r[jnp.newaxis], scanned_r))
    beta_over_r_values = beta_over_r_from_integral(
        alpha_values,
        Krr_values,
        r_values,
        dr,
    )

    U_state = (
        A_values,
        phi_values,
        alpha_values,
        Krr_values,
        beta_over_r_values,
        Er_values,
        source_terms,
        r_values,
    )

    return U_state
