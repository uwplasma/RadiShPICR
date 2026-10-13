import jax
import jax.numpy as jnp
from jax import lax

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.vacuum_conditions import (
    total_particle_charge,
    total_particle_mass,
    vacuum_rescale_factors,
)
from RadiShPICR.ConstraintBasedRelativity.particle_sources import (
    deposit_source_numerators,
    particle_coordinate_radius,
    source_terms_at_point as _source_terms_at_point,
    source_values_at_point,
)


def dr_A(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    return 2.0 * phi * jnp.sqrt(A)


def dr_sqrt_phi(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    # Er is the covariant radial field, so E_i E^i = Er^2 / A^2.
    mass_energy_density = rho + 0.5 * Er**2 / A**2

    A, phi, mass_energy_density, r = jnp.broadcast_arrays(
        A,
        phi,
        mass_energy_density,
        r,
    )
    output_shape = r.shape

    def derivative_at_point(A_point, phi_point, rho_point, r_point):
        return lax.cond(
            r_point == 0.0,
            lambda: -2.0 * jnp.pi * jnp.sqrt(A_point) ** 5 * rho_point / 3.0,
            lambda: (
                -2.0 * jnp.pi * jnp.sqrt(A_point) ** 5 * rho_point
                - 2.0 * phi_point / r_point
            ),
        )

    derivative = jax.vmap(derivative_at_point)(
        A.reshape(-1),
        phi.reshape(-1),
        mass_energy_density.reshape(-1),
        r.reshape(-1),
    )

    return derivative.reshape(output_shape)


def dr_alpha(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    # Covariant radial stress is -Er**2/2; only the scalar energy density
    # contracts the electric field with gamma^rr = 1/A**2.
    total_Srr = Srr - 0.5 * Er**2

    first_term = 4.0 * jnp.pi * alpha * total_Srr * r * A
    second_term = -2.0 * alpha * phi * jnp.sqrt(A)
    third_term = -2.0 * alpha * phi**2 * r
    denominator = A * (
        1.0 + 2.0 * r * phi / jnp.sqrt( A )
    )

    return (first_term + second_term + third_term) / denominator


def Krr_from_state(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms

    return 4.0 * jnp.pi * r * Sr / (
        1.0 + 2.0 * r * phi / jnp.sqrt( A )
    )


def dr_beta_over_r(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    Krr = Krr_from_state(U_state)
    alpha, Krr, r = jnp.broadcast_arrays(alpha, Krr, r)
    output_shape = r.shape

    def derivative_at_point(alpha_point, Krr_point, r_point):
        return lax.cond(
            r_point == 0.0,
            lambda: jnp.zeros_like(r_point),
            lambda: alpha_point * Krr_point / r_point,
        )

    derivative = jax.vmap(derivative_at_point)(
        alpha.reshape(-1),
        Krr.reshape(-1),
        r.reshape(-1),
    )

    return derivative.reshape(output_shape)


def beta_over_r_from_integral(alpha, Krr, r, dr):
    """Shift condition solved as a tail integral after the radial Heun solve."""

    def integrand_at_point(alpha_point, Krr_point, r_point):
        return lax.cond(
            r_point == 0.0,
            lambda: jnp.zeros_like(r_point),
            lambda: alpha_point * Krr_point / r_point,
        )

    integrand = jax.vmap(integrand_at_point)(alpha, Krr, r)

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

    A, phi, Er, charge_density, r = jnp.broadcast_arrays(
        A,
        phi,
        Er,
        charge_density,
        r,
    )
    output_shape = r.shape

    def derivative_at_point(A_point, phi_point, Er_point, charge_point, r_point):
        return lax.cond(
            r_point == 0.0,
            lambda: A_point**2 * charge_point / 3.0,
            lambda: (
                A_point**2 * charge_point
                - 2.0 * Er_point / r_point
                - 2.0 * phi_point * Er_point / jnp.sqrt(A_point)
            ),
        )

    derivative = jax.vmap(derivative_at_point)(
        A.reshape(-1),
        phi.reshape(-1),
        Er.reshape(-1),
        charge_density.reshape(-1),
        r.reshape(-1),
    )

    return derivative.reshape(output_shape)


def heuns_method(U_state, dr, numerators, grid):
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
    source_terms_predictor = source_values_at_point(
        numerators,
        A_predictor,
        r_predictor,
        grid,
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
    source_terms_corrected = source_values_at_point(
        numerators,
        A_corrected,
        r_predictor,
        grid,
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


def _integrate_metric_sweep(
    numerators,
    grid,
    center_A,
    center_alpha,
):
    """Integrate one radial Heun shot from the supplied origin values."""

    r_grid = grid.r_full
    dr = jnp.asarray(grid.dr, dtype=r_grid.dtype)
    initial_A = jnp.asarray(center_A, dtype=r_grid.dtype)
    initial_phi = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_alpha = jnp.asarray(center_alpha, dtype=r_grid.dtype)
    initial_Krr = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_beta_over_r = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_Er = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_r = r_grid[0]
    initial_source_terms = source_values_at_point(
        numerators,
        initial_A,
        initial_r,
        grid,
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
            numerators,
            grid,
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


def integrate_metric_from_origin(particles, grid, center_A, center_alpha):
    """Converge the completed particle map around the original Heun sweep.

    Deposition is fixed during a sweep. Only proper volumes use the local
    predictor/corrector metric. A failed iteration returns a nonfinite state.
    """
    A_guess = jnp.full_like(grid.r_full, center_A)
    tolerance = jnp.maximum(1.0e-10, 100.0 * jnp.finfo(A_guess.dtype).eps)

    def iteration(carry):
        count, A, error = carry
        numerators = deposit_source_numerators(particles, A, grid)
        state = _integrate_metric_sweep(numerators, grid, center_A, center_alpha)
        new_A = state[0]
        old_r = particle_coordinate_radius(particles, A, grid.r_full)
        new_r = particle_coordinate_radius(particles, new_A, grid.r_full)
        profile_error = jnp.max(jnp.abs(new_A - A) / jnp.maximum(jnp.abs(A), 1.0))
        particle_displacement = jnp.where(particles.weight != 0.0, jnp.abs(new_r - old_r), 0.0)
        map_error = jnp.max(particle_displacement, initial=0.0) / grid.dr
        error = jnp.maximum(profile_error, map_error)
        return count + 1, 0.5 * (A + new_A), error

    _, A, error = lax.while_loop(
        lambda carry: (carry[0] < 100) & (carry[2] > tolerance),
        iteration, (jnp.asarray(0), A_guess, jnp.asarray(jnp.inf, dtype=A_guess.dtype)),
    )
    numerators = deposit_source_numerators(particles, A, grid)
    state = _integrate_metric_sweep(numerators, grid, center_A, center_alpha)
    valid = jnp.isfinite(error) & (error <= tolerance)
    mapped_r = particle_coordinate_radius(particles, state[0], grid.r_full)
    valid = valid & jnp.all((particles.weight == 0.0) | jnp.isfinite(mapped_r))
    return jax.tree.map(lambda values: jnp.where(valid, values, jnp.nan), state)


def calculate_metric(
    particles,
    r_grid,
    dr,
    *,
    previous_X_t=None,
    previous_X_r=None,
):
    """Solve the radial constraints with two vacuum-rescaled Heun shots.

    The origin rescaling is updated as one pair: a new ``(X_r, X_t)`` only
    replaces the retained pair when its ``X_t`` exceeds the value from the
    previous accepted metric state.
    """

    r_grid = jnp.asarray(r_grid)
    dr = jnp.asarray(dr, dtype=r_grid.dtype)
    grid = RadialGrid(
        r_full=r_grid,
        r_interior=r_grid[1:-1],
        # The origin is parity-filled; only the outer endpoint remains vacuum.
        dr=dr,
        r_max=r_grid[-1],
    )

    trial_U_state = integrate_metric_from_origin(
        particles,
        grid,
        center_A=jnp.asarray(1.0, dtype=r_grid.dtype),
        center_alpha=jnp.asarray(1.0, dtype=r_grid.dtype),
    )
    trial_A, _, trial_alpha, _, _, _, _, trial_r_grid = trial_U_state

    X_r, X_t = vacuum_rescale_factors(
        trial_A[-1],
        trial_alpha[-1],
        trial_r_grid[-1],
        total_particle_mass(particles),
        total_particle_charge(particles),
    )

    if previous_X_t is None:
        origin_X_r = X_r
        origin_X_t = X_t
    else:
        previous_X_r = jnp.asarray(previous_X_r, dtype=r_grid.dtype)
        previous_X_t = jnp.asarray(previous_X_t, dtype=r_grid.dtype)
        use_new_rescaling = X_t > previous_X_t
        origin_X_r = jnp.where(use_new_rescaling, X_r, previous_X_r)
        origin_X_t = jnp.where(use_new_rescaling, X_t, previous_X_t)

    center_A = trial_A[0] / origin_X_r
    center_alpha = trial_alpha[0] / origin_X_t

    return integrate_metric_from_origin(
        particles,
        grid,
        center_A=center_A,
        center_alpha=center_alpha,
    )
