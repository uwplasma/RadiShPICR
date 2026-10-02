import jax
import jax.numpy as jnp
from jax import lax

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.utils import (
    angular_lorentz_term,
    radial_shell_volume,
)
from RadiShPICR.ConstraintBasedRelativity.vacuum_conditions import (
    total_particle_charge,
    total_particle_mass,
    vacuum_rescale_factors,
)
from RadiShPICR.particles.shape_factors.cartesian_shapes import radial_shape_stencil


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


def _source_terms_at_point(
    particles,
    A_at_point,
    radial_coordinate,
    grid,
):
    rs, _ = particles.get_positions()
    ur_over_A, uphi = particles.get_velocities()
    dr = grid.dr
    # The constrained particle class stores r_s and u_r / A. Convert them
    # with the local Heun value of A before applying the ordinary r stencil.
    r_particle = rs / A_at_point
    ur = A_at_point * ur_over_A

    even_indices, even_stencil_weights = radial_shape_stencil(
        r_particle,
        grid,
        shape_mode=particles.get_shape(),
        parity=1,
    )
    odd_indices, odd_stencil_weights = radial_shape_stencil(
        r_particle,
        grid,
        shape_mode=particles.get_shape(),
        parity=-1,
    )
    floating_index = (radial_coordinate - grid.r_full[0]) / dr
    grid_index = jnp.rint(floating_index).astype(even_indices.dtype)
    # get the weights for the particles that contribute to this grid point

    even_weights = jnp.sum(
        jnp.where(even_indices == grid_index, even_stencil_weights, 0.0),
        axis=0,
    )
    odd_weights = jnp.sum(
        jnp.where(odd_indices == grid_index, odd_stencil_weights, 0.0),
        axis=0,
    )


    lorentz_factor = jnp.sqrt(
        1.0
        + ur**2 / A_at_point**2
        + angular_lorentz_term(uphi, A_at_point, r_particle)
    )
    # The Lorentz factor is computed using the metric at the grid point, which is used to compute the mass density and charge density contributions from the particles.

    cell_volume = radial_shell_volume(
        A_at_point,
        radial_coordinate,
        dr,
    )
    # the volume of the cell is computed using the metric at the grid point, which is used to compute the mass density and charge density contributions from the particles.

    weighted_mass = particles.get_mass() * even_weights
    conformal_mass_density = jnp.sum(weighted_mass * lorentz_factor)
    conformal_charge_density = jnp.sum(particles.get_charge() * even_weights)
    conformal_Srr = jnp.sum(weighted_mass * ur**2 / lorentz_factor)
    conformal_Sr = jnp.sum(particles.get_mass() * odd_weights * ur)
    # compute the mass density, charge density, and stress-energy tensor components in the conformal frame
    mass_density = conformal_mass_density / cell_volume
    charge_density = conformal_charge_density / cell_volume
    Srr = conformal_Srr / cell_volume
    Sr = conformal_Sr / cell_volume
    # compute the mass density, charge density, and stress-energy tensor components in the physical frame by dividing by the cell volume, which is computed using the metric at the grid point.

    return mass_density, charge_density, Srr, Sr


def heuns_method(U_state, dr, particles, grid):
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


def integrate_metric_from_origin(
    particles,
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
    initial_source_terms = _source_terms_at_point(
        particles,
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
            particles,
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
