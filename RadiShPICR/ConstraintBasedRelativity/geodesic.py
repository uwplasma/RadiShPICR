import jax.numpy as jnp

from RadiShPICR.particles.particle_shapes import interpolate_fields_to_particles
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.solve_metric import dr_A, dr_alpha, dr_beta_over_r
from RadiShPICR.ConstraintBasedRelativity.utils import pad_value, safe_radius


def _field_interpolation_grid(r_grid):
    dr_grid = r_grid[1] - r_grid[0]
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr_grid,
        r_max=r_grid[-1],
    )


def compute_geodesic_terms(particles, U_state):
    A_values, phi_values, alpha_values, Krr_values, beta_over_r_values, Er_values, source_terms, r_grid = U_state
    r, phi = particles.get_positions()
    ur, uphi = particles.get_velocities()
    dr_grid = r_grid[1] - r_grid[0]
    shape_mode = particles.get_shape()
    interpolation_grid = _field_interpolation_grid(r_grid)

    beta = beta_over_r_values * r_grid
    grid_derivative_state = (
        A_values,
        phi_values,
        alpha_values,
        Krr_values,
        beta_over_r_values,
        Er_values,
        source_terms,
        r_grid,
    )
    dA_dr = dr_A(grid_derivative_state)
    dalpha_dr = dr_alpha(grid_derivative_state, dr_grid)
    d_shift_dr = dr_beta_over_r(grid_derivative_state, dr_grid) * r_grid + beta_over_r_values

    (
        A_at_particle,
        lapse_at_particle,
        shift_at_particle,
        dA_dr_at_particle,
        d_lapse_dr_at_particle,
        d_shift_dr_at_particle,
    ) = interpolate_fields_to_particles(
        jnp.stack(
            (
                A_values,
                alpha_values,
                beta,
                dA_dr,
                dalpha_dr,
                d_shift_dr,
            )
        ),
        r,
        interpolation_grid,
        shape_mode=shape_mode,
    )

    safe_r_particle = safe_radius(r, 0.5 * dr_grid)
    A_for_denominators = pad_value(A_at_particle)
    W = jnp.sqrt(
        1.0
        + ur**2 / A_for_denominators**2
        + uphi**2 / (safe_r_particle**2 * A_for_denominators**2)
    )

    dr_dt = lapse_at_particle * ur / (A_for_denominators**2 * W) - shift_at_particle

    du_r_dt = -W * d_lapse_dr_at_particle + ur * d_shift_dr_at_particle
    du_r_dt = du_r_dt + (
        lapse_at_particle * ur**2 * dA_dr_at_particle / (A_for_denominators**3 * W)
    )
    du_r_dt = du_r_dt + (
        lapse_at_particle
        * uphi**2
        / W
        * (
            1.0 / (safe_r_particle**3 * A_for_denominators**2)
            + dA_dr_at_particle / (safe_r_particle**2 * A_for_denominators**3)
        )
    )

    return dr_dt, du_r_dt
