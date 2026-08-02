import jax
import jax.numpy as jnp
from jax import lax

from RadiShPICR.particles.particle_shapes import interpolate_fields_to_particles
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.solve_metric import dr_A, dr_alpha, dr_beta_over_r
from RadiShPICR.ConstraintBasedRelativity.utils import angular_lorentz_term


def _field_interpolation_grid(r_grid):
    dr_grid = r_grid[1] - r_grid[0]
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr_grid,
        r_max=r_grid[-1],
    )


def _angular_radial_force(
    uphi,
    r,
    A,
    lapse,
    dA_dr,
    lorentz_factor,
):
    """Return the angular radial force without evaluating it when uphi is zero."""

    def angular_force_at_particle(uphi_p, r_p, A_p, alpha_p, dA_dr_p, W_p):
        return lax.cond(
            uphi_p == 0.0,
            lambda: jnp.zeros_like(uphi_p),
            lambda: (
                alpha_p
                * uphi_p**2
                / W_p
                * (
                    1.0 / (r_p**3 * A_p**2)
                    + dA_dr_p / (r_p**2 * A_p**3)
                )
            ),
        )

    return jax.vmap(angular_force_at_particle)(
        uphi,
        r,
        A,
        lapse,
        dA_dr,
        lorentz_factor,
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
        field_parities=jnp.asarray((1, 1, -1, -1, -1, 1)),
    )


    W = jnp.sqrt(
        1.0
        + ur**2 / A_at_particle**2
        + angular_lorentz_term(uphi, A_at_particle, r)
    )

    dr_dt = lapse_at_particle * ur / (A_at_particle**2 * W) - shift_at_particle

    du_r_dt = -W * d_lapse_dr_at_particle + ur * d_shift_dr_at_particle
    du_r_dt = du_r_dt + (
        lapse_at_particle * ur**2 * dA_dr_at_particle / (A_at_particle**3 * W)
    )
    du_r_dt = du_r_dt + _angular_radial_force(
        uphi,
        r,
        A_at_particle,
        lapse_at_particle,
        dA_dr_at_particle,
        W,
    )

    return dr_dt, du_r_dt
