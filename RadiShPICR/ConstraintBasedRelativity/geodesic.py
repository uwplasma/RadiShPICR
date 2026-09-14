import jax
import jax.numpy as jnp
from jax import lax

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.solve_metric import dr_A, dr_alpha
from RadiShPICR.ConstraintBasedRelativity.utils import angular_lorentz_term
from RadiShPICR.particles.shape_factors.cartesian_shapes import interpolate_fields_to_particles


def _field_interpolation_grid(r_grid):
    dr_grid = r_grid[1] - r_grid[0]
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr_grid,
        r_max=r_grid[-1],
    )


def isotropic_particle_radius(particles, U_state):
    """Invert the completed map ``r_s = A(r) r`` for each particle."""

    A_values, _, _, _, _, _, _, r_grid = U_state
    rs_grid = A_values * r_grid

    return jnp.interp(
        particles.r,
        rs_grid,
        r_grid,
        left=jnp.nan,
        right=jnp.nan,
    )


def _angular_radial_force(uphi, rs, lapse, radial_jacobian, lorentz_factor):
    """Return the angular force without evaluating it when ``uphi`` is zero."""

    def angular_force_at_particle(uphi_p, rs_p, alpha_p, jacobian_p, W_p):
        return lax.cond(
            uphi_p == 0.0,
            lambda: jnp.zeros_like(uphi_p),
            lambda: alpha_p * jacobian_p * uphi_p**2 / (rs_p**3 * W_p),
        )

    return jax.vmap(angular_force_at_particle)(
        uphi,
        rs,
        lapse,
        radial_jacobian,
        lorentz_factor,
    )


def _azimuthal_derivative(uphi, rs, lapse, lorentz_factor):
    """Return the lapse-freezing azimuthal derivative at each particle."""

    def azimuthal_derivative_at_particle(uphi_p, rs_p, alpha_p, W_p):
        return lax.cond(
            uphi_p == 0.0,
            lambda: jnp.zeros_like(uphi_p),
            lambda: alpha_p * uphi_p / (rs_p**2 * W_p),
        )

    return jax.vmap(azimuthal_derivative_at_particle)(
        uphi,
        rs,
        lapse,
        lorentz_factor,
    )


def compute_geodesic_terms(particles, U_state, dur_dt_EM=None):
    """Evaluate the polar-slicing equations for stored ``r_s`` and ``u_r / A``."""

    (
        A_values,
        phi_values,
        alpha_values,
        Krr_values,
        beta_over_r_values,
        Er_values,
        source_terms,
        r_grid,
    ) = U_state
    rs = particles.r
    ur_over_A = particles.ur
    uphi = particles.uphi
    interpolation_grid = _field_interpolation_grid(r_grid)
    r_particle = isotropic_particle_radius(particles, U_state)

    dA_dr = dr_A(U_state)
    dalpha_dr = dr_alpha(U_state, interpolation_grid.dr)
    (
        A_at_particle,
        lapse_at_particle,
        Krr_at_particle,
        dA_dr_at_particle,
        d_lapse_dr_at_particle,
    ) = interpolate_fields_to_particles(
        jnp.stack(
            (
                A_values,
                alpha_values,
                Krr_values,
                dA_dr,
                dalpha_dr,
            )
        ),
        r_particle,
        interpolation_grid,
        shape_mode=particles.get_shape(),
        field_parities=jnp.asarray((1, 1, 1, -1, -1)),
    )

    W = jnp.sqrt(
        1.0
        + ur_over_A**2
        + angular_lorentz_term(uphi, 1.0, rs)
    )
    radial_jacobian = 1.0 + rs * dA_dr_at_particle / A_at_particle**2

    # Paper IV equation (24) has K_T = 0 in polar slicing.
    drs_dt = (
        lapse_at_particle
        * ur_over_A
        * radial_jacobian
        / W
    )

    dur_over_A_dt = -W * d_lapse_dr_at_particle / A_at_particle
    dur_over_A_dt = dur_over_A_dt + _angular_radial_force(
        uphi,
        rs,
        lapse_at_particle,
        radial_jacobian,
        W,
    )
    dur_over_A_dt = dur_over_A_dt + (
        lapse_at_particle * Krr_at_particle * ur_over_A
    )

    if dur_dt_EM is not None:
        dur_over_A_dt = dur_over_A_dt + dur_dt_EM / A_at_particle

    dphi_dt = _azimuthal_derivative(
        uphi,
        rs,
        lapse_at_particle,
        W,
    )

    return drs_dt, dphi_dt, dur_over_A_dt
