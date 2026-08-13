import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.particles.particle_shapes import (
    _interpolate_cell_centered_fields_to_particles,
)
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _radial_grid_from_metric(metric: Z4C_Metric):
    return RadialGrid(
        r_full=metric.r,
        r_interior=metric.r,
        dr=metric.dr,
        r_max=metric.r[-1],
    )


def _areal_radius_grid(metric: Z4C_Metric):
    """Return ``R = r sqrt(gT / chi)`` using det(conformal_gamma) = 1."""

    return metric.r / jnp.sqrt(
        metric.chi * jnp.sqrt(metric.conformal_grr)
    )


def _areal_radius_coordinates(metric: Z4C_Metric):
    """Return the regular areal-radius map including the coordinate origin."""

    isotropic_radius = jnp.concatenate(
        (jnp.zeros((1,), dtype=metric.r.dtype), metric.r)
    )
    areal_radius = jnp.concatenate(
        (jnp.zeros((1,), dtype=metric.r.dtype), _areal_radius_grid(metric))
    )

    return isotropic_radius, areal_radius


def _origin_branch_isotropic_radius(rs, metric: Z4C_Metric):
    """Invert ``R(r)`` on the strictly increasing branch from the origin."""

    r_grid, rs_grid = _areal_radius_coordinates(metric)

    edge_indices = jnp.arange(rs_grid.size - 1)
    increasing_edge = jnp.logical_and(
        jnp.isfinite(rs_grid[:-1]),
        jnp.logical_and(
            jnp.isfinite(rs_grid[1:]),
            jnp.diff(rs_grid) > 0.0,
        ),
    )
    branch_end_index = jnp.min(
        jnp.where(increasing_edge, rs_grid.size - 1, edge_indices)
    )
    # If every edge is increasing, the branch ends at the outermost point.
    # Otherwise it ends immediately before the first flat or decreasing edge.

    grid_indices = jnp.arange(rs_grid.size)
    origin_branch_rs = jnp.where(
        grid_indices <= branch_end_index,
        rs_grid,
        jnp.inf,
    )
    upper_index = jnp.searchsorted(origin_branch_rs, rs, side="right")
    upper_index = jnp.clip(upper_index, 1, jnp.maximum(branch_end_index, 1))
    lower_index = upper_index - 1

    delta_rs = rs_grid[upper_index] - rs_grid[lower_index]
    valid_interval = jnp.logical_and(
        upper_index <= branch_end_index,
        delta_rs > 0.0,
    )
    safe_delta_rs = jnp.where(valid_interval, delta_rs, 1.0)
    fraction = (rs - rs_grid[lower_index]) / safe_delta_rs
    r_particle = r_grid[lower_index] + fraction * (
        r_grid[upper_index] - r_grid[lower_index]
    )

    valid_radius = jnp.logical_and(
        branch_end_index > 0,
        jnp.logical_and(
            jnp.isfinite(rs),
            jnp.logical_and(rs >= 0.0, rs <= rs_grid[branch_end_index]),
        ),
    )
    r_particle = jnp.where(valid_radius, r_particle, jnp.nan)

    return r_particle, valid_radius


def isotropic_particle_state(particles, metric: Z4C_Metric):
    """Convert stored lapse-freezing variables to isotropic ``r`` and ``u_r``."""

    rs, _ = particles.get_positions()
    ubar, _ = particles.get_velocities()
    grid = _radial_grid_from_metric(metric)

    r_particle, valid_state = _origin_branch_isotropic_radius(rs, metric)
    # The explicit origin preserves valid particles inside the first
    # cell-centered metric point. A turnover beyond a particle no longer
    # invalidates its local inverse map.

    interpolation_radius = jnp.where(valid_state, r_particle, 0.0)
    (
        alpha_p,
        chi_p,
        conformal_grr_p,
    ) = _interpolate_cell_centered_fields_to_particles(
        jnp.stack((metric.alpha, metric.chi, metric.conformal_grr)),
        interpolation_radius,
        grid,
        shape_mode=particles.get_shape(),
        field_parities=jnp.asarray((1, 1, 1)),
    )
    ur = (ubar / alpha_p) * jnp.sqrt(conformal_grr_p / chi_p)
    ur = jnp.where(valid_state, ur, jnp.nan)

    return r_particle, ur


def lapse_freezing_particle_state(r_particle, ur, metric, shape_mode):
    """Convert ordinary isotropic particle variables to stored ``R`` and ``ubar``."""

    grid = _radial_grid_from_metric(metric)
    r_grid, rs_grid = _areal_radius_coordinates(metric)
    valid_radius = (r_particle >= 0.0) & (r_particle <= r_grid[-1])
    valid_state = valid_radius
    interpolation_radius = jnp.where(valid_state, r_particle, 0.0)

    rs = jnp.interp(
        interpolation_radius,
        r_grid,
        rs_grid,
    )
    rs = jnp.where(valid_state, rs, jnp.nan)

    (
        alpha_p,
        chi_p,
        conformal_grr_p,
    ) = _interpolate_cell_centered_fields_to_particles(
        jnp.stack((metric.alpha, metric.chi, metric.conformal_grr)),
        interpolation_radius,
        grid,
        shape_mode=shape_mode,
        field_parities=jnp.asarray((1, 1, 1)),
    )
    ubar = alpha_p * ur * jnp.sqrt(chi_p / conformal_grr_p)
    ubar = jnp.where(valid_state, ubar, jnp.nan)

    return rs, ubar


def _explicit_lapse_freezing_rhs(
    rs,
    ubar,
    alpha_p,
    beta_p,
    chi_p,
    conformal_grr_p,
    dalphadr_p,
    dalphadt_p,
    dbetadr_p,
    dchidr_p,
    dgrrdr_p,
    dchidt_p,
    dgrrdt_p,
):
    """Evaluate the shift-aware ``rs`` and ``ubar`` notebook expressions."""

    lapse_magnitude = jnp.abs(alpha_p)
    coordinate_energy = jnp.sqrt(alpha_p**2 + ubar**2)

    # LapseFreezingZ4C.nb, eqRsFinal after
    # conformal_gt = 1 / sqrt(conformal_grr).
    metric_advection = conformal_grr_p * (
        4.0 * beta_p * jnp.sqrt(chi_p) * conformal_grr_p**0.75
        - beta_p * chi_p * dgrrdr_p * rs
        + chi_p * dgrrdt_p * rs
        - 2.0 * beta_p * dchidr_p * conformal_grr_p * rs
        + 2.0 * dchidt_p * conformal_grr_p * rs
    )
    radial_motion = (
        -4.0 * chi_p * conformal_grr_p**1.25
        + dgrrdr_p * jnp.sqrt(chi_p**3 * conformal_grr_p) * rs
        + 2.0
        * dchidr_p
        * jnp.sqrt(chi_p * conformal_grr_p**3)
        * rs
    ) * ubar * lapse_magnitude / coordinate_energy

    drs_dt = -(metric_advection + radial_motion) / (
        4.0 * chi_p * conformal_grr_p**2
    )

    # LapseFreezingZ4C.nb, eqUbarFinal. The lapse time derivative is required
    # because ubar = alpha * u_r * sqrt(chi / conformal_grr).
    lapse_advection = (
        2.0
        * chi_p
        * (-beta_p * dalphadr_p + dalphadt_p)
        * conformal_grr_p
        * ubar
    )
    metric_momentum = (
        (
            beta_p * chi_p * dgrrdr_p
            - chi_p * dgrrdt_p
            + 2.0 * chi_p * dbetadr_p * conformal_grr_p
            - beta_p * dchidr_p * conformal_grr_p
            + dchidt_p * conformal_grr_p
        )
        * alpha_p
        * ubar
    )
    lapse_force = (
        -2.0
        * dalphadr_p
        * jnp.sqrt(
            chi_p**3 * conformal_grr_p / coordinate_energy**2
        )
        * lapse_magnitude**3
    )
    dubar_dt = (lapse_advection + metric_momentum + lapse_force) / (
        2.0
        * chi_p
        * conformal_grr_p
        * alpha_p
    )

    return dubar_dt, drs_dt


def compute_geodesic_terms(
    particles,
    metric: Z4C_Metric,
    metric_derivative: Z4C_Metric,
):
    """Evaluate the radial notebook RHS for lapse-freezing variables."""

    rs, _ = particles.get_positions()
    ubar, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()
    grid = _radial_grid_from_metric(metric)

    r_particle, _ = isotropic_particle_state(particles, metric)

    (
        alpha_p,
        beta_p,
        chi_p,
        conformal_grr_p,
        dalphadr_p,
        dalphadt_p,
        dbetadr_p,
        dchidr_p,
        dgrrdr_p,
        dchidt_p,
        dgrrdt_p,
    ) = _interpolate_cell_centered_fields_to_particles(
        jnp.stack(
            (
                metric.alpha,
                metric.beta,
                metric.chi,
                metric.conformal_grr,
                first_derivative(metric.alpha, metric.dr, parity=1),
                metric_derivative.alpha,
                first_derivative(metric.beta, metric.dr, parity=-1),
                first_derivative(metric.chi, metric.dr, parity=1),
                first_derivative(
                    metric.conformal_grr,
                    metric.dr,
                    parity=1,
                ),
                metric_derivative.chi,
                metric_derivative.conformal_grr,
            )
        ),
        r_particle,
        grid,
        shape_mode=particle_shape,
        field_parities=jnp.asarray(
            (1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1)
        ),
    )

    # Explicit shift-aware equations generated in LapseFreezingZ4C.nb.
    dubar_dt, drs_dt = _explicit_lapse_freezing_rhs(
        rs,
        ubar,
        alpha_p,
        beta_p,
        chi_p,
        conformal_grr_p,
        dalphadr_p,
        dalphadt_p,
        dbetadr_p,
        dchidr_p,
        dgrrdr_p,
        dchidt_p,
        dgrrdt_p,
    )

    radial_particle = uphi == 0.0
    invalid = jnp.full_like(ubar, jnp.nan)
    dubar_dt = jnp.where(radial_particle, dubar_dt, invalid)
    drs_dt = jnp.where(radial_particle, drs_dt, invalid)
    du_phi_dt = jnp.where(radial_particle, jnp.zeros_like(uphi), invalid)
    dphi_dt = jnp.where(radial_particle, jnp.zeros_like(uphi), invalid)
    # The notebook equations are radial. Unsupported angular states must fail
    # visibly under JIT instead of silently omitting the centrifugal force.

    return dubar_dt, du_phi_dt, drs_dt, dphi_dt
