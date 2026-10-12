"""Time-symmetric Oppenheimer-Snyder initial data.

A homogeneous dust sphere of gravitational mass ``M`` with its surface at areal
radius ``R_s``, on a time-symmetric slice.  The analytic profile fixes the
particle distribution; the spatial metric comes from a constrained shoot that
matches the exterior to ``M``.

Both OS demos build their initial state from here: the Z4C runner evolves the
result through :func:`constrained_state_to_z4c`, and the constrained Heun
runner uses the particles and profile directly in its own solver chart (pass
``initial_X_r`` for that chart; the Z4C shoot normalizes it to one).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from RadiShPICR.ConstraintBasedRelativity import integrate_metric_from_origin
from RadiShPICR.ConstraintBasedRelativity.vacuum_conditions import (
    vacuum_rescale_factors,
)
from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

__all__ = [
    "constrained_state_to_z4c",
    "homogeneous_rest_mass_profile",
    "initial_center_A",
    "initialize_oppenheimer_snyder_particles",
    "isotropic_radius_from_areal",
    "shoot_constrained_schwarzschild_initial_data",
    "surface_isotropic_radius",
]

SHOOTING_TOLERANCE = 1.0e-10
SHOOTING_MAX_ITERATIONS = 64
SHOOTING_METHOD = "bracketed_bisection_with_lapse_normalization"

integrate_metric_jit = jax.jit(integrate_metric_from_origin)


def surface_isotropic_radius(surface_areal_radius, total_mass):
    """Isotropic radius of the stellar surface."""

    return 0.5 * (
        surface_areal_radius
        - total_mass
        + math.sqrt(
            surface_areal_radius
            * (surface_areal_radius - 2.0 * total_mass)
        )
    )


def homogeneous_rest_mass_profile(
    mass_density,
    total_mass,
    surface_areal_radius,
    num_quadrature_points=120000,
):
    """Trapezoid-integrate the cumulative rest mass of a homogeneous sphere.

    Returns ``(areal_grid, cumulative_rest_mass, total_rest_mass)``.  The rest
    mass exceeds the gravitational mass by the binding energy, so the last
    entry is not ``total_mass``.
    """

    mass_density = float(mass_density)
    areal_grid = np.linspace(0.0, surface_areal_radius, num_quadrature_points)
    compactness = 2.0 * total_mass / surface_areal_radius
    proper_volume_factor = np.sqrt(
        1.0 - compactness * (areal_grid / surface_areal_radius) ** 2
    )
    rest_mass_integrand = (
        4.0 * math.pi * areal_grid**2 * mass_density / proper_volume_factor
    )

    cumulative_rest_mass = np.zeros_like(areal_grid)
    cumulative_rest_mass[1:] = np.cumsum(
        0.5
        * (rest_mass_integrand[:-1] + rest_mass_integrand[1:])
        * np.diff(areal_grid)
    )

    return areal_grid, cumulative_rest_mass, float(cumulative_rest_mass[-1])


def isotropic_radius_from_areal(areal_radius, total_mass, surface_areal_radius):
    """Initial isotropic radius of a homogeneous, time-symmetric dust sphere."""

    areal_radius = np.asarray(areal_radius, dtype=float)
    matched_surface_radius = surface_isotropic_radius(
        surface_areal_radius, total_mass
    )
    compactness = 2.0 * total_mass / surface_areal_radius
    interior_root = np.sqrt(
        1.0 - compactness * (areal_radius / surface_areal_radius) ** 2
    )
    surface_root = math.sqrt(1.0 - compactness)

    numerator = areal_radius / (1.0 + interior_root)
    denominator = surface_areal_radius / (1.0 + surface_root)

    return matched_surface_radius * numerator / denominator


def initial_center_A(total_mass, surface_areal_radius):
    """Analytic OS value used only to start the constrained spatial shoot."""

    compactness = 2.0 * total_mass / surface_areal_radius
    surface_root = math.sqrt(1.0 - compactness)
    matched_surface_radius = surface_isotropic_radius(
        surface_areal_radius, total_mass
    )

    return (
        2.0
        * surface_areal_radius
        / (matched_surface_radius * (1.0 + surface_root))
    )


def initialize_oppenheimer_snyder_particles(
    grid,
    mass_density,
    total_mass,
    surface_areal_radius,
    total_particles,
    shape_mode,
    initial_X_r=1.0,
    name="oppenheimer_snyder_dust",
):
    """Build mass-stratified dust: equal-rest-mass particles, one shell per cell.

    ``initial_X_r`` is the solver-chart scale of the constrained formulation,
    where the grid runs out to ``r_max / X_r``.  The Z4C shoot normalizes
    ``X_r`` to one, so that path leaves it at the default.

    Returns ``(particles, particle_areal_radius, total_rest_mass)``.
    """

    areal_grid, cumulative_rest_mass, total_rest_mass = (
        homogeneous_rest_mass_profile(
            mass_density, total_mass, surface_areal_radius
        )
    )
    isotropic_grid = isotropic_radius_from_areal(
        areal_grid, total_mass, surface_areal_radius
    )

    # Populate every deposition shell inside the stellar surface.
    grid_indices = np.arange(grid.r_full.size)
    shell_inner_raw = np.maximum((grid_indices - 0.5) * grid.dr, 0.0)
    shell_outer_raw = (grid_indices + 0.5) * grid.dr
    stellar_surface_raw = isotropic_grid[-1] / initial_X_r
    shell_outer_raw = np.minimum(shell_outer_raw, stellar_surface_raw)
    occupied_shell = shell_outer_raw > shell_inner_raw
    shell_inner_raw = shell_inner_raw[occupied_shell]
    shell_outer_raw = shell_outer_raw[occupied_shell]

    shell_inner_areal = np.interp(
        initial_X_r * shell_inner_raw, isotropic_grid, areal_grid
    )
    shell_outer_areal = np.interp(
        initial_X_r * shell_outer_raw, isotropic_grid, areal_grid
    )
    shell_inner_mass = np.interp(
        shell_inner_areal, areal_grid, cumulative_rest_mass
    )
    shell_outer_mass = np.interp(
        shell_outer_areal, areal_grid, cumulative_rest_mass
    )

    num_occupied_shells = shell_inner_mass.size
    if total_particles < num_occupied_shells:
        raise ValueError(
            "total_particles must cover every occupied deposition shell"
        )

    particles_per_shell = np.full(
        num_occupied_shells,
        total_particles // num_occupied_shells,
        dtype=int,
    )
    particles_per_shell[: total_particles % num_occupied_shells] += 1

    particle_areal_radius = []
    particle_weight = []
    for inner_mass, outer_mass, shell_count in zip(
        shell_inner_mass, shell_outer_mass, particles_per_shell
    ):
        shell_fraction = (np.arange(shell_count, dtype=float) + 0.5) / shell_count
        particle_enclosed_mass = (
            inner_mass + shell_fraction * (outer_mass - inner_mass)
        )
        particle_areal_radius.append(
            np.interp(particle_enclosed_mass, cumulative_rest_mass, areal_grid)
        )
        particle_weight.append(
            np.full(shell_count, (outer_mass - inner_mass) / shell_count)
        )

    particle_areal_radius = np.concatenate(particle_areal_radius)
    particle_weight = np.concatenate(particle_weight)
    particles = particle_species(
        name=name,
        charge=0.0,
        mass=jnp.ones((total_particles,)),
        weight=jnp.asarray(particle_weight),
        # ConstraintBasedRelativity stores r_s and u_r / A in the shared
        # particle class fields r and ur; Z4C stores isotropic r and u_r.
        r=jnp.asarray(particle_areal_radius),
        ur=jnp.zeros((total_particles,)),
        phi=jnp.zeros((total_particles,)),
        uphi=jnp.zeros((total_particles,)),
        shape_mode=shape_mode,
    )

    return particles, particle_areal_radius, total_rest_mass


def shoot_constrained_schwarzschild_initial_data(
    particles,
    grid,
    total_mass,
    center_A_guess,
    shooting_tolerance=SHOOTING_TOLERANCE,
    shooting_max_iterations=SHOOTING_MAX_ITERATIONS,
):
    """Match the time-symmetric OS slice to the requested exterior mass.

    The spatial equations do not depend on the lapse. Bisect the central A
    with alpha(0)=1, then normalize the linear lapse equation in one final
    Heun shot. The particle rest mass is not the exterior gravitational mass.
    """

    def shoot(center_A, center_alpha):
        U_state = integrate_metric_jit(
            particles, grid,
            jnp.asarray(center_A, dtype=grid.r_full.dtype),
            jnp.asarray(center_alpha, dtype=grid.r_full.dtype),
        )
        A, _, alpha, _, _, _, _, radial_nodes = U_state
        X_r, X_t = vacuum_rescale_factors(
            A[-1], alpha[-1], radial_nodes[-1], total_mass, 0.0,
        )
        finite = all(
            np.all(np.isfinite(np.asarray(field)))
            for field in jax.tree.leaves(U_state)
        )
        if (
            not finite
            or not np.all(np.asarray(A) > 0.0)
            or not np.all(np.asarray(alpha) > 0.0)
            or not np.isfinite(float(X_r))
            or not np.isfinite(float(X_t))
            or float(X_t) <= 0.0
        ):
            raise RuntimeError("OS matching shot has nonfinite or nonpositive geometry.")
        return U_state, float(X_r), float(X_t)

    lower_A = 0.9 * center_A_guess
    upper_A = 1.1 * center_A_guess
    _, lower_X_r, _ = shoot(lower_A, 1.0)
    _, upper_X_r, _ = shoot(upper_A, 1.0)
    lower_residual = lower_X_r - 1.0
    upper_residual = upper_X_r - 1.0
    if lower_residual * upper_residual > 0.0:
        raise RuntimeError(
            "OS matching requires a sign-changing central-A bracket: "
            f"residuals are {lower_residual:.6e}, {upper_residual:.6e}."
        )

    for _ in range(shooting_max_iterations):
        center_A = 0.5 * (lower_A + upper_A)
        _, X_r, X_t = shoot(center_A, 1.0)
        residual = X_r - 1.0
        if abs(residual) <= shooting_tolerance:
            break
        if lower_residual * residual <= 0.0:
            upper_A = center_A
        else:
            lower_A = center_A
            lower_residual = residual
    else:
        raise RuntimeError(
            f"OS spatial matching did not converge in {shooting_max_iterations} "
            f"iterations (tolerance {shooting_tolerance:.3e})."
        )

    U_state, X_r, X_t = shoot(center_A, 1.0 / X_t)
    if max(abs(X_r - 1.0), abs(X_t - 1.0)) > shooting_tolerance:
        raise RuntimeError(
            f"OS final matching failed: X_r={X_r:.12e}, X_t={X_t:.12e}."
        )

    return U_state


def constrained_state_to_z4c(U_state, z4c_r, kappa=1.0, eta=2.0, nu=0.02):
    """Translate the matched, time-symmetric constrained slice to Z4c."""

    A, _, alpha, _, _, _, _, r_nodes = U_state

    A = jnp.interp(z4c_r, r_nodes, A)
    alpha = jnp.interp(z4c_r, r_nodes, alpha)

    zeros = jnp.zeros_like(z4c_r)
    ones = jnp.ones_like(z4c_r)

    # The constrained ansatz is isotropic: gamma_rr = gamma_T = A**2.
    # Thus chi = det(gamma_ij)**(-1/3) = A**(-2), while both conformal
    # components are one.  Oppenheimer-Snyder initial data are time symmetric.
    return Z4C_Metric(
        alpha=alpha,
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=1.0 / A**2,
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Gamma=zeros,
        kappa=jnp.asarray(kappa, dtype=z4c_r.dtype),
        eta=jnp.asarray(eta, dtype=z4c_r.dtype),
        nu=jnp.asarray(nu, dtype=z4c_r.dtype),
        r=z4c_r,
        dr=z4c_r[1] - z4c_r[0],
    )
