from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def find_package_root() -> Path:
    current_directory = Path(__file__).resolve().parent
    for candidate in (current_directory, *current_directory.parents):
        if (candidate / "RadiShPICR" / "Z4C").is_dir():
            return candidate

        package_root = candidate / "code" / "RadiShPICR"
        if (package_root / "RadiShPICR" / "Z4C").is_dir():
            return package_root

    raise RuntimeError("Could not find the RadiShPICR package root.")


package_root = find_package_root()
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity import (
    build_radial_grid,
    integrate_metric_from_origin,
)
from RadiShPICR.ConstraintBasedRelativity.geodesic import isotropic_particle_radius
from RadiShPICR.ConstraintBasedRelativity.vacuum_conditions import (
    vacuum_rescale_factors,
)
from RadiShPICR.Z4C.energy_momentum_tensor import compute_radial_matter_terms
from RadiShPICR.Z4C.geodesic import _radial_grid_from_metric
from RadiShPICR.Z4C.particle_boundaries import (
    INNER_AREAL_GHOST_CELLS,
    deleting_inner_areal_radius_boundary,
)
from RadiShPICR.Z4C.time_evolve import rk4_step
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    _interpolate_cell_centered_fields_to_particles,
)


integrate_metric_jit = jax.jit(integrate_metric_from_origin)
compute_radial_matter_terms_jit = jax.jit(compute_radial_matter_terms)
rk4_step_jit = jax.jit(
    rk4_step,
    static_argnames=("particle_boundary",),
)


TOTAL_STAR_MASS = 1.0
SURFACE_AREAL_RADIUS = 10.0
TARGET_SCHWARZSCHILD_TIME = 150 * TOTAL_STAR_MASS

# Extend the 800-node spacing over 20M to 3995 Z4c cells over 100M.
REFERENCE_GRID_POINTS = 800
R_MAX = 100.0
NUM_Z4C_CELLS = 5 * (REFERENCE_GRID_POINTS - 1)
PARTICLE_SHELL_COUNT = 300
PARTICLES_PER_SHELL = 300

CFL = 0.2
FREE_FALL_FRACTION = 0.05
MINIMUM_TRIAL_TIME_STEP = 1.0e-7
SAVE_EVERY = 10
SHOOTING_TOLERANCE = 1.0e-10
SHOOTING_MAX_ITERATIONS = 64
SHOOTING_METHOD = "bracketed_bisection_with_lapse_normalization"
INNER_PARTICLE_DEPOSITION = "unrenormalized_shape_overlap"
PARTICLE_DEPOSITION_SCHEMES = {
    "nearest": "ordinary_nearest",
    "linear": "ordinary_cic",
    "quadratic": "ruyten_density_conserving_quadratic_nonnegative",
}
INNER_PARTICLE_ABSORPTION = (
    "zero_physical_overlap_or_beyond_two_ghost_edge"
)

KAPPA = 1.0
ETA = 2.0
NU = 0.02
ZERO_SHIFT = 0  # 1 freezes the initial shift; 0 evolves it with Z4C.


def surface_isotropic_radius(surface_areal_radius, total_mass):
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
    mass_density = float(mass_density)
    areal_grid = np.linspace(
        0.0,
        surface_areal_radius,
        num_quadrature_points,
    )
    compactness = 2.0 * total_mass / surface_areal_radius
    proper_volume_factor = np.sqrt(
        1.0
        - compactness * (areal_grid / surface_areal_radius) ** 2
    )
    rest_mass_integrand = (
        4.0
        * math.pi
        * areal_grid**2
        * mass_density
        / proper_volume_factor
    )

    cumulative_rest_mass = np.zeros_like(areal_grid)
    cumulative_rest_mass[1:] = np.cumsum(
        0.5
        * (rest_mass_integrand[:-1] + rest_mass_integrand[1:])
        * np.diff(areal_grid)
    )

    return (
        areal_grid,
        cumulative_rest_mass,
        float(cumulative_rest_mass[-1]),
    )


def isotropic_radius_from_areal(
    areal_radius,
    total_mass,
    surface_areal_radius,
):
    """Initial isotropic radius of a homogeneous, time-symmetric dust sphere."""

    areal_radius = np.asarray(areal_radius, dtype=float)
    matched_surface_radius = surface_isotropic_radius(
        surface_areal_radius,
        total_mass,
    )
    compactness = 2.0 * total_mass / surface_areal_radius
    interior_root = np.sqrt(
        1.0
        - compactness * (areal_radius / surface_areal_radius) ** 2
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
        surface_areal_radius,
        total_mass,
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
):
    """Build the same mass-stratified dust particles as the constrained demo."""

    areal_grid, cumulative_rest_mass, total_rest_mass = (
        homogeneous_rest_mass_profile(
            mass_density,
            total_mass,
            surface_areal_radius,
        )
    )
    isotropic_grid = isotropic_radius_from_areal(
        areal_grid,
        total_mass,
        surface_areal_radius,
    )

    grid_indices = np.arange(grid.r_full.size)
    shell_inner_raw = np.maximum((grid_indices - 0.5) * grid.dr, 0.0)
    shell_outer_raw = (grid_indices + 0.5) * grid.dr
    shell_outer_raw = np.minimum(shell_outer_raw, isotropic_grid[-1])
    occupied_shell = shell_outer_raw > shell_inner_raw
    shell_inner_raw = shell_inner_raw[occupied_shell]
    shell_outer_raw = shell_outer_raw[occupied_shell]

    shell_inner_areal = np.interp(
        shell_inner_raw,
        isotropic_grid,
        areal_grid,
    )
    shell_outer_areal = np.interp(
        shell_outer_raw,
        isotropic_grid,
        areal_grid,
    )
    shell_inner_mass = np.interp(
        shell_inner_areal,
        areal_grid,
        cumulative_rest_mass,
    )
    shell_outer_mass = np.interp(
        shell_outer_areal,
        areal_grid,
        cumulative_rest_mass,
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
        shell_inner_mass,
        shell_outer_mass,
        particles_per_shell,
    ):
        shell_fraction = (
            np.arange(shell_count, dtype=float) + 0.5
        ) / shell_count
        particle_enclosed_mass = (
            inner_mass
            + shell_fraction * (outer_mass - inner_mass)
        )
        particle_areal_radius.append(
            np.interp(
                particle_enclosed_mass,
                cumulative_rest_mass,
                areal_grid,
            )
        )
        particle_weight.append(
            np.full(
                shell_count,
                (outer_mass - inner_mass) / shell_count,
            )
        )

    particle_areal_radius = np.concatenate(particle_areal_radius)
    particle_weight = np.concatenate(particle_weight)
    particles = particle_species(
        name="oppenheimer_snyder_dust",
        charge=0.0,
        mass=jnp.ones((total_particles,)),
        weight=jnp.asarray(particle_weight),
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


def constrained_state_to_z4c(
    U_state,
    z4c_r,
    kappa=KAPPA,
    eta=ETA,
    nu=NU,
):
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


def physical_spatial_metric(metric):
    grr = metric.conformal_grr / metric.chi
    gT = metric.conformal_gt / metric.chi
    return grr, gT


def schwarzschild_rescale_factors_from_z4c(metric, exterior_mass):
    """Match a Z4c snapshot using its outer lapse and physical radial metric."""

    grr, gT = physical_spatial_metric(metric)
    X_r, X_t = vacuum_rescale_factors(
        jnp.sqrt(grr[-1]),
        metric.alpha[-1],
        metric.r[-1],
        exterior_mass,
        0.0,
    )
    outer_metric_mismatch = (grr[-1] - gT[-1]) / (
        0.5 * (grr[-1] + gT[-1])
    )

    return X_r, X_t, outer_metric_mismatch


def rescale_z4c_to_schwarzschild_coordinates(
    metric,
    particles,
    exterior_mass,
):
    """Return standard-particle diagnostic copies in the Schwarzschild chart."""

    X_r, X_t, outer_metric_mismatch = (
        schwarzschild_rescale_factors_from_z4c(metric, exterior_mass)
    )

    diagnostic_metric = metric._replace(
        alpha=metric.alpha / X_t,
        beta=metric.beta * X_r / X_t,
        chi=metric.chi * X_r**2,
        Gamma=metric.Gamma / X_r,
        r=metric.r * X_r,
        dr=metric.dr * X_r,
    )

    # Covariant radial momentum scales inversely with the radial coordinate.
    diagnostic_particles = particle_species(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=particles.r * X_r,
        ur=particles.ur / X_r,
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )

    return (
        diagnostic_metric,
        diagnostic_particles,
        X_r,
        X_t,
        outer_metric_mismatch,
    )


def build_initial_state(
    r_max=R_MAX,
    num_z4c_cells=NUM_Z4C_CELLS,
    particles_per_shell=PARTICLES_PER_SHELL,
    shape_mode="nearest",
    total_mass=TOTAL_STAR_MASS,
    surface_areal_radius=SURFACE_AREAL_RADIUS,
    shooting_tolerance=SHOOTING_TOLERANCE,
    shooting_max_iterations=SHOOTING_MAX_ITERATIONS,
):
    """Shoot a Schwarzschild-matched constrained OS slice and convert to Z4c."""

    mass_density = total_mass / (
        (4.0 / 3.0) * jnp.pi * surface_areal_radius**3
    )
    constrained_grid = build_radial_grid(
        r_max,
        num_z4c_cells + 1,
    )
    total_particles = PARTICLE_SHELL_COUNT * particles_per_shell

    particles, particle_areal_radius, total_rest_mass = (
        initialize_oppenheimer_snyder_particles(
            constrained_grid,
            mass_density,
            total_mass,
            surface_areal_radius,
            total_particles,
            shape_mode,
        )
    )

    constrained_U_state = shoot_constrained_schwarzschild_initial_data(
        particles,
        constrained_grid,
        total_mass,
        initial_center_A(total_mass, surface_areal_radius),
        shooting_tolerance,
        shooting_max_iterations,
    )

    constrained_r = constrained_U_state[-1]
    z4c_r = 0.5 * (constrained_r[:-1] + constrained_r[1:])
    metric = constrained_state_to_z4c(constrained_U_state, z4c_r)

    r_particle = isotropic_particle_radius(particles, constrained_U_state)
    A_at_particle = jnp.interp(
        r_particle,
        constrained_U_state[-1],
        constrained_U_state[0],
    )
    ur = A_at_particle * particles.ur
    particles = particle_species(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=r_particle,
        ur=ur,
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )

    return (
        metric,
        particles,
        constrained_U_state,
        particle_areal_radius,
        total_rest_mass,
    )


@jax.jit
def state_is_acceptable(metric, particles):
    evolved_metric_fields = (
        metric.alpha,
        metric.beta,
        metric.conformal_grr,
        metric.conformal_gt,
        metric.chi,
        metric.Kh,
        metric.Arr,
        metric.At,
        metric.theta,
        metric.Gamma,
    )
    finite_metric = jnp.all(jnp.isfinite(jnp.stack(evolved_metric_fields)))
    finite_particles = jnp.all(jnp.isfinite(jnp.stack(
        (particles.r, particles.ur, particles.phi, particles.uphi)
    )))

    grr, gT = physical_spatial_metric(metric)
    positive_geometry = (
        jnp.all(grr > 0.0)
        & jnp.all(gT > 0.0)
        & jnp.all(metric.alpha > 0.0)
    )

    return finite_metric & finite_particles & positive_geometry


def freefall_collapse_time_step(particles, metric):
    matter_terms = compute_radial_matter_terms_jit(
        particles,
        metric,
        inner_open=True,
    )
    rho_max = float(np.max(np.asarray(matter_terms.rho)))
    if rho_max <= 0.0:
        return math.inf

    return FREE_FALL_FRACTION * math.sqrt(3.0 * math.pi / (32.0 * rho_max))


def prepare_output_directory(output_directory):
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_directory}. "
            "Move or clear it before starting a new collapse run."
        )

    metric_directory = output_directory / "metric"
    phase_space_directory = output_directory / "phase_space"
    metric_directory.mkdir(parents=True, exist_ok=True)
    phase_space_directory.mkdir(parents=True, exist_ok=True)

    return metric_directory, phase_space_directory


def write_schwarzschild_snapshot(
    metric,
    particles,
    metric_directory,
    phase_space_directory,
    step,
    schwarzschild_time,
    total_mass=TOTAL_STAR_MASS,
):
    (
        diagnostic_metric,
        diagnostic_particles,
        X_r,
        X_t,
        outer_metric_mismatch,
    ) = rescale_z4c_to_schwarzschild_coordinates(
        metric,
        particles,
        total_mass,
    )
    matter_terms = compute_radial_matter_terms_jit(
        diagnostic_particles,
        diagnostic_metric,
        inner_open=True,
    )
    grr, gT = physical_spatial_metric(diagnostic_metric)
    areal_radius = diagnostic_metric.r * jnp.sqrt(gT)
    chi_p, conformal_grr_p, conformal_gt_p = (
        _interpolate_cell_centered_fields_to_particles(
            jnp.stack(
                (
                    diagnostic_metric.chi,
                    diagnostic_metric.conformal_grr,
                    diagnostic_metric.conformal_gt,
                )
            ),
            diagnostic_particles.r,
            _radial_grid_from_metric(diagnostic_metric),
            shape_mode=diagnostic_particles.get_shape(),
            field_parities=jnp.asarray((1, 1, 1)),
        )
    )
    particle_areal_radius = diagnostic_particles.r * jnp.sqrt(
        conformal_gt_p / chi_p
    )
    radial_orthonormal_momentum = diagnostic_particles.ur * jnp.sqrt(
        chi_p / conformal_grr_p
    )

    metric_path = Path(metric_directory) / f"metric_step_{step:06d}.npz"
    np.savez_compressed(
        metric_path,
        r=np.asarray(diagnostic_metric.r),
        areal_radius=np.asarray(areal_radius),
        alpha=np.asarray(diagnostic_metric.alpha),
        beta=np.asarray(diagnostic_metric.beta),
        grr=np.asarray(grr),
        gT=np.asarray(gT),
        conformal_grr=np.asarray(diagnostic_metric.conformal_grr),
        conformal_gT=np.asarray(diagnostic_metric.conformal_gt),
        chi=np.asarray(diagnostic_metric.chi),
        Kh=np.asarray(diagnostic_metric.Kh),
        Arr=np.asarray(diagnostic_metric.Arr),
        AT=np.asarray(diagnostic_metric.At),
        theta=np.asarray(diagnostic_metric.theta),
        Gamma=np.asarray(diagnostic_metric.Gamma),
        mass_density=np.asarray(matter_terms.rho),
        Srr=np.asarray(matter_terms.Srr),
        ST=np.asarray(matter_terms.Stt),
        Sr=np.asarray(matter_terms.Sr),
        step=int(step),
        time=float(schwarzschild_time),
        schwarzschild_time=float(schwarzschild_time),
        X_r=float(X_r),
        X_t=float(X_t),
        outer_metric_relative_mismatch=float(outer_metric_mismatch),
        total_mass=float(total_mass),
        saved_coordinates="schwarzschild_isotropic_diagnostic",
    )

    phase_space_path = Path(phase_space_directory) / (
        f"phase_space_{diagnostic_particles.name}_step_{step:06d}.npz"
    )
    np.savez_compressed(
        phase_space_path,
        r=np.asarray(diagnostic_particles.r),
        ur=np.asarray(diagnostic_particles.ur),
        areal_radius=np.asarray(particle_areal_radius),
        radial_orthonormal_momentum=np.asarray(radial_orthonormal_momentum),
        phi=np.asarray(diagnostic_particles.phi),
        uphi=np.asarray(diagnostic_particles.uphi),
        weight=np.asarray(diagnostic_particles.weight),
        step=int(step),
        time=float(schwarzschild_time),
        schwarzschild_time=float(schwarzschild_time),
        species_name=diagnostic_particles.name,
        saved_coordinates="schwarzschild_isotropic_diagnostic",
        particle_state_variables="r_ur",
    )

    return float(X_r), float(X_t), float(outer_metric_mismatch)


def run_simulation(args):
    wall_start = time.perf_counter()
    metric_directory, phase_space_directory = prepare_output_directory(
        args.output_directory
    )
    metric, particles, constrained_U_state, particle_areal_radius, total_rest_mass = (
        build_initial_state(
            r_max=args.r_max,
            num_z4c_cells=args.num_cells,
            particles_per_shell=args.particles_per_shell,
            shape_mode=args.shape_mode,
            shooting_tolerance=args.shooting_tolerance,
            shooting_max_iterations=args.shooting_max_iterations,
        )
    )

    total_effective_mass = float(np.sum(np.asarray(particles.get_mass())))
    initial_particle_weight = float(np.sum(np.asarray(particles.weight)))
    initial_grr, initial_gT = physical_spatial_metric(metric)
    initial_isotropy_error = float(
        np.max(np.abs(np.asarray(initial_grr - initial_gT)))
    )
    constrained_A = constrained_U_state[0]
    constrained_alpha = constrained_U_state[2]
    constrained_r = constrained_U_state[-1]
    initial_X_r, initial_X_t = vacuum_rescale_factors(
        constrained_A[-1],
        constrained_alpha[-1],
        constrained_r[-1],
        TOTAL_STAR_MASS,
        0.0,
    )

    np.savez_compressed(
        Path(args.output_directory) / "initial_data_metadata.npz",
        total_mass=TOTAL_STAR_MASS,
        total_rest_mass=float(total_rest_mass),
        total_effective_mass=total_effective_mass,
        surface_areal_radius=SURFACE_AREAL_RADIUS,
        particle_areal_radius=np.asarray(particle_areal_radius),
        constrained_r=np.asarray(constrained_U_state[-1]),
        z4c_r=np.asarray(metric.r),
        initial_isotropy_error=initial_isotropy_error,
        shooting_method=SHOOTING_METHOD,
        shooting_tolerance=args.shooting_tolerance,
        shooting_max_iterations=args.shooting_max_iterations,
        center_A=float(constrained_U_state[0][0]),
        center_alpha=float(constrained_U_state[2][0]),
        constrained_outer_X_r=float(initial_X_r),
        constrained_outer_X_t=float(initial_X_t),
        target_schwarzschild_time=float(args.target_time),
        zero_shift=ZERO_SHIFT,
        particle_state_variables="r_ur",
        particle_boundary="areal_inner_open",
        particle_inner_ghost_cells=INNER_AREAL_GHOST_CELLS,
        particle_deposition_boundary=INNER_PARTICLE_DEPOSITION,
        particle_deposition_scheme=PARTICLE_DEPOSITION_SCHEMES[args.shape_mode],
        particle_absorption_rule=INNER_PARTICLE_ABSORPTION,
        shape_mode=args.shape_mode,
    )

    step = 0
    schwarzschild_time = 0.0
    X_r, X_t, outer_metric_mismatch = write_schwarzschild_snapshot(
        metric,
        particles,
        metric_directory,
        phase_space_directory,
        step,
        schwarzschild_time,
    )

    time_tolerance = 1.0e-10 * TOTAL_STAR_MASS
    maximum_steps = math.inf if args.max_steps is None else args.max_steps

    with tqdm(
        total=args.target_time,
        initial=schwarzschild_time,
        desc="evolving Z4c stellar collapse",
        unit="t",
    ) as progress_bar:
        while (
            args.target_time - schwarzschild_time > time_tolerance
            and step < maximum_steps
        ):
            remaining_schwarzschild_time = (
                args.target_time - schwarzschild_time
            )
            cfl_dt = args.cfl * float(metric.dr)
            freefall_dt = freefall_collapse_time_step(particles, metric)
            trial_dt = min(
                cfl_dt,
                freefall_dt,
                remaining_schwarzschild_time / X_t,
            )

            accepted = False
            endpoint_trial = trial_dt < args.minimum_dt
            while trial_dt >= args.minimum_dt or endpoint_trial:
                endpoint_trial = False
                trial_particles, trial_metric, _, _ = rk4_step_jit(
                    particles,
                    metric,
                    trial_dt,
                    EM_on=False,
                    GR_on=True,
                    zero_shift=ZERO_SHIFT,
                    particle_boundary=deleting_inner_areal_radius_boundary,
                    inner_open=True,
                )

                if not state_is_acceptable(trial_metric, trial_particles):
                    trial_dt *= 0.5
                    continue

                (
                    _,
                    trial_X_t,
                    trial_outer_metric_mismatch,
                ) = schwarzschild_rescale_factors_from_z4c(
                    trial_metric,
                    TOTAL_STAR_MASS,
                )
                trial_X_t = float(trial_X_t)
                trial_schwarzschild_dt = 0.5 * (X_t + trial_X_t) * trial_dt

                if (
                    trial_schwarzschild_dt
                    > remaining_schwarzschild_time + time_tolerance
                ):
                    trial_dt *= (
                        remaining_schwarzschild_time
                        / trial_schwarzschild_dt
                    )
                    endpoint_trial = trial_dt < args.minimum_dt
                    continue

                accepted = True
                break

            if not accepted:
                print(
                    "No finite positive-lapse Z4c trial state found above "
                    f"dt={args.minimum_dt:.3e}; preserving step {step} at "
                    f"Schwarzschild time {schwarzschild_time:.8e}."
                )
                break

            previous_schwarzschild_time = schwarzschild_time
            particles = trial_particles
            metric = trial_metric
            X_t = trial_X_t
            outer_metric_mismatch = float(trial_outer_metric_mismatch)
            schwarzschild_time += trial_schwarzschild_dt
            step += 1

            if args.target_time - schwarzschild_time <= time_tolerance:
                schwarzschild_time = args.target_time

            should_save = (
                step % args.save_every == 0
                or schwarzschild_time >= args.target_time
                or step >= maximum_steps
            )
            if should_save:
                X_r, X_t, outer_metric_mismatch = write_schwarzschild_snapshot(
                    metric,
                    particles,
                    metric_directory,
                    phase_space_directory,
                    step,
                    schwarzschild_time,
                )

            minimum_alpha = float(
                np.min(np.asarray(metric.alpha / X_t))
            )
            progress_bar.update(
                schwarzschild_time - previous_schwarzschild_time
            )
            progress_bar.set_postfix(
                step=step,
                dt=f"{trial_dt:.3e}",
                min_alpha=f"{minimum_alpha:.3e}",
                outer_g_mismatch=f"{outer_metric_mismatch:.3e}",
            )

    final_metric_path = metric_directory / f"metric_step_{step:06d}.npz"
    if not final_metric_path.exists():
        X_r, X_t, outer_metric_mismatch = write_schwarzschild_snapshot(
            metric,
            particles,
            metric_directory,
            phase_space_directory,
            step,
            schwarzschild_time,
        )

    diagnostic_metric, _, _, _, _ = rescale_z4c_to_schwarzschild_coordinates(
        metric,
        particles,
        TOTAL_STAR_MASS,
    )
    completed = args.target_time - schwarzschild_time <= time_tolerance
    final_active_particle_weight = float(
        np.sum(np.asarray(particles.weight))
    )
    run_summary = {
        "zero_shift": ZERO_SHIFT,
        "particle_boundary": "areal_inner_open",
        "particle_inner_ghost_cells": INNER_AREAL_GHOST_CELLS,
        "particle_deposition_boundary": INNER_PARTICLE_DEPOSITION,
        "particle_deposition_scheme": PARTICLE_DEPOSITION_SCHEMES[args.shape_mode],
        "shooting_method": SHOOTING_METHOD,
        "shooting_tolerance": args.shooting_tolerance,
        "shooting_max_iterations": args.shooting_max_iterations,
        "constrained_outer_X_r": float(initial_X_r),
        "constrained_outer_X_t": float(initial_X_t),
        "particle_absorption_rule": INNER_PARTICLE_ABSORPTION,
        "shape_mode": args.shape_mode,
        "completed": bool(completed),
        "final_step": int(step),
        "final_schwarzschild_time": float(schwarzschild_time),
        "target_schwarzschild_time": float(args.target_time),
        "minimum_lapse": float(np.min(np.asarray(diagnostic_metric.alpha))),
        "minimum_chi": float(np.min(np.asarray(diagnostic_metric.chi))),
        "maximum_chi": float(np.max(np.asarray(diagnostic_metric.chi))),
        "active_particles": int(np.count_nonzero(np.asarray(particles.weight))),
        "total_particles": int(particles.r.size),
        "initial_particle_weight": initial_particle_weight,
        "final_active_particle_weight": final_active_particle_weight,
        "deleted_particle_weight": (
            initial_particle_weight - final_active_particle_weight
        ),
        "wall_runtime_seconds": float(time.perf_counter() - wall_start),
    }
    with (Path(args.output_directory) / "run_summary.json").open("w") as stream:
        json.dump(run_summary, stream, indent=2)
        stream.write("\n")

    print(json.dumps(run_summary, indent=2))
    return metric, particles, step, schwarzschild_time


def parse_arguments():
    output_directory = (
        Path(__file__).resolve().parent
        / "outputs"
        / "z4c_oppenheimer_snyder"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--r-max", type=float, default=R_MAX)
    parser.add_argument("--num-cells", type=int, default=NUM_Z4C_CELLS)
    parser.add_argument(
        "--particles-per-shell",
        type=int,
        default=PARTICLES_PER_SHELL,
    )
    parser.add_argument(
        "--shape-mode",
        choices=("nearest", "linear", "quadratic"),
        default="quadratic",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=TARGET_SCHWARZSCHILD_TIME,
    )
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument(
        "--minimum-dt",
        type=float,
        default=MINIMUM_TRIAL_TIME_STEP,
    )
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument(
        "--shooting-tolerance",
        type=float,
        default=SHOOTING_TOLERANCE,
        help="maximum initial Schwarzschild matching error in X_r and X_t",
    )
    parser.add_argument(
        "--shooting-max-iterations",
        type=int,
        default=SHOOTING_MAX_ITERATIONS,
        help="maximum spatial bisection iterations for the initial slice",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=output_directory,
    )

    args = parser.parse_args()
    args.save_every = max(1, args.save_every)
    if not math.isfinite(args.shooting_tolerance) or args.shooting_tolerance <= 0.0:
        parser.error("--shooting-tolerance must be finite and positive")
    if args.shooting_max_iterations < 1:
        parser.error("--shooting-max-iterations must be at least 1")
    return args


if __name__ == "__main__":
    run_simulation(parse_arguments())
