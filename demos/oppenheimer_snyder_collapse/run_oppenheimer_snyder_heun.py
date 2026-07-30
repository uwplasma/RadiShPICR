from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm


def find_package_root() -> Path:
    current_directory = Path(__file__).resolve().parent
    for candidate in (current_directory, *current_directory.parents):
        package_root = candidate / "code" / "RadiShPICR"
        if package_root.is_dir():
            return package_root

    raise RuntimeError("Could not find repository root containing code/RadiShPICR/.")


package_root = find_package_root()
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import jax
import jax.numpy as jnp
from RadiShPICR.ConstraintBasedRelativity import (
    build_radial_grid,
    calculate_metric,
    rescale_to_schwarzschild_coordinates,
    schwarzschild_rescale_factors,
    step_rk4_with_metric,
)
from RadiShPICR.ConstraintBasedRelativity.geodesic import compute_geodesic_terms
from RadiShPICR.diagnostics import write_phase_space
from RadiShPICR.particles import particle_species

calculate_metric = jax.jit(calculate_metric)
step_rk4_with_metric = jax.jit(step_rk4_with_metric)


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


def initial_solver_coordinate_scale(total_mass, surface_areal_radius):
    """Map the raw A(0)=1 chart to the initial Schwarzschild coordinates."""

    compactness = 2.0 * total_mass / surface_areal_radius
    surface_root = math.sqrt(1.0 - compactness)
    matched_surface_radius = surface_isotropic_radius(
        surface_areal_radius,
        total_mass,
    )
    central_A = (
        2.0
        * surface_areal_radius
        / (matched_surface_radius * (1.0 + surface_root))
    )

    return 1.0 / central_A


def initialize_oppenheimer_snyder_particles(
    grid,
    initial_X_r,
    mass_density,
    total_mass,
    surface_areal_radius,
    total_particles,
    shape_mode,
):
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

    # Populate every nearest-deposition shell inside the stellar surface.
    grid_indices = np.arange(grid.r_full.size)
    shell_inner_raw = np.maximum(
        (grid_indices - 0.5) * grid.dr,
        0.0,
    )
    shell_outer_raw = (grid_indices + 0.5) * grid.dr
    stellar_surface_raw = isotropic_grid[-1] / initial_X_r
    shell_outer_raw = np.minimum(
        shell_outer_raw,
        stellar_surface_raw,
    )
    occupied_shell = shell_outer_raw > shell_inner_raw
    shell_inner_raw = shell_inner_raw[occupied_shell]
    shell_outer_raw = shell_outer_raw[occupied_shell]

    shell_inner_areal = np.interp(
        initial_X_r * shell_inner_raw,
        isotropic_grid,
        areal_grid,
    )
    shell_outer_areal = np.interp(
        initial_X_r * shell_outer_raw,
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
    particle_isotropic_radius = isotropic_radius_from_areal(
        particle_areal_radius,
        total_mass,
        surface_areal_radius,
    )
    particle_solver_radius = particle_isotropic_radius / initial_X_r

    particle_mass = jnp.ones((total_particles,))
    particle_ur = jnp.zeros((total_particles,))
    particle_phi = jnp.zeros((total_particles,))
    particle_uphi = jnp.zeros((total_particles,))

    particles = particle_species(
        name="oppenheimer_snyder_dust",
        charge=0.0,
        mass=particle_mass,
        weight=jnp.asarray(particle_weight),
        r=jnp.asarray(particle_solver_radius),
        ur=particle_ur,
        phi=particle_phi,
        uphi=particle_uphi,
        shape_mode=shape_mode,
    )

    return particles, particle_areal_radius, total_rest_mass


SHAPE_FACTOR = 0
# particle shape factor
FREE_FALL_FRACTION = 0.05
# fraction of free fall time step to use for the simulation
CROSSING_FRACTION = 0.25
# fraction of one radial cell that particles may cross in one time step
MINIMUM_TRIAL_TIME_STEP = 1.0e-6
# stop when no finite positive Schwarzschild-lapse trial remains above this step
SAVE_EVERY = 1
# save every completed step by default for compatibility

total_star_mass = 1.0
run_time        = 50.0 / total_star_mass # 50 units of M
surface_areal_radius = 10.0
number_density = total_star_mass / (4/3 * jnp.pi * surface_areal_radius**3)
# define the mass of the ball of dust and the outer radius of the ball of dust
ppc = 80
# define the number of particles per cell
Nr    = 500
r_max = 20.0
dr    = r_max / Nr
Nr_particles = int( surface_areal_radius / dr )
# define the number of grid points within the sphere and the maximum radius of the grid
total_particles = Nr_particles * ppc
# define the total number of particles in the simulation

if SHAPE_FACTOR == 0:
    shape_mode = "nearest"
elif SHAPE_FACTOR == 1:
    shape_mode = "linear"
else:
    shape_mode = "quadratic"

initial_X_r = initial_solver_coordinate_scale(
    total_star_mass,
    surface_areal_radius,
)
solver_r_max = r_max / initial_X_r
grid = build_radial_grid(solver_r_max, Nr)

particles, particle_areal_radius, total_rest_mass = (
    initialize_oppenheimer_snyder_particles(
        grid,
        initial_X_r,
        number_density,
        total_star_mass,
        surface_areal_radius,
        total_particles,
        shape_mode,
    )
)

# define the initial particle species

parser = argparse.ArgumentParser()
parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
args = parser.parse_args()
save_every = max(1, args.save_every)

output_directory = (
    Path(__file__).resolve().parent
    / "outputs"
    / "heun_oppenheimer_snyder"
)
U_state_directory = output_directory / "U_state"
phase_space_directory = output_directory / "phase_space"
U_state_directory.mkdir(parents=True, exist_ok=True)
phase_space_directory.mkdir(parents=True, exist_ok=True)


def apparent_horizon(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state
    # unpack the metric state into its components
    return 1.0 + 2.0 * np.asarray(r_grid) * np.asarray(phi) / np.sqrt(np.asarray(A))
# define a function to compute the apparent horizon function based on the metric state

def freefall_collapse_time_step(
    U_state,
):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state
    mass_density, charge_density, Srr, Sr = source_terms
    # unpack the metric state into its components
    rho_max = float(np.max(np.asarray(mass_density)))
    return FREE_FALL_FRACTION * math.sqrt(3.0 * math.pi / (32.0 * rho_max))


def particle_crossing_time_step(
    particles,
    U_state,
    dr,
):
    dr_dt, _ = compute_geodesic_terms(particles, U_state)
    maximum_speed = float(np.max(np.abs(np.asarray(dr_dt))))
    if maximum_speed == 0.0:
        return math.inf

    return CROSSING_FRACTION * dr / maximum_speed


def copy_particles(particles):
    return particle_species(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=jnp.array(particles.r),
        ur=jnp.array(particles.ur),
        phi=jnp.array(particles.phi),
        uphi=jnp.array(particles.uphi),
        shape_mode=particles.shape_mode,
    )


def schwarzschild_lapse(U_state):
    alpha = U_state[2]
    _, X_t = schwarzschild_rescale_factors(
        U_state,
        total_star_mass,
    )

    return alpha / X_t


def solver_state_is_acceptable(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state
    mass_density, charge_density, Srr, Sr = source_terms
    solver_arrays = (
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
        r_grid,
    )

    finite_state = all(
        np.all(np.isfinite(np.asarray(values)))
        for values in solver_arrays
    )
    diagnostic_alpha = np.asarray(schwarzschild_lapse(U_state))
    finite_state = finite_state and np.all(np.isfinite(diagnostic_alpha))
    minimum_alpha = float(np.min(diagnostic_alpha))

    return finite_state and minimum_alpha > 0.0


def write_schwarzschild_snapshot(
    solver_U_state,
    solver_particles,
    step,
    solver_time,
    schwarzschild_time,
):
    U_state, diagnostic_particles, _ = rescale_to_schwarzschild_coordinates(
        solver_U_state,
        solver_particles,
        total_star_mass,
    )
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state
    mass_density, charge_density, Srr, Sr = source_terms

    U_state_path = U_state_directory / f"U_state_step_{step:06d}.npz"
    np.savez_compressed(
        U_state_path,
        r=np.asarray(r_grid),
        A=np.asarray(A),
        phi=np.asarray(phi),
        alpha=np.asarray(alpha),
        Krr=np.asarray(Krr),
        beta_over_r=np.asarray(beta_over_r),
        Er=np.asarray(Er),
        mass_density=np.asarray(mass_density),
        charge_density=np.asarray(charge_density),
        Srr=np.asarray(Srr),
        Sr=np.asarray(Sr),
        step=int(step),
        time=float(schwarzschild_time),
        schwarzschild_time=float(schwarzschild_time),
        solver_time=float(solver_time),
    )

    phase_space_path = phase_space_directory / (
        f"phase_space_{diagnostic_particles.name}_step_{step:06d}.npz"
    )
    np.savez_compressed(
        phase_space_path,
        r=np.asarray(diagnostic_particles.r),
        ur=np.asarray(diagnostic_particles.ur),
        weight=np.asarray(diagnostic_particles.weight),
        step=int(step),
        time=float(schwarzschild_time),
        schwarzschild_time=float(schwarzschild_time),
        solver_time=float(solver_time),
        species_name=diagnostic_particles.name,
    )




solver_U_state = calculate_metric(particles, grid.r_full, grid.dr)
# solve for the initial metric for the particles

solver_time = 0.0
step = 0
dt = freefall_collapse_time_step(solver_U_state)
# define the initial time step based on the free fall time step of the system
schwarzschild_time = 0.0
_, X_t = schwarzschild_rescale_factors(
    solver_U_state,
    total_star_mass,
)
write_schwarzschild_snapshot(
    solver_U_state,
    particles,
    step,
    solver_time,
    schwarzschild_time,
)


with tqdm(
    total=run_time,
    initial=solver_time,
    desc="evolving stellar collapse",
    unit="t",
) as progress_bar:
    while solver_time < run_time:
        freefall_dt = freefall_collapse_time_step(solver_U_state)
        crossing_dt = particle_crossing_time_step(
            particles,
            solver_U_state,
            grid.dr,
        )
        trial_dt = min(freefall_dt, crossing_dt, run_time - solver_time)
        # limit the trial by collapse, particle crossing, and the remaining run time

        accepted = False
        while trial_dt >= MINIMUM_TRIAL_TIME_STEP:
            trial_particles = copy_particles(particles)
            trial_particles, trial_U_state = step_rk4_with_metric(
                trial_particles,
                solver_U_state,
                grid.r_full,
                grid.dr,
                trial_dt,
            )

            if solver_state_is_acceptable(trial_U_state):
                accepted = True
                break

            trial_dt *= 0.5

        if not accepted:
            print(
                "No finite positive Schwarzschild-lapse trial state found above "
                f"dt={MINIMUM_TRIAL_TIME_STEP:.3e}; preserving the last "
                f"accepted state at step {step}, solver time {solver_time:.3e}, "
                f"schwarzschild time {schwarzschild_time:.3e}."
            )
            break

        previous_solver_time = solver_time
        previous_X_t = float(X_t)
        particles = trial_particles
        solver_U_state = trial_U_state
        solver_time += trial_dt
        step += 1
        _, X_t = schwarzschild_rescale_factors(
            solver_U_state,
            total_star_mass,
        )
        schwarzschild_time += 0.5 * (
            previous_X_t + float(X_t)
        ) * trial_dt

        should_save = (
            step % save_every == 0
            or solver_time >= run_time
        )
        if should_save:
            write_schwarzschild_snapshot(
                solver_U_state,
                particles,
                step,
                solver_time,
                schwarzschild_time,
            )

        minimum_alpha = float(
            np.min(np.asarray(schwarzschild_lapse(solver_U_state)))
        )
        progress_bar.update(solver_time - previous_solver_time)
        progress_bar.set_postfix(
            step=step,
            dt=f"{trial_dt:.3e}",
            min_alpha=f"{minimum_alpha:.3e}",
        )
