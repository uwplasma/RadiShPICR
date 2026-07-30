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

SHAPE_FACTOR = 0
# particle shape factor
FREE_FALL_FRACTION = 0.1
# fraction of free fall time step to use for the simulation
CROSSING_FRACTION = 0.25
# fraction of one radial cell that particles may cross in one time step
MINIMUM_TRIAL_TIME_STEP = 1.0e-6
# stop when no finite positive-lapse trial can be found above this time step
SAVE_EVERY = 1
# save every completed step by default for compatibility

total_star_mass = 1.0
run_time        = 50.0 / total_star_mass # 50 units of M
surface_areal_radius = 10.0
number_density = total_star_mass / (4/3 * jnp.pi * surface_areal_radius**3)
# define the mass of the ball of dust and the outer radius of the ball of dust
ppc = 50
# define the number of particles per cell
Nr    = 500
r_max = 20.0
dr    = r_max / Nr
Nr_particles = int( surface_areal_radius / dr )
# define the number of grid points within the sphere and the maximum radius of the grid
total_particles = Nr_particles * ppc
# define the total number of particles in the simulation
surface_range = jnp.linspace(0.0, surface_areal_radius, Nr_particles)
# define the radial grid within the sphere of dust
shell_volumes = 4/3 * jnp.pi * (surface_range**3)
shell_volumes = jnp.diff(jnp.concatenate([jnp.array([0.0]), shell_volumes]))
# define the volume of each shell in the radial grid
# each shell volume will have ppc particles.
per_particle_volume = shell_volumes / ppc
# define the partition of volume of each particle based on the shell volume and the number of particles per cell
particle_volume = jnp.repeat(per_particle_volume, ppc)
# define the volume of each particle by repeating the per particle volume for each particle per cell
particle_mass = jnp.ones((total_particles,))
# define a uniform mass for each micro-particle
particle_weight = particle_volume * number_density
# define the weight of each macro-particle based on the particle volume and the number density
particle_solver_radius = jnp.repeat(surface_range, ppc)
# define the solver radius for each particle by repeating the surface range for each particle per cell
particle_ur = jnp.zeros((total_particles,))
particle_phi = jnp.zeros((total_particles,))
particle_uphi = jnp.zeros((total_particles,))
# define the particle velocities and positions

if SHAPE_FACTOR == 0:
    shape_mode = "nearest"
elif SHAPE_FACTOR == 1:
    shape_mode = "linear"
else:
    shape_mode = "quadratic"

particles = particle_species(
    name="oppenheimer_snyder_dust",
    charge=0.0,
    mass=jnp.asarray(particle_mass),
    weight=particle_weight,
    r=jnp.asarray(particle_solver_radius),
    ur=particle_ur,
    phi=particle_phi,
    uphi=particle_uphi,
    shape_mode=shape_mode,
)

# define the initial particle species

parser = argparse.ArgumentParser()
parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
args = parser.parse_args()
save_every = max(1, args.save_every)

grid = build_radial_grid( r_max, Nr )
# define the radial grid

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
    minimum_alpha = float(np.min(np.asarray(alpha)))

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
                "No finite positive-lapse trial state found above "
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
            np.min(np.asarray(solver_U_state[2]))
        )
        progress_bar.update(solver_time - previous_solver_time)
        progress_bar.set_postfix(
            step=step,
            dt=f"{trial_dt:.3e}",
            min_alpha=f"{minimum_alpha:.3e}",
        )
