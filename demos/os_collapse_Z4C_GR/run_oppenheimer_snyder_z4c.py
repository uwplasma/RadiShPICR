"""Oppenheimer-Snyder dust collapse in Z4C.

A homogeneous, time-symmetric dust sphere satisfying the production Z4C
Hamiltonian constraint at prescribed gravitational mass, evolved with an
absorbing inner areal boundary. The constrained shoot supplies the lapse.

Snapshots hold raw state -- metric and particles -- in one openPMD series.
Diagnostics are recomputed from it by the plotting scripts.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import jax
import jax.numpy as jnp
from tqdm import tqdm

from RadiShPICR.ConstraintBasedRelativity import build_radial_grid
from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.initial_data import (
    constrained_state_to_z4c,
    initial_center_A,
    initialize_oppenheimer_snyder_particles,
    shoot_constrained_schwarzschild_initial_data,
    solve_star_initial_data,
    surface_isotropic_radius,
)
from RadiShPICR.initial_data.oppenheimer_snyder import SHOOTING_METHOD
from RadiShPICR.Z4C.particle_boundaries import (
    INNER_AREAL_GHOST_CELLS,
    deleting_inner_areal_radius_boundary,
)
from RadiShPICR.Z4C.time_evolve import rk4_step
from RadiShPICR.Z4C.utils import freefall_collapse_time_step

TOTAL_STAR_MASS = 1.0
SURFACE_AREAL_RADIUS = 10.0
DURATION = 150.0  # Final evolution t/M.

Nr = 4000       # Number of Z4C radial cells.
PPC = 300       # Particles per occupied initial deposition shell.
R_WIND = 100.0  # Outer radial coordinate.
SHAPE_MODE = "ruyten_density_conserving_quadratic_nonnegative"
MAX_STEPS = None
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/evolution_data"

CFL = 0.2
MINIMUM_TRIAL_TIME_STEP = 1.0e-7
SAVE_EVERY = 10
SHOOTING_TOLERANCE = 1.0e-12
SHOOTING_MAX_ITERATIONS = 100

KAPPA = 0.1
ETA = 2.0
NU = 0.02
ZERO_SHIFT = 0  # 1 freezes the initial shift; 0 evolves it with Z4C.

rk4_step_jit = jax.jit(rk4_step, static_argnames=("particle_boundary",))


def build_initial_state():
    """Solve the OS constraints with the production Z4C particle sources."""

    mass_density = TOTAL_STAR_MASS / (
        (4.0 / 3.0) * jnp.pi * SURFACE_AREAL_RADIUS**3
    )
    grid = build_radial_grid(R_WIND, Nr + 1)

    # Match the initializer's nodal deposition shells, including the partial
    # surface shell, in the initial isotropic chart (X_r = 1).
    surface_r = surface_isotropic_radius(SURFACE_AREAL_RADIUS, TOTAL_STAR_MASS)
    grid_indices = np.arange(grid.r_full.size)
    shell_inner = np.maximum((grid_indices - 0.5) * grid.dr, 0.0)
    shell_outer = np.minimum((grid_indices + 0.5) * grid.dr, surface_r)
    occupied_shell_count = int(np.count_nonzero(shell_outer > shell_inner))

    particles, _, _ = initialize_oppenheimer_snyder_particles(
        grid, mass_density, TOTAL_STAR_MASS, SURFACE_AREAL_RADIUS,
        occupied_shell_count * PPC, SHAPE_MODE,
    )
    U_state = shoot_constrained_schwarzschild_initial_data(
        particles, grid, TOTAL_STAR_MASS,
        initial_center_A(TOTAL_STAR_MASS, SURFACE_AREAL_RADIUS),
        SHOOTING_TOLERANCE, SHOOTING_MAX_ITERATIONS,
    )

    constrained_r = U_state[-1]
    z4c_r = 0.5 * (constrained_r[:-1] + constrained_r[1:])
    metric = constrained_state_to_z4c(U_state, z4c_r, KAPPA, ETA, NU)

    # The Hamiltonian solve maps the original areal particles onto the
    # completed Z4C geometry and calibrates rest mass to the prescribed M.
    metric, particles, _ = solve_star_initial_data(
        particles, metric, TOTAL_STAR_MASS, tolerance=SHOOTING_TOLERANCE,
    )
    rest_mass = float(jnp.sum(particles.get_mass()))

    return metric, particles, float(rest_mass)


@jax.jit
def state_is_acceptable(metric, particles):
    """Finite fields everywhere and positive lapse, conformal factor, geometry."""

    finite = jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value))
        for value in (*metric, particles.r, particles.ur, particles.phi,
                      particles.uphi, particles.weight)
    ]))
    positive = jnp.all(jnp.stack((metric.alpha, metric.chi,
                                  metric.conformal_grr, metric.conformal_gt)) > 0.0)

    return finite & positive


def run_simulation():
    if OUTPUT_DIRECTORY.exists() and any(OUTPUT_DIRECTORY.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty output directory: {OUTPUT_DIRECTORY}"
        )

    wall_start = time.perf_counter()
    metric, particles, total_rest_mass = build_initial_state()

    target_time = DURATION * TOTAL_STAR_MASS
    initial_particle_weight = float(np.sum(np.asarray(particles.weight)))
    grr = metric.conformal_grr / metric.chi
    gT = metric.conformal_gt / metric.chi
    metadata = dict(
        total_mass=TOTAL_STAR_MASS, total_rest_mass=total_rest_mass,
        total_effective_mass=float(np.sum(np.asarray(particles.get_mass()))),
        surface_areal_radius=SURFACE_AREAL_RADIUS,
        initial_isotropy_error=float(np.max(np.abs(np.asarray(grr - gT)))),
        initial_metric="star_hamiltonian_completed_map",
        initial_solver="z4c_hamiltonian_completed_map",
        total_charge=0.0, initial_gravitational_mass=TOTAL_STAR_MASS,
        shooting_method=SHOOTING_METHOD,
        shooting_tolerance=SHOOTING_TOLERANCE,
        shooting_max_iterations=SHOOTING_MAX_ITERATIONS,
        zero_shift=ZERO_SHIFT, kappa=KAPPA, eta=ETA, nu=NU,
        num_cells=Nr, particles_per_shell=PPC, r_max=R_WIND,
        occupied_shell_count=int(particles.r.size // PPC),
        shape_mode=SHAPE_MODE, particle_state_variables="r_ur",
        particle_boundary="areal_inner_open",
        particle_inner_ghost_cells=INNER_AREAL_GHOST_CELLS,
        particle_deposition_boundary="unrenormalized_shape_overlap",
        particle_deposition_scheme=SHAPE_MODE,
        particle_absorption_rule="zero_physical_overlap_or_beyond_two_ghost_edge",
        metric_boundary="sommerfeld", time_coordinate="evolution",
        target_time=target_time, duration_over_mass=DURATION, cfl=CFL,
        dr=float(metric.dr),
    )

    step, evolution_time = 0, 0.0
    time_tolerance = 1.0e-10 * TOTAL_STAR_MASS
    maximum_steps = math.inf if MAX_STEPS is None else MAX_STEPS
    stop_reason = "target_time"

    with BSSNFieldSeries(OUTPUT_DIRECTORY, metadata=metadata) as fields, \
            tqdm(total=DURATION, desc="evolving Z4c stellar collapse",
                 unit="t/M") as progress:
        fields.write(metric, 0, time=0.0, particles=particles)

        while target_time - evolution_time > time_tolerance and step < maximum_steps:
            remaining = target_time - evolution_time
            trial_dt = min(CFL * float(metric.dr),
                           freefall_collapse_time_step(particles, metric),
                           remaining)
            accepted = False
            endpoint_trial = trial_dt < MINIMUM_TRIAL_TIME_STEP
            while trial_dt >= MINIMUM_TRIAL_TIME_STEP or endpoint_trial:
                endpoint_trial = False
                trial_particles, trial_metric, _, _ = rk4_step_jit(
                    particles, metric, trial_dt,
                    E_r=jnp.zeros_like(metric.r), EM_on=False, GR_on=True,
                    zero_shift=ZERO_SHIFT,
                    particle_boundary=deleting_inner_areal_radius_boundary,
                    inner_open=True,
                )
                if not state_is_acceptable(trial_metric, trial_particles):
                    trial_dt *= 0.5
                    continue
                accepted = True
                break

            if not accepted:
                stop_reason = "no_acceptable_trial_above_minimum_dt"
                print(
                    "No finite positive-lapse Z4c trial state found above "
                    f"dt={MINIMUM_TRIAL_TIME_STEP:.3e}; preserving step {step} at "
                    f"evolution time {evolution_time:.8e}."
                )
                break

            metric, particles = trial_metric, trial_particles
            evolution_time += trial_dt
            step += 1
            if target_time - evolution_time <= time_tolerance:
                evolution_time = target_time

            if step % SAVE_EVERY == 0:
                fields.write(metric, step, time=evolution_time, dt=trial_dt,
                             particles=particles)

            progress.update(trial_dt / TOTAL_STAR_MASS)
            progress.set_postfix(
                step=step, dt=f"{trial_dt:.3e}",
                min_alpha=f"{float(np.min(np.asarray(metric.alpha))):.3e}",
            )

        completed = target_time - evolution_time <= time_tolerance
        if not completed and step >= maximum_steps:
            stop_reason = "max_steps"
        fields.write(metric, step, time=evolution_time, particles=particles)

        final_weight = float(np.sum(np.asarray(particles.weight)))
        result = dict(
            completed=bool(completed), stop_reason=stop_reason,
            final_step=int(step), final_time=float(evolution_time),
            final_time_over_mass=float(evolution_time / TOTAL_STAR_MASS),
            minimum_lapse=float(np.min(np.asarray(metric.alpha))),
            minimum_chi=float(np.min(np.asarray(metric.chi))),
            maximum_chi=float(np.max(np.asarray(metric.chi))),
            active_particles=int(np.count_nonzero(np.asarray(particles.weight))),
            total_particles=int(particles.r.size),
            initial_particle_weight=initial_particle_weight,
            final_active_particle_weight=final_weight,
            deleted_particle_weight=initial_particle_weight - final_weight,
            wall_runtime_seconds=float(time.perf_counter() - wall_start),
        )
        fields.set_metadata(**result)

    print(json.dumps({**metadata, **result}, indent=2))

    return result


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    summary = run_simulation()
    if not summary["completed"] and summary["stop_reason"] != "max_steps":
        raise SystemExit(
            "Evolution stopped; the last accepted state was preserved in the series."
        )
