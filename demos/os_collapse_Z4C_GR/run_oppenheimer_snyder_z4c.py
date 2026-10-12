"""Oppenheimer-Snyder dust collapse in Z4C.

A homogeneous, time-symmetric dust sphere seeded from a constrained Heun shoot
matched to Schwarzschild at the outer boundary, then evolved in Z4C evolution
time with an absorbing inner areal boundary.

Snapshots hold raw state -- metric and particles -- in one openPMD series.
Diagnostics are recomputed from it by the plotting scripts.
"""

from __future__ import annotations

import argparse
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
from RadiShPICR.ConstraintBasedRelativity.geodesic import isotropic_particle_radius
from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.initial_data import (
    constrained_state_to_z4c,
    initial_center_A,
    initialize_oppenheimer_snyder_particles,
    shoot_constrained_schwarzschild_initial_data,
)
from RadiShPICR.initial_data.oppenheimer_snyder import SHOOTING_METHOD
from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.particle_boundaries import (
    INNER_AREAL_GHOST_CELLS,
    deleting_inner_areal_radius_boundary,
)
from RadiShPICR.Z4C.time_evolve import rk4_step
from RadiShPICR.Z4C.utils import freefall_collapse_time_step

TOTAL_STAR_MASS = 1.0
SURFACE_AREAL_RADIUS = 10.0
DURATION = 150.0  # Final evolution t/M.

# Extend the 800-node spacing over 20M to 3995 Z4c cells over 100M.
REFERENCE_GRID_POINTS = 800
R_MAX = 100.0
NUM_Z4C_CELLS = 5 * (REFERENCE_GRID_POINTS - 1)
PARTICLE_SHELL_COUNT = 300
PARTICLES_PER_SHELL = 300

CFL = 0.2
MINIMUM_TRIAL_TIME_STEP = 1.0e-7
SAVE_EVERY = 10
SHOOTING_TOLERANCE = 1.0e-10
SHOOTING_MAX_ITERATIONS = 64
PARTICLE_DEPOSITION_SCHEMES = {
    "nearest": "ordinary_nearest",
    "linear": "ordinary_cic",
    "quadratic": "ruyten_density_conserving_quadratic_nonnegative",
}

KAPPA = 1.0
ETA = 2.0
NU = 0.02
ZERO_SHIFT = 0  # 1 freezes the initial shift; 0 evolves it with Z4C.

rk4_step_jit = jax.jit(rk4_step, static_argnames=("particle_boundary",))


def build_initial_state(r_max, num_z4c_cells, particles_per_shell, shape_mode,
                        shooting_tolerance, shooting_max_iterations):
    """Shoot a Schwarzschild-matched constrained OS slice and convert to Z4c."""

    mass_density = TOTAL_STAR_MASS / (
        (4.0 / 3.0) * jnp.pi * SURFACE_AREAL_RADIUS**3
    )
    grid = build_radial_grid(r_max, num_z4c_cells + 1)

    particles, _, total_rest_mass = initialize_oppenheimer_snyder_particles(
        grid, mass_density, TOTAL_STAR_MASS, SURFACE_AREAL_RADIUS,
        PARTICLE_SHELL_COUNT * particles_per_shell, shape_mode,
    )
    U_state = shoot_constrained_schwarzschild_initial_data(
        particles, grid, TOTAL_STAR_MASS,
        initial_center_A(TOTAL_STAR_MASS, SURFACE_AREAL_RADIUS),
        shooting_tolerance, shooting_max_iterations,
    )

    A, _, _, _, _, _, _, constrained_r = U_state
    z4c_r = 0.5 * (constrained_r[:-1] + constrained_r[1:])
    metric = constrained_state_to_z4c(U_state, z4c_r, KAPPA, ETA, NU)

    # The constrained chart stores areal radius and u_r / A; Z4C wants
    # isotropic radius and the covariant u_r.
    particle_r = isotropic_particle_radius(particles, U_state)
    particles = particle_species(
        name=particles.name, charge=particles.charges, mass=particles.masses,
        weight=particles.weight, r=particle_r,
        ur=jnp.interp(particle_r, constrained_r, A) * particles.ur,
        phi=particles.phi, uphi=particles.uphi, shape_mode=particles.shape_mode,
    )

    return metric, particles, float(total_rest_mass)


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


def run_simulation(args):
    wall_start = time.perf_counter()
    metric, particles, total_rest_mass = build_initial_state(
        args.r_max, args.num_cells, args.particles_per_shell, args.shape_mode,
        args.shooting_tolerance, args.shooting_max_iterations,
    )

    target_time = args.duration * TOTAL_STAR_MASS
    initial_particle_weight = float(np.sum(np.asarray(particles.weight)))
    grr = metric.conformal_grr / metric.chi
    gT = metric.conformal_gt / metric.chi
    metadata = dict(
        total_mass=TOTAL_STAR_MASS, total_rest_mass=total_rest_mass,
        total_effective_mass=float(np.sum(np.asarray(particles.get_mass()))),
        surface_areal_radius=SURFACE_AREAL_RADIUS,
        initial_isotropy_error=float(np.max(np.abs(np.asarray(grr - gT)))),
        initial_metric="os_heun_schwarzschild_matched",
        shooting_method=SHOOTING_METHOD,
        shooting_tolerance=args.shooting_tolerance,
        shooting_max_iterations=args.shooting_max_iterations,
        zero_shift=ZERO_SHIFT, kappa=KAPPA, eta=ETA, nu=NU,
        num_cells=args.num_cells, particles_per_shell=args.particles_per_shell,
        shape_mode=args.shape_mode, particle_state_variables="r_ur",
        particle_boundary="areal_inner_open",
        particle_inner_ghost_cells=INNER_AREAL_GHOST_CELLS,
        particle_deposition_boundary="unrenormalized_shape_overlap",
        particle_deposition_scheme=PARTICLE_DEPOSITION_SCHEMES[args.shape_mode],
        particle_absorption_rule="zero_physical_overlap_or_beyond_two_ghost_edge",
        metric_boundary="sommerfeld", time_coordinate="evolution",
        target_time=target_time, duration_over_mass=args.duration, cfl=args.cfl,
        dr=float(metric.dr),
    )

    step, evolution_time = 0, 0.0
    time_tolerance = 1.0e-10 * TOTAL_STAR_MASS
    maximum_steps = math.inf if args.max_steps is None else args.max_steps
    stop_reason = "target_time"

    with BSSNFieldSeries(args.output_directory, metadata=metadata) as fields, \
            tqdm(total=args.duration, desc="evolving Z4c stellar collapse",
                 unit="t/M") as progress:
        fields.write(metric, 0, time=0.0, particles=particles)

        while target_time - evolution_time > time_tolerance and step < maximum_steps:
            remaining = target_time - evolution_time
            trial_dt = min(args.cfl * float(metric.dr),
                           freefall_collapse_time_step(particles, metric),
                           remaining)
            accepted = False
            endpoint_trial = trial_dt < args.minimum_dt
            while trial_dt >= args.minimum_dt or endpoint_trial:
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
                    f"dt={args.minimum_dt:.3e}; preserving step {step} at "
                    f"evolution time {evolution_time:.8e}."
                )
                break

            metric, particles = trial_metric, trial_particles
            evolution_time += trial_dt
            step += 1
            if target_time - evolution_time <= time_tolerance:
                evolution_time = target_time

            if step % args.save_every == 0:
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


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-max", type=float, default=R_MAX)
    parser.add_argument("--num-cells", type=int, default=NUM_Z4C_CELLS)
    parser.add_argument("--particles-per-shell", type=int, default=PARTICLES_PER_SHELL)
    parser.add_argument("--shape-mode", choices=tuple(PARTICLE_DEPOSITION_SCHEMES),
                        default="quadratic")
    parser.add_argument("--duration", type=float, default=DURATION,
                        help="Final evolution time in stellar masses (default: 150).")
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--minimum-dt", type=float, default=MINIMUM_TRIAL_TIME_STEP)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--shooting-tolerance", type=float, default=SHOOTING_TOLERANCE)
    parser.add_argument("--shooting-max-iterations", type=int,
                        default=SHOOTING_MAX_ITERATIONS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-directory", type=Path,
                        default=DEMO_DIRECTORY / "outputs/z4c_oppenheimer_snyder")
    args = parser.parse_args()

    args.save_every = max(1, args.save_every)
    if not np.isfinite(args.shooting_tolerance) or args.shooting_tolerance <= 0.0:
        parser.error("shooting-tolerance must be finite and positive")
    if args.shooting_max_iterations < 1:
        parser.error("shooting-max-iterations must be at least one")
    if args.duration <= 0.0 or args.cfl <= 0.0:
        parser.error("duration and cfl must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output directory: {args.output_directory}")

    return args


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    summary = run_simulation(parse_arguments())
    if not summary["completed"] and summary["stop_reason"] != "max_steps":
        raise SystemExit(
            "Evolution stopped; the last accepted state was preserved in the series."
        )
