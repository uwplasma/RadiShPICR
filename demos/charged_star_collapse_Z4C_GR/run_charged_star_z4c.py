"""Charged Oppenheimer-Snyder collapse in Z4C with a self-consistent E field.

A homogeneous dust sphere carrying uniform charge-to-mass ratio, seeded from
a completed-map constrained lapse. The production-grid Hamiltonian solve
fixes gravitational mass and charge through an RN exterior before evolution.

Snapshots hold raw state -- metric, E_r, charge density, particles -- in one
openPMD series.  Diagnostics are recomputed from it by the plotting scripts.
"""

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
)
from RadiShPICR.Z4C.electric_field import (
    compute_radial_charge_density,
    solve_radial_electric_field,
)
from RadiShPICR.Z4C.particle_boundaries import (
    INNER_AREAL_GHOST_CELLS,
    deleting_inner_areal_radius_boundary,
)
from RadiShPICR.Z4C.time_evolve import rk4_step
from RadiShPICR.Z4C.utils import freefall_collapse_time_step

OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/evolution_data"

MASS_SCALE = 1.0  # Initial gravitational mass for the seed and normalization.
CHARGE_TO_MASS = 0.2  # Signed Q_geom / seed rest mass, before mass calibration.
SURFACE_AREAL_RADIUS = 10.0
EPSILON_0 = 1.0  # G=c=epsilon_0=1, as in the constrained Heun solver.

R_MAX = 100.0
NUM_CELLS = 4000
PARTICLE_SHELL_COUNT = 300
PARTICLES_PER_SHELL = 300
SHAPE_MODE = "quadratic"  # "nearest", "linear", or "quadratic".
INITIAL_TOLERANCE = 1.0e-12
SHOOTING_MAX_ITERATIONS = 100

DURATION = 150.0  # Final Z4C evolution t/M, using the initial gravitational mass.
CFL = 0.2
MINIMUM_DT = 1.0e-7
SAVE_EVERY = 10
ZERO_SHIFT = 0  # 1 freezes the initial shift; 0 evolves it with Z4C.
KAPPA = 1.0
ETA = 2.0
NU = 0.02

rk4_step_jit = jax.jit(rk4_step, static_argnames=("particle_boundary",))


def build_initial_state():
    """Solve consistent charged-star data at the prescribed mass and charge."""

    grid = build_radial_grid(R_MAX, NUM_CELLS + 1)
    density = MASS_SCALE / ((4.0 / 3.0) * jnp.pi * SURFACE_AREAL_RADIUS**3)
    particles, _, rest_mass = initialize_oppenheimer_snyder_particles(
        grid, density, MASS_SCALE, SURFACE_AREAL_RADIUS,
        PARTICLE_SHELL_COUNT * PARTICLES_PER_SHELL, SHAPE_MODE,
        name="charged_dust",
    )
    particles.charges = jnp.asarray(CHARGE_TO_MASS * np.sqrt(4.0 * np.pi))

    # Keep the constrained lapse prescription as a gauge seed. The final
    # spatial metric and rest mass are solved below with the RN exterior.
    U_state = shoot_constrained_schwarzschild_initial_data(
        particles, grid, MASS_SCALE,
        initial_center_A(MASS_SCALE, SURFACE_AREAL_RADIUS),
        INITIAL_TOLERANCE, SHOOTING_MAX_ITERATIONS,
    )
    r = U_state[-1]
    z4c_r = 0.5 * (r[:-1] + r[1:])
    metric = constrained_state_to_z4c(U_state, z4c_r, KAPPA, ETA, NU)

    metric, particles, _ = solve_star_initial_data(
        particles, metric, MASS_SCALE, epsilon_0=EPSILON_0, tolerance=INITIAL_TOLERANCE,
    )
    rest_mass = float(jnp.sum(particles.get_mass()))

    return metric, particles, MASS_SCALE, float(rest_mass)


@jax.jit
def state_checks(metric, particles, charge_density, E_r):
    """Finite fields, positive geometry, and particles clear of the outer edge."""

    finite = jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value))
        for value in (*metric, particles.r, particles.ur, particles.phi,
                      particles.uphi, particles.weight, charge_density, E_r)
    ]))
    positive = jnp.all(jnp.stack((metric.alpha, metric.chi,
                                  metric.conformal_grr, metric.conformal_gt)) > 0.0)
    # Inner ghost crossings are handled by the OS boundary at each RK stage.
    inside_outer = jnp.all((particles.weight == 0.0)
                           | (particles.r < metric.r[-1] - 2.0 * metric.dr))

    return jnp.stack((finite, positive, inside_outer))


def run_simulation():
    if not (0.0 < 2.0 * MASS_SCALE < SURFACE_AREAL_RADIUS):
        raise ValueError(
            "Require 0 < 2*MASS_SCALE < SURFACE_AREAL_RADIUS for the OS seed profile."
        )
    if (NUM_CELLS < 8 or PARTICLES_PER_SHELL < 1 or SAVE_EVERY < 1
            or not np.isfinite(DURATION) or DURATION <= 0 or CFL <= 0
            or MINIMUM_DT <= 0 or R_MAX <= 0 or SHOOTING_MAX_ITERATIONS < 1
            or INITIAL_TOLERANCE <= 0):
        raise ValueError("Times and grid/output counts must be positive (NUM_CELLS >= 8).")
    if OUTPUT_DIRECTORY.exists() and any(OUTPUT_DIRECTORY.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output directory: {OUTPUT_DIRECTORY}")

    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()

    metric, particles, mass, rest_mass = build_initial_state()
    target_time = DURATION * mass
    initial_charge = float(jnp.sum(particles.get_charge()))
    if abs(initial_charge) / np.sqrt(4.0 * np.pi) >= mass:
        raise ValueError("The comparison requires subextremal RN initial data, |Q_geom| < M.")

    charge_density = compute_radial_charge_density(particles, metric, inner_open=True)
    E_r = solve_radial_electric_field(metric, charge_density, EPSILON_0)
    if not bool(jnp.all(state_checks(metric, particles, charge_density, E_r))):
        raise ValueError(
            "Initial state is invalid or its particle stencils reach the outer boundary."
        )

    metadata = dict(
        initial_metric="star_hamiltonian_completed_map", constraint_solved=True,
        mass_scale=mass, initial_gravitational_mass=mass, total_mass=mass,
        star_rest_mass=rest_mass, total_charge=initial_charge,
        initial_charge=initial_charge, normalization_charge=initial_charge,
        geometric_charge=initial_charge / np.sqrt(4.0 * np.pi),
        charge_to_mass=initial_charge / (np.sqrt(4.0 * np.pi) * rest_mass),
        seed_charge_to_mass=CHARGE_TO_MASS, epsilon_0=EPSILON_0,
        reference_mass=mass, reference_mass_source="initial_gravitational_mass",
        surface_areal_radius=SURFACE_AREAL_RADIUS,
        initial_tolerance=INITIAL_TOLERANCE,
        initial_solver="z4c_hamiltonian_completed_map",
        zero_shift=ZERO_SHIFT, kappa=KAPPA, eta=ETA, nu=NU,
        num_cells=NUM_CELLS, particles_per_shell=PARTICLES_PER_SHELL,
        shape_mode=SHAPE_MODE, particle_boundary="areal_inner_open",
        inner_particle_ghost_cells=INNER_AREAL_GHOST_CELLS,
        inner_particle_deposition="unrenormalized_shape_overlap",
        inner_particle_absorption="zero_physical_overlap_or_beyond_two_ghost_edge",
        charge_absorption="remove_particles_without_field_correction",
        electric_field_update="current_rk4_covariant_E_r",
        inner_open=True, metric_boundary="sommerfeld",
        time_coordinate="evolution", target_time=target_time,
        duration_over_mass=DURATION, cfl=CFL, particle_state_variables="r_ur",
    )

    step, evolution_time = 0, 0.0
    time_tolerance = 1.0e-12 * max(1.0, target_time)
    stop_reason = "target_time"
    failure_names = ("nonfinite_state", "nonpositive_geometry", "outer_particle_boundary")

    with BSSNFieldSeries(OUTPUT_DIRECTORY, metadata=metadata) as fields, \
            tqdm(total=DURATION, desc="charged collapse (Z4C time)", unit="t/M") as progress:
        fields.write(metric, 0, time=0.0, E_r=E_r,
                     charge_density=charge_density, particles=particles)

        while target_time - evolution_time > time_tolerance:
            remaining = target_time - evolution_time
            trial_dt = min(CFL * float(metric.dr),
                           freefall_collapse_time_step(particles, metric),
                           remaining)
            accepted = False
            endpoint_trial = trial_dt < MINIMUM_DT
            while trial_dt >= MINIMUM_DT or endpoint_trial:
                endpoint_trial = False
                trial = rk4_step_jit(
                    particles, metric, trial_dt, EM_on=True, GR_on=True, E_r=E_r,
                    epsilon_0=EPSILON_0, zero_shift=ZERO_SHIFT,
                    particle_boundary=deleting_inner_areal_radius_boundary,
                    inner_open=True,
                )
                checks = np.asarray(state_checks(trial[1], trial[0], trial[2], trial[3]))
                if not np.all(checks):
                    stop_reason = failure_names[int(np.flatnonzero(~checks)[0])]
                    trial_dt *= 0.5
                    continue
                accepted = True
                break
            if not accepted:
                break

            particles, metric, charge_density, E_r = trial
            evolution_time += trial_dt
            step += 1
            progress.update(trial_dt / mass)
            if step % SAVE_EVERY == 0:
                fields.write(metric, step, time=evolution_time, dt=trial_dt,
                             E_r=E_r, charge_density=charge_density,
                             particles=particles)

        completed = target_time - evolution_time <= time_tolerance
        if completed:
            evolution_time = target_time
            stop_reason = "target_time"
        fields.write(metric, step, time=evolution_time, E_r=E_r,
                     charge_density=charge_density, particles=particles)

        result = dict(
            completed=completed, stop_reason=stop_reason, final_step=step,
            final_time=evolution_time, final_time_over_mass=evolution_time / mass,
            total_particles=int(particles.r.size),
            active_particles=int(jnp.count_nonzero(particles.weight)),
            final_active_particle_weight=float(jnp.sum(particles.weight)),
            final_particle_charge=float(jnp.sum(particles.get_charge())),
            removed_particle_charge=initial_charge - float(jnp.sum(particles.get_charge())),
            minimum_lapse=float(jnp.min(metric.alpha)),
            minimum_chi=float(jnp.min(metric.chi)),
            wall_runtime_seconds=time.perf_counter() - started,
        )
        fields.set_metadata(**result)

    print(json.dumps({**metadata, **result}, indent=2))

    return result


if __name__ == "__main__":
    summary = run_simulation()
    if not summary["completed"]:
        raise SystemExit(
            "Evolution stopped; the last accepted state was preserved in the series."
        )
