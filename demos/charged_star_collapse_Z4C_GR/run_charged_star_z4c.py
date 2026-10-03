"""Charged Heun-initialized stellar collapse in the OS Schwarzschild chart."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import jax
import jax.numpy as jnp

from demos.os_collapse_Z4C_GR import run_oppenheimer_snyder_z4c as os_collapse
from RadiShPICR.ConstraintBasedRelativity.geodesic import isotropic_particle_radius
from RadiShPICR.Z4C.particle_boundaries import deleting_inner_areal_radius_boundary
from RadiShPICR.ConstraintBasedRelativity import build_radial_grid
from RadiShPICR.ConstraintBasedRelativity.vacuum_conditions import vacuum_rescale_factors
from RadiShPICR.Z4C.curvature_invariants import misner_sharp_mass
from RadiShPICR.Z4C.electric_field import (
    compute_radial_charge_density, solve_radial_electric_field,
    compute_electrostatic_matter_terms, radial_gauss_residual,
)
from RadiShPICR.Z4C.energy_momentum_tensor import compute_radial_matter_terms, compute_hamiltonian_constraint
from RadiShPICR.Z4C.geodesic import _radial_grid_from_metric
from RadiShPICR.Z4C.time_evolve import rk4_step
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    _interpolate_cell_centered_fields_to_particles,
)

SNAPSHOT_NAMES = {
    "alpha": "alpha", "beta": "beta", "chi": "chi",
    "conformal_grr": "conformal_grr", "conformal_gt": "conformal_gT",
    "Kh": "Kh", "Arr": "Arr", "At": "AT", "theta": "theta", "Gamma": "Gamma",
}
rk4_step_jit = jax.jit(rk4_step, static_argnames=("particle_boundary",))


def build_initial_state(args):
    """Use the OS Heun shoot with charged sources and Schwarzschild matching."""
    grid = build_radial_grid(args.r_max, args.num_cells + 1)
    density = args.mass_scale / ((4.0 / 3.0) * jnp.pi * args.surface_areal_radius**3)
    particles, particle_R, rest_mass = os_collapse.initialize_oppenheimer_snyder_particles(
        grid, density, args.mass_scale, args.surface_areal_radius,
        os_collapse.PARTICLE_SHELL_COUNT * args.particles_per_shell, args.shape_mode,
    )
    particles.name = "charged_dust"
    particles.charges = jnp.asarray(args.charge_to_mass * np.sqrt(4.0 * np.pi))
    charge = float(jnp.sum(particles.get_charge()))

    # The Heun equations include E_r, its energy, and radial tension. Only the
    # outer coordinate normalization is Schwarzschild (charge argument zero).
    U_state = os_collapse.shoot_constrained_schwarzschild_initial_data(
        particles, grid, args.mass_scale,
        os_collapse.initial_center_A(args.mass_scale, args.surface_areal_radius),
        args.initial_tolerance, args.shooting_max_iterations,
    )
    A, _, alpha, _, _, E_r, _, r = U_state
    z4c_r = 0.5 * (r[:-1] + r[1:])
    metric = os_collapse.constrained_state_to_z4c(
        U_state, z4c_r, args.kappa, args.eta, args.nu,
    )
    particle_r = isotropic_particle_radius(particles, U_state)
    particles = particle_species(
        name=particles.name, charge=particles.charges, mass=particles.masses,
        weight=particles.weight, r=particle_r,
        ur=jnp.interp(particle_r, r, A) * particles.ur,
        phi=particles.phi, uphi=particles.uphi, shape_mode=particles.shape_mode,
    )
    X_r, X_t = vacuum_rescale_factors(A[-1], alpha[-1], r[-1], args.mass_scale, 0.0)
    # In the constrained isotropic metric Q_flux = 4 pi r^2 A E_r.
    charge_flux = float(4.0 * jnp.pi * r[-1]**2 * A[-1] * E_r[-1])
    initial_data = dict(
        initial_solver="heun_" + os_collapse.SHOOTING_METHOD,
        initial_X_r=float(X_r), initial_X_t=float(X_t),
        initial_outer_charge_flux=charge_flux,
        initial_charge_flux_residual=charge_flux - charge,
        constrained_r=np.asarray(r), constrained_A=np.asarray(A),
        constrained_alpha=np.asarray(alpha), constrained_E_r=np.asarray(E_r),
        initial_particle_areal_radius=np.asarray(particle_R),
    )
    return metric, particles, initial_data, args.mass_scale, rest_mass


@jax.jit
def snapshot_fields(metric, particles, E_r, epsilon_0):
    charge_density = compute_radial_charge_density(particles, metric, inner_open=True)
    matter = compute_radial_matter_terms(particles, metric, inner_open=True)
    field_matter = compute_electrostatic_matter_terms(metric, E_r, epsilon_0)
    volume = proper_radial_shell_volume(metric)
    shell_charge = charge_density * volume
    enclosed_charge = jnp.cumsum(shell_charge) - 0.5 * shell_charge
    grr = metric.conformal_grr / metric.chi
    gT = metric.conformal_gt / metric.chi
    R = metric.r * jnp.sqrt(gT)
    mass_ms = misner_sharp_mass(metric)
    gauss_residual = radial_gauss_residual(metric, E_r, charge_density, epsilon_0)
    return dict(
        grr=grr, gT=gT, areal_radius=R,
        g_tt=-metric.alpha**2 + grr * metric.beta**2,
        g_tr=grr * metric.beta, g_theta_theta=metric.r**2 * gT,
        mass_density=matter.rho, charge_density=charge_density, E_r=E_r,
        em_energy_density=field_matter.rho,
        Srr=matter.Srr + field_matter.Srr, ST=matter.Stt + field_matter.Stt,
        Sr=matter.Sr, enclosed_charge=enclosed_charge,
        deposited_charge=jnp.sum(shell_charge), electric_field_energy=jnp.sum(field_matter.rho * volume),
        truncated_shape_charge=jnp.sum(particles.get_charge()) - jnp.sum(shell_charge),
        gauss_residual=gauss_residual,
        gauss_residual_linf=jnp.max(jnp.abs(gauss_residual)),
        gauss_residual_rms=jnp.sqrt(jnp.mean(gauss_residual**2)),
        misner_sharp_mass=mass_ms, areal_gradient_norm=1.0 - 2.0 * mass_ms / R,
        rn_mass=mass_ms + enclosed_charge**2 / (8.0 * jnp.pi * epsilon_0 * R),
    )


@jax.jit
def state_checks(metric, particles, charge_density, E_r):
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


@jax.jit
def state_is_acceptable(metric, particles, charge_density, E_r):
    return jnp.all(state_checks(metric, particles, charge_density, E_r))


def write_snapshot(metric, particles, E_r, output_directory, step, evolution_time, schwarzschild_time, args):
    metric, particles, X_r, X_t, mismatch = os_collapse.rescale_z4c_to_schwarzschild_coordinates(
        metric, particles, args.reference_mass,
    )
    fields = {key: np.asarray(getattr(metric, name)) for name, key in SNAPSHOT_NAMES.items()}
    # E_r is a spatial covector, so r' = X_r*r implies E'_r = E_r/X_r.
    diagnostics = {key: np.asarray(value) for key, value in snapshot_fields(metric, particles, E_r / X_r, args.epsilon_0).items()}
    diagnostics["removed_particle_charge"] = args.initial_charge - float(jnp.sum(particles.get_charge()))
    common = dict(
        step=step, time=schwarzschild_time, schwarzschild_time=schwarzschild_time,
        evolution_time=evolution_time, saved_coordinates="schwarzschild_isotropic_diagnostic",
        reference_mass=args.reference_mass, total_mass=args.reference_mass,
        epsilon_0=args.epsilon_0, X_r=float(X_r), X_t=float(X_t),
        normalization_charge=0.0, outer_metric_mismatch=float(mismatch),
        initial_charge=args.initial_charge,
        total_charge=float(jnp.sum(particles.get_charge())),
        active_particle_charge=float(jnp.sum(particles.get_charge())),
        active_particles=int(jnp.count_nonzero(particles.weight)),
    )
    np.savez_compressed(
        output_directory / "metric" / f"metric_step_{step:06d}.npz",
        r=np.asarray(metric.r), **fields, **diagnostics, **common,
    )
    chi_p, grr_p, gt_p = _interpolate_cell_centered_fields_to_particles(
        jnp.stack((metric.chi, metric.conformal_grr, metric.conformal_gt)),
        particles.r, _radial_grid_from_metric(metric), shape_mode=particles.get_shape(),
        field_parities=jnp.asarray((1, 1, 1)),
    )
    np.savez_compressed(
        output_directory / "phase_space" / f"phase_space_charged_dust_step_{step:06d}.npz",
        r=np.asarray(particles.r), ur=np.asarray(particles.ur), phi=np.asarray(particles.phi),
        uphi=np.asarray(particles.uphi), weight=np.asarray(particles.weight),
        mass=np.asarray(particles.masses), charge=np.asarray(particles.charges),
        areal_radius=np.asarray(particles.r * jnp.sqrt(gt_p / chi_p)),
        radial_orthonormal_momentum=np.asarray(particles.ur * jnp.sqrt(chi_p / grr_p)),
        species_name=particles.name, particle_state_variables="r_ur", **common,
    )
    return diagnostics


def run_simulation(args):
    jax.config.update("jax_enable_x64", True)
    start = time.perf_counter()
    output = args.output_directory
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output directory: {output}")
    metric, particles, initial_data, mass, rest_mass = build_initial_state(args)
    args.reference_mass = mass
    args.initial_charge = float(jnp.sum(particles.get_charge()))
    if abs(args.initial_charge) / np.sqrt(4.0 * np.pi) >= mass:
        raise ValueError("The comparison requires subextremal RN initial data, |Q_geom| < M.")
    charge_density = compute_radial_charge_density(particles, metric, inner_open=True)
    E_r = solve_radial_electric_field(metric, charge_density, args.epsilon_0)
    initial_raw = snapshot_fields(metric, particles, E_r, args.epsilon_0)
    if not bool(state_is_acceptable(metric, particles, initial_raw["charge_density"], initial_raw["E_r"])):
        raise ValueError("Initial state is invalid or its particle stencils reach the outer boundary.")
    (output / "metric").mkdir(parents=True, exist_ok=True)
    (output / "phase_space").mkdir(exist_ok=True)
    hamiltonian = np.asarray(compute_hamiltonian_constraint(metric)) - 16.0 * np.pi * np.asarray(
        initial_raw["mass_density"] + initial_raw["em_energy_density"]
    )
    metadata = dict(
        initial_metric="charged_heun_schwarzschild_matched", constraint_solved=True,
        mass_scale=mass, initial_gravitational_mass=mass, total_mass=mass,
        star_rest_mass=float(rest_mass), total_charge=args.initial_charge,
        initial_charge=args.initial_charge, normalization_charge=0.0,
        geometric_charge=args.initial_charge / np.sqrt(4.0 * np.pi),
        charge_to_mass=args.charge_to_mass, epsilon_0=args.epsilon_0,
        reference_mass=mass, reference_mass_source="initial_gravitational_mass",
        surface_areal_radius=args.surface_areal_radius, initial_tolerance=args.initial_tolerance,
        initial_solver=initial_data["initial_solver"],
        initial_X_r=initial_data["initial_X_r"], initial_X_t=initial_data["initial_X_t"],
        initial_charge_flux_residual=initial_data["initial_charge_flux_residual"],
        initial_hamiltonian_linf=float(np.max(np.abs(hamiltonian))),
        initial_hamiltonian_rms=float(np.sqrt(np.mean(hamiltonian**2))),
        zero_shift=args.zero_shift, kappa=args.kappa, eta=args.eta, nu=args.nu,
        num_cells=args.num_cells, particles_per_shell=args.particles_per_shell,
        shape_mode=args.shape_mode, particle_boundary="areal_inner_open",
        inner_particle_ghost_cells=os_collapse.INNER_AREAL_GHOST_CELLS,
        inner_particle_deposition=os_collapse.INNER_PARTICLE_DEPOSITION,
        inner_particle_absorption=os_collapse.INNER_PARTICLE_ABSORPTION,
        charge_absorption="remove_particles_without_field_correction",
        electric_field_update="current_rk4_covariant_E_r",
        gauss_diagnostic="differential_epsilon_0_div_E_minus_rho_q",
        inner_open=True, metric_boundary="sommerfeld",
        saved_coordinates="schwarzschild_isotropic_diagnostic",
        time_coordinate="schwarzschild", target_time=args.target_time,
        cfl=args.cfl, particle_state_variables="r_ur",
    )
    initial = write_snapshot(metric, particles, E_r, output, 0, 0.0, 0.0, args)
    metadata["initial_gauss_residual_linf"] = float(initial["gauss_residual_linf"])
    metadata["initial_gauss_residual_rms"] = float(initial["gauss_residual_rms"])
    np.savez_compressed(
        output / "initial_data_metadata.npz",
        **{**metadata, **initial_data}, z4c_r=np.asarray(metric.r),
        initial_hamiltonian_constraint=hamiltonian,
        initial_deposited_charge=float(initial["deposited_charge"]),
        initial_E_r=np.asarray(E_r),
    )
    summary_path = output / "run_summary.json"
    summary_path.write_text(json.dumps({**metadata, "completed": False, "final_step": 0,
                                      "final_time": 0.0, "final_evolution_time": 0.0}, indent=2) + "\n")
    step, evolution_time, schwarzschild_time = 0, 0.0, 0.0
    _, X_t, _ = os_collapse.schwarzschild_rescale_factors_from_z4c(metric, mass)
    X_t = float(X_t)
    maximum_steps = math.inf if args.max_steps is None else args.max_steps
    time_tolerance = 1.0e-12 * max(1.0, args.target_time)
    stop_reason = "target_time"
    failure_names = ("nonfinite_state", "nonpositive_geometry", "outer_particle_boundary")
    with tqdm(total=args.target_time, desc="charged collapse (Schwarzschild time)", unit="t") as progress:
        while args.target_time - schwarzschild_time > time_tolerance and step < maximum_steps:
            remaining = args.target_time - schwarzschild_time
            trial_dt = min(args.cfl * float(metric.dr),
                           os_collapse.freefall_collapse_time_step(particles, metric),
                           remaining / X_t)
            accepted = False
            endpoint_trial = trial_dt < args.minimum_dt
            while trial_dt >= args.minimum_dt or endpoint_trial:
                endpoint_trial = False
                trial_particles, trial_metric, trial_rho_q, trial_E = rk4_step_jit(
                    particles, metric, trial_dt, EM_on=True, GR_on=True,
                    E_r=E_r,
                    epsilon_0=args.epsilon_0, zero_shift=args.zero_shift,
                    particle_boundary=deleting_inner_areal_radius_boundary, inner_open=True,
                )
                checks = np.asarray(state_checks(trial_metric, trial_particles, trial_rho_q, trial_E))
                if not np.all(checks):
                    stop_reason = failure_names[int(np.flatnonzero(~checks)[0])] + "_at_minimum_dt"
                    trial_dt *= 0.5
                    continue
                trial_X_r, trial_X_t, _ = os_collapse.schwarzschild_rescale_factors_from_z4c(trial_metric, mass)
                trial_X_t = float(trial_X_t)
                if not (np.isfinite(float(trial_X_r)) and float(trial_X_r) > 0.0
                        and np.isfinite(trial_X_t) and trial_X_t > 0.0):
                    stop_reason = "invalid_schwarzschild_normalization_at_minimum_dt"
                    trial_dt *= 0.5
                    continue
                schwarzschild_dt = 0.5 * (X_t + trial_X_t) * trial_dt
                if schwarzschild_dt > remaining + time_tolerance:
                    trial_dt *= remaining / schwarzschild_dt
                    endpoint_trial = trial_dt < args.minimum_dt
                    continue
                accepted = True
                break
            if not accepted:
                break
            metric, particles, E_r = trial_metric, trial_particles, trial_E
            evolution_time += trial_dt
            schwarzschild_time += schwarzschild_dt
            X_t = trial_X_t
            step += 1
            progress.update(schwarzschild_dt)
            if step % args.save_every == 0:
                diagnostics = write_snapshot(metric, particles, E_r, output, step, evolution_time, schwarzschild_time, args)
                summary_path.write_text(json.dumps({**metadata, "completed": False, "final_step": step,
                    "final_gauss_residual_linf": float(diagnostics["gauss_residual_linf"]),
                    "final_gauss_residual_rms": float(diagnostics["gauss_residual_rms"]),
                    "removed_particle_charge": float(diagnostics["removed_particle_charge"]),
                    "truncated_shape_charge": float(diagnostics["truncated_shape_charge"]),
                    "final_time": schwarzschild_time, "final_evolution_time": evolution_time}, indent=2) + "\n")
    completed = args.target_time - schwarzschild_time <= time_tolerance
    if completed:
        schwarzschild_time = args.target_time
        stop_reason = "target_time"
    elif step >= maximum_steps:
        stop_reason = "max_steps"
    final = write_snapshot(metric, particles, E_r, output, step, evolution_time, schwarzschild_time, args)
    summary = dict(
        **metadata, completed=completed, stop_reason=stop_reason,
        final_step=step, final_time=schwarzschild_time, final_evolution_time=evolution_time,
        total_particles=int(particles.r.size), active_particles=int(jnp.count_nonzero(particles.weight)),
        final_active_particle_weight=float(jnp.sum(particles.weight)),
        final_particle_charge=float(jnp.sum(particles.get_charge())),
        final_deposited_charge=float(final["deposited_charge"]),
        removed_particle_charge=float(final["removed_particle_charge"]),
        truncated_shape_charge=float(final["truncated_shape_charge"]),
        final_gauss_residual_linf=float(final["gauss_residual_linf"]),
        final_gauss_residual_rms=float(final["gauss_residual_rms"]),
        gauss_residual_linf_change=float(final["gauss_residual_linf"] - initial["gauss_residual_linf"]),
        initial_electric_field_energy=float(initial["electric_field_energy"]),
        final_electric_field_energy=float(final["electric_field_energy"]),
        minimum_lapse=float(jnp.min(metric.alpha)), minimum_chi=float(jnp.min(metric.chi)),
        wall_runtime_seconds=time.perf_counter() - start,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEMO_DIRECTORY / "outputs/z4c_charged_star_schwarzschild")
    parser.add_argument("--mass-scale", type=float, default=1.0, help="Initial gravitational mass used for OS profile and Schwarzschild normalization.")
    parser.add_argument("--charge-to-mass", type=float, default=0.2, help="Signed Q_geom / stellar rest mass.")
    parser.add_argument("--surface-areal-radius", type=float, default=10.0)
    parser.add_argument("--initial-tolerance", "--shooting-tolerance", type=float, default=os_collapse.SHOOTING_TOLERANCE)
    parser.add_argument("--shooting-max-iterations", type=int, default=os_collapse.SHOOTING_MAX_ITERATIONS)
    parser.add_argument("--r-max", type=float, default=100.0)
    parser.add_argument("--num-cells", type=int, default=os_collapse.NUM_Z4C_CELLS)
    parser.add_argument("--particles-per-shell", type=int, default=os_collapse.PARTICLES_PER_SHELL)
    parser.add_argument("--shape-mode", choices=("nearest", "linear", "quadratic"), default="quadratic")
    parser.add_argument("--target-time", type=float, default=150.0, help="Target Schwarzschild diagnostic time, G=c=1.")
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--minimum-dt", type=float, default=1.e-7)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--zero-shift", type=int, choices=(0, 1), default=0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--nu", type=float, default=0.02)
    args = parser.parse_args(argv)
    args.epsilon_0 = 1.0  # Use G=c=epsilon_0=1, as in the constrained Heun solver.
    if not (0.0 < 2.0 * args.mass_scale < args.surface_areal_radius):
        parser.error("Require 0 < 2*mass-scale < surface-areal-radius for the OS seed profile.")
    if (args.num_cells < 8 or args.particles_per_shell < 1 or args.save_every < 1
            or args.target_time <= 0 or args.cfl <= 0 or args.minimum_dt <= 0 or args.r_max <= 0
            or args.shooting_max_iterations < 1 or args.initial_tolerance <= 0 or (args.max_steps is not None and args.max_steps < 0)):
        parser.error("Times and grid/output counts must be positive (num-cells >= 8).")
    return args


if __name__ == "__main__":
    result = run_simulation(parse_arguments())
    if not result["completed"] and result["stop_reason"] != "max_steps":
        raise SystemExit("Evolution stopped; the last accepted state was preserved. See run_summary.json.")
