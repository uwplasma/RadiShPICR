"""Cold relativistic radial two-stream instability on fixed Minkowski Z4C data.

The Z4C metric uses the live ordinary ``(r, u_r)`` particle convention but is
held exactly flat in this first electrostatic validation.  Two equal electron
beams move at ``+/- beam_velocity`` through a prescribed, infinitely massive
ion charge density.  The plasma occupies a guarded annulus, so no artificial
periodic seam or particle reflection enters the measured growth interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm


def find_package_root() -> Path:
    current_directory = Path(__file__).resolve().parent
    for candidate in (current_directory, *current_directory.parents):
        if (candidate / "RadiShPICR" / "Z4C").is_dir():
            return candidate

    raise RuntimeError("Could not find the RadiShPICR package root.")


package_root = find_package_root()
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C import (
    Z4C_Metric,
    compute_radial_charge_density,
    electric_field_energy,
    electrostatic_particles_rk4_step,
    solve_radial_electric_field,
)
from RadiShPICR.Z4C.energy_momentum_tensor import _proper_radial_shell_volume
from RadiShPICR.Z4C.utils import generate_r_grid


OUTGOING_ELECTRONS = 0
INCOMING_ELECTRONS = 1
POPULATION_LABELS = ("outgoing_electrons", "incoming_electrons")

DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "z4c_two_stream"
)


@dataclass(frozen=True)
class TwoStreamParameters:
    r_max: float = 80.0
    plasma_r_min: float = 20.0
    plasma_r_max: float = 60.0
    analysis_r_min: float = 30.0
    analysis_r_max: float = 50.0
    num_cells: int = 1600
    particles_per_cell: int = 16
    epsilon_0: float = 1.0
    plasma_frequency: float = 1.0
    electron_charge_to_mass: float = -1.0
    beam_velocity: float = 0.2
    wavenumber: float = math.pi
    perturbation_amplitude: float = 1.0e-3
    shape_mode: str = "quadratic"
    dt: float = 0.025
    final_time: float = 30.0
    save_every: int = 10
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY


def validate_parameters(params: TwoStreamParameters) -> None:
    if not (
        0.0
        < params.plasma_r_min
        < params.analysis_r_min
        < params.analysis_r_max
        < params.plasma_r_max
        < params.r_max
    ):
        raise ValueError(
            "expected 0 < plasma_r_min < analysis_r_min < analysis_r_max "
            "< plasma_r_max < r_max"
        )
    if params.num_cells < 8 or params.particles_per_cell < 1:
        raise ValueError("the grid and every occupied cell must contain particles")
    if params.epsilon_0 <= 0.0 or params.plasma_frequency <= 0.0:
        raise ValueError("epsilon_0 and plasma_frequency must be positive")
    if params.electron_charge_to_mass >= 0.0:
        raise ValueError("electron_charge_to_mass must be negative")
    if not 0.0 < params.beam_velocity < 1.0:
        raise ValueError("beam_velocity must be a physical speed in (0, 1)")
    if params.wavenumber <= 0.0 or params.perturbation_amplitude < 0.0:
        raise ValueError("wavenumber must be positive and amplitude nonnegative")
    if params.shape_mode not in ("nearest", "linear", "quadratic"):
        raise ValueError("shape_mode must be nearest, linear, or quadratic")
    if params.dt <= 0.0 or params.final_time < 0.0 or params.save_every < 1:
        raise ValueError("dt and save_every must be positive and final_time nonnegative")

    num_steps = params.final_time / params.dt
    if not math.isclose(num_steps, round(num_steps), rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("final_time must be an integer multiple of dt")


def make_flat_metric(params: TwoStreamParameters) -> Z4C_Metric:
    r = generate_r_grid(0.0, params.r_max, params.num_cells)
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)

    return Z4C_Metric(
        alpha=ones,
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=ones,
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Gamma=zeros,
        kappa=jnp.asarray(0.0, dtype=r.dtype),
        eta=jnp.asarray(0.0, dtype=r.dtype),
        nu=jnp.asarray(0.0, dtype=r.dtype),
        r=r,
        dr=r[1] - r[0],
    )


def quiet_start_one_beam(
    params: TwoStreamParameters,
    metric: Z4C_Metric,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-volume subcell positions and one-beam macro masses."""

    r = np.asarray(metric.r)
    dr = float(metric.dr)
    shell_inner = np.maximum(r - 0.5 * dr, params.plasma_r_min)
    shell_outer = np.minimum(r + 0.5 * dr, params.plasma_r_max)
    occupied = shell_outer > shell_inner
    shell_inner = shell_inner[occupied]
    shell_outer = shell_outer[occupied]

    subcell_fraction = (
        np.arange(params.particles_per_cell, dtype=float) + 0.5
    ) / params.particles_per_cell
    inner_cubed = shell_inner[:, np.newaxis] ** 3
    outer_cubed = shell_outer[:, np.newaxis] ** 3
    radius = np.cbrt(
        inner_cubed
        + subcell_fraction[np.newaxis, :] * (outer_cubed - inner_cubed)
    ).ravel()

    # omega_p^2 = (q/m)^2 rho_m / epsilon_0 for the combined electrons.
    charge_to_mass = abs(params.electron_charge_to_mass)
    electron_mass_density = (
        params.epsilon_0 * params.plasma_frequency**2 / charge_to_mass**2
    )
    shell_volume = (4.0 * np.pi / 3.0) * (
        shell_outer**3 - shell_inner**3
    )
    one_beam_mass = (
        0.5
        * electron_mass_density
        * shell_volume[:, np.newaxis]
        / params.particles_per_cell
    )
    one_beam_mass = np.broadcast_to(
        one_beam_mass,
        (shell_volume.size, params.particles_per_cell),
    ).ravel()

    return radius, one_beam_mass


def make_electron_particles(
    params: TwoStreamParameters,
    metric: Z4C_Metric,
    perturb: bool = True,
) -> tuple[particle_species, np.ndarray]:
    base_radius, one_beam_mass = quiet_start_one_beam(params, metric)
    radius = base_radius
    if perturb:
        displacement = (
            params.perturbation_amplitude
            / params.wavenumber
            * np.sin(params.wavenumber * (base_radius - params.plasma_r_min))
        )
        radius = base_radius + displacement

    if np.any(radius <= 0.0) or np.any(radius >= params.r_max):
        raise ValueError("the requested perturbation places electrons outside the grid")

    gamma_b = 1.0 / math.sqrt(1.0 - params.beam_velocity**2)
    beam_momentum = gamma_b * params.beam_velocity
    r = np.concatenate((radius, radius))
    ur = np.concatenate(
        (
            np.full_like(radius, beam_momentum),
            np.full_like(radius, -beam_momentum),
        )
    )
    weight = np.concatenate((one_beam_mass, one_beam_mass))
    population_id = np.concatenate(
        (
            np.full(radius.size, OUTGOING_ELECTRONS, dtype=np.int32),
            np.full(radius.size, INCOMING_ELECTRONS, dtype=np.int32),
        )
    )

    particles = particle_species(
        name="electrons",
        charge=jnp.asarray(params.electron_charge_to_mass),
        mass=jnp.asarray(1.0),
        weight=jnp.asarray(weight),
        r=jnp.asarray(r),
        ur=jnp.asarray(ur),
        phi=jnp.zeros_like(jnp.asarray(r)),
        uphi=jnp.zeros_like(jnp.asarray(r)),
        shape_mode=params.shape_mode,
    )

    return particles, population_id


def cold_two_stream_growth_rate(params: TwoStreamParameters) -> float:
    """Return the local cold symmetric-beam amplitude growth rate."""

    gamma_b = 1.0 / math.sqrt(1.0 - params.beam_velocity**2)
    omega_squared = params.plasma_frequency**2 / gamma_b**3
    kv_squared = (params.wavenumber * params.beam_velocity) ** 2
    unstable_omega_squared = (
        0.5
        * math.sqrt(omega_squared**2 + 8.0 * omega_squared * kv_squared)
        - kv_squared
        - 0.5 * omega_squared
    )

    return math.sqrt(max(unstable_omega_squared, 0.0))


def relativistic_kinetic_energy(particles: particle_species) -> jnp.ndarray:
    lorentz_factor = jnp.sqrt(1.0 + particles.ur**2 + particles.uphi**2 / particles.r**2)
    return jnp.sum(particles.get_mass() * (lorentz_factor - 1.0))


def target_mode_diagnostics(
    metric: Z4C_Metric,
    E_r,
    params: TwoStreamParameters,
) -> tuple[float, float, float, float]:
    """Project the electric field onto the seeded sine/cosine mode."""

    r = np.asarray(metric.r)
    field = np.asarray(E_r)
    volume = np.asarray(_proper_radial_shell_volume(metric))
    inside = (r >= params.analysis_r_min) & (r <= params.analysis_r_max)
    r = r[inside]
    field = field[inside]
    volume = volume[inside]

    phase = params.wavenumber * (r - params.plasma_r_min)
    basis = np.column_stack((np.sin(phase), np.cos(phase)))
    gram = basis.T @ (volume[:, np.newaxis] * basis)
    rhs = basis.T @ (volume * field)
    coefficients = np.linalg.solve(gram, rhs)
    mode_field = basis @ coefficients
    mode_amplitude = float(np.linalg.norm(coefficients))
    mode_energy = float(
        np.sum(0.5 * params.epsilon_0 * mode_field**2 * volume)
    )
    analysis_energy = float(
        np.sum(0.5 * params.epsilon_0 * field**2 * volume)
    )
    mode_fraction = mode_energy / analysis_energy if analysis_energy > 0.0 else 0.0

    return mode_amplitude, mode_energy, mode_fraction, analysis_energy


def gauss_law_residual(
    metric: Z4C_Metric,
    charge_density,
    E_r,
    epsilon_0: float,
) -> tuple[float, float]:
    """Return the algebraic finite-volume Gauss-solve consistency residual."""

    centered_field = np.asarray(E_r)
    face_field = np.empty(centered_field.size + 1, dtype=centered_field.dtype)
    face_field[0] = 0.0
    for cell in range(centered_field.size):
        face_field[cell + 1] = 2.0 * centered_field[cell] - face_field[cell]

    r = np.asarray(metric.r)
    dr = float(metric.dr)
    face_radius = np.concatenate((np.array([0.0]), r + 0.5 * dr))
    face_flux = 4.0 * np.pi * face_radius**2 * face_field
    volume = np.asarray(_proper_radial_shell_volume(metric))
    reconstructed_charge_density = (
        epsilon_0 * np.diff(face_flux) / volume
    )
    residual = reconstructed_charge_density - np.asarray(charge_density)
    absolute_residual = float(np.max(np.abs(residual)))
    density_scale = float(np.max(np.abs(np.asarray(charge_density))))
    relative_residual = absolute_residual / max(density_scale, 1.0e-30)

    return absolute_residual, relative_residual


def particle_boundary_margin(
    particles: particle_species,
    r_max: float,
) -> float:
    radius = np.asarray(particles.r)
    return float(min(np.min(radius), r_max - np.max(radius)))


def collect_diagnostic_row(
    step: int,
    time: float,
    particles: particle_species,
    metric: Z4C_Metric,
    ion_charge_density,
    charge_density,
    E_r,
    params: TwoStreamParameters,
    initial_total_energy: float | None,
) -> tuple[dict[str, float | int | bool], float]:
    field_energy = float(
        electric_field_energy(metric, E_r, epsilon_0=params.epsilon_0)
    )
    kinetic_energy = float(relativistic_kinetic_energy(particles))
    total_energy = field_energy + kinetic_energy
    if initial_total_energy is None:
        initial_total_energy = total_energy

    mode_amplitude, mode_energy, mode_fraction, analysis_field_energy = (
        target_mode_diagnostics(
            metric,
            E_r,
            params,
        )
    )
    gauss_absolute, gauss_relative = gauss_law_residual(
        metric,
        charge_density,
        E_r,
        params.epsilon_0,
    )
    volume = np.asarray(_proper_radial_shell_volume(metric))
    total_charge = float(np.sum(np.asarray(charge_density) * volume))
    electron_charge = float(np.sum(np.asarray(particles.get_charge())))
    ion_charge = float(np.sum(np.asarray(ion_charge_density) * volume))
    minimum_r = float(np.min(np.asarray(particles.r)))
    maximum_r = float(np.max(np.asarray(particles.r)))
    boundary_margin = particle_boundary_margin(particles, params.r_max)
    finite_state = bool(
        np.all(np.isfinite(np.asarray(particles.r)))
        and np.all(np.isfinite(np.asarray(particles.ur)))
        and np.all(np.isfinite(np.asarray(E_r)))
        and np.all(np.isfinite(np.asarray(charge_density)))
    )

    row = {
        "step": int(step),
        "time": float(time),
        "minimum_r": minimum_r,
        "maximum_r": maximum_r,
        "boundary_margin": float(boundary_margin),
        "maximum_abs_Er": float(np.max(np.abs(np.asarray(E_r)))),
        "electric_field_energy": field_energy,
        "analysis_electric_field_energy": analysis_field_energy,
        "relativistic_kinetic_energy": kinetic_energy,
        "field_plus_particle_energy": total_energy,
        "relative_total_energy_drift": (
            (total_energy - initial_total_energy) / initial_total_energy
        ),
        "target_mode_amplitude": mode_amplitude,
        "target_mode_power": mode_energy,
        "target_mode_energy": mode_energy,
        "target_mode_energy_fraction": mode_fraction,
        "electron_charge": electron_charge,
        "ion_charge": ion_charge,
        "total_charge": total_charge,
        "gauss_residual_linf": gauss_absolute,
        "relative_gauss_residual_linf": gauss_relative,
        "finite_state": finite_state,
    }

    return row, initial_total_energy


def append_diagnostic_row(path: Path, row: dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_metric_snapshot(
    metric: Z4C_Metric,
    ion_charge_density,
    charge_density,
    E_r,
    output_directory: Path,
    step: int,
    time: float,
) -> Path:
    path = output_directory / f"metric_step_{step:06d}.npz"
    electron_charge_density = charge_density - ion_charge_density
    np.savez_compressed(
        path,
        r=np.asarray(metric.r),
        alpha=np.asarray(metric.alpha),
        beta=np.asarray(metric.beta),
        conformal_grr=np.asarray(metric.conformal_grr),
        conformal_gt=np.asarray(metric.conformal_gt),
        chi=np.asarray(metric.chi),
        Kh=np.asarray(metric.Kh),
        Arr=np.asarray(metric.Arr),
        At=np.asarray(metric.At),
        theta=np.asarray(metric.theta),
        Gamma=np.asarray(metric.Gamma),
        E_r=np.asarray(E_r),
        charge_density=np.asarray(charge_density),
        electron_charge_density=np.asarray(electron_charge_density),
        ion_charge_density=np.asarray(ion_charge_density),
        step=int(step),
        time=float(time),
        fixed_minkowski=True,
        particle_state_variables="r_ur",
    )
    return path


def write_phase_space_snapshot(
    particles: particle_species,
    population_id: np.ndarray,
    output_directory: Path,
    step: int,
    time: float,
) -> Path:
    radial_momentum = np.asarray(particles.ur)
    local_radial_velocity = radial_momentum / np.sqrt(1.0 + radial_momentum**2)
    path = output_directory / f"phase_space_step_{step:06d}.npz"
    np.savez_compressed(
        path,
        r=np.asarray(particles.r),
        ur=radial_momentum,
        radial_orthonormal_momentum=radial_momentum,
        local_radial_velocity=local_radial_velocity,
        weight=np.asarray(particles.weight),
        charge_to_mass=np.broadcast_to(
            np.asarray(particles.charges / particles.masses),
            radial_momentum.shape,
        ),
        population_id=np.asarray(population_id),
        population_labels=np.asarray(POPULATION_LABELS),
        step=int(step),
        time=float(time),
        species_name=particles.name,
        saved_coordinates="z4c_minkowski_radial",
        particle_state_variables="r_ur",
    )
    return path


def write_run_parameters(
    params: TwoStreamParameters,
    metric: Z4C_Metric,
    particles: particle_species,
    ion_charge_density,
) -> Path:
    values = asdict(params)
    values["output_directory"] = str(params.output_directory)
    values["r_min"] = 0.0
    values["grid_spacing"] = float(metric.dr)
    values["num_steps"] = int(round(params.final_time / params.dt))
    values["num_electrons"] = int(particles.r.size)
    values["electron_mass_density"] = (
        params.epsilon_0
        * params.plasma_frequency**2
        / params.electron_charge_to_mass**2
    )
    values["beam_lorentz_factor"] = 1.0 / math.sqrt(
        1.0 - params.beam_velocity**2
    )
    values["beam_momentum"] = (
        values["beam_lorentz_factor"] * params.beam_velocity
    )
    values["beam_to_beam_relative_velocity"] = (
        2.0 * params.beam_velocity / (1.0 + params.beam_velocity**2)
    )
    values["theoretical_amplitude_growth_rate"] = cold_two_stream_growth_rate(
        params
    )
    values["theoretical_energy_growth_rate"] = (
        2.0 * values["theoretical_amplitude_growth_rate"]
    )
    values["normalization"] = {
        "speed_of_light": 1.0,
        "epsilon_0": params.epsilon_0,
        "combined_electron_plasma_frequency": params.plasma_frequency,
    }
    values["geometry"] = {
        "coordinate_system": "spherical_radial",
        "particle_domain": "guarded_annulus",
        "analysis_window": [params.analysis_r_min, params.analysis_r_max],
        "local_theory_benchmark": "homogeneous_slab_approximation",
    }
    values["fixed_metric"] = "minkowski_z4c"
    values["particle_boundary"] = "none_guarded_annulus"
    values["ion_background"] = "fixed_discrete_neutralizing_charge_density"
    values["nonlinear_validation_target"] = "BGK_trapped_particle_island"
    values["gauss_residual_definition"] = (
        "cell_center_to_face_reconstruction_of_finite_volume_solve"
    )
    values["particle_state_variables"] = "r_ur"
    values["total_electron_charge"] = float(
        np.sum(np.asarray(particles.get_charge()))
    )
    values["total_ion_charge"] = float(
        np.sum(
            np.asarray(ion_charge_density)
            * np.asarray(_proper_radial_shell_volume(metric))
        )
    )

    path = Path(params.output_directory) / "run_parameters.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2)
        stream.write("\n")
    return path


def metric_is_exactly_minkowski(metric: Z4C_Metric) -> bool:
    ones = (metric.alpha, metric.conformal_grr, metric.conformal_gt, metric.chi)
    zeros = (
        metric.beta,
        metric.Kh,
        metric.Arr,
        metric.At,
        metric.theta,
        metric.Gamma,
    )
    return all(np.all(np.asarray(field) == 1.0) for field in ones) and all(
        np.all(np.asarray(field) == 0.0) for field in zeros
    )


def run_two_stream(params: TwoStreamParameters) -> dict[str, object]:
    validate_parameters(params)
    output_directory = Path(params.output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_directory}. "
            "Move it or select a fresh --output-dir before starting the run."
        )

    metric = make_flat_metric(params)
    particles, population_id = make_electron_particles(params, metric, perturb=True)
    unperturbed_particles, _ = make_electron_particles(
        params,
        metric,
        perturb=False,
    )

    compute_charge_jit = jax.jit(compute_radial_charge_density)
    step_jit = jax.jit(electrostatic_particles_rk4_step)
    ion_charge_density = -compute_charge_jit(unperturbed_particles, metric)
    electron_charge_density = compute_charge_jit(particles, metric)
    charge_density = electron_charge_density + ion_charge_density
    E_r = solve_radial_electric_field(
        metric,
        charge_density,
        epsilon_0=params.epsilon_0,
    )
    jax.block_until_ready(E_r)

    if not metric_is_exactly_minkowski(metric):
        raise RuntimeError("the fixed Z4C initial data are not exactly Minkowski")

    num_steps = int(round(params.final_time / params.dt))
    stage_safe_margin = 2.0 * float(metric.dr)
    if num_steps > 0:
        # On Minkowski data |dr/dt| < 1, so one additional dt guarantees that
        # every intermediate RK4 stage stays outside the two-cell guard.
        stage_safe_margin += params.dt
    if particle_boundary_margin(particles, params.r_max) <= stage_safe_margin:
        raise RuntimeError(
            "the initial electrons do not leave enough room for a boundary-safe "
            "RK4 stage"
        )

    metric_directory = output_directory / "metric"
    phase_space_directory = output_directory / "phase_space"
    metric_directory.mkdir(parents=True, exist_ok=True)
    phase_space_directory.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_directory / "diagnostics.csv"

    write_run_parameters(params, metric, particles, ion_charge_density)

    initial_total_energy = None
    row, initial_total_energy = collect_diagnostic_row(
        0,
        0.0,
        particles,
        metric,
        ion_charge_density,
        charge_density,
        E_r,
        params,
        initial_total_energy,
    )
    append_diagnostic_row(diagnostics_path, row)
    write_metric_snapshot(
        metric,
        ion_charge_density,
        charge_density,
        E_r,
        metric_directory,
        0,
        0.0,
    )
    write_phase_space_snapshot(
        particles,
        population_id,
        phase_space_directory,
        0,
        0.0,
    )

    time = 0.0
    with tqdm(
        total=params.final_time,
        initial=time,
        desc="evolving fixed-Minkowski Z4C two-stream",
        unit="t",
    ) as progress_bar:
        for step in range(1, num_steps + 1):
            previous_time = time
            if particle_boundary_margin(particles, params.r_max) <= (
                2.0 * float(metric.dr) + params.dt
            ):
                raise RuntimeError(
                    "the next RK4 stages would enter the guarded computational "
                    "boundary; the last written state remains valid"
                )
            particles, charge_density, E_r = step_jit(
                particles,
                metric,
                ion_charge_density,
                params.dt,
                params.epsilon_0,
            )
            jax.block_until_ready(E_r)
            time = step * params.dt
            row, _ = collect_diagnostic_row(
                step,
                time,
                particles,
                metric,
                ion_charge_density,
                charge_density,
                E_r,
                params,
                initial_total_energy,
            )

            boundary_encounter = row["boundary_margin"] <= 2.0 * float(metric.dr)
            if not row["finite_state"] or boundary_encounter:
                raise RuntimeError(
                    "the next step is non-finite or has reached the guarded "
                    "computational boundary; the last written snapshot remains valid"
                )

            append_diagnostic_row(diagnostics_path, row)
            should_save = step % params.save_every == 0 or step == num_steps
            if should_save:
                write_metric_snapshot(
                    metric,
                    ion_charge_density,
                    charge_density,
                    E_r,
                    metric_directory,
                    step,
                    time,
                )
                write_phase_space_snapshot(
                    particles,
                    population_id,
                    phase_space_directory,
                    step,
                    time,
                )

            progress_bar.update(time - previous_time)
            progress_bar.set_postfix(
                step=step,
                electric_energy=f"{row['electric_field_energy']:.3e}",
                mode_fraction=f"{row['target_mode_energy_fraction']:.3f}",
                energy_drift=f"{row['relative_total_energy_drift']:.2e}",
            )

    return {
        "output_directory": output_directory,
        "diagnostics_path": diagnostics_path,
        "metric_directory": metric_directory,
        "phase_space_directory": phase_space_directory,
        "steps": num_steps,
        "time": time,
        "finite_state": bool(row["finite_state"]),
        "fixed_minkowski": metric_is_exactly_minkowski(metric),
        "theoretical_amplitude_growth_rate": cold_two_stream_growth_rate(params),
        "final_electric_field_energy": float(row["electric_field_energy"]),
        "final_mode_fraction": float(row["target_mode_energy_fraction"]),
    }


def parse_args() -> argparse.Namespace:
    defaults = TwoStreamParameters()
    parser = argparse.ArgumentParser(
        description=(
            "Run a cold radial two-stream instability on fixed Minkowski Z4C data."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=defaults.output_directory)
    parser.add_argument("--r-max", type=float, default=defaults.r_max)
    parser.add_argument("--plasma-r-min", type=float, default=defaults.plasma_r_min)
    parser.add_argument("--plasma-r-max", type=float, default=defaults.plasma_r_max)
    parser.add_argument("--analysis-r-min", type=float, default=defaults.analysis_r_min)
    parser.add_argument("--analysis-r-max", type=float, default=defaults.analysis_r_max)
    parser.add_argument("--num-cells", type=int, default=defaults.num_cells)
    parser.add_argument(
        "--particles-per-cell",
        type=int,
        default=defaults.particles_per_cell,
        help="quiet-start macro-particles per cell in each electron beam",
    )
    parser.add_argument("--epsilon-0", type=float, default=defaults.epsilon_0)
    parser.add_argument(
        "--plasma-frequency",
        type=float,
        default=defaults.plasma_frequency,
        help="combined-electron nonrelativistic plasma frequency",
    )
    parser.add_argument(
        "--electron-charge-to-mass",
        type=float,
        default=defaults.electron_charge_to_mass,
    )
    parser.add_argument(
        "--beam-velocity",
        type=float,
        default=defaults.beam_velocity,
        help="magnitude of each beam velocity in the fixed-ion frame",
    )
    parser.add_argument("--wavenumber", type=float, default=defaults.wavenumber)
    parser.add_argument(
        "--perturbation-amplitude",
        type=float,
        default=defaults.perturbation_amplitude,
        help="dimensionless density-seed amplitude in xi=(epsilon/k) sin(k r)",
    )
    parser.add_argument(
        "--shape-mode",
        choices=("nearest", "linear", "quadratic"),
        default=defaults.shape_mode,
    )
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--final-time", type=float, default=defaults.final_time)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = replace(
        TwoStreamParameters(),
        output_directory=args.output_dir,
        r_max=args.r_max,
        plasma_r_min=args.plasma_r_min,
        plasma_r_max=args.plasma_r_max,
        analysis_r_min=args.analysis_r_min,
        analysis_r_max=args.analysis_r_max,
        num_cells=args.num_cells,
        particles_per_cell=args.particles_per_cell,
        epsilon_0=args.epsilon_0,
        plasma_frequency=args.plasma_frequency,
        electron_charge_to_mass=args.electron_charge_to_mass,
        beam_velocity=args.beam_velocity,
        wavenumber=args.wavenumber,
        perturbation_amplitude=args.perturbation_amplitude,
        shape_mode=args.shape_mode,
        dt=args.dt,
        final_time=args.final_time,
        save_every=args.save_every,
    )
    summary = run_two_stream(params)
    print("run_summary =")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
