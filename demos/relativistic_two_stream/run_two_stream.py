"""Cold relativistic radial two-stream instability on Z4C initial data.

The Z4C metric uses the live ordinary ``(r, u_r)`` particle convention and is
initialized by solving the time-symmetric Hamiltonian constraint.  With
gravity disabled, that solved metric is held fixed so only dynamical
back-reaction and metric feedback into the plasma are omitted.  Two equal
neutral electron-ion streams move at ``+/- beam_velocity``.  Co-moving ions
remove the unperturbed charge separation at the plasma edges; an electron-only
displacement seeds the instability.  All four populations evolve through the
same coupled RK4 step.  The plasma
occupies a guarded annulus, so no artificial periodic seam or particle
reflection enters the measured growth interval.
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

from RadiShPICR.particles.shape_factors.common import (
    proper_radial_shell_volume,
)
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    _interpolate_cell_centered_fields_to_particles,
)
from RadiShPICR.Z4C import (
    METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    METRIC_BOUNDARY_SOMMERFELD,
    Z4C_Metric,
    compute_electrostatic_matter_terms,
    compute_radial_charge_density,
    electric_field_energy,
    misner_sharp_mass,
    rk4_step,
    solve_radial_electric_field,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    MatterTerms,
    compute_radial_matter_terms,
)
from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.geodesic import _radial_grid_from_metric
from RadiShPICR.Z4C.utils import generate_r_grid


OUTGOING_ELECTRONS = 0
INCOMING_ELECTRONS = 1
OUTGOING_IONS = 2
INCOMING_IONS = 3
POPULATION_LABELS = (
    "outgoing_electrons", "incoming_electrons", "outgoing_ions", "incoming_ions"
)

DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "z4c_two_stream_neutral_streams"
)
DIAGNOSTIC_SCHEMA_VERSION = 2
INITIAL_DATA_RELAXATION = 0.5
INITIAL_DATA_BOUNDARY_NEWTON_ITERATIONS = 16
METRIC_BOUNDARY_CODES = {
    "sommerfeld": METRIC_BOUNDARY_SOMMERFELD,
    "constraint_preserving": METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
}


@dataclass(frozen=True)
class TwoStreamParameters:
    dynamic_gr: bool = True
    metric_boundary: str = "constraint_preserving"
    r_max: float = 80.0
    plasma_r_min: float = 20.0
    plasma_r_max: float = 60.0
    analysis_r_min: float = 30.0
    analysis_r_max: float = 50.0
    num_cells: int = 1600
    particles_per_cell: int = 16
    # This geometric normalization keeps the plasma frequency and q/m fixed
    # while making the default annulus weakly self-gravitating.
    epsilon_0: float = 1.0e-8
    plasma_frequency: float = 1.0
    electron_charge_to_mass: float = -1.0
    ion_to_electron_mass_ratio: float = 1836.0
    beam_velocity: float = 0.2
    wavenumber: float = math.pi
    perturbation_amplitude: float = 1.0e-3
    shape_mode: str = "quadratic"
    initial_data_tolerance: float = 1.0e-10
    initial_data_max_iterations: int = 200
    dt: float = 0.025
    final_time: float = 70.0
    save_every: int = 10
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY


def validate_parameters(params: TwoStreamParameters) -> None:
    if params.metric_boundary not in METRIC_BOUNDARY_CODES:
        raise ValueError(
            "metric_boundary must be sommerfeld or constraint_preserving"
        )
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
    if params.ion_to_electron_mass_ratio <= 0.0:
        raise ValueError("ion_to_electron_mass_ratio must be positive")
    if not 0.0 < params.beam_velocity < 1.0:
        raise ValueError("beam_velocity must be a physical speed in (0, 1)")
    if params.wavenumber <= 0.0 or params.perturbation_amplitude < 0.0:
        raise ValueError("wavenumber must be positive and amplitude nonnegative")
    if params.shape_mode not in ("nearest", "linear", "quadratic"):
        raise ValueError("shape_mode must be nearest, linear, or quadratic")
    if params.initial_data_tolerance <= 0.0:
        raise ValueError("initial_data_tolerance must be positive")
    if params.initial_data_max_iterations < 1:
        raise ValueError("initial_data_max_iterations must be positive")
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


def _particles_with_initial_metric_momentum(
    particles: particle_species,
    metric: Z4C_Metric,
    radial_orthonormal_momentum,
) -> particle_species:
    """Lower the prescribed local radial momentum with the initial metric."""

    radial_metric_at_particles = _interpolate_cell_centered_fields_to_particles(
        (metric.conformal_grr / metric.chi)[jnp.newaxis, :],
        particles.r,
        _radial_grid_from_metric(metric),
        shape_mode=particles.get_shape(),
        field_parities=jnp.asarray((1,)),
    )[0]
    covariant_radial_momentum = (
        jnp.sqrt(radial_metric_at_particles) * radial_orthonormal_momentum
    )

    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=particles.r,
        ur=covariant_radial_momentum,
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )


def _chi_sommerfeld_residual(metric: Z4C_Metric, conformal_factor):
    """Return the production outer-boundary residual for ``chi = psi^-4``."""

    chi = conformal_factor**-4
    dchi_dr = first_derivative(chi, metric.dr, parity=1)

    return dchi_dr[-1] + (chi[-1] - 1.0) / metric.r[-1]


def _solve_radial_poisson_equation(
    metric: Z4C_Metric,
    source,
    boundary_tolerance,
):
    """Solve ``laplacian(psi) = source`` with the production chi boundary."""

    inner_radius = jnp.maximum(metric.r - 0.5 * metric.dr, 0.0)
    outer_radius = metric.r + 0.5 * metric.dr
    coordinate_shell_volume = (4.0 * jnp.pi / 3.0) * (
        outer_radius**3 - inner_radius**3
    )

    # The face flux is the finite-volume integral of the spherical Laplacian.
    enclosed_source = jnp.cumsum(source * coordinate_shell_volume) / (4.0 * jnp.pi)
    face_gradient = jnp.concatenate(
        (
            jnp.zeros_like(enclosed_source[:1]),
            enclosed_source / outer_radius**2,
        )
    )

    # Integrate the source-determined gradients inward from an unknown outer
    # value.  The old psi Robin value is a close starting point for Newton's
    # method, while the converged additive constant makes chi satisfy the exact
    # production one-sided Sommerfeld stencil.
    outer_face_field = 1.0 - outer_radius[-1] * face_gradient[-1]
    outer_cell_field = outer_face_field - 0.5 * metric.dr * face_gradient[-1]
    cell_jumps = metric.dr * face_gradient[1:-1]
    inward_change = jnp.concatenate(
        (
            jnp.cumsum(cell_jumps[::-1])[::-1],
            jnp.zeros_like(outer_cell_field[jnp.newaxis]),
        )
    )

    minimum_outer_cell_field = (
        jnp.max(inward_change) + jnp.sqrt(jnp.finfo(source.dtype).eps)
    )
    outer_cell_field = jnp.maximum(
        outer_cell_field,
        minimum_outer_cell_field,
    )

    def newton_iteration(_, current_outer_cell_field):
        conformal_factor = current_outer_cell_field - inward_change
        residual = _chi_sommerfeld_residual(metric, conformal_factor)

        dchi_douter_cell = -4.0 * conformal_factor**-5
        dresidual_douter_cell = first_derivative(
            dchi_douter_cell,
            metric.dr,
            parity=1,
        )[-1]
        dresidual_douter_cell += dchi_douter_cell[-1] / metric.r[-1]

        candidate = (
            current_outer_cell_field
            - residual / dresidual_douter_cell
        )
        positive_candidate = jnp.where(
            candidate > minimum_outer_cell_field,
            candidate,
            0.5 * (
                current_outer_cell_field + minimum_outer_cell_field
            ),
        )
        finite_candidate = jnp.where(
            jnp.isfinite(positive_candidate),
            positive_candidate,
            current_outer_cell_field,
        )

        return jnp.where(
            jnp.abs(residual) > boundary_tolerance,
            finite_candidate,
            current_outer_cell_field,
        )

    outer_cell_field = jax.lax.fori_loop(
        0,
        INITIAL_DATA_BOUNDARY_NEWTON_ITERATIONS,
        newton_iteration,
        outer_cell_field,
    )

    return outer_cell_field - inward_change


def _radial_poisson_residual(metric: Z4C_Metric, field, source):
    inner_radius = jnp.maximum(metric.r - 0.5 * metric.dr, 0.0)
    outer_radius = metric.r + 0.5 * metric.dr
    coordinate_shell_volume = (4.0 * jnp.pi / 3.0) * (
        outer_radius**3 - inner_radius**3
    )

    # The additive normalization does not change the source-determined outer
    # flux.  Reconstruct that flux directly instead of using the removed psi
    # Robin condition.
    outer_gradient = jnp.sum(source * coordinate_shell_volume) / (
        4.0 * jnp.pi * outer_radius[-1] ** 2
    )
    face_gradient = jnp.concatenate(
        (
            jnp.zeros_like(field[:1]),
            (field[1:] - field[:-1]) / metric.dr,
            outer_gradient[jnp.newaxis],
        )
    )
    face_radius = jnp.concatenate((inner_radius[:1], outer_radius))
    laplacian = (
        4.0
        * jnp.pi
        * jnp.diff(face_radius**2 * face_gradient)
        / coordinate_shell_volume
    )

    return laplacian - source


def _time_symmetric_constraint_update(
    flat_metric: Z4C_Metric,
    reference_particles: particle_species,
    radial_orthonormal_momentum,
    conformal_factor,
    epsilon_0: float,
    boundary_tolerance: float,
):
    metric = flat_metric._replace(chi=conformal_factor**-4)
    particles = _particles_with_initial_metric_momentum(
        reference_particles,
        metric,
        radial_orthonormal_momentum,
    )

    charge_density = compute_radial_charge_density(particles, metric)
    E_r = solve_radial_electric_field(
        metric,
        charge_density,
        epsilon_0=epsilon_0,
    )
    matter_terms = total_matter_terms(
        particles,
        metric,
        E_r,
        epsilon_0,
    )

    # For gamma_ij = psi^4 delta_ij and K_ij = 0, the Hamiltonian constraint is
    # laplacian(psi) = -2 pi psi^5 rho.  rho includes particles and E-field energy.
    hamiltonian_source = -2.0 * jnp.pi * conformal_factor**5 * matter_terms.rho
    updated_conformal_factor = _solve_radial_poisson_equation(
        flat_metric,
        hamiltonian_source,
        boundary_tolerance,
    )

    return updated_conformal_factor, particles, charge_density, E_r, matter_terms


def solve_time_symmetric_initial_data(
    params: TwoStreamParameters,
    flat_metric: Z4C_Metric,
    particles: particle_species,
):
    """Solve the coupled conformally flat Hamiltonian and Gauss constraints."""

    radial_orthonormal_momentum = particles.ur
    conformal_factor = jnp.ones_like(flat_metric.r)
    constraint_update = jax.jit(_time_symmetric_constraint_update)

    for iteration in range(1, params.initial_data_max_iterations + 1):
        updated_conformal_factor, _, _, _, _ = constraint_update(
            flat_metric,
            particles,
            radial_orthonormal_momentum,
            conformal_factor,
            params.epsilon_0,
            params.initial_data_tolerance,
        )
        jax.block_until_ready(updated_conformal_factor)

        update = updated_conformal_factor - conformal_factor
        relative_update = float(
            jnp.max(jnp.abs(update))
            / jnp.maximum(jnp.max(jnp.abs(updated_conformal_factor)), 1.0)
        )
        if not np.isfinite(relative_update) or np.any(
            np.asarray(updated_conformal_factor) <= 0.0
        ):
            raise RuntimeError("the time-symmetric initial-data solve became invalid")

        if relative_update <= params.initial_data_tolerance:
            conformal_factor = updated_conformal_factor
            break

        conformal_factor = (
            conformal_factor
            + INITIAL_DATA_RELAXATION * update
        )
    else:
        raise RuntimeError(
            "the time-symmetric initial-data solve did not converge within "
            f"{params.initial_data_max_iterations} iterations"
        )

    metric = flat_metric._replace(chi=conformal_factor**-4)
    _, particles, charge_density, E_r, matter_terms = constraint_update(
        flat_metric,
        particles,
        radial_orthonormal_momentum,
        conformal_factor,
        params.epsilon_0,
        params.initial_data_tolerance,
    )
    jax.block_until_ready(E_r)

    hamiltonian_source = -2.0 * jnp.pi * conformal_factor**5 * matter_terms.rho
    solved_conformal_factor = _solve_radial_poisson_equation(
        flat_metric,
        hamiltonian_source,
        params.initial_data_tolerance,
    )
    fixed_point_residual = float(
        jnp.max(jnp.abs(solved_conformal_factor - conformal_factor))
        / jnp.maximum(jnp.max(jnp.abs(conformal_factor)), 1.0)
    )
    poisson_residual = _radial_poisson_residual(
        flat_metric,
        conformal_factor,
        hamiltonian_source,
    )
    hamiltonian_constraint = -8.0 * conformal_factor**-5 * poisson_residual
    hamiltonian_constraint_linf = float(
        jnp.max(jnp.abs(hamiltonian_constraint))
    )
    relative_hamiltonian_constraint_linf = float(
        jnp.max(jnp.abs(hamiltonian_constraint))
        / jnp.maximum(16.0 * jnp.pi * jnp.max(jnp.abs(matter_terms.rho)), 1.0e-30)
    )
    chi_sommerfeld_residual = float(
        jnp.abs(_chi_sommerfeld_residual(metric, conformal_factor))
    )
    if (
        not np.isfinite(chi_sommerfeld_residual)
        or chi_sommerfeld_residual > params.initial_data_tolerance
    ):
        raise RuntimeError(
            "the time-symmetric initial data do not satisfy the production "
            "chi Sommerfeld boundary condition"
        )

    initial_data_diagnostics = {
        "iterations": iteration,
        "relative_fixed_point_residual": fixed_point_residual,
        "hamiltonian_constraint_linf": hamiltonian_constraint_linf,
        "relative_hamiltonian_constraint_linf": (
            relative_hamiltonian_constraint_linf
        ),
        "chi_sommerfeld_residual": chi_sommerfeld_residual,
    }

    return (
        metric,
        particles,
        charge_density,
        E_r,
        initial_data_diagnostics,
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


def make_plasma_particles(
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
    # Each unperturbed stream is charge and current neutral.  The two signs
    # also cancel the total momentum density, including the displaced seed.
    # ur is specific momentum: ions use the same gamma*v as their electrons.
    r = np.concatenate((radius, radius, base_radius, base_radius))
    ur = np.concatenate(
        (
            np.full_like(radius, beam_momentum),
            np.full_like(radius, -beam_momentum),
            np.full_like(base_radius, beam_momentum),
            np.full_like(base_radius, -beam_momentum),
        )
    )
    weight = np.tile(one_beam_mass, 4)
    charge = np.concatenate(
        (
            np.full_like(radius, params.electron_charge_to_mass),
            np.full_like(radius, params.electron_charge_to_mass),
            np.full_like(base_radius, -params.electron_charge_to_mass),
            np.full_like(base_radius, -params.electron_charge_to_mass),
        )
    )
    mass = np.concatenate(
        (
            np.ones_like(radius),
            np.ones_like(radius),
            np.full_like(base_radius, params.ion_to_electron_mass_ratio),
            np.full_like(base_radius, params.ion_to_electron_mass_ratio),
        )
    )
    population_id = np.concatenate(
        (
            np.full(radius.size, OUTGOING_ELECTRONS, dtype=np.int32),
            np.full(radius.size, INCOMING_ELECTRONS, dtype=np.int32),
            np.full(base_radius.size, OUTGOING_IONS, dtype=np.int32),
            np.full(base_radius.size, INCOMING_IONS, dtype=np.int32),
        )
    )

    particles = particle_species(
        name="two_stream_plasma",
        charge=jnp.asarray(charge),
        mass=jnp.asarray(mass),
        weight=jnp.asarray(weight),
        r=jnp.asarray(r),
        ur=jnp.asarray(ur),
        phi=jnp.zeros_like(jnp.asarray(r)),
        uphi=jnp.zeros_like(jnp.asarray(r)),
        shape_mode=params.shape_mode,
    )

    return particles, population_id


def cold_two_stream_growth_rate(params: TwoStreamParameters) -> float:
    """Return the local cold neutral-stream amplitude growth rate."""

    gamma_b = 1.0 / math.sqrt(1.0 - params.beam_velocity**2)
    # Co-moving electrons and ions have the same Doppler denominator in the
    # longitudinal susceptibility; their plasma frequencies squared add.
    omega_squared = (
        params.plasma_frequency**2
        * (1.0 + 1.0 / params.ion_to_electron_mass_ratio)
        / gamma_b**3
    )
    kv_squared = (params.wavenumber * params.beam_velocity) ** 2
    unstable_omega_squared = (
        0.5
        * math.sqrt(omega_squared**2 + 8.0 * omega_squared * kv_squared)
        - kv_squared
        - 0.5 * omega_squared
    )

    return math.sqrt(max(unstable_omega_squared, 0.0))


def relativistic_kinetic_energy(
    particles: particle_species,
    metric: Z4C_Metric,
) -> jnp.ndarray:
    grr_p, gt_p = _interpolate_cell_centered_fields_to_particles(
        jnp.stack(
            (
                metric.conformal_grr / metric.chi,
                metric.conformal_gt / metric.chi,
            )
        ),
        particles.r,
        _radial_grid_from_metric(metric),
        shape_mode=particles.get_shape(),
        field_parities=jnp.asarray((1, 1)),
    )
    lorentz_factor = jnp.sqrt(
        1.0
        + particles.ur**2 / grr_p
        + particles.uphi**2 / (particles.r**2 * gt_p)
    )
    return jnp.sum(particles.get_mass() * (lorentz_factor - 1.0))


def total_matter_terms(
    particles: particle_species,
    metric: Z4C_Metric,
    E_r,
    epsilon_0: float,
) -> MatterTerms:
    particle_matter = compute_radial_matter_terms(particles, metric)
    field_matter = compute_electrostatic_matter_terms(
        metric,
        E_r,
        epsilon_0=epsilon_0,
    )

    return MatterTerms(
        rho=particle_matter.rho + field_matter.rho,
        Srr=particle_matter.Srr + field_matter.Srr,
        Stt=particle_matter.Stt + field_matter.Stt,
        Sr=particle_matter.Sr + field_matter.Sr,
        St=particle_matter.St + field_matter.St,
    )


def target_mode_diagnostics(
    metric: Z4C_Metric,
    E_r,
    params: TwoStreamParameters,
) -> tuple[float, float, float, float]:
    """Project the electric field onto the seeded sine/cosine mode."""

    r = np.asarray(metric.r)
    field = np.asarray(E_r)
    volume = np.asarray(proper_radial_shell_volume(metric))
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
    volume = np.asarray(proper_radial_shell_volume(metric))
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


def particles_in_populations(
    particles: particle_species,
    population_id: np.ndarray,
    populations,
) -> particle_species:
    """Return the same particle layout with other populations zero-weighted."""

    selected = jnp.asarray(np.isin(population_id, populations))
    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=jnp.where(selected, particles.weight, 0.0),
        r=particles.r,
        ur=particles.ur,
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )


def collect_diagnostic_row(
    step: int,
    time: float,
    particles: particle_species,
    metric: Z4C_Metric,
    population_id: np.ndarray,
    charge_density,
    E_r,
    params: TwoStreamParameters,
    initial_total_energy: float | None,
) -> tuple[dict[str, float | int | bool], float]:
    field_energy = float(
        electric_field_energy(metric, E_r, epsilon_0=params.epsilon_0)
    )
    kinetic_energy = float(relativistic_kinetic_energy(particles, metric))
    rest_mass_energy = float(jnp.sum(particles.get_mass()))
    outer_misner_sharp_mass = float(misner_sharp_mass(metric)[-1])
    gravitational_binding_energy = (
        outer_misner_sharp_mass
        - rest_mass_energy
        - kinetic_energy
        - field_energy
    )
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
    electron_particles = particles_in_populations(
        particles,
        population_id,
        (OUTGOING_ELECTRONS, INCOMING_ELECTRONS),
    )
    ion_particles = particles_in_populations(
        particles,
        population_id,
        (OUTGOING_IONS, INCOMING_IONS),
    )
    volume = np.asarray(proper_radial_shell_volume(metric))
    total_charge = float(np.sum(np.asarray(charge_density) * volume))
    electron_charge = float(np.sum(np.asarray(electron_particles.get_charge())))
    ion_charge = float(np.sum(np.asarray(ion_particles.get_charge())))
    minimum_r = float(np.min(np.asarray(particles.r)))
    maximum_r = float(np.max(np.asarray(particles.r)))
    boundary_margin = particle_boundary_margin(particles, params.r_max)
    finite_state = bool(
        np.all(np.isfinite(np.asarray(particles.r)))
        and np.all(np.isfinite(np.asarray(particles.ur)))
        and np.all(np.isfinite(np.asarray(E_r)))
        and np.all(np.isfinite(np.asarray(charge_density)))
        and all(
            np.all(np.isfinite(np.asarray(getattr(metric, field_name))))
            for field_name in Z4C_Metric._fields[:10]
        )
        and np.isfinite(outer_misner_sharp_mass)
        and np.isfinite(gravitational_binding_energy)
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
        "rest_mass_energy": rest_mass_energy,
        "relativistic_kinetic_energy": kinetic_energy,
        "misner_sharp_mass": outer_misner_sharp_mass,
        "gravitational_binding_energy": gravitational_binding_energy,
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
    particles: particle_species,
    population_id: np.ndarray,
    charge_density,
    E_r,
    output_directory: Path,
    step: int,
    time: float,
    dynamic_gr: bool,
    epsilon_0: float,
    initial_metric: str,
) -> Path:
    path = output_directory / f"metric_step_{step:06d}.npz"
    electron_particles = particles_in_populations(
        particles,
        population_id,
        (OUTGOING_ELECTRONS, INCOMING_ELECTRONS),
    )
    ion_particles = particles_in_populations(
        particles,
        population_id,
        (OUTGOING_IONS, INCOMING_IONS),
    )
    electron_charge_density = compute_radial_charge_density(
        electron_particles,
        metric,
    )
    ion_charge_density = compute_radial_charge_density(ion_particles, metric)
    matter_terms = total_matter_terms(
        particles,
        metric,
        E_r,
        epsilon_0,
    )
    areal_radius = metric.r * jnp.sqrt(metric.conformal_gt / metric.chi)
    mass_profile = misner_sharp_mass(metric)
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
        areal_radius=np.asarray(areal_radius),
        E_r=np.asarray(E_r),
        misner_sharp_mass=np.asarray(mass_profile),
        charge_density=np.asarray(charge_density),
        electron_charge_density=np.asarray(electron_charge_density),
        ion_charge_density=np.asarray(ion_charge_density),
        matter_rho=np.asarray(matter_terms.rho),
        matter_Srr=np.asarray(matter_terms.Srr),
        matter_Stt=np.asarray(matter_terms.Stt),
        matter_Sr=np.asarray(matter_terms.Sr),
        matter_St=np.asarray(matter_terms.St),
        step=int(step),
        time=float(time),
        diagnostic_schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        initial_metric=initial_metric,
        dynamic_gr=dynamic_gr,
        fixed_minkowski=metric_is_exactly_minkowski(metric),
        particle_state_variables="r_ur",
    )
    return path


def write_phase_space_snapshot(
    particles: particle_species,
    metric: Z4C_Metric,
    population_id: np.ndarray,
    output_directory: Path,
    step: int,
    time: float,
    dynamic_gr: bool,
) -> Path:
    chi_p, conformal_grr_p = _interpolate_cell_centered_fields_to_particles(
        jnp.stack((metric.chi, metric.conformal_grr)),
        particles.r,
        _radial_grid_from_metric(metric),
        shape_mode=particles.get_shape(),
        field_parities=jnp.asarray((1, 1)),
    )
    radial_momentum = np.asarray(particles.ur)
    radial_orthonormal_momentum = np.asarray(
        particles.ur * jnp.sqrt(chi_p / conformal_grr_p)
    )
    local_radial_velocity = radial_orthonormal_momentum / np.sqrt(
        1.0 + radial_orthonormal_momentum**2
    )
    path = output_directory / f"phase_space_step_{step:06d}.npz"
    np.savez_compressed(
        path,
        r=np.asarray(particles.r),
        ur=radial_momentum,
        radial_orthonormal_momentum=radial_orthonormal_momentum,
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
        saved_coordinates="z4c_radial",
        dynamic_gr=dynamic_gr,
        particle_state_variables="r_ur",
    )
    return path


def write_run_parameters(
    params: TwoStreamParameters,
    metric: Z4C_Metric,
    particles: particle_species,
    population_id: np.ndarray,
    initial_metric: str,
    initial_data_diagnostics: dict[str, float | int],
) -> Path:
    values = asdict(params)
    values["diagnostic_schema_version"] = DIAGNOSTIC_SCHEMA_VERSION
    values["output_directory"] = str(params.output_directory)
    values["r_min"] = 0.0
    values["grid_spacing"] = float(metric.dr)
    values["num_steps"] = int(round(params.final_time / params.dt))
    electron_mask = np.isin(population_id, (OUTGOING_ELECTRONS, INCOMING_ELECTRONS))
    ion_mask = np.isin(population_id, (OUTGOING_IONS, INCOMING_IONS))
    values["num_electrons"] = int(np.count_nonzero(electron_mask))
    values["num_ions"] = int(np.count_nonzero(ion_mask))
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
    values["initial_metric"] = initial_metric
    values["initial_data"] = {
        "spatial_metric": "conformally_flat_hamiltonian_constraint",
        "extrinsic_curvature": "zero",
        "shift": "zero",
        "lapse": "one",
        "outer_boundary": "discrete_chi_sommerfeld",
        **initial_data_diagnostics,
    }
    values["metric_evolution"] = (
        "dynamic_z4c" if params.dynamic_gr else "fixed_initial_z4c"
    )
    values["metric_outer_boundary"] = params.metric_boundary
    values["fixed_metric"] = (
        None if params.dynamic_gr else "time_symmetric_conformally_flat_z4c"
    )
    values["particle_boundary"] = "none_guarded_annulus"
    values["ion_background"] = "two_comoving_neutral_streams"
    values["nonlinear_validation_target"] = "BGK_trapped_particle_island"
    values["gauss_residual_definition"] = (
        "cell_center_to_face_reconstruction_of_finite_volume_solve"
    )
    values["particle_state_variables"] = "r_ur"
    values["total_electron_charge"] = float(
        np.sum(np.asarray(particles.get_charge())[electron_mask])
    )
    values["total_ion_charge"] = float(
        np.sum(np.asarray(particles.get_charge())[ion_mask])
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

    flat_metric = make_flat_metric(params)
    particles, population_id = make_plasma_particles(
        params,
        flat_metric,
        perturb=True,
    )

    step_jit = jax.jit(rk4_step)
    metric_boundary = METRIC_BOUNDARY_CODES[params.metric_boundary]
    (
        metric,
        particles,
        charge_density,
        E_r,
        initial_data_diagnostics,
    ) = solve_time_symmetric_initial_data(
        params,
        flat_metric,
        particles,
    )
    initial_metric = "time_symmetric_conformally_flat_z4c"

    areal_radius = np.asarray(
        metric.r * jnp.sqrt(metric.conformal_gt / metric.chi)
    )
    if np.any(np.diff(areal_radius) <= 0.0):
        raise RuntimeError(
            "the time-symmetric initial data contain a trapped/non-monotone "
            "areal-radius region; lower epsilon_0 or enlarge r_max before "
            "using the guarded-annulus two-stream evolution"
        )

    num_steps = int(round(params.final_time / params.dt))
    stage_safe_margin = 2.0 * float(metric.dr)
    if num_steps > 0:
        # The initial conformal solution has alpha=1 and gamma_rr >= 1, so
        # |dr/dt| < 1.  One dt keeps every initial RK stage outside the guard.
        stage_safe_margin += params.dt
    if particle_boundary_margin(particles, params.r_max) <= stage_safe_margin:
        raise RuntimeError(
            "the initial particles do not leave enough room for a boundary-safe "
            "RK4 stage"
        )

    metric_directory = output_directory / "metric"
    phase_space_directory = output_directory / "phase_space"
    metric_directory.mkdir(parents=True, exist_ok=True)
    phase_space_directory.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_directory / "diagnostics.csv"

    write_run_parameters(
        params,
        metric,
        particles,
        population_id,
        initial_metric,
        initial_data_diagnostics,
    )

    initial_total_energy = None
    row, initial_total_energy = collect_diagnostic_row(
        0,
        0.0,
        particles,
        metric,
        population_id,
        charge_density,
        E_r,
        params,
        initial_total_energy,
    )
    append_diagnostic_row(diagnostics_path, row)
    write_metric_snapshot(
        metric,
        particles,
        population_id,
        charge_density,
        E_r,
        metric_directory,
        0,
        0.0,
        params.dynamic_gr,
        params.epsilon_0,
        initial_metric,
    )
    write_phase_space_snapshot(
        particles,
        metric,
        population_id,
        phase_space_directory,
        0,
        0.0,
        params.dynamic_gr,
    )

    time = 0.0
    with tqdm(
        total=params.final_time,
        initial=time,
        desc=(
            "evolving dynamical Z4C two-stream"
            if params.dynamic_gr
            else "evolving fixed-background Z4C two-stream"
        ),
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
            particles, metric, charge_density, E_r = step_jit(
                particles,
                metric,
                params.dt,
                EM_on=True,
                GR_on=params.dynamic_gr,
                epsilon_0=params.epsilon_0,
                metric_boundary=metric_boundary,
            )
            jax.block_until_ready(E_r)
            time = step * params.dt
            row, _ = collect_diagnostic_row(
                step,
                time,
                particles,
                metric,
                population_id,
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
                    particles,
                    population_id,
                    charge_density,
                    E_r,
                    metric_directory,
                    step,
                    time,
                    params.dynamic_gr,
                    params.epsilon_0,
                    initial_metric,
                )
                write_phase_space_snapshot(
                    particles,
                    metric,
                    population_id,
                    phase_space_directory,
                    step,
                    time,
                    params.dynamic_gr,
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
        "dynamic_gr": params.dynamic_gr,
        "metric_boundary": params.metric_boundary,
        "fixed_minkowski": metric_is_exactly_minkowski(metric),
        "initial_metric": initial_metric,
        "initial_data_iterations": initial_data_diagnostics["iterations"],
        "initial_hamiltonian_constraint_linf": initial_data_diagnostics[
            "hamiltonian_constraint_linf"
        ],
        "initial_relative_hamiltonian_constraint_linf": (
            initial_data_diagnostics["relative_hamiltonian_constraint_linf"]
        ),
        "initial_chi_sommerfeld_residual": initial_data_diagnostics[
            "chi_sommerfeld_residual"
        ],
        "final_misner_sharp_mass": float(row["misner_sharp_mass"]),
        "final_gravitational_binding_energy": float(
            row["gravitational_binding_energy"]
        ),
        "theoretical_amplitude_growth_rate": cold_two_stream_growth_rate(params),
        "final_electric_field_energy": float(row["electric_field_energy"]),
        "final_mode_fraction": float(row["target_mode_energy_fraction"]),
    }


def parse_args() -> argparse.Namespace:
    defaults = TwoStreamParameters()
    parser = argparse.ArgumentParser(
        description=(
            "Run a cold radial two-stream instability on Z4C initial data."
        )
    )
    parser.add_argument(
        "--dynamic-gr",
        action=argparse.BooleanOptionalAction,
        default=defaults.dynamic_gr,
        help=(
            "evolve Z4C from solved time-symmetric initial data; disable to hold "
            "that initial metric fixed"
        ),
    )
    parser.add_argument(
        "--metric-boundary",
        choices=("sommerfeld", "constraint-preserving"),
        default=defaults.metric_boundary.replace("_", "-"),
        help="outer boundary for the dynamical Z4C metric",
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
        help="quiet-start macro-particles per cell in each electron or ion stream",
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
        "--ion-to-electron-mass-ratio",
        type=float,
        default=defaults.ion_to_electron_mass_ratio,
    )
    parser.add_argument(
        "--beam-velocity",
        type=float,
        default=defaults.beam_velocity,
        help="local speed of each neutral stream in the common zero-momentum frame",
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
    parser.add_argument(
        "--initial-data-tolerance",
        type=float,
        default=defaults.initial_data_tolerance,
    )
    parser.add_argument(
        "--initial-data-max-iterations",
        type=int,
        default=defaults.initial_data_max_iterations,
    )
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--final-time", type=float, default=defaults.final_time)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = replace(
        TwoStreamParameters(),
        dynamic_gr=args.dynamic_gr,
        metric_boundary=args.metric_boundary.replace("-", "_"),
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
        ion_to_electron_mass_ratio=args.ion_to_electron_mass_ratio,
        beam_velocity=args.beam_velocity,
        wavenumber=args.wavenumber,
        perturbation_amplitude=args.perturbation_amplitude,
        shape_mode=args.shape_mode,
        initial_data_tolerance=args.initial_data_tolerance,
        initial_data_max_iterations=args.initial_data_max_iterations,
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
