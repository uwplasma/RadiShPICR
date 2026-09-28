from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.Z4C.spatial_metric import dchidt


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "relativistic_two_stream"
    / "run_two_stream.py"
)


def load_two_stream_module():
    module_name = "z4c_relativistic_two_stream_demo"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reduced_parameters(module, output_directory=None):
    replacements = {
        "r_max": 8.0,
        "plasma_r_min": 2.0,
        "plasma_r_max": 6.0,
        "analysis_r_min": 3.0,
        "analysis_r_max": 5.0,
        "num_cells": 160,
        "particles_per_cell": 4,
        "dt": 0.01,
        "final_time": 0.02,
        "save_every": 1,
    }
    if output_directory is not None:
        replacements["output_directory"] = output_directory

    return replace(module.TwoStreamParameters(), **replacements)


def test_metric_boundary_cli_selects_constraint_preserving(monkeypatch):
    module = load_two_stream_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), "--metric-boundary", "constraint-preserving"],
    )

    args = module.parse_args()

    assert args.metric_boundary == "constraint-preserving"


def test_flat_metric_is_exact_cell_centered_minkowski_data():
    module = load_two_stream_module()
    params = reduced_parameters(module)
    metric = module.make_flat_metric(params)
    expected_r = (jnp.arange(params.num_cells) + 0.5) * (
        params.r_max / params.num_cells
    )

    assert jnp.array_equal(metric.r, expected_r)
    assert math.isclose(float(metric.dr), params.r_max / params.num_cells)
    for field in (
        metric.alpha,
        metric.conformal_grr,
        metric.conformal_gt,
        metric.chi,
    ):
        assert jnp.array_equal(field, jnp.ones_like(metric.r))
    for field in (
        metric.beta,
        metric.Kh,
        metric.Arr,
        metric.At,
        metric.theta,
        metric.Gamma,
    ):
        assert jnp.array_equal(field, jnp.zeros_like(metric.r))


def test_quiet_start_is_cold_symmetric_and_uses_spherical_volume_quantiles():
    module = load_two_stream_module()
    params = reduced_parameters(module)
    metric = module.make_flat_metric(params)
    quiet_radius, beam_weight = module.quiet_start_one_beam(params, metric)
    particles, population_id = module.make_plasma_particles(
        params,
        metric,
        perturb=False,
    )

    num_particles_per_beam = quiet_radius.size
    outgoing_r = np.asarray(particles.r[:num_particles_per_beam])
    incoming_r = np.asarray(
        particles.r[num_particles_per_beam:2 * num_particles_per_beam]
    )
    outgoing_ur = np.asarray(particles.ur[:num_particles_per_beam])
    incoming_ur = np.asarray(
        particles.ur[num_particles_per_beam:2 * num_particles_per_beam]
    )
    ion_r = np.asarray(particles.r[2 * num_particles_per_beam:])
    ion_ur = np.asarray(particles.ur[2 * num_particles_per_beam:])
    gamma_b = 1.0 / math.sqrt(1.0 - params.beam_velocity**2)

    assert np.array_equal(outgoing_r, quiet_radius)
    assert np.array_equal(incoming_r, quiet_radius)
    assert np.allclose(outgoing_ur, gamma_b * params.beam_velocity)
    assert np.allclose(incoming_ur, -outgoing_ur)
    assert np.array_equal(ion_r, np.tile(quiet_radius, 2))
    assert np.allclose(ion_ur, np.concatenate((outgoing_ur, incoming_ur)))
    assert np.array_equal(
        np.asarray(particles.weight[:num_particles_per_beam]),
        beam_weight,
    )
    assert np.array_equal(
        np.asarray(particles.weight[2 * num_particles_per_beam:]),
        np.tile(beam_weight, 2),
    )
    assert np.allclose(
        np.asarray(particles.masses[2 * num_particles_per_beam:]),
        params.ion_to_electron_mass_ratio,
    )
    assert np.allclose(
        np.asarray(particles.charges[2 * num_particles_per_beam:]),
        -params.electron_charge_to_mass,
    )
    assert np.array_equal(
        population_id,
        np.concatenate(
            (
                np.full(num_particles_per_beam, module.OUTGOING_ELECTRONS),
                np.full(num_particles_per_beam, module.INCOMING_ELECTRONS),
                np.full(num_particles_per_beam, module.OUTGOING_IONS),
                np.full(num_particles_per_beam, module.INCOMING_IONS),
            )
        ),
    )

    radii_by_cell = quiet_radius.reshape(
        -1,
        params.particles_per_cell,
    )
    radius_cubed_spacing = np.diff(radii_by_cell**3, axis=1)
    assert np.allclose(
        radius_cubed_spacing,
        radius_cubed_spacing[:, :1],
    )

    # Splitting the old ions changes their kinetic energy, not rest mass or
    # charge.  Each moving electron-ion pair cancels charge and current.
    expected_electron_mass = (
        params.epsilon_0 * params.plasma_frequency**2
        / params.electron_charge_to_mass**2
        * (4.0 * np.pi / 3.0)
        * (params.plasma_r_max**3 - params.plasma_r_min**3)
    )
    assert np.isclose(
        np.sum(np.asarray(particles.get_mass())),
        expected_electron_mass * (1.0 + params.ion_to_electron_mass_ratio),
        rtol=1.0e-14, atol=0.0,
    )
    charge = np.asarray(particles.get_charge()).reshape(4, -1)
    velocity = np.asarray(particles.ur).reshape(4, -1)
    velocity = velocity / np.sqrt(1.0 + velocity**2)
    assert np.array_equal(charge[:2], -charge[2:])
    assert np.array_equal(charge[:2] * velocity[:2], -charge[2:] * velocity[2:])


def test_initial_ions_cancel_quiet_electrons_and_seeded_k_dominates():
    module = load_two_stream_module()
    params = reduced_parameters(module)
    metric = module.make_flat_metric(params)
    quiet_plasma, quiet_population_id = module.make_plasma_particles(
        params,
        metric,
        perturb=False,
    )
    quiet_charge_density = module.compute_radial_charge_density(
        quiet_plasma,
        metric,
    )
    quiet_electrons = module.particles_in_populations(
        quiet_plasma,
        quiet_population_id,
        (module.OUTGOING_ELECTRONS, module.INCOMING_ELECTRONS),
    )
    quiet_ions = module.particles_in_populations(
        quiet_plasma,
        quiet_population_id,
        (module.OUTGOING_IONS, module.INCOMING_IONS),
    )
    quiet_electron_density = module.compute_radial_charge_density(
        quiet_electrons,
        metric,
    )
    ion_charge_density = module.compute_radial_charge_density(quiet_ions, metric)
    perturbed_plasma, _ = module.make_plasma_particles(
        params,
        metric,
        perturb=True,
    )
    total_charge_density = module.compute_radial_charge_density(
        perturbed_plasma,
        metric,
    )

    assert jnp.allclose(
        ion_charge_density,
        -quiet_electron_density,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert jnp.allclose(quiet_charge_density, 0.0, rtol=0.0, atol=1.0e-14)

    r = np.asarray(metric.r)
    inside = (r >= params.analysis_r_min) & (r < params.analysis_r_max)
    seeded_charge = np.asarray(total_charge_density)[inside]
    spectrum = np.abs(np.fft.rfft(seeded_charge - np.mean(seeded_charge)))
    mode_number = int(
        round(
            params.wavenumber
            * (params.analysis_r_max - params.analysis_r_min)
            / (2.0 * np.pi)
        )
    )
    competing_modes = np.delete(spectrum[1:], mode_number - 1)

    assert mode_number == 1
    assert spectrum[mode_number] > 5.0 * np.max(competing_modes)


def test_epsilon_normalization_preserves_the_flat_space_electric_field():
    module = load_two_stream_module()
    weak_params = reduced_parameters(module)
    unit_params = replace(weak_params, epsilon_0=1.0)
    metric = module.make_flat_metric(weak_params)
    weak_particles, _ = module.make_plasma_particles(weak_params, metric)
    unit_particles, _ = module.make_plasma_particles(unit_params, metric)

    weak_charge_density = module.compute_radial_charge_density(
        weak_particles,
        metric,
    )
    unit_charge_density = module.compute_radial_charge_density(
        unit_particles,
        metric,
    )
    weak_E_r = module.solve_radial_electric_field(
        metric,
        weak_charge_density,
        epsilon_0=weak_params.epsilon_0,
    )
    unit_E_r = module.solve_radial_electric_field(
        metric,
        unit_charge_density,
        epsilon_0=unit_params.epsilon_0,
    )

    assert jnp.allclose(
        weak_charge_density,
        weak_params.epsilon_0 * unit_charge_density,
    )
    assert jnp.allclose(weak_E_r, unit_E_r)


def test_zero_source_poisson_solve_preserves_exact_flat_conformal_factor():
    module = load_two_stream_module()
    params = reduced_parameters(module)
    metric = module.make_flat_metric(params)

    conformal_factor = module._solve_radial_poisson_equation(
        metric,
        jnp.zeros_like(metric.r),
        params.initial_data_tolerance,
    )

    assert jnp.array_equal(conformal_factor, jnp.ones_like(metric.r))
    assert abs(
        module._chi_sommerfeld_residual(metric, conformal_factor)
    ) <= params.initial_data_tolerance


def test_time_symmetric_initial_data_solves_hamiltonian_and_momentum_constraints():
    module = load_two_stream_module()
    params = replace(
        reduced_parameters(module),
        num_cells=80,
        particles_per_cell=2,
    )
    flat_metric = module.make_flat_metric(params)
    reference_particles, _ = module.make_plasma_particles(
        params,
        flat_metric,
        perturb=True,
    )

    (
        metric,
        particles,
        charge_density,
        E_r,
        initial_data_diagnostics,
    ) = module.solve_time_symmetric_initial_data(
        params,
        flat_metric,
        reference_particles,
    )

    assert (
        initial_data_diagnostics["iterations"]
        < params.initial_data_max_iterations
    )
    assert (
        initial_data_diagnostics["relative_fixed_point_residual"]
        <= params.initial_data_tolerance
    )
    assert initial_data_diagnostics["hamiltonian_constraint_linf"] < 1.0e-11
    assert (
        initial_data_diagnostics["relative_hamiltonian_constraint_linf"]
        < 1.0e-7
    )
    assert np.all(np.asarray(metric.chi) > 0.0)
    assert not np.all(np.asarray(metric.chi) == 1.0)
    areal_radius = np.asarray(
        metric.r * jnp.sqrt(metric.conformal_gt / metric.chi)
    )
    assert np.all(np.diff(areal_radius) > 0.0)
    assert (
        initial_data_diagnostics["chi_sommerfeld_residual"]
        <= params.initial_data_tolerance
    )
    for field in (
        metric.beta,
        metric.Kh,
        metric.Arr,
        metric.At,
        metric.theta,
        metric.Gamma,
    ):
        assert np.all(np.asarray(field) == 0.0)

    radial_metric_at_particles = (
        module._interpolate_cell_centered_fields_to_particles(
            (metric.conformal_grr / metric.chi)[jnp.newaxis, :],
            particles.r,
            module._radial_grid_from_metric(metric),
            shape_mode=particles.get_shape(),
            field_parities=jnp.asarray((1,)),
        )[0]
    )
    local_radial_momentum = particles.ur / jnp.sqrt(radial_metric_at_particles)
    assert jnp.allclose(local_radial_momentum, reference_particles.ur)

    matter_terms = module.total_matter_terms(
        particles,
        metric,
        E_r,
        params.epsilon_0,
    )
    particle_matter = module.compute_radial_matter_terms(particles, metric)
    momentum_scale = jnp.max(jnp.abs(matter_terms.rho))
    assert jnp.max(jnp.abs(particle_matter.Sr)) < 1.0e-12 * momentum_scale
    assert abs(dchidt(metric, matter_terms)[-1]) <= params.initial_data_tolerance
    assert jnp.all(jnp.isfinite(charge_density))
    assert jnp.all(jnp.isfinite(E_r))


def test_sommerfeld_matched_initial_data_removes_first_step_mass_kick():
    module = load_two_stream_module()
    params = replace(
        reduced_parameters(module),
        # Radius and time are both reduced by ten while epsilon_0 increases
        # by one hundred, preserving the default gravitational compactness.
        epsilon_0=1.0e-6,
        dt=0.0025,
        final_time=0.0025,
    )
    flat_metric = module.make_flat_metric(params)
    reference_particles, _ = module.make_plasma_particles(
        params,
        flat_metric,
        perturb=True,
    )
    metric, particles, _, _, _ = module.solve_time_symmetric_initial_data(
        params,
        flat_metric,
        reference_particles,
    )

    initial_mass = module.misner_sharp_mass(metric)[-1]
    _, final_metric, _, _ = module.jax.jit(module.rk4_step)(
        particles,
        metric,
        params.dt,
        EM_on=True,
        GR_on=True,
        epsilon_0=params.epsilon_0,
    )
    module.jax.block_until_ready(final_metric.chi)
    final_mass = module.misner_sharp_mass(final_metric)[-1]
    relative_mass_drift = abs((final_mass - initial_mass) / initial_mass)

    assert relative_mass_drift < 1.0e-4

    _, cp_metric, _, _ = module.jax.jit(module.rk4_step)(
        particles,
        metric,
        params.dt,
        EM_on=True,
        GR_on=True,
        epsilon_0=params.epsilon_0,
        metric_boundary=module.METRIC_BOUNDARY_CONSTRAINT_PRESERVING,
    )
    module.jax.block_until_ready(cp_metric.chi)
    cp_mass = module.misner_sharp_mass(cp_metric)[-1]
    cp_relative_mass_drift = abs((cp_mass - initial_mass) / initial_mass)

    assert cp_relative_mass_drift < 1.0e-8


def test_constraint_preserving_boundary_reduces_crossing_time_mass_drift():
    module = load_two_stream_module()
    params = replace(
        reduced_parameters(module),
        particles_per_cell=2,
        epsilon_0=1.0e-6,
        dt=0.0025,
        final_time=5.0,
    )
    flat_metric = module.make_flat_metric(params)
    reference_particles, _ = module.make_plasma_particles(
        params,
        flat_metric,
        perturb=True,
    )
    metric, particles, _, _, _ = module.solve_time_symmetric_initial_data(
        params,
        flat_metric,
        reference_particles,
    )
    initial_mass = module.misner_sharp_mass(metric)[-1]
    num_steps = int(round(params.final_time / params.dt))

    def evolve(selected_boundary):
        def advance_one(carry, _):
            stage_particles, stage_metric = carry
            stage_particles, stage_metric, _, _ = module.rk4_step(
                stage_particles,
                stage_metric,
                params.dt,
                EM_on=True,
                GR_on=True,
                epsilon_0=params.epsilon_0,
                metric_boundary=selected_boundary,
            )
            mass = module.misner_sharp_mass(stage_metric)[-1]
            return (stage_particles, stage_metric), mass

        (_, final_metric), mass = module.jax.lax.scan(
            advance_one,
            (particles, metric),
            xs=None,
            length=num_steps,
        )
        return final_metric, mass

    evolve_jit = module.jax.jit(evolve)
    _, sommerfeld_mass = evolve_jit(module.METRIC_BOUNDARY_SOMMERFELD)
    cp_metric, cp_mass = evolve_jit(
        module.METRIC_BOUNDARY_CONSTRAINT_PRESERVING
    )
    module.jax.block_until_ready(cp_metric.chi)

    sommerfeld_drift = jnp.max(
        jnp.abs((sommerfeld_mass - initial_mass) / initial_mass)
    )
    cp_drift = jnp.max(jnp.abs((cp_mass - initial_mass) / initial_mass))

    assert jnp.all(jnp.isfinite(jnp.stack(cp_metric[:10])))
    assert cp_drift < 2.0e-3
    assert cp_drift < sommerfeld_drift / 20.0


@pytest.mark.parametrize("mass_ratio", [1.0, 1836.0])
def test_cold_relativistic_growth_rate_matches_dispersion_root(mass_ratio):
    module = load_two_stream_module()
    params = replace(module.TwoStreamParameters(), ion_to_electron_mass_ratio=mass_ratio)
    gamma_b = 1.0 / math.sqrt(1.0 - params.beam_velocity**2)
    effective_omega_squared = (
        params.plasma_frequency**2
        * (1.0 + 1.0 / params.ion_to_electron_mass_ratio)
        / gamma_b**3
    )
    kv_squared = (
        params.wavenumber * params.beam_velocity
    ) ** 2
    expected_growth_squared = (
        0.5
        * math.sqrt(
            effective_omega_squared**2
            + 8.0 * effective_omega_squared * kv_squared
        )
        - kv_squared
        - 0.5 * effective_omega_squared
    )
    growth_rate = module.cold_two_stream_growth_rate(params)

    assert math.isclose(growth_rate**2, expected_growth_squared)
    assert growth_rate > 0.0
    omega = 1j * growth_rate
    dielectric = 1.0
    for mass in (1.0, mass_ratio):
        for velocity in (-params.beam_velocity, params.beam_velocity):
            dielectric -= (
                params.plasma_frequency**2 / (2.0 * mass * gamma_b**3)
                / (omega - params.wavenumber * velocity)**2
            )
    assert abs(dielectric) < 1.0e-12


@pytest.mark.parametrize("dynamic_gr", [False, True])
def test_unperturbed_neutral_streams_do_not_generate_edge_fields(dynamic_gr):
    module = load_two_stream_module()
    params = replace(
        module.TwoStreamParameters(), num_cells=400, particles_per_cell=4,
        dynamic_gr=dynamic_gr,
    )
    flat_metric = module.make_flat_metric(params)
    particles, _ = module.make_plasma_particles(params, flat_metric, perturb=False)
    metric, particles, _, _, _ = module.solve_time_symmetric_initial_data(
        params, flat_metric, particles,
    )
    initial_metric = metric
    initial_charge = float(jnp.sum(particles.get_charge()))
    step = jax.jit(module.rk4_step)
    for _ in range(40):
        particles, metric, charge_density, E_r = step(
            particles, metric, params.dt, EM_on=True, GR_on=dynamic_gr,
            epsilon_0=params.epsilon_0,
            metric_boundary=module.METRIC_BOUNDARY_CODES[params.metric_boundary],
        )
    assert np.all(np.isfinite(np.asarray(particles.r)))
    assert float(jnp.max(jnp.abs(E_r))) < 1.0e-10
    assert float(jnp.max(jnp.abs(charge_density))) / params.epsilon_0 < 1.0e-10
    assert abs(float(jnp.sum(particles.get_charge())) - initial_charge) < 1.0e-15
    if not dynamic_gr:
        for initial_field, final_field in zip(initial_metric, metric):
            assert np.array_equal(initial_field, final_field)


def test_tiny_run_writes_consistent_metadata_and_stays_inside_guards(tmp_path):
    module = load_two_stream_module()
    output_directory = tmp_path / "z4c_two_stream"
    params = replace(
        reduced_parameters(module, output_directory=output_directory),
        dynamic_gr=False,
    )

    summary = module.run_two_stream(params)

    assert summary["finite_state"]
    assert summary["steps"] == 2
    assert summary["fixed_minkowski"] is False
    with (output_directory / "run_parameters.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["fixed_metric"] == "time_symmetric_conformally_flat_z4c"
    assert metadata["initial_metric"] == "time_symmetric_conformally_flat_z4c"
    assert (
        metadata["initial_data"]["spatial_metric"]
        == "conformally_flat_hamiltonian_constraint"
    )
    assert metadata["initial_data"]["extrinsic_curvature"] == "zero"
    assert metadata["initial_data"]["iterations"] > 0
    assert metadata["initial_data"]["relative_fixed_point_residual"] <= (
        params.initial_data_tolerance
    )
    assert metadata["initial_data"]["hamiltonian_constraint_linf"] < 1.0e-11
    assert metadata["initial_data"]["outer_boundary"] == "discrete_chi_sommerfeld"
    assert metadata["initial_data"]["chi_sommerfeld_residual"] <= (
        params.initial_data_tolerance
    )
    assert summary["initial_chi_sommerfeld_residual"] <= (
        params.initial_data_tolerance
    )
    assert metadata["metric_evolution"] == "fixed_initial_z4c"
    assert metadata["metric_outer_boundary"] == params.metric_boundary
    assert summary["metric_boundary"] == params.metric_boundary
    assert metadata["dynamic_gr"] is False
    assert metadata["diagnostic_schema_version"] == 2
    assert metadata["particle_boundary"] == "none_guarded_annulus"
    assert metadata["ion_background"] == "two_comoving_neutral_streams"
    assert metadata["ion_to_electron_mass_ratio"] == 1836.0
    assert metadata["num_ions"] == metadata["num_electrons"]
    assert metadata["normalization"]["speed_of_light"] == 1.0
    assert metadata["normalization"]["combined_electron_plasma_frequency"] == (
        params.plasma_frequency
    )
    assert metadata["geometry"]["coordinate_system"] == "spherical_radial"
    assert metadata["nonlinear_validation_target"] == (
        "BGK_trapped_particle_island"
    )
    assert metadata["beam_to_beam_relative_velocity"] == (
        2.0 * params.beam_velocity / (1.0 + params.beam_velocity**2)
    )

    metric_paths = sorted((output_directory / "metric").glob("metric_step_*.npz"))
    phase_paths = sorted(
        (output_directory / "phase_space").glob("phase_space_step_*.npz")
    )
    assert len(metric_paths) == len(phase_paths) == 3

    initial_metric_fields = None
    for metric_path, phase_path in zip(metric_paths, phase_paths):
        with np.load(metric_path) as metric_snapshot, np.load(
            phase_path
        ) as phase_snapshot:
            assert metric_snapshot["step"] == phase_snapshot["step"]
            assert metric_snapshot["time"] == phase_snapshot["time"]
            assert not metric_snapshot["dynamic_gr"]
            assert not metric_snapshot["fixed_minkowski"]
            assert np.all(metric_snapshot["alpha"] == 1.0)
            assert np.all(metric_snapshot["chi"] > 0.0)
            assert not np.all(metric_snapshot["chi"] == 1.0)
            assert metric_snapshot["diagnostic_schema_version"] == 2
            metric_fields = tuple(
                np.array(metric_snapshot[field_name])
                for field_name in (
                    "alpha",
                    "beta",
                    "conformal_grr",
                    "conformal_gt",
                    "chi",
                    "Kh",
                    "Arr",
                    "At",
                    "theta",
                    "Gamma",
                )
            )
            if initial_metric_fields is None:
                initial_metric_fields = metric_fields
            else:
                for metric_field, initial_field in zip(
                    metric_fields,
                    initial_metric_fields,
                ):
                    assert np.array_equal(metric_field, initial_field)
            for field_name in (
                "areal_radius",
                "misner_sharp_mass",
                "matter_rho",
                "matter_Srr",
                "matter_Stt",
                "matter_Sr",
                "matter_St",
            ):
                assert metric_snapshot[field_name].shape == metric_snapshot["r"].shape
                assert np.all(np.isfinite(metric_snapshot[field_name]))
            assert "kretschmann_scalar" not in metric_snapshot.files
            assert np.allclose(
                metric_snapshot["electron_charge_density"]
                + metric_snapshot["ion_charge_density"],
                metric_snapshot["charge_density"],
            )
            assert tuple(phase_snapshot["population_labels"]) == (
                "outgoing_electrons",
                "incoming_electrons",
                "outgoing_ions",
                "incoming_ions",
            )
            for population in (module.OUTGOING_IONS, module.INCOMING_IONS):
                assert np.count_nonzero(phase_snapshot["population_id"] == population)
            if phase_snapshot["step"] > 0:
                ion_momentum = phase_snapshot["ur"][
                    np.isin(
                        phase_snapshot["population_id"],
                        (module.OUTGOING_IONS, module.INCOMING_IONS),
                    )
                ]
                assert np.max(np.abs(ion_momentum)) > 0.0
            assert np.min(phase_snapshot["r"]) > 0.0
            assert np.max(phase_snapshot["r"]) < params.r_max
            assert np.all(phase_snapshot["r"] > params.plasma_r_min - 0.1)
            assert np.all(phase_snapshot["r"] < params.plasma_r_max + 0.1)

    diagnostics = np.genfromtxt(
        output_directory / "diagnostics.csv",
        delimiter=",",
        names=True,
    )
    assert {
        "electric_field_energy",
        "analysis_electric_field_energy",
        "rest_mass_energy",
        "relativistic_kinetic_energy",
        "misner_sharp_mass",
        "gravitational_binding_energy",
        "target_mode_amplitude",
        "target_mode_power",
        "total_charge",
        "relative_gauss_residual_linf",
        "relative_total_energy_drift",
        "minimum_r",
        "maximum_r",
    }.issubset(diagnostics.dtype.names)
    assert diagnostics.size == 3
    assert np.array_equal(diagnostics["step"], np.arange(3))
    assert np.allclose(diagnostics["time"], np.asarray([0.0, 0.01, 0.02]))
    assert np.all(diagnostics["finite_state"])
    assert np.allclose(
        diagnostics["misner_sharp_mass"] - diagnostics["rest_mass_energy"],
        diagnostics["gravitational_binding_energy"]
        + diagnostics["relativistic_kinetic_energy"]
        + diagnostics["electric_field_energy"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dynamic_gr_evolves_from_time_symmetric_metric(tmp_path):
    module = load_two_stream_module()
    output_directory = tmp_path / "dynamic_z4c_two_stream"
    params = replace(
        reduced_parameters(module, output_directory=output_directory),
        dynamic_gr=True,
        metric_boundary="constraint_preserving",
        dt=1.0e-4,
        final_time=1.0e-4,
    )

    summary = module.run_two_stream(params)

    assert summary["finite_state"]
    assert summary["dynamic_gr"] is True
    assert summary["fixed_minkowski"] is False

    with (output_directory / "run_parameters.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["initial_metric"] == "time_symmetric_conformally_flat_z4c"
    assert (
        metadata["initial_data"]["spatial_metric"]
        == "conformally_flat_hamiltonian_constraint"
    )
    assert metadata["initial_data"]["extrinsic_curvature"] == "zero"
    assert metadata["initial_data"]["relative_fixed_point_residual"] <= (
        params.initial_data_tolerance
    )
    assert metadata["initial_data"]["hamiltonian_constraint_linf"] < 1.0e-11
    assert metadata["initial_data"]["outer_boundary"] == "discrete_chi_sommerfeld"
    assert metadata["initial_data"]["chi_sommerfeld_residual"] <= (
        params.initial_data_tolerance
    )
    assert metadata["metric_evolution"] == "dynamic_z4c"
    assert metadata["metric_outer_boundary"] == "constraint_preserving"
    assert summary["metric_boundary"] == "constraint_preserving"
    assert metadata["fixed_metric"] is None
    assert metadata["diagnostic_schema_version"] == 2

    metric_paths = sorted((output_directory / "metric").glob("metric_step_*.npz"))
    with np.load(metric_paths[0]) as initial_snapshot:
        assert initial_snapshot["dynamic_gr"]
        assert not initial_snapshot["fixed_minkowski"]
        for field_name in ("alpha", "conformal_grr", "conformal_gt"):
            assert np.all(initial_snapshot[field_name] == 1.0)
        assert np.all(initial_snapshot["chi"] > 0.0)
        assert not np.all(initial_snapshot["chi"] == 1.0)
        for field_name in ("beta", "Kh", "Arr", "At", "theta", "Gamma"):
            assert np.all(initial_snapshot[field_name] == 0.0)
        assert "kretschmann_scalar" not in initial_snapshot.files
    with np.load(metric_paths[-1]) as final_snapshot:
        assert not np.all(final_snapshot["alpha"] == 1.0)
        assert "kretschmann_scalar" not in final_snapshot.files

    diagnostics = np.atleast_1d(
        np.genfromtxt(
            output_directory / "diagnostics.csv",
            delimiter=",",
            names=True,
        )
    )
    assert np.allclose(
        diagnostics["misner_sharp_mass"] - diagnostics["rest_mass_energy"],
        diagnostics["gravitational_binding_energy"]
        + diagnostics["relativistic_kinetic_energy"]
        + diagnostics["electric_field_energy"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("dynamic_gr", (False, True))
def test_run_rejects_a_trapped_initial_slice(tmp_path, dynamic_gr):
    module = load_two_stream_module()
    params = replace(
        reduced_parameters(
            module,
            output_directory=tmp_path / "trapped_initial_slice",
        ),
        epsilon_0=1.0,
        num_cells=40,
        particles_per_cell=1,
        final_time=0.0,
        dynamic_gr=dynamic_gr,
    )

    with pytest.raises(RuntimeError, match="trapped/non-monotone"):
        module.run_two_stream(params)

    assert not params.output_directory.exists()


def test_run_aborts_before_an_rk4_stage_can_enter_the_boundary(tmp_path):
    module = load_two_stream_module()
    params = replace(
        module.TwoStreamParameters(),
        dynamic_gr=False,
        output_directory=tmp_path / "too_close_to_boundary",
        r_max=1.0,
        plasma_r_min=0.11,
        plasma_r_max=0.89,
        analysis_r_min=0.2,
        analysis_r_max=0.8,
        num_cells=20,
        particles_per_cell=2,
        dt=0.02,
        final_time=0.02,
        save_every=1,
    )

    with pytest.raises(RuntimeError, match="boundary-safe RK4 stage"):
        module.run_two_stream(params)

    assert not params.output_directory.exists()

    safe_params = reduced_parameters(
        module,
        output_directory=params.output_directory,
    )
    safe_params = replace(safe_params, dynamic_gr=False)
    module.run_two_stream(safe_params)
    assert (params.output_directory / "diagnostics.csv").exists()


def test_run_refuses_to_overwrite_a_nonempty_output_directory(tmp_path):
    module = load_two_stream_module()
    output_directory = tmp_path / "existing_run"
    output_directory.mkdir()
    marker = output_directory / "keep.txt"
    marker.write_text("preserve existing output\n")
    params = reduced_parameters(module, output_directory=output_directory)

    with pytest.raises(RuntimeError, match="Output directory is not empty"):
        module.run_two_stream(params)

    assert marker.read_text() == "preserve existing output\n"
