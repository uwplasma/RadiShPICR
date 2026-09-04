from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


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
    assert np.array_equal(ion_r, quiet_radius)
    assert np.allclose(ion_ur, 0.0)
    assert np.array_equal(
        np.asarray(particles.weight[:num_particles_per_beam]),
        beam_weight,
    )
    assert np.array_equal(
        np.asarray(particles.weight[2 * num_particles_per_beam:]),
        2.0 * beam_weight,
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
                np.full(num_particles_per_beam, module.IONS),
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
        (module.IONS,),
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


def test_cold_relativistic_growth_rate_matches_dispersion_root():
    module = load_two_stream_module()
    params = module.TwoStreamParameters()
    gamma_b = 1.0 / math.sqrt(1.0 - params.beam_velocity**2)
    effective_omega_squared = params.plasma_frequency**2 / gamma_b**3
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


def test_tiny_run_writes_consistent_metadata_and_stays_inside_guards(tmp_path):
    module = load_two_stream_module()
    output_directory = tmp_path / "z4c_two_stream"
    params = reduced_parameters(module, output_directory=output_directory)

    summary = module.run_two_stream(params)

    assert summary["finite_state"]
    assert summary["steps"] == 2
    with (output_directory / "run_parameters.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["fixed_metric"] == "minkowski_z4c"
    assert metadata["initial_metric"] == "minkowski_z4c"
    assert metadata["metric_evolution"] == "fixed_minkowski"
    assert metadata["dynamic_gr"] is False
    assert metadata["diagnostic_schema_version"] == 2
    assert metadata["particle_boundary"] == "none_guarded_annulus"
    assert metadata["ion_background"] == "mobile_particle_ions_initialized_at_rest"
    assert metadata["ion_to_electron_mass_ratio"] == 1836.0
    assert metadata["num_ions"] * 2 == metadata["num_electrons"]
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

    for metric_path, phase_path in zip(metric_paths, phase_paths):
        with np.load(metric_path) as metric_snapshot, np.load(
            phase_path
        ) as phase_snapshot:
            assert metric_snapshot["step"] == phase_snapshot["step"]
            assert metric_snapshot["time"] == phase_snapshot["time"]
            assert not metric_snapshot["dynamic_gr"]
            assert metric_snapshot["fixed_minkowski"]
            assert np.all(metric_snapshot["alpha"] == 1.0)
            assert np.all(metric_snapshot["chi"] == 1.0)
            assert metric_snapshot["diagnostic_schema_version"] == 2
            for field_name in (
                "areal_radius",
                "misner_sharp_mass",
                "kretschmann_scalar",
                "matter_rho",
                "matter_Srr",
                "matter_Stt",
                "matter_Sr",
                "matter_St",
            ):
                assert metric_snapshot[field_name].shape == metric_snapshot["r"].shape
                assert np.all(np.isfinite(metric_snapshot[field_name]))
            assert np.allclose(
                metric_snapshot["electron_charge_density"]
                + metric_snapshot["ion_charge_density"],
                metric_snapshot["charge_density"],
            )
            assert tuple(phase_snapshot["population_labels"]) == (
                "outgoing_electrons",
                "incoming_electrons",
                "ions",
            )
            assert np.count_nonzero(phase_snapshot["population_id"] == module.IONS)
            if phase_snapshot["step"] > 0:
                ion_momentum = phase_snapshot["ur"][
                    phase_snapshot["population_id"] == module.IONS
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


def test_dynamic_gr_evolves_from_flat_metric(tmp_path):
    module = load_two_stream_module()
    output_directory = tmp_path / "dynamic_z4c_two_stream"
    params = replace(
        reduced_parameters(module, output_directory=output_directory),
        dynamic_gr=True,
        dt=1.0e-4,
        final_time=1.0e-4,
    )

    summary = module.run_two_stream(params)

    assert summary["finite_state"]
    assert summary["dynamic_gr"] is True
    assert summary["fixed_minkowski"] is False

    with (output_directory / "run_parameters.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["initial_metric"] == "minkowski_z4c"
    assert metadata["metric_evolution"] == "dynamic_z4c"
    assert metadata["fixed_metric"] is None
    assert metadata["diagnostic_schema_version"] == 2

    metric_paths = sorted((output_directory / "metric").glob("metric_step_*.npz"))
    with np.load(metric_paths[0]) as initial_snapshot:
        assert initial_snapshot["dynamic_gr"]
        assert not initial_snapshot["fixed_minkowski"]
        for field_name in ("alpha", "conformal_grr", "conformal_gt", "chi"):
            assert np.all(initial_snapshot[field_name] == 1.0)
        for field_name in ("beta", "Kh", "Arr", "At", "theta", "Gamma"):
            assert np.all(initial_snapshot[field_name] == 0.0)
        assert np.all(np.isfinite(initial_snapshot["kretschmann_scalar"]))
    with np.load(metric_paths[-1]) as final_snapshot:
        assert not np.all(final_snapshot["alpha"] == 1.0)
        assert np.all(np.isfinite(final_snapshot["kretschmann_scalar"]))

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


def test_run_aborts_before_an_rk4_stage_can_enter_the_boundary(tmp_path):
    module = load_two_stream_module()
    params = replace(
        module.TwoStreamParameters(),
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
