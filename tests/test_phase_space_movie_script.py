from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "relativistic_two_stream"
    / "make_two_stream_diagnostics.py"
)


def load_diagnostic_script_module():
    spec = importlib.util.spec_from_file_location(
        "make_two_stream_diagnostics",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_snapshot(
    phase_space_directory: Path,
    step: int,
    time: float,
    radius=None,
    radial_momentum=None,
    neutral_streams=False,
):
    phase_space_directory.mkdir(parents=True, exist_ok=True)
    if radius is None:
        radius = np.array([0.1, 0.3, 0.6, 0.9]) + 0.01 * step
    if radial_momentum is None:
        radial_momentum = np.array([0.22, 0.18, -0.21, -0.17])
    population_id = np.array([0, 0, 1, 1])
    population_labels = ["outgoing_electrons", "incoming_electrons"]
    if neutral_streams:
        population_id = np.arange(4)
        population_labels += ["outgoing_ions", "incoming_ions"]
    np.savez_compressed(
        phase_space_directory / f"phase_space_step_{step:06d}.npz",
        r=np.asarray(radius),
        ur=np.asarray(radial_momentum),
        population_id=population_id,
        population_labels=np.asarray(population_labels),
        step=step,
        time=time,
    )


def write_metric_snapshot(metric_directory: Path, step: int, time: float):
    metric_directory.mkdir(parents=True, exist_ok=True)
    radius = np.linspace(0.05, 0.95, 20)
    phase = 2.0 * np.pi * radius + 0.2 * step
    perturbation = 1.0e-3 * np.sin(phase)
    zeros = np.zeros_like(radius)
    np.savez_compressed(
        metric_directory / f"metric_step_{step:06d}.npz",
        r=radius,
        alpha=1.0 + perturbation,
        beta=0.2 * perturbation,
        conformal_grr=1.0 - perturbation,
        conformal_gt=1.0 + 0.5 * perturbation,
        chi=1.0 - 0.4 * perturbation,
        Kh=0.3 * perturbation,
        Arr=-0.2 * perturbation,
        At=0.1 * perturbation,
        theta=0.05 * perturbation,
        Gamma=-0.4 * perturbation,
        E_r=0.5 * np.sin(phase),
        areal_radius=radius * (1.0 + 0.1 * perturbation),
        misner_sharp_mass=0.01 * radius,
        matter_rho=1.0e-3 * (1.0 + np.cos(phase)),
        matter_Srr=-1.0e-4 * np.cos(phase),
        matter_Stt=1.0e-4 * np.cos(phase),
        matter_Sr=zeros,
        matter_St=zeros,
        step=step,
        time=time,
        diagnostic_schema_version=2,
    )


def test_snapshot_discovery_is_numeric_and_retains_final_frame(tmp_path):
    module = load_diagnostic_script_module()
    phase_space_directory = tmp_path / "phase_space"
    write_snapshot(phase_space_directory, 10, 1.0)
    write_snapshot(phase_space_directory, 0, 0.0)
    write_snapshot(phase_space_directory, 2, 0.2)

    paths = module.discover_phase_space_paths(
        phase_space_directory,
        frame_stride=2,
    )
    frames = module.load_phase_space_frames(
        phase_space_directory,
        frame_stride=2,
    )

    assert [module.phase_space_step(path) for path in paths] == [0, 10]
    assert [frame["step"] for frame in frames] == [0, 10]
    assert frames[0]["population_labels"] == (
        "outgoing_electrons",
        "incoming_electrons",
    )
    assert np.allclose(frames[-1]["r"], [0.2, 0.4, 0.7, 1.0])

    metric_directory = tmp_path / "metric"
    write_metric_snapshot(metric_directory, 10, 1.0)
    write_metric_snapshot(metric_directory, 0, 0.0)
    write_metric_snapshot(metric_directory, 2, 0.2)
    metric_paths = module.discover_metric_paths(metric_directory, frame_stride=2)
    metric_frames = module.load_metric_frames(metric_directory, frame_stride=2)

    assert [module.metric_step(path) for path in metric_paths] == [0, 10]
    assert [frame["step"] for frame in metric_frames] == [0, 10]


def test_metric_snapshot_schema_requires_source_aware_fields(tmp_path):
    module = load_diagnostic_script_module()
    snapshot_path = tmp_path / "metric_step_000000.npz"
    np.savez_compressed(snapshot_path, r=np.array([0.1, 0.2]), step=0, time=0.0)

    with pytest.raises(RuntimeError, match="rerun the two-stream simulation"):
        module.load_metric_frame(snapshot_path)


def test_metric_and_particle_snapshots_must_be_synchronized(tmp_path):
    module = load_diagnostic_script_module()
    phase_space_directory = tmp_path / "phase_space"
    metric_directory = tmp_path / "metric"
    write_snapshot(phase_space_directory, 0, 0.0)
    write_snapshot(phase_space_directory, 1, 0.1)
    write_metric_snapshot(metric_directory, 0, 0.0)
    write_metric_snapshot(metric_directory, 2, 0.2)

    with pytest.raises(ValueError, match="not synchronized"):
        module.validate_synchronized_frames(
            module.load_phase_space_frames(phase_space_directory),
            module.load_metric_frames(metric_directory),
        )


def test_metric_movie_limits_are_fixed_from_the_complete_frame_set():
    module = load_diagnostic_script_module()
    value_chunks = [
        np.array([-1.0, 0.5]),
        np.array([-3.0, 2.0]),
        np.array([0.25, 1.5]),
    ]

    limit = module._fixed_symmetric_limit(value_chunks)

    assert limit == pytest.approx(3.15)
    assert all(np.max(np.abs(values)) < limit for values in value_chunks)
    assert module._cell_centered_radial_limits(
        np.array([0.05, 0.15, 0.25])
    ) == pytest.approx((0.0, 0.3))


def test_misner_sharp_energy_plot_tracks_mass_and_relative_drift(
    tmp_path,
    monkeypatch,
):
    module = load_diagnostic_script_module()
    time = np.array([0.0, 0.5, 1.0])
    mass = np.array([10.0, 10.1, 9.8])
    captured = {}
    subplots = module.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(module.plt, "subplots", capture_subplots)
    plot_path = tmp_path / "misner_sharp_energy.png"

    module.make_misner_sharp_energy_plot(time, mass, plot_path, dpi=60)

    axes = captured["axes"]
    assert plot_path.stat().st_size > 0
    assert np.array_equal(axes[0].lines[0].get_ydata(), mass)
    assert np.allclose(axes[1].lines[0].get_ydata(), [0.0, 0.01, -0.02])
    assert "2.000e-02" in axes[1].texts[0].get_text()


def test_snapshot_schema_requires_raw_radial_momentum(tmp_path):
    module = load_diagnostic_script_module()
    snapshot_path = tmp_path / "phase_space_step_000000.npz"
    np.savez_compressed(
        snapshot_path,
        r=np.array([0.1]),
        population_id=np.array([0]),
        population_labels=np.array(["electrons"]),
        step=0,
        time=0.0,
    )

    with pytest.raises(ValueError, match="ur"):
        module.load_phase_space_frame(snapshot_path)


def test_exponential_energy_fit_recovers_field_growth_rate():
    module = load_diagnostic_script_module()
    time = np.linspace(0.0, 12.0, 121)
    gamma = 0.37
    energy = 2.0e-10 * np.exp(2.0 * gamma * time)

    automatic_fit = module.fit_log_energy_growth(time, energy)
    manual_fit = module.fit_log_energy_growth(
        time,
        energy,
        fit_start=2.0,
        fit_end=8.0,
    )

    assert automatic_fit["auto_selected"] is True
    assert automatic_fit["energy_e_folds"] >= 2.0
    assert automatic_fit["r_squared"] >= 0.95
    assert automatic_fit["gamma"] == pytest.approx(gamma, rel=1.0e-12)
    assert manual_fit["auto_selected"] is False
    assert manual_fit["fit_start"] == pytest.approx(2.0)
    assert manual_fit["fit_end"] == pytest.approx(8.0)
    assert manual_fit["gamma"] == pytest.approx(gamma, rel=1.0e-12)


def test_growth_fit_rejects_missing_or_nonfinite_mode_purity_data():
    module = load_diagnostic_script_module()
    time = np.linspace(0.0, 6.0, 31)
    energy = np.exp(time)

    with pytest.raises(ValueError, match="mode_fraction is required"):
        module.fit_log_energy_growth(
            time,
            energy,
            minimum_mode_fraction=0.5,
        )

    mode_fraction = np.ones_like(time)
    mode_fraction[10] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        module.fit_log_energy_growth(
            time,
            energy,
            mode_fraction=mode_fraction,
            minimum_mode_fraction=0.5,
        )


def test_manual_low_purity_window_does_not_meet_campaign_acceptance():
    module = load_diagnostic_script_module()
    time = np.linspace(0.0, 6.0, 31)
    gamma = 0.4
    energy = np.exp(2.0 * gamma * time)
    mode_fraction = np.full_like(time, 0.25)

    growth_fit = module.fit_log_energy_growth(
        time,
        energy,
        fit_start=0.0,
        fit_end=6.0,
        mode_fraction=mode_fraction,
        minimum_mode_fraction=0.5,
    )
    module.add_theory_comparison(
        growth_fit,
        {"theoretical_amplitude_growth_rate": gamma},
    )

    assert growth_fit["meets_growth_criteria"] is False
    assert growth_fit["growth_rate_within_30_percent"] is True
    assert growth_fit["growth_fit_acceptance"] is False


def test_renderer_requires_mode_fraction_for_the_default_purity_gate(tmp_path):
    module = load_diagnostic_script_module()
    output_directory = tmp_path / "missing_mode_fraction"
    output_directory.mkdir()
    time = np.linspace(0.0, 6.0, 31)
    energy = np.exp(time)
    rest_mass = np.full_like(time, 10.0)
    kinetic_energy = np.ones_like(time)
    gravitational_energy = -0.5 * np.ones_like(time)
    misner_sharp_mass = rest_mass + kinetic_energy + energy + gravitational_energy
    np.savetxt(
        output_directory / "diagnostics.csv",
        np.column_stack(
            (
                time,
                energy,
                energy,
                rest_mass,
                kinetic_energy,
                misner_sharp_mass,
                gravitational_energy,
            )
        ),
        delimiter=",",
        header=(
            "time,electric_field_energy,analysis_electric_field_energy,"
            "rest_mass_energy,relativistic_kinetic_energy,misner_sharp_mass,"
            "gravitational_binding_energy"
        ),
        comments="",
    )
    with (output_directory / "run_parameters.json").open("w") as stream:
        json.dump({"wavenumber": 1.0, "diagnostic_schema_version": 2}, stream)

    with pytest.raises(ValueError, match="mode-purity threshold"):
        module.render_two_stream_diagnostics(output_directory)


def test_renderer_rejects_output_from_the_old_schema(tmp_path):
    module = load_diagnostic_script_module()
    output_directory = tmp_path / "old_output"
    output_directory.mkdir()
    with (output_directory / "run_parameters.json").open("w") as stream:
        json.dump({"wavenumber": 1.0}, stream)

    with pytest.raises(RuntimeError, match="rerun the two-stream simulation"):
        module.render_two_stream_diagnostics(output_directory)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the movie smoke test",
)
@pytest.mark.parametrize("neutral_streams", [False, True])
def test_render_writes_all_plots_and_two_frame_h264_movies(tmp_path, neutral_streams):
    module = load_diagnostic_script_module()
    output_directory = tmp_path / "two_stream_output"
    phase_space_directory = output_directory / "phase_space"
    output_directory.mkdir()

    time = np.linspace(0.0, 6.0, 31)
    gamma = 0.4
    energy = 1.0e-9 * np.exp(2.0 * gamma * time)
    total_energy = 3.0 * energy
    kinetic_energy = np.full_like(time, 0.2)
    rest_mass_energy = np.full_like(time, 10.0)
    gravitational_energy = -0.1 + 0.005 * np.sin(time)
    misner_sharp_mass = (
        rest_mass_energy + gravitational_energy + total_energy + kinetic_energy
    )
    np.savetxt(
        output_directory / "diagnostics.csv",
        np.column_stack(
            (
                time,
                total_energy,
                energy,
                np.ones_like(time),
                rest_mass_energy,
                kinetic_energy,
                misner_sharp_mass,
                gravitational_energy,
            )
        ),
        delimiter=",",
        header=(
            "time,electric_field_energy,analysis_electric_field_energy,"
            "target_mode_energy_fraction,rest_mass_energy,"
            "relativistic_kinetic_energy,misner_sharp_mass,"
            "gravitational_binding_energy"
        ),
        comments="",
    )
    with (output_directory / "run_parameters.json").open("w") as stream:
        json.dump(
            {
                "r_min": 0.0,
                "r_max": 1.0,
                "analysis_r_min": 0.1,
                "analysis_r_max": 0.9,
                "plasma_r_min": 0.0,
                "wavenumber": 4.0 * np.pi,
                "theoretical_amplitude_growth_rate": gamma,
                "diagnostic_schema_version": 2,
            },
            stream,
        )

    write_snapshot(phase_space_directory, 0, 0.0, neutral_streams=neutral_streams)
    write_snapshot(
        phase_space_directory,
        1,
        0.1,
        radius=[0.12, 0.32, 0.58, 0.88],
        radial_momentum=[0.20, 0.24, -0.18, -0.23],
        neutral_streams=neutral_streams,
    )
    metric_directory = output_directory / "metric"
    write_metric_snapshot(metric_directory, 0, 0.0)
    write_metric_snapshot(metric_directory, 1, 0.1)

    paths = module.render_two_stream_diagnostics(
        output_directory,
        max_frames=2,
        fps=2,
        dpi=60,
    )

    assert paths["energy_plot"].stat().st_size > 0
    assert paths["misner_sharp_energy_plot"].stat().st_size > 0
    assert paths["energy_composition_plot"].stat().st_size > 0
    assert paths["growth_fit"].stat().st_size > 0
    movie_names = (
        "phase_space_movie",
        "metric_geometry_movie",
        "extrinsic_z4c_movie",
        "electric_field_movie",
    )
    for movie_name in movie_names:
        assert paths[movie_name].stat().st_size > 0
    with paths["growth_fit"].open() as stream:
        growth_fit = json.load(stream)
    assert growth_fit["gamma"] == pytest.approx(gamma, rel=1.0e-12)
    assert growth_fit["fit_energy_column"] == "analysis_electric_field_energy"
    assert growth_fit["fit_window_minimum_mode_fraction"] == pytest.approx(1.0)
    assert growth_fit["energy_amplification"] > np.exp(2.0)
    assert growth_fit["growth_rate_relative_error"] == pytest.approx(0.0)
    assert growth_fit["growth_rate_within_30_percent"] is True
    assert growth_fit["growth_fit_acceptance"] is True

    for movie_name in movie_names:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,nb_frames",
                "-of",
                "json",
                str(paths[movie_name]),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["codec_name"] == "h264"
        assert stream["pix_fmt"] == "yuv420p"
        assert int(stream["nb_frames"]) == 2
