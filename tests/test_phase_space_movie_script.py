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
):
    phase_space_directory.mkdir(parents=True, exist_ok=True)
    if radius is None:
        radius = np.array([0.1, 0.3, 0.6, 0.9]) + 0.01 * step
    if radial_momentum is None:
        radial_momentum = np.array([0.22, 0.18, -0.21, -0.17])
    np.savez_compressed(
        phase_space_directory / f"phase_space_step_{step:06d}.npz",
        r=np.asarray(radius),
        ur=np.asarray(radial_momentum),
        population_id=np.array([0, 0, 1, 1]),
        population_labels=np.array(["outgoing_electrons", "incoming_electrons"]),
        step=step,
        time=time,
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
    np.savetxt(
        output_directory / "diagnostics.csv",
        np.column_stack((time, energy, energy)),
        delimiter=",",
        header=(
            "time,electric_field_energy,analysis_electric_field_energy"
        ),
        comments="",
    )
    with (output_directory / "run_parameters.json").open("w") as stream:
        json.dump({"wavenumber": 1.0}, stream)

    with pytest.raises(ValueError, match="mode-purity threshold"):
        module.render_two_stream_diagnostics(output_directory)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the movie smoke test",
)
def test_render_writes_png_growth_fit_and_two_frame_h264_movie(tmp_path):
    module = load_diagnostic_script_module()
    output_directory = tmp_path / "two_stream_output"
    phase_space_directory = output_directory / "phase_space"
    output_directory.mkdir()

    time = np.linspace(0.0, 6.0, 31)
    gamma = 0.4
    energy = 1.0e-9 * np.exp(2.0 * gamma * time)
    np.savetxt(
        output_directory / "diagnostics.csv",
        np.column_stack((time, 3.0 * energy, energy, np.ones_like(time))),
        delimiter=",",
        header=(
            "time,electric_field_energy,analysis_electric_field_energy,"
            "target_mode_energy_fraction"
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
            },
            stream,
        )

    write_snapshot(phase_space_directory, 0, 0.0)
    write_snapshot(
        phase_space_directory,
        1,
        0.1,
        radius=[0.12, 0.32, 0.58, 0.88],
        radial_momentum=[0.20, 0.24, -0.18, -0.23],
    )

    paths = module.render_two_stream_diagnostics(
        output_directory,
        max_frames=2,
        fps=2,
        dpi=60,
    )

    assert paths["energy_plot"].stat().st_size > 0
    assert paths["growth_fit"].stat().st_size > 0
    assert paths["phase_space_movie"].stat().st_size > 0
    with paths["growth_fit"].open() as stream:
        growth_fit = json.load(stream)
    assert growth_fit["gamma"] == pytest.approx(gamma, rel=1.0e-12)
    assert growth_fit["fit_energy_column"] == "analysis_electric_field_energy"
    assert growth_fit["fit_window_minimum_mode_fraction"] == pytest.approx(1.0)
    assert growth_fit["energy_amplification"] > np.exp(2.0)
    assert growth_fit["growth_rate_relative_error"] == pytest.approx(0.0)
    assert growth_fit["growth_rate_within_30_percent"] is True
    assert growth_fit["growth_fit_acceptance"] is True

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
            str(paths["phase_space_movie"]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert int(stream["nb_frames"]) == 2
