"""Plot two-stream electric growth and phase space for BGK inspection.

The energy figure always includes the total-domain electric energy.  For the
guarded spherical annulus, the default exponential fit uses the central
analysis-window energy so that charge-separation transients at the finite
plasma edges are not mistaken for the local two-stream mode.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "z4c_two_stream"
)
PHASE_SPACE_PATTERN = re.compile(r"phase_space_step_(\d+)\.npz$")
POPULATION_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#72b7b2",
)


def phase_space_step(snapshot_path: Path) -> int:
    match = PHASE_SPACE_PATTERN.fullmatch(snapshot_path.name)
    if match is None:
        raise ValueError(f"Not a phase-space snapshot: {snapshot_path}")
    return int(match.group(1))


def discover_phase_space_paths(
    phase_space_directory: Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[Path]:
    """Select numerically ordered snapshots while always retaining the final one."""

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    snapshots = [
        path
        for path in phase_space_directory.glob("phase_space_step_*.npz")
        if PHASE_SPACE_PATTERN.fullmatch(path.name)
    ]
    snapshots.sort(key=phase_space_step)
    if not snapshots:
        raise RuntimeError(
            f"No phase-space snapshots found in {phase_space_directory}"
        )

    selected = snapshots[::frame_stride]
    if selected[-1] != snapshots[-1]:
        selected.append(snapshots[-1])

    if max_frames is not None and len(selected) > max_frames:
        if max_frames == 1:
            selected = [selected[-1]]
        else:
            indices = np.linspace(0, len(selected) - 1, max_frames, dtype=int)
            selected = [selected[index] for index in np.unique(indices)]
            if selected[-1] != snapshots[-1]:
                selected[-1] = snapshots[-1]

    return selected


def load_phase_space_frame(snapshot_path: Path) -> dict[str, object]:
    """Read one snapshot using the two-stream output schema."""

    required_fields = {
        "r",
        "ur",
        "population_id",
        "population_labels",
        "step",
        "time",
    }
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        missing_fields = required_fields.difference(snapshot.files)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{snapshot_path} is missing fields: {missing}")

        radius = np.asarray(snapshot["r"], dtype=float)
        radial_momentum = np.asarray(snapshot["ur"], dtype=float)
        population_id = np.asarray(snapshot["population_id"], dtype=np.int64)
        population_labels = tuple(
            str(label) for label in np.asarray(snapshot["population_labels"])
        )
        step = int(np.asarray(snapshot["step"]).item())
        time = float(np.asarray(snapshot["time"]).item())

    if radius.ndim != 1 or radial_momentum.shape != radius.shape:
        raise ValueError(f"{snapshot_path} must contain matching one-dimensional r and ur")
    if population_id.shape != radius.shape:
        raise ValueError(
            f"{snapshot_path} population_id must have the same shape as r"
        )
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(radial_momentum)):
        raise ValueError(f"{snapshot_path} contains non-finite phase-space values")
    if step != phase_space_step(snapshot_path):
        raise ValueError(f"{snapshot_path} filename and stored step disagree")
    if not np.isfinite(time):
        raise ValueError(f"{snapshot_path} contains a non-finite time")
    if not population_labels:
        raise ValueError(f"{snapshot_path} contains no population labels")
    if population_id.size and (
        np.min(population_id) < 0 or np.max(population_id) >= len(population_labels)
    ):
        raise ValueError(f"{snapshot_path} contains an unknown population_id")

    return {
        "r": radius,
        "ur": radial_momentum,
        "population_id": population_id,
        "population_labels": population_labels,
        "step": step,
        "time": time,
    }


def load_phase_space_frames(
    phase_space_directory: Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[dict[str, object]]:
    paths = discover_phase_space_paths(
        phase_space_directory,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    frames = [load_phase_space_frame(path) for path in paths]

    reference_labels = frames[0]["population_labels"]
    reference_population_id = frames[0]["population_id"]
    for previous, frame in zip(frames, frames[1:]):
        if frame["step"] <= previous["step"] or frame["time"] <= previous["time"]:
            raise ValueError("phase-space snapshots must increase in step and time")
        if frame["population_labels"] != reference_labels:
            raise ValueError("population labels change between phase-space snapshots")
        if not np.array_equal(frame["population_id"], reference_population_id):
            raise ValueError("particle population ordering changes between snapshots")

    return frames


def load_diagnostics(
    diagnostics_path: Path,
    energy_column: str = "electric_field_energy",
) -> tuple[np.ndarray, np.ndarray]:
    diagnostics = np.genfromtxt(diagnostics_path, delimiter=",", names=True)
    diagnostics = np.atleast_1d(diagnostics)
    if diagnostics.dtype.names is None:
        raise ValueError(f"Could not read the header from {diagnostics_path}")

    required_columns = {"time", energy_column}
    missing_columns = required_columns.difference(diagnostics.dtype.names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{diagnostics_path} is missing columns: {missing}")

    time = np.asarray(diagnostics["time"], dtype=float)
    energy = np.asarray(diagnostics[energy_column], dtype=float)
    if time.size < 2 or not np.all(np.isfinite(time)):
        raise ValueError("diagnostics must contain at least two finite times")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("diagnostic times must be strictly increasing")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("electric-field energy must be finite and nonnegative")

    return time, energy


def _linear_fit(time: np.ndarray, log_energy: np.ndarray) -> tuple[float, float, float]:
    centered_time = time - np.mean(time)
    centered_energy = log_energy - np.mean(log_energy)
    denominator = float(np.dot(centered_time, centered_time))
    if denominator == 0.0:
        raise ValueError("growth-fit times must span a nonzero interval")

    slope = float(np.dot(centered_time, centered_energy) / denominator)
    intercept = float(np.mean(log_energy) - slope * np.mean(time))
    residual = log_energy - (slope * time + intercept)
    total_variation = float(np.dot(centered_energy, centered_energy))
    if total_variation == 0.0:
        r_squared = 0.0
    else:
        r_squared = 1.0 - float(np.dot(residual, residual)) / total_variation

    return slope, intercept, r_squared


def _growth_fit_result(
    time: np.ndarray,
    energy: np.ndarray,
    original_indices: np.ndarray,
    start: int,
    end: int,
    auto_selected: bool,
    minimum_e_folds: float,
    minimum_r_squared: float,
    saturation_index: int,
    mode_fraction: np.ndarray | None,
    minimum_mode_fraction: float | None,
) -> dict[str, object]:
    selected_time = time[start : end + 1]
    selected_log_energy = np.log(energy[start : end + 1])
    slope, intercept, r_squared = _linear_fit(selected_time, selected_log_energy)
    fitted_energy_e_folds = slope * (selected_time[-1] - selected_time[0])
    energy_amplification = float(energy[end] / energy[start])
    energy_e_folds = float(np.log(energy_amplification))

    mode_fraction_qualifies = True
    minimum_selected_mode_fraction = None
    mean_selected_mode_fraction = None
    if mode_fraction is not None:
        selected_mode_fraction = mode_fraction[start : end + 1]
        minimum_selected_mode_fraction = float(np.min(selected_mode_fraction))
        mean_selected_mode_fraction = float(np.mean(selected_mode_fraction))
        if minimum_mode_fraction is not None:
            mode_fraction_qualifies = (
                minimum_selected_mode_fraction >= minimum_mode_fraction
            )

    return {
        "fit_method": "longest_qualifying_window" if auto_selected else "user_window",
        "auto_selected": auto_selected,
        "fit_start": float(selected_time[0]),
        "fit_end": float(selected_time[-1]),
        "start_index": int(original_indices[start]),
        "end_index": int(original_indices[end]),
        "num_samples": int(end - start + 1),
        "log_energy_slope": slope,
        "gamma": 0.5 * slope,
        "log_energy_intercept": intercept,
        "r_squared": r_squared,
        "energy_amplification": energy_amplification,
        "energy_e_folds": energy_e_folds,
        "fitted_energy_e_folds": fitted_energy_e_folds,
        "minimum_energy_e_folds": float(minimum_e_folds),
        "minimum_r_squared": float(minimum_r_squared),
        "minimum_mode_fraction": minimum_mode_fraction,
        "fit_window_minimum_mode_fraction": minimum_selected_mode_fraction,
        "fit_window_mean_mode_fraction": mean_selected_mode_fraction,
        "meets_growth_criteria": bool(
            slope > 0.0
            and energy_e_folds >= minimum_e_folds
            and r_squared >= minimum_r_squared
            and mode_fraction_qualifies
        ),
        "growth_search_end_index": int(original_indices[saturation_index]),
        "growth_search_end_time": float(time[saturation_index]),
    }


def fit_log_energy_growth(
    time: np.ndarray,
    electric_field_energy: np.ndarray,
    fit_start: float | None = None,
    fit_end: float | None = None,
    minimum_e_folds: float = 2.0,
    minimum_r_squared: float = 0.95,
    minimum_samples: int = 5,
    mode_fraction: np.ndarray | None = None,
    minimum_mode_fraction: float | None = None,
) -> dict[str, object]:
    """Fit ``log(U_E)`` and return the electric-field growth rate ``gamma``.

    Since electric-field energy grows as ``exp(2 gamma t)``, ``gamma`` is half
    the fitted log-energy slope.  Automatic selection examines all windows up
    to the largest observed energy peak and chooses the longest one satisfying
    the requested e-folding, linearity, and optional mode-purity thresholds.
    """

    time = np.asarray(time, dtype=float)
    energy = np.asarray(electric_field_energy, dtype=float)
    if time.shape != energy.shape or time.ndim != 1:
        raise ValueError("time and electric_field_energy must be matching 1-D arrays")
    if minimum_mode_fraction is not None and mode_fraction is None:
        raise ValueError(
            "mode_fraction is required when a mode-purity threshold is requested"
        )
    if mode_fraction is not None:
        mode_fraction = np.asarray(mode_fraction, dtype=float)
        if mode_fraction.shape != time.shape:
            raise ValueError("mode_fraction must match the diagnostic time array")
        if not np.all(np.isfinite(mode_fraction)):
            raise ValueError("mode_fraction must contain only finite values")
    if time.size < minimum_samples or not np.all(np.diff(time) > 0.0):
        raise ValueError("growth-fit times must be strictly increasing")

    positive = np.isfinite(time) & np.isfinite(energy) & (energy > 0.0)
    original_indices = np.flatnonzero(positive)
    time = time[positive]
    energy = energy[positive]
    if mode_fraction is not None:
        mode_fraction = mode_fraction[positive]
    if time.size < minimum_samples:
        raise ValueError("not enough positive electric-field energy samples to fit")

    saturation_index = int(np.argmax(energy))
    if fit_start is not None or fit_end is not None:
        lower = -np.inf if fit_start is None else fit_start
        upper = np.inf if fit_end is None else fit_end
        selected = np.flatnonzero((time >= lower) & (time <= upper))
        if selected.size < minimum_samples:
            raise ValueError(
                f"the requested fit window must contain at least {minimum_samples} samples"
            )
        return _growth_fit_result(
            time,
            energy,
            original_indices,
            int(selected[0]),
            int(selected[-1]),
            False,
            minimum_e_folds,
            minimum_r_squared,
            saturation_index,
            mode_fraction,
            minimum_mode_fraction,
        )

    if saturation_index + 1 < minimum_samples:
        raise RuntimeError("no resolved growth interval was found before the peak")

    # Prefix sums make the exhaustive window search O(N^2) without repeatedly
    # fitting the same data.  The correlation squared is the R^2 of a linear
    # least-squares fit to log electric-field energy.
    fit_time = time[: saturation_index + 1]
    log_energy = np.log(energy[: saturation_index + 1])
    prefix_time = np.concatenate(([0.0], np.cumsum(fit_time)))
    prefix_log_energy = np.concatenate(([0.0], np.cumsum(log_energy)))
    prefix_time_squared = np.concatenate(([0.0], np.cumsum(fit_time**2)))
    prefix_log_energy_squared = np.concatenate(
        ([0.0], np.cumsum(log_energy**2))
    )
    prefix_product = np.concatenate(([0.0], np.cumsum(fit_time * log_energy)))
    prefix_below_mode_threshold = None
    if mode_fraction is not None and minimum_mode_fraction is not None:
        below_mode_threshold = (
            mode_fraction[: saturation_index + 1] < minimum_mode_fraction
        )
        prefix_below_mode_threshold = np.concatenate(
            ([0], np.cumsum(below_mode_threshold, dtype=int))
        )

    best_window = None
    best_score = None
    for start in range(fit_time.size - minimum_samples + 1):
        ends = np.arange(start + minimum_samples - 1, fit_time.size)
        count = ends - start + 1
        sum_time = prefix_time[ends + 1] - prefix_time[start]
        sum_energy = prefix_log_energy[ends + 1] - prefix_log_energy[start]
        sum_time_squared = (
            prefix_time_squared[ends + 1] - prefix_time_squared[start]
        )
        sum_energy_squared = (
            prefix_log_energy_squared[ends + 1]
            - prefix_log_energy_squared[start]
        )
        sum_product = prefix_product[ends + 1] - prefix_product[start]

        time_variation = count * sum_time_squared - sum_time**2
        energy_variation = count * sum_energy_squared - sum_energy**2
        covariance = count * sum_product - sum_time * sum_energy
        slopes = covariance / time_variation
        r_squared = np.zeros_like(slopes)
        varying = energy_variation > 0.0
        r_squared[varying] = (
            covariance[varying] ** 2
            / (time_variation[varying] * energy_variation[varying])
        )
        durations = fit_time[ends] - fit_time[start]
        energy_e_folds = log_energy[ends] - log_energy[start]
        valid = (
            (slopes > 0.0)
            & (energy_e_folds >= minimum_e_folds)
            & (r_squared >= minimum_r_squared)
        )
        if prefix_below_mode_threshold is not None:
            contains_low_mode_fraction = (
                prefix_below_mode_threshold[ends + 1]
                - prefix_below_mode_threshold[start]
            ) > 0
            valid &= ~contains_low_mode_fraction

        for end_index in np.flatnonzero(valid):
            end = int(ends[end_index])
            score = (
                float(durations[end_index]),
                int(count[end_index]),
                float(r_squared[end_index]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_window = (start, end)

    if best_window is None:
        raise RuntimeError(
            "no log-energy window before the largest observed peak has at least "
            f"{minimum_e_folds:g} e-foldings and R^2 >= {minimum_r_squared:g}"
        )

    return _growth_fit_result(
        time,
        energy,
        original_indices,
        best_window[0],
        best_window[1],
        True,
        minimum_e_folds,
        minimum_r_squared,
        saturation_index,
        mode_fraction,
        minimum_mode_fraction,
    )


def make_electric_field_energy_plot(
    time: np.ndarray,
    energy: np.ndarray,
    growth_fit: dict[str, object],
    plot_path: Path,
    total_energy: np.ndarray | None = None,
    fitted_energy_label: str = r"analysis-window $U_E$",
    dpi: int = 180,
) -> None:
    positive = energy > 0.0
    fit_time = time[
        (time >= growth_fit["fit_start"])
        & (time <= growth_fit["fit_end"])
        & positive
    ]
    fitted_energy = np.exp(
        growth_fit["log_energy_intercept"]
        + growth_fit["log_energy_slope"] * fit_time
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if total_energy is not None:
        positive_total = total_energy > 0.0
        axis.semilogy(
            time[positive_total],
            total_energy[positive_total],
            color="#bab0ac",
            linewidth=1.2,
            label=r"total-domain $U_E$",
        )
    axis.semilogy(
        time[positive],
        energy[positive],
        color="#4c78a8",
        label=fitted_energy_label,
    )
    axis.semilogy(
        fit_time,
        fitted_energy,
        "--",
        color="#e45756",
        linewidth=2.0,
        label=(
            (
                "fit: "
                if growth_fit["meets_growth_criteria"]
                else "diagnostic fit (criteria not met): "
            )
            + rf"$\gamma={growth_fit['gamma']:.4g}$, "
            + rf"$R^2={growth_fit['r_squared']:.4f}$"
        ),
    )
    axis.axvspan(
        growth_fit["fit_start"],
        growth_fit["fit_end"],
        color="#e45756",
        alpha=0.08,
    )
    axis.set_xlabel("time")
    axis.set_ylabel(r"electric-field energy $U_E$")
    axis.set_title("Relativistic two-stream electric-field growth")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.savefig(plot_path, dpi=dpi)
    plt.close(figure)


def phase_space_limits(
    frames: list[dict[str, object]],
    radial_minimum: float | None = None,
    radial_maximum: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    radius_chunks = []
    momentum_chunks = []
    for frame in frames:
        inside = np.ones(frame["r"].shape, dtype=bool)
        if radial_minimum is not None:
            inside &= frame["r"] >= radial_minimum
        if radial_maximum is not None:
            inside &= frame["r"] <= radial_maximum
        radius_chunks.append(frame["r"][inside])
        momentum_chunks.append(frame["ur"][inside])

    if not any(chunk.size for chunk in radius_chunks):
        raise ValueError("no particles lie inside the requested analysis window")

    radius = np.concatenate(radius_chunks)
    radial_momentum = np.concatenate(momentum_chunks)
    radial_limits = (float(np.min(radius)), float(np.max(radius)))
    momentum_limits = (float(np.min(radial_momentum)), float(np.max(radial_momentum)))

    radial_span = radial_limits[1] - radial_limits[0]
    momentum_span = momentum_limits[1] - momentum_limits[0]
    radial_padding = 0.02 * radial_span if radial_span > 0.0 else 0.02
    momentum_padding = (
        0.05 * momentum_span
        if momentum_span > 0.0
        else 0.05 * max(abs(momentum_limits[0]), 1.0)
    )

    return (
        (radial_limits[0] - radial_padding, radial_limits[1] + radial_padding),
        (
            momentum_limits[0] - momentum_padding,
            momentum_limits[1] + momentum_padding,
        ),
    )


def make_phase_space_movie(
    frames: list[dict[str, object]],
    movie_path: Path,
    radial_minimum: float,
    radial_maximum: float,
    fps: int = 24,
    dpi: int = 120,
) -> None:
    """Render raw radial phase space with fixed run-wide limits."""

    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Matplotlib could not find ffmpeg for phase_space.mp4")

    _, momentum_limits = phase_space_limits(
        frames,
        radial_minimum=radial_minimum,
        radial_maximum=radial_maximum,
    )
    population_labels = frames[0]["population_labels"]
    figure, raw_axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    raw_artists = []

    for population, label in enumerate(population_labels):
        color = POPULATION_COLORS[population % len(POPULATION_COLORS)]
        raw_artists.append(
            raw_axis.scatter([], [], s=5, alpha=0.55, color=color, label=label)
        )

    raw_axis.set_xlim(radial_minimum, radial_maximum)
    raw_axis.set_ylim(*momentum_limits)
    raw_axis.set_xlabel(r"radial position $r$")
    raw_axis.set_ylabel(r"radial momentum $u_r$")
    raw_axis.set_title("Raw radial phase space")
    raw_axis.grid(alpha=0.2)

    raw_axis.legend(loc="best", markerscale=2.0)
    title = figure.suptitle("")

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
        metadata={"artist": "RadiShPICR"},
    )
    with writer.saving(figure, movie_path, dpi=dpi):
        for frame in frames:
            for population, raw_artist in enumerate(raw_artists):
                in_window = np.logical_and(
                    frame["r"] >= radial_minimum,
                    frame["r"] <= radial_maximum,
                )
                selected = np.logical_and(
                    frame["population_id"] == population,
                    in_window,
                )
                radius = frame["r"][selected]
                radial_momentum = frame["ur"][selected]
                raw_artist.set_offsets(np.column_stack((radius, radial_momentum)))

            title.set_text(
                "Relativistic two-stream phase space, "
                f"step {frame['step']:06d}, t = {frame['time']:.6g}"
            )
            writer.grab_frame()

    plt.close(figure)


def load_run_parameters(parameter_path: Path) -> dict[str, object]:
    with parameter_path.open(encoding="utf-8") as parameter_file:
        return json.load(parameter_file)


def _first_parameter(parameters: dict[str, object], names: tuple[str, ...]):
    for name in names:
        if name in parameters:
            return parameters[name]
    return None


def radial_domain(
    parameters: dict[str, object],
    frames: list[dict[str, object]],
) -> tuple[float, float]:
    radial_minimum = _first_parameter(
        parameters,
        (
            "analysis_r_min",
            "analysis_radial_minimum",
            "r_min",
            "radial_minimum",
            "domain_r_min",
            "radial_domain_minimum",
        ),
    )
    radial_maximum = _first_parameter(
        parameters,
        (
            "analysis_r_max",
            "analysis_radial_maximum",
            "r_max",
            "radial_maximum",
            "domain_r_max",
            "radial_domain_maximum",
            "cloud_areal_radius",
        ),
    )
    if radial_minimum is None:
        radial_minimum = min(float(np.min(frame["r"])) for frame in frames)
    if radial_maximum is None:
        radial_maximum = max(float(np.max(frame["r"])) for frame in frames)

    radial_minimum = float(radial_minimum)
    radial_maximum = float(radial_maximum)
    if not radial_maximum > radial_minimum:
        raise ValueError("run_parameters.json contains an invalid radial domain")
    return radial_minimum, radial_maximum


def add_theory_comparison(
    growth_fit: dict[str, object],
    parameters: dict[str, object],
) -> None:
    growth_fit["growth_fit_acceptance"] = bool(
        growth_fit["meets_growth_criteria"]
    )
    theoretical_gamma = _first_parameter(
        parameters,
        (
            "theoretical_amplitude_growth_rate",
            "theoretical_growth_rate",
        ),
    )
    if theoretical_gamma is None:
        return

    theoretical_gamma = float(theoretical_gamma)
    growth_fit["theoretical_amplitude_growth_rate"] = theoretical_gamma
    growth_fit["theoretical_energy_growth_rate"] = 2.0 * theoretical_gamma
    if theoretical_gamma > 0.0:
        relative_error = abs(growth_fit["gamma"] - theoretical_gamma) / (
            theoretical_gamma
        )
        growth_fit["growth_rate_relative_error"] = float(relative_error)
        growth_fit["growth_rate_within_30_percent"] = bool(relative_error <= 0.30)
        growth_fit["growth_fit_acceptance"] = bool(
            growth_fit["meets_growth_criteria"] and relative_error <= 0.30
        )


def render_two_stream_diagnostics(
    output_directory: Path,
    fit_start: float | None = None,
    fit_end: float | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    fps: int = 24,
    dpi: int = 120,
    minimum_e_folds: float = 2.0,
    minimum_r_squared: float = 0.95,
    fit_energy_column: str | None = None,
    minimum_mode_fraction: float | None = 0.5,
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    parameters = load_run_parameters(output_directory / "run_parameters.json")
    diagnostics_path = output_directory / "diagnostics.csv"
    diagnostic_table = np.atleast_1d(
        np.genfromtxt(diagnostics_path, delimiter=",", names=True)
    )
    available_columns = diagnostic_table.dtype.names or ()
    if fit_energy_column is None:
        if "analysis_electric_field_energy" in available_columns:
            fit_energy_column = "analysis_electric_field_energy"
        else:
            fit_energy_column = "electric_field_energy"

    time, energy = load_diagnostics(
        diagnostics_path,
        energy_column=fit_energy_column,
    )
    _, total_energy = load_diagnostics(
        diagnostics_path,
        energy_column="electric_field_energy",
    )
    mode_fraction = None
    if "target_mode_energy_fraction" in available_columns:
        mode_fraction = np.asarray(
            diagnostic_table["target_mode_energy_fraction"],
            dtype=float,
        )
    elif minimum_mode_fraction is not None:
        raise ValueError(
            "target_mode_energy_fraction is required when a mode-purity "
            "threshold is requested"
        )
    growth_fit = fit_log_energy_growth(
        time,
        energy,
        fit_start=fit_start,
        fit_end=fit_end,
        minimum_e_folds=minimum_e_folds,
        minimum_r_squared=minimum_r_squared,
        mode_fraction=mode_fraction,
        minimum_mode_fraction=minimum_mode_fraction,
    )
    growth_fit["fit_energy_column"] = fit_energy_column
    add_theory_comparison(growth_fit, parameters)

    energy_plot_path = output_directory / "electric_field_energy.png"
    growth_fit_path = output_directory / "growth_fit.json"
    movie_path = output_directory / "phase_space.mp4"

    energy_labels = {
        "electric_field_energy": r"total-domain $U_E$",
        "analysis_electric_field_energy": r"analysis-window $U_E$",
        "target_mode_power": r"seeded-mode $U_{E,k}$",
    }
    plotted_total_energy = (
        None if fit_energy_column == "electric_field_energy" else total_energy
    )
    make_electric_field_energy_plot(
        time,
        energy,
        growth_fit,
        energy_plot_path,
        total_energy=plotted_total_energy,
        fitted_energy_label=energy_labels.get(fit_energy_column, fit_energy_column),
    )
    with growth_fit_path.open("w", encoding="utf-8") as fit_file:
        json.dump(growth_fit, fit_file, indent=2)
        fit_file.write("\n")

    frames = load_phase_space_frames(
        output_directory / "phase_space",
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    radial_minimum, radial_maximum = radial_domain(parameters, frames)
    make_phase_space_movie(
        frames,
        movie_path,
        radial_minimum,
        radial_maximum,
        fps=fps,
        dpi=dpi,
    )

    return {
        "energy_plot": energy_plot_path,
        "growth_fit": growth_fit_path,
        "phase_space_movie": movie_path,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot electric-field growth and render the two-stream phase-space movie."
        )
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--fit-start", type=float)
    parser.add_argument("--fit-end", type=float)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--minimum-e-folds", type=float, default=2.0)
    parser.add_argument("--minimum-r-squared", type=float, default=0.95)
    parser.add_argument(
        "--fit-energy-column",
        choices=(
            "electric_field_energy",
            "analysis_electric_field_energy",
            "target_mode_power",
        ),
        help=(
            "energy series to fit; defaults to analysis-window energy when "
            "present while still plotting the total-domain energy"
        ),
    )
    parser.add_argument("--minimum-mode-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    paths = render_two_stream_diagnostics(
        args.output_directory,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        fps=args.fps,
        dpi=args.dpi,
        minimum_e_folds=args.minimum_e_folds,
        minimum_r_squared=args.minimum_r_squared,
        fit_energy_column=args.fit_energy_column,
        minimum_mode_fraction=args.minimum_mode_fraction,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
