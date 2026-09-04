"""Render two-stream field growth, energy composition, and spacetime state.

The energy figure always includes the total-domain electric energy.  For the
guarded spherical annulus, the default exponential fit uses the central
analysis-window energy so that charge-separation transients at the finite
plasma edges are not mistaken for the local two-stream mode.  Metric and
particle movies use the same numerically ordered snapshots and fixed run-wide
axes.
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
METRIC_PATTERN = re.compile(r"metric_step_(\d+)\.npz$")
DIAGNOSTIC_SCHEMA_VERSION = 2
POPULATION_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#72b7b2",
)

METRIC_GEOMETRY_FIELDS = (
    ("alpha", r"$\alpha - 1$", 1.0),
    ("beta", r"$\beta^r$", 0.0),
    ("chi", r"$\chi - 1$", 1.0),
    ("conformal_grr", r"$\widetilde{\gamma}_{rr} - 1$", 1.0),
    ("conformal_gt", r"$\widetilde{\gamma}_{T} - 1$", 1.0),
)
EXTRINSIC_Z4C_FIELDS = (
    ("Kh", r"$\widehat{K}$", 0.0),
    ("Arr", r"$\widetilde{A}_{rr}$", 0.0),
    ("At", r"$\widetilde{A}_{T}$", 0.0),
    ("theta", r"$\Theta$", 0.0),
    ("Gamma", r"$\widetilde{\Gamma}^{r}$", 0.0),
)
METRIC_ARRAY_FIELDS = (
    "r",
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
    "E_r",
    "kretschmann_scalar",
    "areal_radius",
    "misner_sharp_mass",
    "matter_rho",
    "matter_Srr",
    "matter_Stt",
    "matter_Sr",
    "matter_St",
)


def phase_space_step(snapshot_path: Path) -> int:
    match = PHASE_SPACE_PATTERN.fullmatch(snapshot_path.name)
    if match is None:
        raise ValueError(f"Not a phase-space snapshot: {snapshot_path}")
    return int(match.group(1))


def metric_step(snapshot_path: Path) -> int:
    match = METRIC_PATTERN.fullmatch(snapshot_path.name)
    if match is None:
        raise ValueError(f"Not a metric snapshot: {snapshot_path}")
    return int(match.group(1))


def _discover_snapshot_paths(
    snapshot_directory: Path,
    pattern: re.Pattern,
    step_from_path,
    snapshot_label: str,
    frame_stride: int,
    max_frames: int | None,
) -> list[Path]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    snapshots = [
        path
        for path in snapshot_directory.iterdir()
        if pattern.fullmatch(path.name)
    ]
    snapshots.sort(key=step_from_path)
    if not snapshots:
        raise RuntimeError(
            f"No {snapshot_label} snapshots found in {snapshot_directory}"
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


def discover_phase_space_paths(
    phase_space_directory: Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[Path]:
    """Select numerically ordered snapshots while always retaining the final one."""

    return _discover_snapshot_paths(
        phase_space_directory,
        PHASE_SPACE_PATTERN,
        phase_space_step,
        "phase-space",
        frame_stride,
        max_frames,
    )


def discover_metric_paths(
    metric_directory: Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[Path]:
    """Select numerically ordered metric snapshots including the final one."""

    return _discover_snapshot_paths(
        metric_directory,
        METRIC_PATTERN,
        metric_step,
        "metric",
        frame_stride,
        max_frames,
    )


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


def load_metric_frame(snapshot_path: Path) -> dict[str, object]:
    """Read one source-aware Z4C metric snapshot."""

    required_fields = set(METRIC_ARRAY_FIELDS) | {
        "step",
        "time",
        "diagnostic_schema_version",
    }
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        missing_fields = required_fields.difference(snapshot.files)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise RuntimeError(
                f"{snapshot_path} uses the old diagnostic schema and is missing "
                f"{missing}; rerun the two-stream simulation"
            )

        schema_version = int(
            np.asarray(snapshot["diagnostic_schema_version"]).item()
        )
        if schema_version != DIAGNOSTIC_SCHEMA_VERSION:
            raise RuntimeError(
                f"{snapshot_path} has diagnostic schema {schema_version}; "
                "rerun the two-stream simulation"
            )

        frame = {
            field: np.asarray(snapshot[field], dtype=float)
            for field in METRIC_ARRAY_FIELDS
        }
        frame["step"] = int(np.asarray(snapshot["step"]).item())
        frame["time"] = float(np.asarray(snapshot["time"]).item())

    radius = frame["r"]
    if radius.ndim != 1 or radius.size < 2 or not np.all(np.isfinite(radius)):
        raise ValueError(f"{snapshot_path} must contain a one-dimensional radial grid")
    for field in METRIC_ARRAY_FIELDS[1:]:
        values = frame[field]
        if values.shape != radius.shape:
            raise ValueError(f"{snapshot_path} field {field} does not match r")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{snapshot_path} field {field} is non-finite")
    if frame["step"] != metric_step(snapshot_path):
        raise ValueError(f"{snapshot_path} filename and stored step disagree")
    if not np.isfinite(frame["time"]):
        raise ValueError(f"{snapshot_path} contains a non-finite time")

    return frame


def load_metric_frames(
    metric_directory: Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[dict[str, object]]:
    paths = discover_metric_paths(
        metric_directory,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    frames = [load_metric_frame(path) for path in paths]

    reference_radius = frames[0]["r"]
    for previous, frame in zip(frames, frames[1:]):
        if frame["step"] <= previous["step"] or frame["time"] <= previous["time"]:
            raise ValueError("metric snapshots must increase in step and time")
        if not np.array_equal(frame["r"], reference_radius):
            raise ValueError("the radial metric grid changes between snapshots")

    return frames


def validate_synchronized_frames(
    phase_space_frames: list[dict[str, object]],
    metric_frames: list[dict[str, object]],
) -> None:
    phase_steps = [frame["step"] for frame in phase_space_frames]
    metric_steps = [frame["step"] for frame in metric_frames]
    phase_times = np.asarray([frame["time"] for frame in phase_space_frames])
    metric_times = np.asarray([frame["time"] for frame in metric_frames])

    if phase_steps != metric_steps or not np.array_equal(phase_times, metric_times):
        raise ValueError("metric and phase-space snapshots are not synchronized")


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


def _fixed_symmetric_limit(value_chunks: list[np.ndarray]) -> float:
    maximum = max(float(np.max(np.abs(values))) for values in value_chunks)
    if maximum == 0.0:
        return 1.0e-12
    return 1.05 * maximum


def _cell_centered_radial_limits(radius: np.ndarray) -> tuple[float, float]:
    dr = float(radius[1] - radius[0])
    return max(0.0, float(radius[0] - 0.5 * dr)), float(radius[-1] + 0.5 * dr)


def make_metric_movie(
    frames: list[dict[str, object]],
    movie_path: Path,
    fields: tuple[tuple[str, str, float], ...],
    title_text: str,
    fps: int = 24,
    dpi: int = 120,
) -> None:
    """Render synchronized radial metric fields with fixed run-wide axes."""

    if not FFMpegWriter.isAvailable():
        raise RuntimeError(f"Matplotlib could not find ffmpeg for {movie_path.name}")

    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(7.2, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    radius = frames[0]["r"]
    radial_limits = _cell_centered_radial_limits(radius)
    artists = []
    for axis, (field, label, reference_value) in zip(axes, fields):
        value_chunks = [frame[field] - reference_value for frame in frames]
        limit = _fixed_symmetric_limit(value_chunks)
        artist, = axis.plot(radius, value_chunks[0], color="#4c78a8")
        axis.set_xlim(*radial_limits)
        axis.set_ylim(-limit, limit)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        artists.append((artist, field, reference_value))

    axes[-1].set_xlabel(r"radial coordinate $r$")
    title = figure.suptitle("")
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
        metadata={"artist": "RadiShPICR"},
    )
    with writer.saving(figure, movie_path, dpi=dpi):
        for frame in frames:
            for artist, field, reference_value in artists:
                artist.set_ydata(frame[field] - reference_value)
            title.set_text(
                f"{title_text}, step {frame['step']:06d}, "
                f"t = {frame['time']:.6g}"
            )
            writer.grab_frame()

    plt.close(figure)


def make_electric_kretschmann_movie(
    frames: list[dict[str, object]],
    movie_path: Path,
    fps: int = 24,
    dpi: int = 120,
) -> None:
    """Render the signed electric field and spacetime curvature invariant."""

    if not FFMpegWriter.isAvailable():
        raise RuntimeError(f"Matplotlib could not find ffmpeg for {movie_path.name}")

    radius = frames[0]["r"]
    radial_limits = _cell_centered_radial_limits(radius)
    electric_limit = _fixed_symmetric_limit([frame["E_r"] for frame in frames])
    curvature_chunks = [frame["kretschmann_scalar"] for frame in frames]
    curvature_limit = _fixed_symmetric_limit(curvature_chunks)
    curvature_linthresh = max(curvature_limit / 1.05 * 1.0e-6, 1.0e-30)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    electric_artist, = axes[0].plot(radius, frames[0]["E_r"], color="#e45756")
    curvature_artist, = axes[1].plot(
        radius,
        frames[0]["kretschmann_scalar"],
        color="#4c78a8",
    )
    axes[0].set_ylim(-electric_limit, electric_limit)
    axes[0].set_ylabel(r"signed $E_r$")
    axes[1].set_yscale("symlog", linthresh=curvature_linthresh)
    axes[1].set_ylim(-curvature_limit, curvature_limit)
    axes[1].set_ylabel(r"K = $R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}$")
    axes[1].set_xlabel(r"radial coordinate $r$")
    for axis in axes:
        axis.set_xlim(*radial_limits)
        axis.grid(alpha=0.25)

    title = figure.suptitle("")
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
        metadata={"artist": "RadiShPICR"},
    )
    with writer.saving(figure, movie_path, dpi=dpi):
        for frame in frames:
            electric_artist.set_ydata(frame["E_r"])
            curvature_artist.set_ydata(frame["kretschmann_scalar"])
            title.set_text(
                "Electric field and Kretschmann scalar, "
                f"step {frame['step']:06d}, t = {frame['time']:.6g}"
            )
            writer.grab_frame()

    plt.close(figure)


def make_energy_composition_plot(
    time: np.ndarray,
    gravitational_energy: np.ndarray,
    electric_energy: np.ndarray,
    kinetic_energy: np.ndarray,
    mass_above_rest: np.ndarray,
    plot_path: Path,
    dpi: int = 180,
) -> None:
    """Plot the signed Misner--Sharp energy decomposition and fractions."""

    normalization = (
        np.abs(gravitational_energy) + electric_energy + kinetic_energy
    )
    gravitational_fraction = np.divide(
        gravitational_energy,
        normalization,
        out=np.zeros_like(gravitational_energy),
        where=normalization > 0.0,
    )
    electric_fraction = np.divide(
        electric_energy,
        normalization,
        out=np.zeros_like(electric_energy),
        where=normalization > 0.0,
    )
    kinetic_fraction = np.divide(
        kinetic_energy,
        normalization,
        out=np.zeros_like(kinetic_energy),
        where=normalization > 0.0,
    )

    energy_chunks = [
        gravitational_energy,
        electric_energy,
        kinetic_energy,
        mass_above_rest,
    ]
    energy_limit = _fixed_symmetric_limit(energy_chunks)
    energy_linthresh = max(energy_limit / 1.05 * 1.0e-6, 1.0e-30)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        time,
        gravitational_energy,
        label=r"$E_{\rm GR}$ (binding remainder)",
    )
    axes[0].plot(time, electric_energy, label=r"$E_{\rm electric}$")
    axes[0].plot(time, kinetic_energy, label=r"$E_{\rm kinetic}$")
    axes[0].plot(
        time,
        mass_above_rest,
        "--",
        color="black",
        label=r"$M_{\rm MS}-M_{\rm rest}$",
    )
    axes[0].set_yscale("symlog", linthresh=energy_linthresh)
    axes[0].set_ylim(-energy_limit, energy_limit)
    axes[0].set_ylabel("energy")
    axes[0].set_title("Misner--Sharp energy composition")
    axes[0].legend(loc="best")

    axes[1].plot(time, gravitational_fraction, label=r"$E_{\rm GR}/D$")
    axes[1].plot(time, electric_fraction, label=r"$E_{\rm electric}/D$")
    axes[1].plot(time, kinetic_fraction, label=r"$E_{\rm kinetic}/D$")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("signed fraction")
    axes[1].legend(loc="best")
    axes[1].text(
        0.01,
        0.03,
        r"$D=|E_{\rm GR}|+E_{\rm electric}+E_{\rm kinetic}$",
        transform=axes[1].transAxes,
        fontsize="small",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")

    figure.savefig(plot_path, dpi=dpi)
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
            "r_min",
            "radial_minimum",
            "domain_r_min",
            "radial_domain_minimum",
            "analysis_r_min",
            "analysis_radial_minimum",
        ),
    )
    radial_maximum = _first_parameter(
        parameters,
        (
            "r_max",
            "radial_maximum",
            "domain_r_max",
            "radial_domain_maximum",
            "cloud_areal_radius",
            "analysis_r_max",
            "analysis_radial_maximum",
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
    schema_version = parameters.get("diagnostic_schema_version")
    if schema_version != DIAGNOSTIC_SCHEMA_VERSION:
        raise RuntimeError(
            "This output directory predates the source-aware diagnostic schema; "
            "rerun the two-stream simulation"
        )

    diagnostics_path = output_directory / "diagnostics.csv"
    diagnostic_table = np.atleast_1d(
        np.genfromtxt(diagnostics_path, delimiter=",", names=True)
    )
    available_columns = diagnostic_table.dtype.names or ()
    composition_columns = {
        "rest_mass_energy",
        "misner_sharp_mass",
        "gravitational_binding_energy",
        "electric_field_energy",
        "relativistic_kinetic_energy",
    }
    missing_composition_columns = composition_columns.difference(available_columns)
    if missing_composition_columns:
        missing = ", ".join(sorted(missing_composition_columns))
        raise RuntimeError(
            f"{diagnostics_path} is missing source-aware columns {missing}; "
            "rerun the two-stream simulation"
        )

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

    phase_space_frames = load_phase_space_frames(
        output_directory / "phase_space",
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    metric_frames = load_metric_frames(
        output_directory / "metric",
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    validate_synchronized_frames(phase_space_frames, metric_frames)

    rest_mass_energy = np.asarray(
        diagnostic_table["rest_mass_energy"], dtype=float
    )
    misner_sharp_mass = np.asarray(
        diagnostic_table["misner_sharp_mass"], dtype=float
    )
    gravitational_energy = np.asarray(
        diagnostic_table["gravitational_binding_energy"], dtype=float
    )
    kinetic_energy = np.asarray(
        diagnostic_table["relativistic_kinetic_energy"], dtype=float
    )
    composition_arrays = (
        rest_mass_energy,
        misner_sharp_mass,
        gravitational_energy,
        total_energy,
        kinetic_energy,
    )
    if not all(np.all(np.isfinite(values)) for values in composition_arrays):
        raise ValueError("energy-composition diagnostics must be finite")
    mass_above_rest = misner_sharp_mass - rest_mass_energy
    decomposed_mass = gravitational_energy + total_energy + kinetic_energy
    if not np.allclose(
        mass_above_rest,
        decomposed_mass,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("energy-composition diagnostics do not close")

    energy_plot_path = output_directory / "electric_field_energy.png"
    energy_composition_path = output_directory / "energy_composition.png"
    growth_fit_path = output_directory / "growth_fit.json"
    movie_path = output_directory / "phase_space.mp4"
    metric_geometry_path = output_directory / "metric_geometry.mp4"
    extrinsic_z4c_path = output_directory / "extrinsic_z4c.mp4"
    electric_kretschmann_path = output_directory / "electric_kretschmann.mp4"

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

    make_energy_composition_plot(
        time,
        gravitational_energy,
        total_energy,
        kinetic_energy,
        mass_above_rest,
        energy_composition_path,
        dpi=dpi,
    )

    radial_minimum, radial_maximum = radial_domain(parameters, phase_space_frames)
    make_phase_space_movie(
        phase_space_frames,
        movie_path,
        radial_minimum,
        radial_maximum,
        fps=fps,
        dpi=dpi,
    )
    make_metric_movie(
        metric_frames,
        metric_geometry_path,
        METRIC_GEOMETRY_FIELDS,
        "Metric geometry",
        fps=fps,
        dpi=dpi,
    )
    make_metric_movie(
        metric_frames,
        extrinsic_z4c_path,
        EXTRINSIC_Z4C_FIELDS,
        "Extrinsic curvature and Z4C fields",
        fps=fps,
        dpi=dpi,
    )
    make_electric_kretschmann_movie(
        metric_frames,
        electric_kretschmann_path,
        fps=fps,
        dpi=dpi,
    )

    return {
        "energy_plot": energy_plot_path,
        "energy_composition_plot": energy_composition_path,
        "growth_fit": growth_fit_path,
        "phase_space_movie": movie_path,
        "metric_geometry_movie": metric_geometry_path,
        "extrinsic_z4c_movie": extrinsic_z4c_path,
        "electric_kretschmann_movie": electric_kretschmann_path,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot energy diagnostics and render synchronized two-stream movies."
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
