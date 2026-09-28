"""Overlay electric-field energy for dynamic and static Z4C two-stream runs.

Run with no arguments to compare outputs/dynamic_gr and outputs/no_dynamic_gr.
The static run holds the solved initial metric fixed.  Both curves use the
saved total-domain electric_field_energy, including the proper shell volume
and inverse radial metric, with each run's own coordinate-time samples.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"


def plot_energy_comparison(dynamic_directory, static_directory, plot_path, yscale="linear"):
    """Plot the recorded energies directly, without normalization or fitting."""

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)

    runs = (
        (dynamic_directory, "Dynamic relativity (Z4C)", "#4c78a8", "-"),
        (static_directory, "Static relativity (fixed initial metric)", "#f58518", "--"),
    )
    for directory, label, color, linestyle in runs:
        diagnostics = np.genfromtxt(
            Path(directory) / "diagnostics.csv",
            delimiter=",",
            names=True,
            usecols=("time", "electric_field_energy"),
            ndmin=1,
        )
        axis.plot(
            diagnostics["time"],
            diagnostics["electric_field_energy"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )

    axis.set_yscale(yscale)
    if yscale == "linear":
        axis.set_ylim(bottom=0)
    axis.set_xlabel("Coordinate time (code units)")
    axis.set_ylabel(r"Total-domain electric-field energy $U_E$ (code units)")
    axis.set_title("Two-stream instability: dynamic vs static relativity")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamic-dir", type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "dynamic_gr",
        help="Dynamic Z4C run directory containing diagnostics.csv.",
    )
    parser.add_argument(
        "--static-dir", type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "no_dynamic_gr",
        help="Fixed initial metric run directory containing diagnostics.csv.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "electric_field_energy_comparison.png",
        help="Destination figure path (default: outputs/electric_field_energy_comparison.png).",
    )
    parser.add_argument("--yscale", choices=("log", "linear"), default="linear")
    args = parser.parse_args()

    plot_path = plot_energy_comparison(
        args.dynamic_dir, args.static_dir, args.output, args.yscale,
    )
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
