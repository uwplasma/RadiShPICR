"""Plot OS central and minimum lapse histories from saved Z4c evolution.

The central-proper-time plot follows Fig. 6 and Eqs. (95)-(97) of
Wu and Most, arXiv:2611.12831v1. The analytic curve assumes an initially
unit lapse; it is a reference, not an established bound for this demo's
spatially varying initial lapse. The minimum-lapse plot uses coordinate time.

Read the current raw-state openPMD series without evolving or rescaling it.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import openpmd_api as io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEMO_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT = DEMO_DIRECTORY / "outputs/evolution_data/bssn_fields.h5"
DEFAULT_OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/lapse_history"


def read_lapse_history(path):
    """Return scalar histories, initial gravitational mass, and initial radius.

    The physical radial grid begins at dr/2. Thus alpha[0] estimates the
    central lapse; it is not necessarily the spatial minimum. At the true
    symmetry center beta=0 and d(tau_c)=alpha_c dt. Integrating alpha[0]
    estimates this central proper time, not a first-cell observer's clock.
    """

    path = Path(path)
    if path.is_dir():
        path = path / "bssn_fields.h5"
    if not path.is_file():
        raise FileNotFoundError(f"No saved OS evolution series at {path}")

    series = io.Series(str(path), io.Access.read_only,
                       '{"defer_iteration_parsing": true}')
    try:
        if ("savedCoordinates" not in series.attributes
                or series.get_attribute("savedCoordinates") != "evolution"):
            raise ValueError("Expected raw evolution coordinates (savedCoordinates='evolution').")
        if "total_mass" not in series.attributes or "surface_areal_radius" not in series.attributes:
            raise ValueError("The OS reference needs saved total_mass and surface_areal_radius.")
        mass = float(series.get_attribute("total_mass"))
        surface_areal_radius = float(series.get_attribute("surface_areal_radius"))
        if not (np.isfinite(mass) and mass > 0.0
                and np.isfinite(surface_areal_radius) and surface_areal_radius > 2.0 * mass):
            raise ValueError("The OS reference requires saved M > 0 and R0 > 2M.")

        steps = sorted(series.iterations)
        times, central_lapse, minimum_lapse = [], [], []
        for step in steps:
            iteration = series.iterations[step]
            iteration.open()
            alpha = iteration.meshes["alpha"][io.Mesh.SCALAR].load_chunk()
            series.flush()
            if alpha.ndim != 1 or alpha.size == 0 or not np.all(np.isfinite(alpha)):
                raise ValueError(f"Step {step} needs a finite, nonempty radial lapse profile.")

            times.append(float(iteration.time))
            central_lapse.append(float(alpha[0]))
            # All saved cells are physical. Particle-deletion ghost counts
            # do not remove metric cells from this minimum.
            minimum_lapse.append(float(np.min(alpha)))
            iteration.close()
    finally:
        series.close()

    times = np.asarray(times)
    central_lapse = np.asarray(central_lapse)
    minimum_lapse = np.asarray(minimum_lapse)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("The series needs a nonempty history with finite timestamps.")
    if times[0] != 0.0:
        raise ValueError("Central proper time requires the initial snapshot at t=0.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Saved timestamps must be strictly increasing.")

    # Integrate every saved interval; accepted steps and output spacing can vary.
    proper_time = np.zeros_like(times)
    proper_time[1:] = np.cumsum(
        0.5 * (central_lapse[:-1] + central_lapse[1:]) * np.diff(times)
    )
    history = dict(
        step=np.asarray(steps, dtype=np.int64),
        time_over_mass=times / mass,
        central_proper_time_over_mass=proper_time / mass,
        central_lapse=central_lapse,
        minimum_lapse=minimum_lapse,
    )
    return history, mass, surface_areal_radius


def os_lapse_reference(mass, surface_areal_radius):
    """Unit-initial-lapse OS reference, parameterized through its first zero."""

    eta_zero = np.arccos(2.0 * np.exp(-1.0 / 6.0) - 1.0)
    eta = np.linspace(0.0, eta_zero, 1000)
    radius_ratio = 0.5 * (1.0 + np.cos(eta))
    proper_time_over_mass = np.sqrt((surface_areal_radius / mass)**3 / 8.0) * (
        eta + np.sin(eta)
    )
    alpha_reference = 1.0 + 6.0 * np.log(radius_ratio)
    return proper_time_over_mass, alpha_reference


def save_lapse_history(history, mass, surface_areal_radius, output_directory):
    """Save two separate figures and the numerical scalar histories."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    reference_time, reference_lapse = os_lapse_reference(mass, surface_areal_radius)
    marker = "." if len(history["step"]) == 1 else None

    for name, time_key, lapse_key, xlabel, ylabel, title in (
        ("central_lapse_proper_time", "central_proper_time_over_mass", "central_lapse",
         r"Estimated central proper time $\tau_c/M$", r"Central lapse $\alpha_c$",
         "OS collapse: central lapse"),
        ("minimum_lapse_coordinate_time", "time_over_mass", "minimum_lapse",
         r"Coordinate time $t/M$", r"Minimum lapse $\min_r\alpha$",
         "OS collapse: minimum lapse"),
    ):
        figure, axis = plt.subplots(figsize=(7.6, 5.0))
        figure.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.25)
        axis.plot(history[time_key], history[lapse_key], color="tab:blue",
                  linewidth=1.8, marker=marker, label="Z4c numerical")
        if lapse_key == "central_lapse":
            axis.plot(reference_time, reference_lapse, ":", color="black",
                      linewidth=1.8, label="unit-initial-lapse OS reference")
            note = (
                r"Central values use the innermost cell; $\tau_c$ is a finite-grid estimate."
                "\nReference assumes an initially unit lapse; it is not an established bound for this run."
            )
        else:
            note = "Minimum over all saved radial cells, with the saved lapse normalization."
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title + rf" ($R_0/M={surface_areal_radius / mass:g}$)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
        figure.text(0.5, 0.05, note, ha="center", va="bottom", fontsize=8.5)
        for extension in ("png", "pdf"):
            path = output_directory / f"{name}.{extension}"
            figure.savefig(path, dpi=180)
            print(f"Saved {path}")
        plt.close(figure)

    path = output_directory / "lapse_history.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT,
                        help="saved run directory or bssn_fields.h5 (default: demo outputs/evolution_data)")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY,
                        help="destination for figures and CSV (default: demo outputs/lapse_history)")
    args = parser.parse_args()

    history, mass, surface_areal_radius = read_lapse_history(args.input)
    save_lapse_history(history, mass, surface_areal_radius, args.output_directory)
    print(f"Source: {args.input.resolve()}")
    print(f"Read {len(history['step'])} snapshots through t/M={history['time_over_mass'][-1]:.12g}; "
          f"estimated central tau/M={history['central_proper_time_over_mass'][-1]:.12g}")
    print(f"Initial gravitational mass M={mass:.12g}; R0/M={surface_areal_radius / mass:.12g}")
    print("The unit-initial-lapse OS reference is not an established bound for this run.")


if __name__ == "__main__":
    main()
