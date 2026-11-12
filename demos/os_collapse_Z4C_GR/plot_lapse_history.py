"""Compare the OS central lapse against its analytic reference.

One figure: numerical central lapse versus estimated central proper time, the
analytic Oppenheimer-Snyder lapse on the same axes, and the residual below.
The analytic curve follows Eqs. (95)-(97) of Wu and Most, arXiv:2611.12831v1,
and assumes an initially unit lapse. It is a like-for-like comparison only for
the runner's ``paper-os`` preset; for the constrained preset, whose initial
lapse varies in space, it is a reference and not an established bound.

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


def read_lapse_history(path):
    """Return scalar histories, initial gravitational mass, and initial radius.

    The physical radial grid begins at dr/2. Thus alpha[0] estimates the
    central lapse; it is not necessarily the spatial minimum. At the true
    symmetry center beta=0 and d(tau_c)=alpha_c dt. Integrating alpha[0]
    estimates this central proper time, not a first-cell observer's clock.

    The analytic reference ends at its own first zero. Samples beyond that
    proper time have no reference value, so their residual is NaN rather than
    a comparison against a clamped endpoint.
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
        times, central_lapse = [], []
        for step in steps:
            iteration = series.iterations[step]
            iteration.open()
            alpha = iteration.meshes["alpha"][io.Mesh.SCALAR].load_chunk()
            series.flush()
            if alpha.ndim != 1 or alpha.size == 0 or not np.all(np.isfinite(alpha)):
                raise ValueError(f"Step {step} needs a finite, nonempty radial lapse profile.")

            times.append(float(iteration.time))
            central_lapse.append(float(alpha[0]))
            iteration.close()
    finally:
        series.close()

    times = np.asarray(times)
    central_lapse = np.asarray(central_lapse)
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
    proper_time_over_mass = proper_time / mass

    # np.interp clamps to the endpoint value, which is the reference's first
    # zero. Mask instead, so a residual past that point is undefined rather
    # than silently equal to the raw numerical lapse.
    reference_time, reference_lapse = os_lapse_reference(mass, surface_areal_radius)
    analytic_central_lapse = np.where(
        proper_time_over_mass <= reference_time[-1],
        np.interp(proper_time_over_mass, reference_time, reference_lapse),
        np.nan,
    )

    history = dict(
        step=np.asarray(steps, dtype=np.int64),
        time_over_mass=times / mass,
        central_proper_time_over_mass=proper_time_over_mass,
        central_lapse=central_lapse,
        analytic_central_lapse=analytic_central_lapse,
        lapse_residual=central_lapse - analytic_central_lapse,
    )
    return history, mass, surface_areal_radius


def save_lapse_history(history, mass, surface_areal_radius, output_directory):
    """Save the comparison figure and the numerical scalar histories."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    reference_time, reference_lapse = os_lapse_reference(mass, surface_areal_radius)
    proper_time = history["central_proper_time_over_mass"]
    marker = "." if len(history["step"]) == 1 else None

    figure, (axis, residual_axis) = plt.subplots(
        2, 1, sharex=True, figsize=(7.6, 6.4),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.21, hspace=0.08)

    # Resample the reference onto the compared window. Drawing it in full would
    # autoscale a short run against the reference's entire descent to zero.
    window_end = min(float(np.max(proper_time)), float(reference_time[-1]))
    if window_end > 0.0:
        window = np.linspace(0.0, window_end, 400)
        reference_time, reference_lapse = window, np.interp(
            window, reference_time, reference_lapse,
        )

    axis.plot(proper_time, history["central_lapse"], color="tab:blue",
              linewidth=1.8, marker=marker, label="Z4c numerical")
    axis.plot(reference_time, reference_lapse, ":", color="black",
              linewidth=1.8, label="unit-initial-lapse OS analytic")
    axis.set_ylabel(r"Central lapse $\alpha_c$")
    axis.set_title("OS collapse: central lapse, analytic vs numerical"
                   rf" ($R_0/M={surface_areal_radius / mass:g}$)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.plot(proper_time, history["lapse_residual"], color="tab:red",
                       linewidth=1.5, marker=marker)
    residual_axis.set_xlabel(r"Estimated central proper time $\tau_c/M$")
    residual_axis.set_ylabel("numerical $-$ analytic", fontsize=9)
    residual_axis.grid(alpha=0.25)

    # Hold the axis to the compared data; the analytic curve runs to its own
    # first zero, which can be far beyond a short run.
    if proper_time.size > 1 and proper_time[-1] > proper_time[0]:
        residual_axis.set_xlim(proper_time[0], proper_time[-1])

    figure.text(
        0.5, 0.04,
        r"Central values use the innermost cell; $\tau_c$ is a finite-grid estimate."
        "\nThe analytic curve assumes an initially unit lapse, and is undefined"
        " past its first zero.",
        ha="center", va="bottom", fontsize=8.5,
    )
    for extension in ("png", "pdf"):
        path = output_directory / f"central_lapse_proper_time.{extension}"
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
                        help="destination for the figure and CSV (default: demo outputs/lapse_history)")
    args = parser.parse_args()

    history, mass, surface_areal_radius = read_lapse_history(args.input)
    save_lapse_history(history, mass, surface_areal_radius, args.output_directory)
    print(f"Source: {args.input.resolve()}")
    print(f"Read {len(history['step'])} snapshots through t/M={history['time_over_mass'][-1]:.12g}; "
          f"estimated central tau/M={history['central_proper_time_over_mass'][-1]:.12g}")
    print(f"Initial gravitational mass M={mass:.12g}; R0/M={surface_areal_radius / mass:.12g}")

    residual = history["lapse_residual"]
    compared = np.isfinite(residual)
    if np.any(compared):
        proper_time = history["central_proper_time_over_mass"][compared]
        print(f"Compared {int(np.count_nonzero(compared))} of {residual.size} snapshots "
              f"over tau_c/M in [{np.min(proper_time):.12g}, {np.max(proper_time):.12g}]: "
              f"max |numerical - analytic|={np.max(np.abs(residual[compared])):.6e}, "
              f"rms={np.sqrt(np.mean(residual[compared]**2)):.6e}")
    else:
        print("No snapshot lies inside the analytic reference's proper-time domain; "
              "the residual is undefined everywhere.")
    print("The unit-initial-lapse OS reference is not an established bound for "
          "a run whose initial lapse varies in space.")


if __name__ == "__main__":
    main()
