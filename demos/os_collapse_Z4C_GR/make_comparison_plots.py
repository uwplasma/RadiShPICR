"""Overlay final OS and initial-mass Schwarzschild fields against areal radius.

Run after run_schwarzschild_puncture.py. Use --radial-limit to restrict R/M.
Both runs are saved in evolution coordinates; areal radius only changes the
horizontal axis.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from RadiShPICR.diagnostics.read_bssn import final_snapshot, read_metadata

VARIABLES = (
    ("alpha", r"$\alpha$"),
    ("beta", r"$\beta^r$"),
    ("chi", r"$\chi$"),
    ("conformal_grr", r"$\widetilde{\gamma}_{rr}$"),
    ("conformal_gt", r"$\widetilde{\gamma}_{T}$"),
    ("Kh", r"$\widehat{K}$"),
    ("Arr", r"$\widetilde{A}_{rr}$"),
    ("At", r"$\widetilde{A}_{T}$"),
    ("theta", r"$\Theta$"),
    ("Gamma", r"$\widetilde{\Gamma}^{r}$"),
)


def read_final_profiles(run_directory):
    """Return the last state of a run, with its time and run metadata."""
    snapshot = final_snapshot(run_directory, with_particles=False)
    metadata = read_metadata(run_directory)

    fields = {name: np.asarray(getattr(snapshot.metric, name))
              for name, _ in VARIABLES}
    fields["r"] = np.asarray(snapshot.metric.r)
    fields["areal_radius"] = np.asarray(snapshot.areal_radius)

    if not all(np.all(np.isfinite(value)) for value in fields.values()):
        raise ValueError(f"Nonfinite final fields in {run_directory}")
    if any(np.any(fields[name] <= 0.0)
           for name in ("chi", "conformal_grr", "conformal_gt")):
        raise ValueError(f"Nonpositive spatial geometry in {run_directory}")

    return fields, snapshot.time, metadata


def make_comparison(os_directory, puncture_directory, output_directory, radial_limit=None):
    os_data = read_final_profiles(os_directory)
    puncture_data = read_final_profiles(puncture_directory)
    mass = float(os_data[2]["total_mass"])
    puncture_summary = puncture_data[2]
    if (not np.isfinite(mass) or mass <= 0.0
            or puncture_summary.get("mass_source") != "initial_gravitational_mass"
            or not np.isclose(puncture_summary["mass"], mass, rtol=1e-12, atol=0.0)):
        raise ValueError("The puncture must use the OS initial gravitational mass.")

    profiles = (os_data[0], puncture_data[0])
    radii = [fields["areal_radius"] / mass for fields in profiles]
    lower = max(float(radius.min()) for radius in radii)
    upper = min(float(radius.max()) for radius in radii)
    if radial_limit is not None:
        upper = min(upper, radial_limit)
    if upper <= lower:
        raise ValueError("The requested range has no shared areal-radius interval.")

    labels = []
    for name, (_, time, summary) in zip(
        ("OS collapse", "Schwarzschild puncture"), (os_data, puncture_data),
    ):
        status = ""
        if summary.get("completed") is False:
            status = " [INCOMPLETE]"
        elif summary.get("completed") is None:
            status = " [completion unknown]"
        labels.append(f"{name}: evolution t/M = {time / mass:.5f}{status}")

    figure, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True,
                                constrained_layout=True)
    for axis, (name, label) in zip(axes.flat, VARIABLES):
        for fields, radius, style, color, legend in zip(
            profiles, radii, ("-", "--"), ("tab:blue", "tab:orange"), labels,
        ):
            # Keep grid order even if an interior areal-radius curve folds.
            axis.plot(radius, fields[name], style, color=color,
                      linewidth=1.5, label=legend)
        axis.set_xlim(lower, upper)
        visible = [fields[name][(radius >= lower) & (radius <= upper)]
                   for fields, radius in zip(profiles, radii)]
        values = np.concatenate(visible)
        if values.size:
            vmin, vmax = float(values.min()), float(values.max())
            padding = 0.05 * (vmax - vmin) if vmax > vmin else 0.05 * max(abs(vmin), 1.0)
            axis.set_ylim(vmin - padding, vmax + padding)
        axis.set_ylabel(label)
        axis.set_title(name)
        axis.grid(alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel(r"Areal radius $R/M$, initial stellar gravitational mass $M$")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"Final OS collapse and Schwarzschild puncture\nInitial stellar gravitational mass M = {mass:.9g}\n"
        "Gauge-dependent evolution-coordinate components; areal radius only changes the horizontal axis.",
        fontsize=12,
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "final_bssn_comparison.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os-directory", type=Path,
                        default=DEMO_DIRECTORY / "outputs/z4c_oppenheimer_snyder")
    parser.add_argument("--puncture-directory", type=Path,
                        default=DEMO_DIRECTORY / "outputs/schwarzschild_puncture_initial_mass")
    parser.add_argument("--output-directory", type=Path,
                        default=DEMO_DIRECTORY / "outputs/schwarzschild_comparison_initial_mass")
    parser.add_argument("--radial-limit", type=float,
                        help="Maximum R/M shown (default: full shared radial extent).")
    args = parser.parse_args()
    make_comparison(args.os_directory, args.puncture_directory,
                    args.output_directory, args.radial_limit)


if __name__ == "__main__":
    main()
