"""Overlay final OS and initial-mass Schwarzschild fields against areal radius.

Run after run_schwarzschild_puncture.py. Use --radial-limit to restrict R/M.
The OS diagnostic rescaling is reversed; the field components retain their
evolution-coordinate definitions when plotted against areal radius.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEMO_DIRECTORY = Path(__file__).resolve().parent
VARIABLES = (
    ("alpha", r"$\alpha$"),
    ("beta", r"$\beta^r$"),
    ("chi", r"$\chi$"),
    ("conformal_grr", r"$\widetilde{\gamma}_{rr}$"),
    ("conformal_gT", r"$\widetilde{\gamma}_{T}$"),
    ("Kh", r"$\widehat{K}$"),
    ("Arr", r"$\widetilde{A}_{rr}$"),
    ("AT", r"$\widetilde{A}_{T}$"),
    ("theta", r"$\Theta$"),
    ("Gamma", r"$\widetilde{\Gamma}^{r}$"),
)


def read_final_snapshot(output_directory):
    """Read the recorded final step, or the latest saved step without a summary."""
    output_directory = Path(output_directory)
    summary_path = output_directory / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "final_step" in summary:
        path = output_directory / "metric" / f"metric_step_{summary['final_step']:06d}.npz"
    else:
        snapshots = sorted((output_directory / "metric").glob("metric_step_*.npz"))
        if not snapshots:
            raise RuntimeError(f"No metric snapshots found in {output_directory}")
        path = snapshots[-1]

    with np.load(path) as snapshot:
        fields = {name: snapshot[name] for name, _ in VARIABLES}
        r = snapshot["r"]
        coordinates = str(snapshot["saved_coordinates"])
        time = float(snapshot["time"])
        if coordinates == "schwarzschild_isotropic_diagnostic":
            X_r, X_t = float(snapshot["X_r"]), float(snapshot["X_t"])
            r = r / X_r
            fields["alpha"] *= X_t
            fields["beta"] *= X_t / X_r
            fields["chi"] /= X_r**2
            fields["Gamma"] *= X_r
        elif coordinates != "evolution":
            raise ValueError(f"Unsupported saved coordinates: {coordinates}")

    if not all(np.all(np.isfinite(value)) for value in (r, *fields.values())):
        raise ValueError(f"Nonfinite final fields in {path}")
    if any(np.any(fields[name] <= 0.0) for name in ("chi", "conformal_grr", "conformal_gT")):
        raise ValueError(f"Nonpositive spatial geometry in {path}")
    fields["r"] = r
    fields["areal_radius"] = r * np.sqrt(fields["conformal_gT"] / fields["chi"])
    return fields, time, coordinates, summary


def make_comparison(os_directory, puncture_directory, output_directory, radial_limit=None):
    with np.load(Path(os_directory) / "initial_data_metadata.npz") as initial:
        mass = float(initial["total_mass"])
    os_data = read_final_snapshot(os_directory)
    puncture_data = read_final_snapshot(puncture_directory)
    puncture_summary = puncture_data[3]
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
    for name, (_, time, coordinates, summary) in zip(
        ("OS collapse", "Schwarzschild puncture"), (os_data, puncture_data),
    ):
        status = ""
        if summary.get("completed") is False:
            status = " [INCOMPLETE]"
        elif summary.get("completed") is None:
            status = " [completion unknown]"
        time_label = "evolution" if coordinates == "evolution" else "diagnostic"
        labels.append(f"{name}: {time_label} t/M = {time / mass:.5f}{status}")

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
