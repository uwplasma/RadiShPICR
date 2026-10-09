"""Compare saved charged-star and evolved RN puncture final states.

Undo diagnostic rescaling only for historical Schwarzschild-chart snapshots.
Areal radius changes the horizontal axis, not the gauge of those components.
Plot only the covariant electric field E_r and final BSSN/Z4C variables.
No evolution or parameter fitting is performed here.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEMO_DIRECTORY = Path(__file__).resolve().parent
STAR_DIRECTORY = DEMO_DIRECTORY / "outputs/z4c_charged_star_evolution_300M"
PUNCTURE_DIRECTORY = DEMO_DIRECTORY / "outputs/reissner_nordstrom_puncture_evolution_300M"
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/rn_comparison_evolution_300M"
RADIAL_LIMIT = 10.0  # Maximum R/M for the inner figures.

VARIABLES = (
    ("alpha", r"$\alpha$"), ("beta", r"$\beta^r$"), ("chi", r"$\chi$"),
    ("conformal_grr", r"$\widetilde{\gamma}_{rr}$"),
    ("conformal_gT", r"$\widetilde{\gamma}_{T}$"),
    ("Kh", r"$\widehat{K}$"), ("Arr", r"$\widetilde{A}_{rr}$"),
    ("AT", r"$\widetilde{A}_{T}$"), ("theta", r"$\Theta$"),
    ("Gamma", r"$\widetilde{\Gamma}^{r}$"),
)


def read_final_snapshot(output_directory):
    output_directory = Path(output_directory)
    summary_path = output_directory / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "final_step" in summary:
        path = output_directory / "metric" / f"metric_step_{summary['final_step']:06d}.npz"
    else:
        snapshots = sorted((output_directory / "metric").glob("metric_step_*.npz"))
        if not snapshots:
            raise RuntimeError(f"No metric snapshots in {output_directory}")
        path = snapshots[-1]

    with np.load(path) as data:
        fields = {name: np.asarray(data[name]).copy() for name in (
            *(name for name, _ in VARIABLES), "r", "E_r",
        )}
        coordinates = str(data["saved_coordinates"])
        info = dict(snapshot=str(path.resolve()), saved_coordinates=coordinates,
                    time=float(data["time"]), evolution_time=float(data["evolution_time"]),
                    epsilon_0=float(data["epsilon_0"]), completed=summary.get("completed"),
                    evolution_parameters={key: summary.get(key) for key in (
                        "kappa", "eta", "nu", "zero_shift", "cfl",
                        "puncture_chi_floor", "puncture_chi_floor_cells",
                    )})
        if coordinates == "schwarzschild_isotropic_diagnostic":
            X_r, X_t = float(data["X_r"]), float(data["X_t"])
            fields["r"] /= X_r
            fields["alpha"] *= X_t
            fields["beta"] *= X_t / X_r
            fields["chi"] /= X_r**2
            fields["Gamma"] *= X_r
            fields["E_r"] *= X_r
        elif coordinates != "evolution":
            raise ValueError(f"Unsupported saved coordinates: {coordinates}")

    if not all(np.all(np.isfinite(value)) for value in fields.values()):
        raise ValueError(f"Nonfinite final fields in {path}")
    if any(np.any(fields[name] <= 0.0) for name in ("chi", "conformal_grr", "conformal_gT")):
        raise ValueError(f"Nonpositive spatial geometry in {path}")
    fields["areal_radius"] = fields["r"] * np.sqrt(fields["conformal_gT"] / fields["chi"])
    return fields, info, summary


def make_comparison(star_directory, puncture_directory, output_directory, radial_limit=10.0):
    with np.load(Path(star_directory) / "initial_data_metadata.npz") as initial:
        mass, charge, epsilon_0 = (float(initial[key]) for key in ("total_mass", "initial_charge", "epsilon_0"))
    datasets = [read_final_snapshot(directory) for directory in (star_directory, puncture_directory)]
    profiles = [data[0] for data in datasets]
    puncture_summary = datasets[1][2]
    if (puncture_summary.get("mass_source") != "initial_gravitational_mass"
            or not np.isclose(puncture_summary["mass"], mass, rtol=1e-12, atol=0.0)
            or not np.isclose(puncture_summary["initial_charge"], charge, rtol=1e-12, atol=0.0)
            or any(not np.isclose(data[1]["epsilon_0"], epsilon_0, rtol=1e-12, atol=0.0) for data in datasets)):
        raise ValueError("The reference must match the star's initial M, Q, and epsilon_0.")

    labels = []
    for name, (_, info, _) in zip(("Charged star", "RN puncture"), datasets):
        label = f"{name}: Z4C t/M={info['evolution_time'] / mass:.5f}"
        if info["saved_coordinates"] != "evolution":
            label += f", diagnostic t/M={info['time'] / mass:.5f}"
        if info["completed"] is False:
            label += " [INCOMPLETE]"
        elif info["completed"] is None:
            label += " [completion unknown]"
        labels.append(label)

    summary = dict(
        mass=mass, initial_charge=charge, epsilon_0=epsilon_0,
        charged_star=datasets[0][1], RN_puncture=datasets[1][1],
        plotted_fields=[name for name, _ in VARIABLES] + ["E_r"],
        field_comparison="evolution-coordinate components versus areal radius; no gauge transformation or mass fit",
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    radii = [fields["areal_radius"] / mass for fields in profiles]
    lower = max(float(radius.min()) for radius in radii)
    upper = min(float(radius.max()) for radius in radii)
    paths = []
    for extent, limit in (("full", upper), ("inner", min(upper, radial_limit))):
        if limit <= lower:
            raise ValueError("The requested plot range has no shared areal-radius interval.")
        for kind, variables, layout, size in (
            ("bssn", VARIABLES, (5, 2), (12, 15)),
            ("electric_field", (("E_r", r"$E_r$"),), (1, 1), (10, 6)),
        ):
            figure, axes = plt.subplots(*layout, figsize=size, sharex=True,
                                        squeeze=False, constrained_layout=True)
            for axis, (key, label) in zip(axes.flat, variables):
                visible = []
                for fields, radius, style, color, legend in zip(profiles, radii, ("-", "--"), ("tab:blue", "tab:orange"), labels):
                    axis.plot(radius, fields[key], style, color=color, linewidth=1.5, label=legend)
                    visible.append(fields[key][(radius >= lower) & (radius <= limit)])
                values = np.concatenate(visible)
                if values.size:
                    vmin, vmax = float(values.min()), float(values.max())
                    padding = max(0.05 * (vmax - vmin), 1e-12 * max(abs(vmin), abs(vmax), 1.0))
                    axis.set_ylim(vmin - padding, vmax + padding)
                axis.set_xlim(lower, limit)
                axis.set_ylabel(label)
                axis.set_title(key)
                axis.grid(alpha=0.3)
            for axis in axes[-1]:
                axis.set_xlabel(r"Areal radius $R/M$ (initial stellar mass)")
            axes[0, 0].legend(fontsize=7)
            subtitle = "Evolution-coordinate BSSN/Z4C components" if kind == "bssn" else "Covariant electric field in evolution coordinates"
            star_nu, rn_nu = (data[2].get("nu") for data in datasets)
            if star_nu is not None and rn_nu is not None and star_nu != rn_nu:
                subtitle += f"\nDissipation differs: star nu={star_nu:g}, RN nu={rn_nu:g}"
            floor = puncture_summary.get("puncture_chi_floor", 0.0)
            if floor > 0.0:
                subtitle += f"\nRN puncture chi floor={floor:g} in the first three cells"
            figure.suptitle(f"Final charged-star collapse and evolved RN puncture\nM={mass:.9g}, q/M={charge / np.sqrt(4 * np.pi * epsilon_0) / mass:.9g}\n{subtitle}", fontsize=12)
            path = output_directory / f"final_{kind}_comparison_{extent}.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(path)
            print(f"Saved {path}")
    return paths, summary


def main():
    make_comparison(STAR_DIRECTORY, PUNCTURE_DIRECTORY, OUTPUT_DIRECTORY, RADIAL_LIMIT)


if __name__ == "__main__":
    main()
