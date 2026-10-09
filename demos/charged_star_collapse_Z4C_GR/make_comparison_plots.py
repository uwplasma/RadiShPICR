"""Compare saved charged-star and evolved RN puncture final states.

Undo the star's diagnostic rescaling before comparing coordinate components.
Areal radius changes the horizontal axis, not the gauge of those components.
Electric flux, normal electric field, and geometric masses also have analytic
RN references. No evolution or parameter fitting is performed here.
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
    ("alpha", r"$\alpha$"), ("beta", r"$\beta^r$"), ("chi", r"$\chi$"),
    ("conformal_grr", r"$\widetilde{\gamma}_{rr}$"),
    ("conformal_gT", r"$\widetilde{\gamma}_{T}$"),
    ("Kh", r"$\widehat{K}$"), ("Arr", r"$\widetilde{A}_{rr}$"),
    ("AT", r"$\widetilde{A}_{T}$"), ("theta", r"$\Theta$"),
    ("Gamma", r"$\widetilde{\Gamma}^{r}$"),
)
PHYSICAL_VARIABLES = (
    ("E_normal", r"$E_{\hat r}$"), ("charge_flux", r"$Q_{\rm flux}$"),
    ("misner_sharp_mass", r"$m_{\rm MS}/M$"),
    ("rn_mass_from_flux", r"$(m_{\rm MS}+Q_{\rm flux}^2/(8\pi\epsilon_0R))/M$"),
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
            *(name for name, _ in VARIABLES), "r", "E_r", "misner_sharp_mass",
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
    R = fields["r"] * np.sqrt(fields["conformal_gT"] / fields["chi"])
    E_normal = fields["E_r"] * np.sqrt(fields["chi"] / fields["conformal_grr"])
    charge_flux = 4.0 * np.pi * info["epsilon_0"] * R**2 * E_normal
    fields.update(
        areal_radius=R, E_normal=E_normal, charge_flux=charge_flux,
        K=fields["Kh"] + 2.0 * fields["theta"],
        rn_mass_from_flux=fields["misner_sharp_mass"] + charge_flux**2 / (8.0 * np.pi * info["epsilon_0"] * R),
    )
    return fields, info, summary


def comparison_statistics(profiles, mass, charge, epsilon_0):
    """Compare the monotone outer branches outside the nominal RN horizon.

    Exclude three derivative-closure cells at each boundary. Field differences
    use a common areal-radius grid and retain their evolution-coordinate gauge.
    """
    horizon = mass + np.sqrt(mass**2 - charge**2 / (4.0 * np.pi * epsilon_0))
    branches = []
    for fields in profiles:
        R = fields["areal_radius"]
        indices = np.arange(max(3, int(np.argmin(R))), len(R) - 3)
        indices = indices[R[indices] >= horizon]
        if len(indices) < 2 or np.any(np.diff(R[indices]) <= 0.0):
            raise ValueError("Comparison statistics require a resolved monotone exterior branch.")
        branches.append({key: value[indices] for key, value in fields.items()})

    lower = max(fields["areal_radius"][0] for fields in branches)
    upper = min(fields["areal_radius"][-1] for fields in branches)
    result = {"nominal_RN_horizon_over_mass": float(horizon / mass)}
    for extent, limit in (("full_exterior", upper), ("inner_exterior", min(upper, 10.0 * mass))):
        if limit <= lower:
            result[extent] = {"available": False}
            continue
        R = np.linspace(lower, limit, 1000)
        samples = [{key: np.interp(R, fields["areal_radius"], value)
                    for key, value in fields.items()} for fields in branches]
        errors = {"R_over_mass_interval": [float(lower / mass), float(limit / mass)]}
        for name, fields in zip(("charged_star", "RN_puncture"), branches):
            # Evaluate analytic errors on native points. Interpolating 1/R^2
            # first would manufacture an electric-field error for exact RN.
            native_R = fields["areal_radius"]
            mask = (native_R >= lower) & (native_R <= limit)
            mass_error = fields["rn_mass_from_flux"][mask] / mass - 1.0
            flux_error = fields["charge_flux"][mask] - charge
            E_error = fields["E_normal"][mask] - charge / (4.0 * np.pi * epsilon_0 * native_R[mask]**2)
            errors[name] = {
                "rn_mass_relative_error_linf": float(np.max(np.abs(mass_error))),
                "rn_mass_relative_error_rms": float(np.sqrt(np.mean(mass_error**2))),
                "rn_mass_relative_error_mean": float(np.mean(mass_error)),
                "charge_flux_error_linf": float(np.max(np.abs(flux_error))),
                "normal_electric_field_error_linf": float(np.max(np.abs(E_error))),
            }
        errors["star_minus_puncture"] = {}
        for key, _ in (*VARIABLES, *PHYSICAL_VARIABLES):
            delta = samples[0][key] - samples[1][key]
            errors["star_minus_puncture"][key] = dict(
                linf=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta**2))),
            )
        result[extent] = errors
    return result


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
        label = f"{name}: evolution t/M={info['evolution_time'] / mass:.5f}"
        if info["saved_coordinates"] != "evolution":
            label += f", diagnostic t/M={info['time'] / mass:.5f}"
        if info["completed"] is False:
            label += " [INCOMPLETE]"
        elif info["completed"] is None:
            label += " [completion unknown]"
        labels.append(label)

    statistics = comparison_statistics(profiles, mass, charge, epsilon_0)
    statistics.update(
        mass=mass, initial_charge=charge, epsilon_0=epsilon_0,
        charged_star=datasets[0][1], RN_puncture=datasets[1][1],
        field_comparison="evolution-coordinate components versus areal radius; no gauge transformation or mass fit",
        mass_diagnostic="Misner-Sharp mass plus charge inferred from electric flux, not particle charge",
        statistics_sampling="Analytic errors on native points; run differences on 1000 common areal-radius samples; three edge cells excluded",
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "comparison_summary.json").write_text(json.dumps(statistics, indent=2) + "\n")
    radii = [fields["areal_radius"] / mass for fields in profiles]
    lower = max(float(radius.min()) for radius in radii)
    upper = min(float(radius.max()) for radius in radii)
    paths = []
    for extent, limit in (("full", upper), ("inner", min(upper, radial_limit))):
        if limit <= lower:
            raise ValueError("The requested plot range has no shared areal-radius interval.")
        analytic_R = np.linspace(lower, limit, 1000) * mass
        analytic = {
            "E_normal": charge / (4.0 * np.pi * epsilon_0 * analytic_R**2),
            "charge_flux": np.full_like(analytic_R, charge),
            "misner_sharp_mass": mass - charge**2 / (8.0 * np.pi * epsilon_0 * analytic_R),
            "rn_mass_from_flux": np.full_like(analytic_R, mass),
        }
        for kind, variables, size in (("bssn", VARIABLES, (12, 15)), ("physical", PHYSICAL_VARIABLES, (12, 8))):
            figure, axes = plt.subplots(len(variables) // 2, 2, figsize=size, sharex=True, constrained_layout=True)
            for axis, (key, label) in zip(axes.flat, variables):
                scale = mass if key in ("misner_sharp_mass", "rn_mass_from_flux") else 1.0
                visible = []
                for fields, radius, style, color, legend in zip(profiles, radii, ("-", "--"), ("tab:blue", "tab:orange"), labels):
                    axis.plot(radius, fields[key] / scale, style, color=color, linewidth=1.5, label=legend)
                    visible.append(fields[key][(radius >= lower) & (radius <= limit)] / scale)
                if kind == "physical":
                    axis.plot(analytic_R / mass, analytic[key] / scale, ":", color="black", label="Analytic RN, initial M and Q")
                    visible.append(analytic[key] / scale)
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
            subtitle = "Gauge-dependent evolution-coordinate components" if kind == "bssn" else "Electric flux and geometric mass; analytic RN uses the initial M and Q"
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
    return paths, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--star-directory", type=Path, default=DEMO_DIRECTORY / "outputs/z4c_charged_star_schwarzschild")
    parser.add_argument("--puncture-directory", type=Path, default=DEMO_DIRECTORY / "outputs/reissner_nordstrom_puncture_initial_mass_charge")
    parser.add_argument("--output-directory", type=Path, default=DEMO_DIRECTORY / "outputs/rn_comparison_initial_mass_charge")
    parser.add_argument("--radial-limit", type=float, default=10.0, help="Maximum R/M for the inner figures (default: 10).")
    args = parser.parse_args()
    make_comparison(args.star_directory, args.puncture_directory, args.output_directory, args.radial_limit)


if __name__ == "__main__":
    main()
