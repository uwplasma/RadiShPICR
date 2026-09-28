"""Compare saved dynamic/static Z4C runs without changing simulation outputs.

Recompute physical energy densities and curved-space Gauss consistency from
snapshots, then write a quantitative review and synchronized comparison movies.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


METRIC_FIELDS = ("alpha", "beta", "chi", "conformal_grr", "conformal_gt",
                 "Kh", "Arr", "At", "theta", "Gamma")
COLORS = ("#2166ac", "#d6604d")
LABELS = ("Dynamic Z4C", "Static solved metric")


def load_snapshots(directory):
    paths = sorted(directory.glob("*.npz"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    frames = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            frames.append({key: data[key] for key in data.files})
    return paths, {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}


def physical_fields(data, params):
    r = data["r"]
    dr = params["grid_spacing"]
    grr = data["conformal_grr"] / data["chi"]
    gt = data["conformal_gt"] / data["chi"]
    shell_volume = 4 * np.pi / 3 * ((r + dr / 2)**3 - np.maximum(r - dr / 2, 0)**3)
    volume = shell_volume * np.sqrt(grr) * gt

    # E_r is covariant; Eulerian energy uses E_i E^i and proper volume.
    data["E_normal"] = data["E_r"] / np.sqrt(grr)
    data["rho_E"] = 0.5 * params["epsilon_0"] * data["E_normal"]**2
    qe_me = params["electron_charge_to_mass"]
    qi_mi = -qe_me / params["ion_to_electron_mass_ratio"]
    data["rho_rest"] = data["electron_charge_density"] / qe_me + data["ion_charge_density"] / qi_mi
    data["rho_kin"] = data["matter_rho"] - data["rho_rest"] - data["rho_E"]
    data["grr"] = grr
    data["volume"] = volume

    inside = (r[0] >= params["analysis_r_min"]) & (r[0] <= params["analysis_r_max"])
    data["U_E"] = np.sum(data["rho_E"] * volume, axis=1)
    data["U_E_window"] = np.sum((data["rho_E"] * volume)[:, inside], axis=1)
    data["K_integral"] = np.sum(data["rho_kin"] * volume, axis=1)
    data["M_rest_integral"] = np.sum(data["rho_rest"] * volume, axis=1)

    # Project the physical normal field with the physical energy inner product.
    phase = params["wavenumber"] * (r[0, inside] - params["plasma_r_min"])
    basis = np.column_stack((np.sin(phase), np.cos(phase)))
    fractions = []
    for field, vol in zip(data["E_normal"][:, inside], volume[:, inside]):
        coef = np.linalg.solve(basis.T @ (vol[:, None] * basis), basis.T @ (vol * field))
        fractions.append(np.sum(vol * (basis @ coef)**2) / np.sum(vol * field**2))
    data["physical_mode_fraction"] = np.asarray(fractions)

    # Invert the adjacent-face averaging used by the saved Gauss solve.
    # Flux is area * E_r / sqrt(gamma_rr), with metric averaged to faces.
    face_E = np.zeros((r.shape[0], r.shape[1] + 1))
    for cell in range(r.shape[1]):
        face_E[:, cell + 1] = 2 * data["E_r"][:, cell] - face_E[:, cell]
    face_metric = {}
    for key in ("conformal_grr", "conformal_gt", "chi"):
        a = data[key]
        face_metric[key] = np.concatenate((a[:, :1], 0.5 * (a[:, :-1] + a[:, 1:]), a[:, -1:]), axis=1)
    rf = np.concatenate((np.zeros_like(r[:, :1]), r + dr / 2), axis=1)
    face_area = 4 * np.pi * rf**2 * face_metric["conformal_gt"] / face_metric["chi"]
    flux = face_area * face_E * np.sqrt(face_metric["chi"] / face_metric["conformal_grr"])
    residual = params["epsilon_0"] * np.diff(flux, axis=1) / volume - data["charge_density"]
    data["gauss_relative_corrected"] = np.max(np.abs(residual), axis=1) / np.max(np.abs(data["charge_density"]), axis=1)
    return data


def fit_growth(time, energy, start, end):
    mask = (time >= start) & (time <= end)
    x, y = time[mask], np.log(energy[mask])
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - slope * x - intercept)**2) / np.sum((y - y.mean())**2)
    return {"start": float(x[0]), "end": float(x[-1]), "samples": int(x.size),
            "gamma": float(slope / 2), "r_squared": float(r2)}


def analyze(runs, output):
    dynamic, static = runs
    assert np.array_equal(dynamic["data"]["time"], static["data"]["time"])
    assert np.array_equal(dynamic["diagnostics"]["time"], static["diagnostics"]["time"])
    summary = {"initial_numeric_snapshots_identical": True, "runs": {}}
    for key in dynamic["raw_keys"]:
        a, b = dynamic["data"][key][0], static["data"][key][0]
        if a.dtype.kind in "fiu":
            assert np.array_equal(a, b), key
    summary["parameter_differences"] = {k: [dynamic["params"][k], static["params"][k]]
        for k in dynamic["params"] if dynamic["params"][k] != static["params"][k]}

    rows = []
    for run in runs:
        a, diag = run["data"], run["diagnostics"]
        indices = a["step"].astype(int)
        assert np.all(np.diff(a["step"]) > 0)
        assert np.array_equal(a["time"], diag["time"][indices])
        for key in run["raw_keys"]:
            if a[key].dtype.kind in "fiu":
                assert np.all(np.isfinite(a[key])), key
        for calculated, recorded in (("U_E", "electric_field_energy"),
                                     ("K_integral", "relativistic_kinetic_energy"),
                                     ("M_rest_integral", "rest_mass_energy")):
            np.testing.assert_allclose(a[calculated], diag[recorded][indices], rtol=2e-11, atol=1e-14)
        metric_change = {k: float(np.max(np.abs(a[k] - a[k][0]))) for k in METRIC_FIELDS}
        if not run["params"]["dynamic_gr"]:
            assert all(v == 0 for v in metric_change.values())
        mass = diag["misner_sharp_mass"]
        peak = np.argmax(diag["electric_field_energy"])
        peak_window = np.argmax(a["U_E_window"])
        late = diag["time"] >= 50
        late_snap = a["time"] >= 50
        result = {
            "diagnostic_samples": len(diag), "snapshot_count": len(indices),
            "final_time": float(a["time"][-1]), "metric_max_change": metric_change,
            "initial_mass": float(mass[0]), "mass_final_relative_change": float(mass[-1] / mass[0] - 1),
            "mass_max_relative_change": float(np.max(np.abs(mass / mass[0] - 1))),
            "electric_energy_peak": float(diag["electric_field_energy"][peak]),
            "electric_energy_peak_time": float(diag["time"][peak]),
            "electric_energy_final": float(a["U_E"][-1]),
            "electric_energy_late_mean": float(np.mean(diag["electric_field_energy"][late])),
            "electric_energy_late_std": float(np.std(diag["electric_field_energy"][late])),
            "physical_window_energy_peak": float(a["U_E_window"][peak_window]),
            "physical_window_energy_peak_time": float(a["time"][peak_window]),
            "physical_window_energy_final": float(a["U_E_window"][-1]),
            "physical_window_energy_late_mean": float(np.mean(a["U_E_window"][late_snap])),
            "kinetic_energy_final": float(diag["relativistic_kinetic_energy"][-1]),
            "kinetic_plus_field_final_relative_change": float(diag["relative_total_energy_drift"][-1]),
            "kinetic_plus_field_max_relative_change": float(np.max(np.abs(diag["relative_total_energy_drift"]))),
            "mode_fraction_final_recorded": float(diag["target_mode_energy_fraction"][-1]),
            "mode_fraction_final_physical": float(a["physical_mode_fraction"][-1]),
            "max_total_charge_relative_to_electron_charge": float(np.max(np.abs(diag["total_charge"] / diag["electron_charge"]))),
            "minimum_boundary_margin": float(np.min(diag["boundary_margin"])),
            "saved_gauss_max_relative": float(np.max(diag["relative_gauss_residual_linf"])),
            "corrected_gauss_max_relative": float(np.max(a["gauss_relative_corrected"])),
            "window_energy_saved_over_physical_final": float(diag["analysis_electric_field_energy"][-1] / a["U_E_window"][-1]),
            "final_ranges": {k: [float(a[k][-1].min()), float(a[k][-1].max())]
                for k in ("alpha", "beta", "chi", "grr", "matter_rho", "rho_kin", "rho_E", "E_normal", "theta")},
            "fits_physical_window": [fit_growth(a["time"], a["U_E_window"], start, end)
                for start, end in ((5, 25), (10, 30), (15, 35), (4.25, 39.25))],
            "saved_growth_fit": run["fit"],
        }
        summary["runs"][run["name"]] = result
        for j, time in enumerate(a["time"]):
            rows.append([run["name"], time, a["U_E"][j], a["U_E_window"][j],
                         a["K_integral"][j], a["physical_mode_fraction"][j], a["gauss_relative_corrected"][j]])
    with (output / "corrected_diagnostics.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "time", "electric_energy", "window_electric_energy", "kinetic_energy",
                         "physical_mode_fraction", "curved_gauss_relative_residual"))
        writer.writerows(rows)
    return summary


def plot_histories(runs, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for run, color, label in zip(runs, COLORS, LABELS):
        a, d = run["data"], run["diagnostics"]
        axes[0, 0].plot(d["time"], d["electric_field_energy"], color=color, label=label)
        axes[0, 1].plot(a["time"], a["U_E_window"], color=color)
        axes[1, 0].plot(a["time"], a["physical_mode_fraction"], color=color)
        axes[1, 1].plot(d["time"], 100 * d["relative_total_energy_drift"], color=color)
        axes[2, 0].plot(d["time"], 1e6 * (d["misner_sharp_mass"] / d["misner_sharp_mass"][0] - 1), color=color)
        axes[2, 1].plot(a["time"], a["alpha"].min(axis=1), color=color)
    titles = ("Total physical electric energy", "Physical electric energy: 30 < r < 50",
              "Physical seeded-mode energy fraction: 30 < r < 50", "Change in kinetic + electric energy (%)",
              "Outer Misner–Sharp mass change (ppm)", "Minimum lapse")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Coordinate time (code units)")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.grid(alpha=0.2)
    for ax in axes[0]:
        ax.set_ylim(bottom=0)
    axes[0, 0].legend()
    fig.savefig(output / "comparison_histories.png", dpi=160)
    plt.close(fig)

    # Separate energy histories make the central-window comparison easy to read.
    for key, name, title in (
        ("U_E", "total_electric_energy", "Total-domain physical electric energy"),
        ("U_E_window", "central_window_electric_energy", "Physical electric energy: 30 < r < 50"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for run, color, label, style in zip(runs, COLORS, LABELS, ("-", "--")):
            a = run["data"]
            time, energy = a["time"], a[key]
            if key == "U_E":
                time = run["diagnostics"]["time"]
                energy = run["diagnostics"]["electric_field_energy"]
            ax.plot(time, energy, color=color, label=label, ls=style)
        ax.set(xlabel="Coordinate time (code units)", ylabel="Electric energy (code units)",
               title=f"{title} — linear scale", xscale="linear", yscale="linear")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for col, (key, title) in enumerate((("E_normal", r"Physical electric field $E_{\hat r}$"),
                                      ("rho_kin", "Relativistic kinetic energy density"),
                                      ("matter_rho", "Total Eulerian energy density"))):
        vmax = max(np.abs(run["data"][key]).max() for run in runs)
        for row, run in enumerate(runs):
            a = run["data"]
            im = axes[row, col].pcolormesh(a["r"][0], a["time"], a[key], shading="auto",
                cmap="RdBu_r" if col == 0 else "magma", vmin=-vmax if col == 0 else 0, vmax=vmax, rasterized=True)
            axes[row, col].set_title(f"{LABELS[row]}\n{title}", fontsize=10)
            axes[row, col].set_xlabel("Coordinate radius r")
            axes[row, col].set_ylabel("Coordinate time")
            fig.colorbar(im, ax=axes[row, col], format="%.1e")
    fig.savefig(output / "spacetime_comparison.png", dpi=160)
    plt.close(fig)


def profile_movie(runs, output, name, fields, fps):
    nrows = (len(fields) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.5 * nrows), constrained_layout=True)
    lines = []
    for ax, (key, title, positive) in zip(axes.flat, fields):
        lo = min(run["data"][key].min() for run in runs)
        hi = max(run["data"][key].max() for run in runs)
        pad = 0.06 * (hi - lo) if hi > lo else 0.06 * max(abs(hi), 1e-10)
        ax.set_ylim(0 if positive else lo - pad, hi + pad)
        ax.set_xlim(0, runs[0]["params"]["r_max"])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Coordinate radius r (code units)")
        ax.axvspan(30, 50, color="gray", alpha=0.07)
        ax.grid(alpha=0.2)
        pair = []
        for run, color, label, style in zip(runs, COLORS, LABELS, ("-", "--")):
            line, = ax.plot(run["data"]["r"][0], run["data"][key][0], color=color,
                            label=label, ls=style, lw=1.3)
            pair.append(line)
        lines.append((key, pair))
    axes.flat[0].legend(fontsize=9)
    title = fig.suptitle("")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-threads", "2"])
    with writer.saving(fig, str(output / f"{name}.mp4"), dpi=120):
        for j, time in enumerate(runs[0]["data"]["time"]):
            for key, pair in lines:
                for run, line in zip(runs, pair):
                    line.set_ydata(run["data"][key][j])
            title.set_text(f"Dynamic vs static metric | t = {time:.2f} | shared linear axes; shaded region: analysis window")
            writer.grab_frame()
            if j in (0, len(runs[0]["data"]["time"]) - 1):
                fig.savefig(output / f"{name}_{'initial' if j == 0 else 'final'}.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {name}.mp4", flush=True)


def phase_movie(runs, output, fps):
    # Read all particles for audit; display a deterministic subset per population.
    samples = []
    stats = {}
    initial = None
    for run in runs:
        frames = []
        electron_gamma = []
        ion_gamma = []
        paths = sorted((run["path"] / "phase_space").glob("*.npz"))
        assert len(paths) == len(run["data"]["time"])
        for j, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as a:
                assert float(a["time"]) == run["data"]["time"][j]
                for key in a.files:
                    if a[key].dtype.kind in "fiu":
                        assert np.all(np.isfinite(a[key]))
                if j == 0:
                    current = {k: a[k] for k in a.files if a[k].dtype.kind in "fiu"}
                    if initial is None:
                        initial = current
                    else:
                        assert all(np.array_equal(current[k], initial[k]) for k in current)
                    weights, population = a["weight"], a["population_id"]
                assert np.array_equal(a["weight"], weights)
                assert np.array_equal(a["population_id"], population)
                alive = a["weight"] > 0
                gamma = np.sqrt(1 + a["radial_orthonormal_momentum"]**2)
                for mask, values in ((alive & (population < 2), electron_gamma), (alive & (population >= 2), ion_gamma)):
                    values.append(float(np.average(gamma[mask] - 1, weights=a["weight"][mask])))
                # Raw coordinates: no folding and no phase remapping.
                frames.append([np.column_stack((a["r"][population == p][::4], a["ur"][population == p][::4])) for p in range(4)])
        samples.append(frames)
        stats[run["name"]] = {"initial_particles": int(population.size), "weights_unchanged": True,
            "electron_mean_gamma_minus_one_initial": electron_gamma[0],
            "electron_mean_gamma_minus_one_final": electron_gamma[-1],
            "ion_mean_gamma_minus_one_initial": ion_gamma[0],
            "ion_mean_gamma_minus_one_final": ion_gamma[-1]}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    scatters = []
    for row in range(2):
        populations = (0, 1) if row == 0 else (2, 3)
        ymin = min(frames[j][p][:, 1].min() for frames in samples for j in range(len(frames)) for p in populations)
        ymax = max(frames[j][p][:, 1].max() for frames in samples for j in range(len(frames)) for p in populations)
        for col in range(2):
            ax = axes[row, col]
            ax.set_xlim(0, 80)
            ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
            ax.set_xlabel("Coordinate radius r")
            ax.set_ylabel(r"Covariant specific momentum $u_r$")
            ax.set_title(f"{LABELS[col]}: {'electrons' if row == 0 else 'ions'}")
            pair = [ax.scatter([], [], s=0.7, alpha=0.65, color=color, label=label)
                for color, label in zip(("#2166ac", "#d6604d"), ("Initially outgoing", "Initially incoming"))]
            scatters.append((col, populations, pair))
    axes[0, 0].legend(markerscale=4, fontsize=9)
    title = fig.suptitle("")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-threads", "2"])
    with writer.saving(fig, str(output / "raw_phase_space.mp4"), dpi=120):
        for j, time in enumerate(runs[0]["data"]["time"]):
            for col, populations, pair in scatters:
                for p, scatter in zip(populations, pair):
                    scatter.set_offsets(samples[col][j][p])
            title.set_text(f"Raw radial phase space | t = {time:.2f} | every fourth particle per population")
            writer.grab_frame()
        fig.savefig(output / "raw_phase_space_final.png", dpi=120)
    plt.close(fig)
    print("Wrote raw_phase_space.mp4", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/review_dynamic_vs_static_20260908_linear"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--skip-movies", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    manifest = []
    for name in ("dynamic_gr", "no_dynamic_gr"):
        path = Path(__file__).resolve().parent / "outputs" / name
        params = json.loads((path / "run_parameters.json").read_text())
        paths, data = load_snapshots(path / "metric")
        raw_keys = tuple(data)
        data = physical_fields(data, params)
        # CSV boolean fields must not be parsed as floating point NaNs.
        with (path / "diagnostics.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert all(row["finite_state"] == "True" for row in rows)
        names = [key for key in rows[0] if key != "finite_state"]
        diagnostics = np.array([tuple(float(row[k]) for k in names) for row in rows], dtype=[(k, float) for k in names])
        runs.append({"name": name, "path": path, "params": params, "data": data,
                     "raw_keys": raw_keys, "diagnostics": diagnostics,
                     "fit": json.loads((path / "growth_fit.json").read_text())})
        for source in paths + sorted((path / "phase_space").glob("*.npz")) + [path / "run_parameters.json", path / "diagnostics.csv", path / "growth_fit.json"]:
            manifest.append({"path": str(source), "size": source.stat().st_size,
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        print(f"Read and audited {name}: {len(paths)} metric frames", flush=True)
    (args.output / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = analyze(runs, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_histories(runs, args.output)
    print("Wrote quantitative summary and static plots", flush=True)
    if not args.skip_movies:
        profile_movie(runs, args.output, "relativistic_energy_density", (
            ("matter_rho", r"Total Eulerian energy density $\rho$ (particles + electric field)", True),
            ("rho_kin", r"Relativistic kinetic energy density $\rho-\rho_{rest}-\rho_E$", True),
            ("rho_E", r"Electric energy density $\rho_E=\epsilon_0 E_{\hat r}^2/2$", True),
            ("rho_rest", r"Deposited rest-mass density $\rho_{rest}$", True)), args.fps)
        profile_movie(runs, args.output, "electric_field_and_charge", (
            ("E_normal", r"Physical normal electric field $E_{\hat r}$", False),
            ("E_r", r"Covariant radial electric field $E_r$", False),
            ("charge_density", "Net charge density per proper volume", False),
            ("matter_Sr", r"Total contravariant radial momentum density $S^r$", False)), args.fps)
        profile_movie(runs, args.output, "metric_and_mass", (
            ("alpha", r"Lapse $\alpha$", False), ("beta", r"Shift $\beta^r$", False),
            ("grr", r"Physical radial metric $\gamma_{rr}$", False), ("areal_radius", "Areal radius R(r)", True),
            ("misner_sharp_mass", "Misner–Sharp mass M(r)", False), ("theta", r"Z4C constraint variable $\Theta$", False)), args.fps)
        summary["phase_space"] = phase_movie(runs, args.output, args.fps)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
