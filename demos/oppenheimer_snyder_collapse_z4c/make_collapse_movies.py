from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "outputs"
    / (
        "z4c_oppenheimer_snyder_areal_inner_open_2ghost_tsc_zero_overlap"
        "_nonnegative_density_conserving_tsc"
    )
)

BSSN_VARIABLES = (
    ("alpha", r"lapse $\alpha$"),
    ("beta", r"shift $\beta^r$"),
    ("chi", r"conformal factor $\chi$"),
    ("conformal_grr", r"$\tilde{\gamma}_{rr}$"),
    ("conformal_gT", r"$\tilde{\gamma}_{T}$"),
    ("Kh", r"$K$"),
    ("Arr", r"$\tilde{A}_{rr}$"),
    ("AT", r"$\tilde{A}_{T}$"),
    ("theta", r"$\Theta$"),
    ("Gamma", r"$\tilde{\Gamma}^{r}$"),
)


def particle_boundary_name(output_directory):
    run_summary_path = output_directory / "run_summary.json"
    if run_summary_path.exists():
        with run_summary_path.open() as stream:
            return json.load(stream)["particle_boundary"]

    initial_data_path = output_directory / "initial_data_metadata.npz"
    if initial_data_path.exists():
        with np.load(initial_data_path) as initial_data:
            if "particle_boundary" in initial_data:
                return str(initial_data["particle_boundary"])

    return "deleting"


def metric_snapshots(output_directory, frame_stride):
    snapshots = sorted((output_directory / "metric").glob("metric_step_*.npz"))
    if not snapshots:
        raise RuntimeError(f"No metric snapshots found in {output_directory}")

    selected = snapshots[::frame_stride]
    if selected[-1] != snapshots[-1]:
        selected.append(snapshots[-1])

    return selected


def variable_plot_limits(snapshots, radial_limit):
    limits = {name: [np.inf, -np.inf] for name, _ in BSSN_VARIABLES}

    for snapshot_path in snapshots:
        with np.load(snapshot_path) as snapshot:
            inside = snapshot["areal_radius"] <= radial_limit
            for name, _ in BSSN_VARIABLES:
                values = snapshot[name][inside]
                limits[name][0] = min(limits[name][0], float(np.min(values)))
                limits[name][1] = max(limits[name][1], float(np.max(values)))

    for name, (lower, upper) in limits.items():
        span = upper - lower
        padding = 0.05 * span if span > 0.0 else 0.05 * max(abs(lower), 1.0)
        limits[name] = (lower - padding, upper + padding)

    return limits


def make_bssn_movie(snapshots, movie_path, boundary_name, radial_limit, fps):
    limits = variable_plot_limits(snapshots, radial_limit)
    figure, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()
    lines = []

    with np.load(snapshots[0]) as snapshot:
        radius = snapshot["areal_radius"]
        for axis, (name, title) in zip(axes, BSSN_VARIABLES):
            line, = axis.plot(radius, snapshot[name], linewidth=1.2)
            axis.set_xlim(0.0, radial_limit)
            axis.set_ylim(*limits[name])
            axis.set_title(title)
            axis.grid(alpha=0.25)
            lines.append(line)

    for axis in axes[-4:]:
        axis.set_xlabel(r"areal radius $R/M$")
    axes[10].axis("off")
    axes[11].axis("off")
    title = figure.suptitle("")

    writer = FFMpegWriter(fps=fps, metadata={"artist": "RadiShPICR"})
    with writer.saving(figure, movie_path, dpi=110):
        for snapshot_path in snapshots:
            with np.load(snapshot_path) as snapshot:
                radius = snapshot["areal_radius"]
                for line, (name, _) in zip(lines, BSSN_VARIABLES):
                    line.set_data(radius, snapshot[name])
                title.set_text(
                    f"Z4c Oppenheimer-Snyder collapse, {boundary_name} particles, "
                    f"t = {float(snapshot['time']):.5f} M"
                )
            writer.grab_frame()

    plt.close(figure)


def density_color_limits(snapshots, radial_limit):
    positive_density = []
    for snapshot_path in snapshots:
        with np.load(snapshot_path) as snapshot:
            inside = snapshot["areal_radius"] <= radial_limit
            density = snapshot["mass_density"][inside]
            positive_density.append(density[density > 0.0])

    positive_density = np.concatenate(positive_density)
    if positive_density.size == 0:
        return 1.0e-16, 1.0

    lower = max(float(np.percentile(positive_density, 1.0)), 1.0e-16)
    upper = float(np.percentile(positive_density, 99.9))
    return lower, max(upper, 10.0 * lower)


def radial_density(snapshot, radial_centers):
    radius = np.concatenate(([0.0], snapshot["areal_radius"]))
    density = np.concatenate(([snapshot["mass_density"][0]], snapshot["mass_density"]))
    increasing = np.concatenate(([True], np.diff(radius) > 0.0))
    radius = radius[increasing]
    density = density[increasing]

    return np.interp(radial_centers, radius, density, left=density[0], right=0.0)


def make_density_movie(snapshots, movie_path, boundary_name, radial_limit, fps):
    theta_edges = np.linspace(0.0, 2.0 * np.pi, 361)
    radial_edges = np.linspace(0.0, radial_limit, 401)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    vmin, vmax = density_color_limits(snapshots, radial_limit)

    figure = plt.figure(figsize=(7.8, 7.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="polar")
    axis.set_ylim(0.0, radial_limit)
    axis.set_facecolor("black")

    with np.load(snapshots[0]) as snapshot:
        density = radial_density(snapshot, radial_centers)
    density_2d = np.broadcast_to(density, (theta_edges.size - 1, density.size))
    image = axis.pcolormesh(
        theta_edges,
        radial_edges,
        density_2d.T,
        shading="flat",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    horizon_theta = np.linspace(0.0, 2.0 * np.pi, 721)
    axis.plot(
        horizon_theta,
        np.full_like(horizon_theta, 2.0),
        color="#66c2ff",
        linewidth=1.5,
        label=r"Schwarzschild event horizon $R=2M$",
    )
    axis.legend(loc="upper right")
    title = axis.set_title("")
    colorbar = figure.colorbar(image, ax=axis, pad=0.10)
    colorbar.set_label(r"Eulerian mass density $\rho M^2$")

    writer = FFMpegWriter(fps=fps, metadata={"artist": "RadiShPICR"})
    with writer.saving(figure, movie_path, dpi=100):
        for snapshot_path in snapshots:
            with np.load(snapshot_path) as snapshot:
                density = radial_density(snapshot, radial_centers)
                image.set_array(
                    np.broadcast_to(
                        density,
                        (theta_edges.size - 1, density.size),
                    ).T.ravel()
                )
                title.set_text(
                    f"Z4c Oppenheimer-Snyder density, {boundary_name} particles\n"
                    f"step {int(snapshot['step']):06d}, t = {float(snapshot['time']):.5f} M"
                )
            writer.grab_frame()

    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Render movies from a Z4c Oppenheimer-Snyder collapse run."
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--boundary-name",
        choices=("reflecting", "deleting", "areal_inner_open"),
        help="override the particle boundary recorded by the collapse run",
    )
    parser.add_argument("--radial-limit", type=float, default=15.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_arguments()
    boundary_name = args.boundary_name
    if boundary_name is None:
        boundary_name = particle_boundary_name(args.output_directory)

    snapshots = metric_snapshots(
        args.output_directory,
        max(1, args.frame_stride),
    )
    make_bssn_movie(
        snapshots,
        args.output_directory / "z4c_bssn_variables.mp4",
        boundary_name,
        args.radial_limit,
        args.fps,
    )
    make_density_movie(
        snapshots,
        args.output_directory / "z4c_spherical_density.mp4",
        boundary_name,
        args.radial_limit,
        args.fps,
    )


if __name__ == "__main__":
    main()
