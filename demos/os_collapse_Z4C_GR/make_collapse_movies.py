"""Render movies from a Z4c Oppenheimer-Snyder collapse run.

Reads the run's openPMD series.  Snapshots hold raw state, so the Eulerian
mass density is re-deposited from the saved particles rather than read.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from RadiShPICR.diagnostics.read_bssn import iter_snapshots, read_metadata
from RadiShPICR.Z4C.energy_momentum_tensor import compute_radial_matter_terms

DEFAULT_OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/z4c_oppenheimer_snyder"
BSSN_VARIABLES = (
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


def load_frames(run_directory, frame_stride):
    """Read the series once, re-depositing the mass density per frame."""

    frames = []
    for snapshot in iter_snapshots(run_directory, stride=frame_stride):
        frame = {name: np.asarray(getattr(snapshot.metric, name))
                 for name, _ in BSSN_VARIABLES}
        frame["areal_radius"] = np.asarray(snapshot.areal_radius)
        frame["step"] = snapshot.step
        frame["time"] = snapshot.time
        matter = compute_radial_matter_terms(
            snapshot.species(), snapshot.metric, inner_open=True
        )
        frame["mass_density"] = np.asarray(matter.rho)
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No snapshots found in {run_directory}")

    return frames


def variable_plot_limits(frames, radial_limit):
    limits = {name: [np.inf, -np.inf] for name, _ in BSSN_VARIABLES}
    for frame in frames:
        inside = frame["areal_radius"] <= radial_limit
        for name, _ in BSSN_VARIABLES:
            values = frame[name][inside]
            limits[name][0] = min(limits[name][0], float(np.min(values)))
            limits[name][1] = max(limits[name][1], float(np.max(values)))

    for name, (lower, upper) in limits.items():
        span = upper - lower
        padding = 0.05 * span if span > 0.0 else 0.05 * max(abs(lower), 1.0)
        limits[name] = (lower - padding, upper + padding)

    return limits


def make_bssn_movie(frames, movie_path, boundary_name, radial_limit, fps):
    limits = variable_plot_limits(frames, radial_limit)
    figure, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()

    lines = []
    for axis, (name, title) in zip(axes, BSSN_VARIABLES):
        line, = axis.plot(frames[0]["areal_radius"], frames[0][name], linewidth=1.2)
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
        for frame in frames:
            for line, (name, _) in zip(lines, BSSN_VARIABLES):
                line.set_data(frame["areal_radius"], frame[name])
            title.set_text(
                f"Z4c Oppenheimer-Snyder collapse, {boundary_name} particles, "
                f"t = {frame['time']:.5f} M"
            )
            writer.grab_frame()

    plt.close(figure)
    print(f"Saved {movie_path}")


def density_color_limits(frames, radial_limit):
    positive_density = []
    for frame in frames:
        inside = frame["areal_radius"] <= radial_limit
        density = frame["mass_density"][inside]
        positive_density.append(density[density > 0.0])

    positive_density = np.concatenate(positive_density)
    if positive_density.size == 0:
        return 1.0e-16, 1.0

    lower = max(float(np.percentile(positive_density, 1.0)), 1.0e-16)
    upper = float(np.percentile(positive_density, 99.9))

    return lower, max(upper, 10.0 * lower)


def radial_density(frame, radial_centers):
    radius = np.concatenate(([0.0], frame["areal_radius"]))
    density = np.concatenate(([frame["mass_density"][0]], frame["mass_density"]))
    increasing = np.concatenate(([True], np.diff(radius) > 0.0))
    radius = radius[increasing]
    density = density[increasing]

    return np.interp(radial_centers, radius, density, left=density[0], right=0.0)


def make_density_movie(frames, movie_path, boundary_name, radial_limit, fps):
    theta_edges = np.linspace(0.0, 2.0 * np.pi, 361)
    radial_edges = np.linspace(0.0, radial_limit, 401)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    vmin, vmax = density_color_limits(frames, radial_limit)

    figure = plt.figure(figsize=(7.8, 7.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="polar")
    axis.set_ylim(0.0, radial_limit)
    axis.set_facecolor("black")

    density = radial_density(frames[0], radial_centers)
    density_2d = np.broadcast_to(density, (theta_edges.size - 1, density.size))
    image = axis.pcolormesh(
        theta_edges, radial_edges, density_2d.T, shading="flat", cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    horizon_theta = np.linspace(0.0, 2.0 * np.pi, 721)
    axis.plot(horizon_theta, np.full_like(horizon_theta, 2.0), color="#66c2ff",
              linewidth=1.5, label=r"Schwarzschild event horizon $R=2M$")
    axis.legend(loc="upper right")
    title = axis.set_title("")
    colorbar = figure.colorbar(image, ax=axis, pad=0.10)
    colorbar.set_label(r"Eulerian mass density $\rho M^2$")

    writer = FFMpegWriter(fps=fps, metadata={"artist": "RadiShPICR"})
    with writer.saving(figure, movie_path, dpi=100):
        for frame in frames:
            density = radial_density(frame, radial_centers)
            image.set_array(
                np.broadcast_to(density, (theta_edges.size - 1, density.size)).T.ravel()
            )
            title.set_text(
                f"Z4c Oppenheimer-Snyder density, {boundary_name} particles\n"
                f"step {frame['step']:06d}, t = {frame['time']:.5f} M"
            )
            writer.grab_frame()

    plt.close(figure)
    print(f"Saved {movie_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, nargs="?",
                        default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--boundary-name",
                        choices=("reflecting", "deleting", "areal_inner_open"),
                        help="override the particle boundary recorded by the run")
    parser.add_argument("--radial-limit", type=float, default=15.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_arguments()
    boundary_name = args.boundary_name
    if boundary_name is None:
        boundary_name = read_metadata(args.run_directory).get(
            "particle_boundary", "deleting"
        )

    frames = load_frames(args.run_directory, max(1, args.frame_stride))
    make_bssn_movie(frames, args.run_directory / "z4c_bssn_variables.mp4",
                    boundary_name, args.radial_limit, args.fps)
    make_density_movie(frames, args.run_directory / "z4c_spherical_density.mp4",
                       boundary_name, args.radial_limit, args.fps)


if __name__ == "__main__":
    main()
