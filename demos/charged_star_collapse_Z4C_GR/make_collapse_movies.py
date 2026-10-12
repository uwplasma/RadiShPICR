"""Render movies from a Z4c charged Oppenheimer-Snyder collapse run.

Reads the run's openPMD series.  Snapshots hold raw state, so the Eulerian
mass density is re-deposited from the saved particles and the deposited charge
is re-integrated against the proper shell volume.
"""

from __future__ import annotations

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
from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume
from RadiShPICR.Z4C.energy_momentum_tensor import compute_radial_matter_terms

OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/z4c_charged_star_evolution_300M"
BOUNDARY_NAME = None  # None uses the particle boundary recorded by the run.
RADIAL_LIMIT = 15.0
FRAME_STRIDE = 1
FPS = 50
MOVIES = ("bssn", "density", "electric-field")

BSSN_VARIABLES = (
    ("alpha", r"lapse $\alpha$"),
    ("beta", r"shift $\beta^r$"),
    ("chi", r"conformal factor $\chi$"),
    ("conformal_grr", r"$\tilde{\gamma}_{rr}$"),
    ("conformal_gt", r"$\tilde{\gamma}_{T}$"),
    ("Kh", r"$\widehat K$"),
    ("Arr", r"$\tilde{A}_{rr}$"),
    ("At", r"$\tilde{A}_{T}$"),
    ("theta", r"$\Theta$"),
    ("Gamma", r"$\tilde{\Gamma}^{r}$"),
)

WRITER_ARGS = dict(codec="libx264", extra_args=["-pix_fmt", "yuv420p"],
                   metadata={"artist": "RadiShPICR"})


def load_frames(run_directory, frame_stride):
    """Read the series once, rebuilding the particle-sourced quantities."""

    frames = []
    for snapshot in iter_snapshots(run_directory, stride=frame_stride):
        frame = {name: np.asarray(getattr(snapshot.metric, name))
                 for name, _ in BSSN_VARIABLES}
        frame["areal_radius"] = np.asarray(snapshot.areal_radius)
        frame["E_r"] = np.asarray(snapshot.E_r)
        frame["step"] = snapshot.step
        frame["time"] = snapshot.time

        particles = snapshot.species()
        matter = compute_radial_matter_terms(particles, snapshot.metric,
                                             inner_open=True)
        frame["mass_density"] = np.asarray(matter.rho)
        shell_volume = np.asarray(proper_radial_shell_volume(snapshot.metric))
        frame["deposited_charge"] = float(
            np.sum(np.asarray(snapshot.charge_density) * shell_volume)
        )
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No snapshots found in {run_directory}")

    return frames


def time_label(frame, mass):
    return f"Z4C t/M = {frame['time'] / mass:.5f}"


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


def make_bssn_movie(frames, movie_path, boundary_name, radial_limit, fps, mass):
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
        axis.set_xlabel(r"areal radius $R$ (code units)")
    axes[10].axis("off")
    axes[11].axis("off")
    title = figure.suptitle("")

    writer = FFMpegWriter(fps=fps, **WRITER_ARGS)
    with writer.saving(figure, movie_path, dpi=110):
        for frame in frames:
            for line, (name, _) in zip(lines, BSSN_VARIABLES):
                line.set_data(frame["areal_radius"], frame[name])
            title.set_text(
                f"Z4c charged-star collapse, {boundary_name} particles\n"
                f"step {frame['step']:06d}, " + time_label(frame, mass)
            )
            writer.grab_frame()

    plt.close(figure)
    print(f"Saved {movie_path}")


def make_electric_field_movie(frames, movie_path, boundary_name, radial_limit,
                              fps, mass):
    """Plot the non-densitized covariant FIDO field E_r on a fixed log scale."""

    lower, upper = np.inf, 0.0
    for frame in frames:
        E_r = frame["E_r"][frame["areal_radius"] <= radial_limit]
        lower = np.minimum(lower, np.min(E_r[E_r > 0.0], initial=np.inf))
        upper = np.maximum(upper, np.max(E_r))

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    line, = axis.plot([], [], linewidth=1.5)
    axis.set_xlim(0.0, radial_limit)
    axis.set_yscale("log", nonpositive="mask")
    axis.set_ylim(lower / 1.2, upper * 1.2)
    axis.set_title("Covariant FIDO field")
    axis.set_xlabel(r"areal radius $R$ (code units)")
    axis.set_ylabel(r"$E_r$ (code units)")
    axis.grid(alpha=0.25)
    title = figure.suptitle("")

    writer = FFMpegWriter(fps=fps, **WRITER_ARGS)
    with writer.saving(figure, movie_path, dpi=110):
        for frame in frames:
            # Saved E_r is already non-densitized, in the isotropic basis.
            line.set_data(frame["areal_radius"], frame["E_r"])
            title.set_text(f"step {frame['step']:06d}, " + time_label(frame, mass))
            writer.grab_frame()

    plt.close(figure)
    print(f"Saved {movie_path}")


def density_color_limits(frames, radial_limit):
    positive_density = []
    for frame in frames:
        density = frame["mass_density"][frame["areal_radius"] <= radial_limit]
        positive_density.append(density[density > 0.0])

    positive_density = np.concatenate(positive_density)
    if positive_density.size == 0:
        return 1.0e-16, 1.0

    lower = max(float(np.percentile(positive_density, 1.0)), 1.0e-16)
    upper = float(np.percentile(positive_density, 99.9))

    return lower, max(upper, 10.0 * lower)


def radial_density(frame, radial_centers):
    # Start at the areal-radius minimum: the inner branch folds back.
    start = int(np.argmin(frame["areal_radius"]))
    radius = frame["areal_radius"][start:]
    density = frame["mass_density"][start:]
    increasing = np.concatenate(([True], np.diff(radius) > 0.0))

    return np.interp(radial_centers, radius[increasing], density[increasing],
                     left=0.0, right=0.0)


def make_density_movie(frames, movie_path, boundary_name, radial_limit, fps, mass):
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
    horizon_line, = axis.plot(
        horizon_theta, np.full_like(horizon_theta, mass), color="#66c2ff",
        linewidth=1.5, label=r"RN reference $R_+$ (not a measured horizon)",
    )
    axis.legend(loc="upper right", fontsize=8)
    title = axis.set_title("")
    colorbar = figure.colorbar(image, ax=axis, pad=0.10)
    colorbar.set_label(r"Eulerian particle mass density $\rho$ (code units)")

    writer = FFMpegWriter(fps=fps, **WRITER_ARGS)
    with writer.saving(figure, movie_path, dpi=100):
        for frame in frames:
            density = radial_density(frame, radial_centers)
            charge = frame["deposited_charge"]
            rn_radius = mass + np.sqrt(max(mass**2 - charge**2 / (4.0 * np.pi), 0.0))
            horizon_line.set_ydata(np.full_like(horizon_theta, rn_radius))
            image.set_array(
                np.broadcast_to(density, (theta_edges.size - 1, density.size)).T.ravel()
            )
            title.set_text(
                f"Z4c charged-star density, {boundary_name} particles\n"
                f"step {frame['step']:06d}, " + time_label(frame, mass)
            )
            writer.grab_frame()

    plt.close(figure)
    print(f"Saved {movie_path}")


def main():
    metadata = read_metadata(OUTPUT_DIRECTORY)
    boundary_name = BOUNDARY_NAME or metadata.get("particle_boundary", "deleting")
    if metadata.get("completed") is False:
        boundary_name += " [INCOMPLETE]"
    mass = float(metadata["reference_mass"])

    frames = load_frames(OUTPUT_DIRECTORY, max(1, FRAME_STRIDE))
    if "bssn" in MOVIES:
        make_bssn_movie(frames, OUTPUT_DIRECTORY / "z4c_bssn_variables.mp4",
                        boundary_name, RADIAL_LIMIT, FPS, mass)
    if "density" in MOVIES:
        make_density_movie(frames, OUTPUT_DIRECTORY / "z4c_spherical_density.mp4",
                           boundary_name, RADIAL_LIMIT, FPS, mass)
    if "electric-field" in MOVIES:
        make_electric_field_movie(frames, OUTPUT_DIRECTORY / "z4c_electric_field.mp4",
                                  boundary_name, RADIAL_LIMIT, FPS, mass)


if __name__ == "__main__":
    main()
