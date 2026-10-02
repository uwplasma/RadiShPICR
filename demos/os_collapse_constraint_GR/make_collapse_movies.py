"""Render saved constrained-GR collapse snapshots without running the solver.

Example:
    python make_collapse_movies.py --frame-stride 994 --output-dir /tmp/os_movies

Fields retain their saved isotropic component basis and are plotted against
code coordinate radius r. Number density uses areal radius R = A*r and is
weighted Eulerian number per proper
volume. Its fixed viewing radius is 1.2 times the initial outer particle
radius. The constraint is the Hamiltonian equation used by this solver.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs/heun_oppenheimer_snyder"
FIELDS = (("A", "$A$"), ("phi", r"$\phi$"), ("alpha", r"$\alpha$"),
          ("Krr", r"$K_{rr}$ (saved variable)"), ("beta", r"$\beta^r$"))


def snapshot_pairs(directory):
    metrics = sorted((directory / "U_state").glob("U_state_step_*.npz"))
    if not metrics:
        raise ValueError(f"No U_state snapshots in {directory}")
    pairs = []
    for path in metrics:
        step = path.stem.rsplit("_", 1)[1]
        particles = sorted((directory / "phase_space").glob(f"*_step_{step}.npz"))
        if not particles:
            raise ValueError(f"Missing particle snapshot for {path.name}")
        pairs.append((path, particles))
    return pairs


def radial_geometry(snapshot):
    r, A = snapshot["r"], snapshot["A"]
    R = r * A
    if (not np.all(np.isfinite(R)) or np.any(A <= 0)
            or np.any(np.diff(r) <= 0) or np.any(np.diff(R) <= 0)):
        raise ValueError(f"Invalid or nonmonotonic radius mapping at step {snapshot['step']}")
    return r, A, R


def proper_shell_volumes(r, A, edges):
    """Integrate 4*pi*A^3*r^2 dr between isotropic shell edges.

    Include every metric node in the quadrature. Trapezoidal integration in
    r^3 is exact for a constant metric, including the cell at the origin.
    """
    nodes = np.unique(np.concatenate((edges, r[(r > edges[0]) & (r < edges[-1])])))
    A3 = np.interp(nodes, r, A) ** 3
    segments = (4 * np.pi / 3) * np.diff(nodes**3) * (A3[1:] + A3[:-1]) / 2
    volume = np.concatenate(([0.0], np.cumsum(segments)))
    return np.diff(volume[np.searchsorted(nodes, edges)])


def number_density(snapshot, particle_paths, radial_edges):
    r, A, R = radial_geometry(snapshot)
    # Clip boundary bins to the saved domain; do not invent metric outside it.
    supported_edges = np.clip(radial_edges, R[0], R[-1])
    isotropic_edges = np.interp(supported_edges, R, r)
    volumes = proper_shell_volumes(r, A, isotropic_edges)
    counts = np.zeros(len(radial_edges) - 1)
    for path in particle_paths:
        with np.load(path) as particles:
            if (int(particles['step']) != int(snapshot['step'])
                    or not np.isclose(particles['time'], snapshot['time'])):
                raise ValueError(f"Particle/metric step or time mismatch: {path}")
            radius, weight = particles['areal_radius'], particles['weight']
            inside = (radius >= supported_edges[0]) & (radius <= supported_edges[-1])
            counts += np.histogram(radius[inside], bins=radial_edges,
                                   weights=weight[inside])[0]
    density = np.divide(counts, volumes, out=np.zeros_like(counts), where=volumes > 0)
    return density, volumes


def hamiltonian_constraint(snapshot):
    r, A, R = radial_geometry(snapshot)
    phi = snapshot['phi']
    rho = snapshot['mass_density'] + 0.5 * snapshot['Er']**2 / A**2
    # Differentiate on the saved isotropic grid, not the areal plotting grid.
    derivative = np.gradient(phi, r, edge_order=2)[1:-1]
    H = -8 * A[1:-1]**(-2.5) * (derivative + 2 * phi[1:-1] / r[1:-1])
    H -= 16 * np.pi * rho[1:-1]
    edges = 0.5 * (r[:-1] + r[1:])
    volumes = proper_shell_volumes(r, A, edges)
    rms = np.sqrt(np.sum(H**2 * volumes) / np.sum(volumes))
    return H, rms, np.max(np.abs(H))


def metric_values(snapshot):
    return [snapshot[name] if name != 'beta' else snapshot['r'] * snapshot['beta_over_r']
            for name, _ in FIELDS]


def make_movies(pairs, output_dir, radial_edges, metric_radial_limit, limits, density_limits, fps):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    lines = []
    for axis, (_, label), bounds in zip(axes.flat, FIELDS, limits):
        line, = axis.plot([], [], lw=1.2)
        lines.append(line)
        axis.set(xlim=(0, metric_radial_limit), ylim=bounds, xlabel='Code coordinate radius r', ylabel=label)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis('off')
    title = fig.suptitle('')

    density_fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    axis = density_fig.add_subplot(111, projection='polar')
    theta = np.linspace(0, 2 * np.pi, 361)
    axis.grid(False)
    axis.set_facecolor('black')
    axis.set_ylim(0, radial_edges[-1])
    axis.tick_params(axis='y', colors='white')
    image = axis.pcolormesh(theta, radial_edges,
                           np.ones((len(radial_edges)-1, len(theta)-1)),
                           norm=LogNorm(*density_limits), cmap='magma', shading='flat')
    density_fig.colorbar(image, ax=axis, pad=0.1, label='Weighted number / proper volume')
    density_title = axis.set_title('')
    settings = dict(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    metric_writer, density_writer = FFMpegWriter(**settings), FFMpegWriter(**settings)
    with metric_writer.saving(fig, str(output_dir / 'metric_variables.mp4'), dpi=100), \
            density_writer.saving(density_fig, str(output_dir / 'number_density.mp4'), dpi=100):
        for path, particles in pairs:
            with np.load(path) as snapshot:
                r = snapshot["r"]
                for line, values in zip(lines, metric_values(snapshot)):
                    line.set_data(r, values)
                density, _ = number_density(snapshot, particles, radial_edges)
                image.set_array(np.ma.masked_less_equal(
                    np.broadcast_to(density[:, None], (len(density), len(theta)-1)), 0).ravel())
                time = float(snapshot['time'])
                title.set_text(f'Constrained-GR variables (saved basis), Schwarzschild time = {time:.5f}')
                density_title.set_text(f'Number density vs areal radius\nSchwarzschild time = {time:.5f}')
            metric_writer.grab_frame()
            density_writer.grab_frame()
    plt.close(fig)
    plt.close(density_fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_directory', nargs='?', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--radial-limit', type=float,
                        help='Maximum code coordinate radius r for metric profiles; density uses the initial particle extent')
    parser.add_argument('--radial-bins', type=int, default=400)
    args = parser.parse_args()
    if min(args.frame_stride, args.fps, args.radial_bins) < 1:
        parser.error('frame-stride, fps, and radial-bins must be positive')
    if args.radial_limit is not None and args.radial_limit <= 0:
        parser.error('radial-limit must be positive')
    output = args.output_dir or args.input_directory / 'movies'
    output.mkdir(parents=True, exist_ok=True)
    pairs = snapshot_pairs(args.input_directory)
    selected = pairs[::args.frame_stride]
    if selected[-1] != pairs[-1]:
        selected.append(pairs[-1])

    history, radial_max = [], 0.0
    for path, _ in pairs:
        with np.load(path) as snapshot:
            r, _, _ = radial_geometry(snapshot)
            radial_max = max(radial_max, r[-1])
            _, rms, maximum = hamiltonian_constraint(snapshot)
            history.append((int(snapshot['step']), float(snapshot['time']), rms, maximum))
    history = np.asarray(history)
    np.savetxt(output / 'hamiltonian_constraint.csv', history, delimiter=',',
               header='step,schwarzschild_time,proper_volume_rms,max_abs', comments='')
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(history[:, 1], history[:, 2], label='Proper-volume RMS')
    axis.plot(history[:, 1], history[:, 3], label='Maximum absolute residual')
    axis.set(xlabel='Schwarzschild time', ylabel='Hamiltonian constraint residual')
    axis.set_yscale('symlog', linthresh=1e-12)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(output / 'hamiltonian_constraint.png', dpi=160)
    plt.close(fig)

    metric_radial_limit = args.radial_limit or radial_max
    initial_outer_radius = 0.0
    for particle_path in pairs[0][1]:
        with np.load(particle_path) as particles:
            active = particles['weight'] > 0
            if np.any(active):
                initial_outer_radius = max(
                    initial_outer_radius, float(np.max(particles['areal_radius'][active])))
    if initial_outer_radius <= 0:
        raise ValueError('Initial snapshot has no positive-weight particles at positive radius')
    radial_edges = np.linspace(0, 1.2 * initial_outer_radius, args.radial_bins + 1)
    limits = np.array([[np.inf, -np.inf]] * len(FIELDS))
    density_min, density_max = np.inf, 0.0
    for path, particles in selected:
        with np.load(path) as snapshot:
            r = snapshot["r"]
            inside = r <= metric_radial_limit
            for bounds, values in zip(limits, metric_values(snapshot)):
                values = values[inside & np.isfinite(values)]
                bounds[0] = min(bounds[0], np.min(values))
                bounds[1] = max(bounds[1], np.max(values))
            density, _ = number_density(snapshot, particles, radial_edges)
            positive = density[density > 0]
            if positive.size:
                density_min = min(density_min, positive.min())
                density_max = max(density_max, positive.max())
    padding = 0.05 * np.maximum(limits[:, 1] - limits[:, 0], 1e-12)
    limits += np.column_stack((-padding, padding))
    density_limits = (density_min, max(density_max, density_min * 1.01)) if density_max else (1e-16, 1.0)
    print(f'Rendering {len(selected)} frames; constraint history has {len(history)} snapshots.', flush=True)
    make_movies(selected, output, radial_edges, metric_radial_limit, limits, density_limits, args.fps)
    print(f'Saved movies and constraint diagnostics in {output}')


if __name__ == '__main__':
    main()
