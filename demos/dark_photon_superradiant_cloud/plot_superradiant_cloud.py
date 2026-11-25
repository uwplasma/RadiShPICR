"""Plot the Schwarzschild Proca cloud from the saved openPMD series and probe.csv."""

from functools import partial
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.dark_photon_superradiant_cloud import simulation_parameters as p
from RadiShPICR.diagnostics.read_bssn import iter_snapshots, read_metadata
from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume
from RadiShPICR.Z4C.dark_photon_field import dark_gauss_residual
from RadiShPICR.Z4C.energy_momentum_tensor import (
    compute_hamiltonian_constraint, compute_radial_momentum_constraint, dark_energy_momentum_tensor,
)


@partial(jax.jit, static_argnames=("backreaction",))
def constraint_norms(metric, dark_fields, mu, backreaction):
    gauss = dark_gauss_residual(metric, dark_fields, jnp.zeros_like(metric.r), mu)
    H = compute_hamiltonian_constraint(metric)
    M_r = compute_radial_momentum_constraint(metric)
    gamma_rr = metric.conformal_grr / metric.chi
    if backreaction:
        matter = dark_energy_momentum_tensor(metric, dark_fields, mu)
        H -= 16.0 * jnp.pi * matter.rho
        M_r -= 8.0 * jnp.pi * gamma_rr * matter.Sr

    # The proper magnitude of the momentum covector is |M_r| / sqrt(γ_rr).
    residuals = jnp.stack((gauss, H, M_r / jnp.sqrt(gamma_rr)))
    volume = proper_radial_shell_volume(metric)
    l2 = jnp.sqrt(jnp.sum(residuals**2 * volume, axis=1))
    linf = jnp.max(jnp.abs(residuals), axis=1)
    return jnp.stack((l2, linf), axis=1)


def plot_snapshots(output, metadata):
    mass, mu = metadata["mass"], metadata["mu"]
    times, profiles, norms = [], [], []
    for snapshot in iter_snapshots(output, with_particles=False):
        if snapshot.dark_fields is None:
            raise ValueError(f"Snapshot {snapshot.step} contains no dark fields.")
        if not times:
            r = np.asarray(snapshot.metric.r) / mass
            dr = float(snapshot.metric.dr) / mass
        times.append(snapshot.time / mass)
        profiles.append(np.asarray(snapshot.dark_fields[1]))
        norms.append(np.asarray(constraint_norms(snapshot.metric, snapshot.dark_fields,
                                                 mu, metadata["backreaction"])))

    times = np.asarray(times)
    if len(times) < 2:
        raise ValueError("The spacetime plot needs at least two radial snapshots.")
    profiles, norms = np.stack(profiles), np.stack(norms)
    r_edges = np.append(r - 0.5 * dr, r[-1] + 0.5 * dr)
    # The last snapshot may follow a shorter interval.
    time_edges = np.r_[times[0], 0.5 * (times[1:] + times[:-1]), times[-1]]
    limit = np.max(np.abs(profiles))
    # A large covariant component near the puncture can hide the exterior
    # oscillations on a linear color scale. Keep small signed fields linear.
    initial_peak = np.max(np.abs(profiles[0]))
    color_norm = SymLogNorm(0.1 * initial_peak, vmin=-limit, vmax=limit) if initial_peak else None
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    field = ax.pcolorfast(r_edges, time_edges, profiles, cmap="RdBu_r", norm=color_norm)
    fig.colorbar(field, ax=ax, label=r"$A'_r$")
    ax.set(xlabel=r"$r/M$", ylabel=r"$t/M$", title="Schwarzschild Proca cloud")
    fig.savefig(output / "A_prime_spacetime.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
    for i, (ax, label) in enumerate(zip(axes, ("Dark Gauss", "Hamiltonian", "Radial momentum"))):
        for k, (style, name) in enumerate((("-", r"$L_2$"), ("--", r"$L_\infty$"))):
            ax.plot(times, norms[:, i, k], style, label=name)
        if np.any(norms[:, i] > 0):
            ax.set_yscale("log", nonpositive="mask")
        ax.set_ylabel(label)
        ax.legend()
    sources = "including Proca sources" if metadata["backreaction"] else "vacuum trumpet"
    axes[0].set_title(f"Full-grid constraint norms (Einstein constraints: {sources})")
    axes[-1].set_xlabel(r"$t/M$")
    fig.savefig(output / "constraint_norms.png", dpi=160)
    plt.close(fig)


def plot_fft(output, metadata):
    mass, mu = metadata["mass"], metadata["mu"]
    times, probe = np.loadtxt(output / "probe.csv", delimiter=",", skiprows=1, ndmin=2).T
    start = times[0] + p.FFT_START_FRACTION * (times[-1] - times[0])
    end = times[0] + p.FFT_END_FRACTION * (times[-1] - times[0])
    selected = (times >= start) & (times <= end)
    t, A_r = times[selected], probe[selected]
    if len(t) < 2:
        raise ValueError("The FFT window needs at least two probe samples.")

    omega = 2.0 * np.pi * np.fft.rfftfreq(len(t), d=t[1] - t[0])
    spectrum = np.abs(np.fft.rfft(A_r))
    peak = omega[1 + np.argmax(spectrum[1:])]
    omega_hyd = mu * (1.0 - (mass * mu)**2 / 8.0)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(mass * omega, spectrum, label=fr"FFT of $A'_r$ ($M\Delta\omega={mass * omega[1]:.4g}$)")
    ax.axvline(mass * peak, color="tab:red", ls="-.", label=fr"peak $M\omega={mass * peak:.5g}$")
    ax.axvline(mass * omega_hyd, color="tab:green", ls="--", label=fr"$\mu(1 - \alpha^2/8)$")
    ax.set(xlabel=r"$M\omega$", ylabel=r"$|\mathrm{FFT}(A'_r)|$",
           xlim=(0, max(2 * mass * mu, 1.1 * mass * peak)),
           title=fr"Probe at $r/M={metadata['probe_r'] / mass:.3f}$, "
                 fr"$t/M={t[0] / mass:g}$ to ${t[-1] / mass:g}$")
    ax.legend()
    fig.savefig(output / "A_prime_fft.png", dpi=160)
    plt.close(fig)
    print(f"Peak Mω = {mass * peak:.6g}, bin spacing MΔω = {mass * omega[1]:.6g}, "
          f"predicted binding shift M(μ − ω_hyd) = {mass * (mu - omega_hyd):.6g}.")


def main():
    jax.config.update("jax_enable_x64", True)
    output = p.OUTPUT_DIRECTORY
    metadata = read_metadata(output)
    plot_snapshots(output, metadata)
    plot_fft(output, metadata)
    print(f"Saved the three cloud figures to {output}.")


if __name__ == "__main__":
    main()
