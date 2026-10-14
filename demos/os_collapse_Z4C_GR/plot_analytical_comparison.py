
"""
Christopher Woolford 10/14/2026

Use a height function h(r) to remove the non-diagonal terms from the metric tensor,
and assuming the final state is time independent.

h' = beta_r / (alpha**2 - beta_r * beta**r)
g_rr = gamma_rr + beta_r**2 / (alpha**2 - beta_r * beta**r)
g_RR = g_rr / (dR/dr)**2


"""

import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np
import openpmd_api as io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from RadiShPICR.Z4C.derivatives import first_derivative


STAR_DIRECTORY = DEMO_DIRECTORY / "outputs/evolution_data"
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/analytical_comparison"
INNER_RADIUS_LIMIT = 10.0  # R/M for the inner panel.


def read_final_state():
    """Load the last iteration without parsing all earlier snapshots."""
    series = io.Series(
        str(STAR_DIRECTORY / "bssn_fields.h5"), io.Access.read_only,
        '{"defer_iteration_parsing": true}',
    )
    step = max(series.iterations)
    iteration = series.iterations[step]
    iteration.open()
    fields = {
        name: iteration.meshes[name][io.Mesh.SCALAR].load_chunk()
        for name in ("r", "alpha", "beta", "chi", "conformal_grr", "conformal_gt")
    }
    series.flush()
    dr = float(iteration.meshes["r"].grid_spacing[0])
    time = float(iteration.time)
    mass = float(series.get_attribute("total_mass"))
    series.close()
    return fields, dr, step, time, mass


def reconstruct_grr(fields, dr):
    """Return R and the stationary g_RR on the outward, exterior branch."""
    alpha = fields["alpha"]
    beta = fields["beta"]  # Saved beta is contravariant beta^r.
    gamma_rr = fields["conformal_grr"] / fields["chi"]
    beta_r = gamma_rr * beta

    R = fields["r"] * np.sqrt(fields["conformal_gt"] / fields["chi"])
    dR_dr = np.asarray(first_derivative(R, dr, parity=-1))
    denominator = alpha**2 - beta_r * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        g_rr = gamma_rr + beta_r**2 / denominator
        g_RR = g_rr / dR_dr**2

    # Omit the two origin-parity stencil cells on the final trumpet. Keep
    # gaps as NaNs so a plot cannot join across a pole or a turning point.
    exterior = ((np.arange(R.size) >= max(2, int(np.argmin(R))))
                & (dR_dr > 0.0) & (denominator > 0.0)
                & np.isfinite(R) & np.isfinite(g_RR) & (g_RR > 0.0))
    return R, np.where(exterior, g_RR, np.nan), denominator


def main():
    jax.config.update("jax_enable_x64", True)
    fields, dr, step, time, mass = read_final_state()
    R, g_RR, denominator = reconstruct_grr(fields, dr)
    # The stationary time shift leaves g_tt unchanged; reuse the g_RR mask.
    g_tt = np.where(np.isfinite(g_RR), -denominator, np.nan)

    # Schwarzschild has its own exterior mask, independent of the numerical pole.
    g_RR_schwarzschild = np.full_like(R, np.nan)
    g_tt_schwarzschild = np.full_like(R, np.nan)
    schwarzschild_exterior = ((np.arange(R.size) >= np.argmin(R)) & (R > 2.0 * mass))
    g_RR_schwarzschild[schwarzschild_exterior] = (
        1.0 / (1.0 - 2.0 * mass / R[schwarzschild_exterior])
    )
    g_tt_schwarzschild[schwarzschild_exterior] = (
        -(1.0 - 2.0 * mass / R[schwarzschild_exterior])
    )
    radius = R / mass

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for component, numerical, analytical, scale, label, formula in (
        ("grr", g_RR, g_RR_schwarzschild, "log", r"$g_{RR}$",
         r"$g_{RR}=[\gamma_{rr}+\beta_r^2/(\alpha^2-\beta_r\beta^r)]/(dR/dr)^2$"),
        ("gtt", g_tt, g_tt_schwarzschild, "linear", r"$g_{tt}$",
         r"$g_{tt}=-\alpha^2+\gamma_{rr}(\beta^r)^2$"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        figure.suptitle(
            "Final Oppenheimer–Snyder collapse and analytical Schwarzschild\n"
            f"Initial gravitational mass M={mass:g}\n" + formula,
            fontsize=12,
        )
        for axis, (extent, upper) in zip(
            axes, (("Inner", INNER_RADIUS_LIMIT), ("Full", radius[-1]))
        ):
            visible = radius <= upper
            axis.plot(radius[visible], numerical[visible], color="tab:blue", linewidth=1.8,
                      label=f"OS: stationary reconstruction, t/M={time / mass:g}")
            axis.plot(radius[visible], analytical[visible], "--",
                      color="tab:orange", linewidth=1.8, label="Analytical Schwarzschild")
            axis.set_yscale(scale)
            axis.set_xlim(2.0, min(upper, radius[-1]))
            axis.set_xlabel(r"Areal radius $R/M$ (initial stellar mass)")
            axis.set_ylabel(label)
            axis.set_title(f"{extent} range", fontsize=12)
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=10)
        path = OUTPUT_DIRECTORY / f"final_{component}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        print(f"Saved {path}")

    print(f"Source: {STAR_DIRECTORY / 'bssn_fields.h5'}, step={step}, t/M={time / mass:g}")
    print(f"Initial gravitational mass M={mass:.12g}; Schwarzschild horizon R/M=2")
    print("Reconstruction assumes the final snapshot is stationary.")
    crossings = np.flatnonzero(denominator[:-1] * denominator[1:] < 0.0)
    for index in crossings:
        print(f"Numerical denominator changes sign at R/M in "
              f"[{radius[index]:.9f}, {radius[index + 1]:.9f}]")
    for component, numerical, analytical in (
        ("g_RR", g_RR, g_RR_schwarzschild),
        ("g_tt", g_tt, g_tt_schwarzschild),
    ):
        difference = numerical - analytical
        relative_difference = difference / analytical
        print(f"Outer boundary R/M={radius[-1]:.9f}: "
              f"{component}={numerical[-1]:.9f}, Schwarzschild={analytical[-1]:.9f}")
        print(f"Outer boundary {component} mismatch: {difference[-1]:+.8e} "
              f"(relative {relative_difference[-1]:+.8e})")
        for target in (3.0, 5.0, 10.0):
            index = int(np.argmin(abs(radius - target)))
            print(f"R/M={radius[index]:.6f}: {component}={numerical[index]:.9f}, "
                  f"Schwarzschild={analytical[index]:.9f}, "
                  f"relative mismatch={relative_difference[index]:+.8e}")


if __name__ == "__main__":
    main()
