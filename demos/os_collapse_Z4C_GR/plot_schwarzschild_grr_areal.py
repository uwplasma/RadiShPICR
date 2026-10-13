"""Compare the final OS radial metric with analytical Schwarzschild.

Treat the final snapshot as stationary and write t = T + h(r), where
    h' = beta_r / (alpha**2 - beta_r * beta**r).
This removes the time-radial cross term and gives
    g_rr = gamma_rr + beta_r**2 / (alpha**2 - beta_r * beta**r).
Then transform the component to areal radius: g_RR = g_rr / (dR/dr)**2.
Integrating h is unnecessary for this component. The stationary assumption
is not an exact coordinate transformation of a still-evolving spacetime.

Run with no arguments. Paths and plot limits are set below. Only the final
iteration is read, and M is the saved initial stellar gravitational mass,
without fitting or boundary rescaling. Units are G = c = 1.
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

STAR_DIRECTORY = DEMO_DIRECTORY / "outputs/z4c_oppenheimer_snyder"
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/analytical_schwarzschild_grr_areal"
INNER_RADIUS_LIMIT = 10.0  # R/M for the inner figure.


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

    # Schwarzschild has its own exterior mask, independent of the numerical pole.
    g_RR_schwarzschild = np.full_like(R, np.nan)
    schwarzschild_exterior = ((np.arange(R.size) >= np.argmin(R)) & (R > 2.0 * mass))
    g_RR_schwarzschild[schwarzschild_exterior] = (
        1.0 / (1.0 - 2.0 * mass / R[schwarzschild_exterior])
    )
    radius = R / mass

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for extent, upper in (("full", radius[-1]), ("inner", INNER_RADIUS_LIMIT)):
        visible = radius <= upper
        figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        axis.plot(radius[visible], g_RR[visible], color="tab:blue", linewidth=1.8,
                  label=f"OS: stationary reconstruction, t/M={time / mass:g}")
        axis.plot(radius[visible], g_RR_schwarzschild[visible], "--",
                  color="tab:orange", linewidth=1.8, label="Analytical Schwarzschild")
        axis.set_yscale("log")
        axis.set_xlim(2.0, min(upper, radius[-1]))
        axis.set_xlabel(r"Areal radius $R/M$ (initial stellar mass)")
        axis.set_ylabel(r"$g_{RR}$")
        axis.set_title(
            "Final Oppenheimer–Snyder collapse and analytical Schwarzschild\n"
            f"Initial gravitational mass M={mass:g}\n"
            r"$g_{RR}=[\gamma_{rr}+\beta_r^2/(\alpha^2-\beta_r\beta^r)]/(dR/dr)^2$",
            fontsize=12,
        )
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=10)
        path = OUTPUT_DIRECTORY / f"final_schwarzschild_grr_areal_{extent}.png"
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
    difference = g_RR - g_RR_schwarzschild
    relative_difference = difference / g_RR_schwarzschild
    print(f"Outer boundary R/M={radius[-1]:.9f}: "
          f"g_RR={g_RR[-1]:.9f}, Schwarzschild={g_RR_schwarzschild[-1]:.9f}")
    print(f"Outer boundary mismatch: {difference[-1]:+.8e} "
          f"(relative {relative_difference[-1]:+.8e})")
    for target in (3.0, 5.0, 10.0):
        index = int(np.argmin(abs(radius - target)))
        print(f"R/M={radius[index]:.6f}: g_RR={g_RR[index]:.9f}, "
              f"Schwarzschild={g_RR_schwarzschild[index]:.9f}, "
              f"relative mismatch={relative_difference[index]:+.8e}")


if __name__ == "__main__":
    main()
