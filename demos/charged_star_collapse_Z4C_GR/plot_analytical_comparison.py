"""Compare the final charged OS radial and time metrics with analytical RN.

Use the stationary reconstruction
    g_rr = gamma_rr + beta_r**2 / (alpha**2 - beta_r * beta**r),
then transform the component to areal radius: g_RR = g_rr / (dR/dr)**2.
The stationary time shift leaves g_tt = -alpha**2 + gamma_rr * beta**2
unchanged, with no rescaling of the time coordinate.
This treats the final snapshot as stationary; it is not an exact time
coordinate transformation of a still-evolving spacetime.

Run with no arguments. Paths and plot limits are set below. Only the final
iteration is read, and M and Q are the initial stellar values, without fits
or boundary rescaling. Units are G = c = 1, q**2 = Q**2/(4 pi epsilon_0).
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
    mass, charge, epsilon_0 = (
        float(series.get_attribute(name))
        for name in ("initial_gravitational_mass", "initial_charge", "epsilon_0")
    )
    series.close()
    return fields, dr, step, time, mass, charge, epsilon_0


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
                & np.isfinite(g_RR) & (g_RR > 0.0))
    return R, np.where(exterior, g_RR, np.nan), denominator


def main():
    jax.config.update("jax_enable_x64", True)
    fields, dr, step, time, mass, charge, epsilon_0 = read_final_state()
    R, g_RR, denominator = reconstruct_grr(fields, dr)
    # The stationary time shift leaves g_tt unchanged; reuse the g_RR mask.
    g_tt = np.where(np.isfinite(g_RR), -denominator, np.nan)
    q_squared = charge**2 / (4.0 * np.pi * epsilon_0)
    horizon = mass + np.sqrt(mass**2 - q_squared)
    f_RN = 1.0 - 2.0 * mass / R + q_squared / R**2

    # RN has its own exterior mask, independent of the numerical pole.
    g_RR_RN = np.full_like(R, np.nan)
    g_tt_RN = np.full_like(R, np.nan)
    rn_exterior = ((np.arange(R.size) >= np.argmin(R)) & (R > horizon))
    g_RR_RN[rn_exterior] = 1.0 / f_RN[rn_exterior]
    g_tt_RN[rn_exterior] = -f_RN[rn_exterior]
    radius = R / mass

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for component, numerical, analytical, scale, label, formula in (
        ("grr", g_RR, g_RR_RN, "log", r"$g_{RR}$",
         r"$g_{RR}=[\gamma_{rr}+\beta_r^2/(\alpha^2-\beta_r\beta^r)]/(dR/dr)^2$"),
        ("gtt", g_tt, g_tt_RN, "linear", r"$g_{tt}$",
         r"$g_{tt}=-\alpha^2+\gamma_{rr}(\beta^r)^2$"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        figure.suptitle(
            "Final charged OS collapse and analytical Reissner–Nordström\n"
            f"M={mass:g}, q/M={np.sqrt(q_squared) / mass:.9f}; "
            "no mass fit or rescaling\n"
            + formula,
            fontsize=12,
        )
        for axis, (extent, upper) in zip(
            axes, (("Inner", INNER_RADIUS_LIMIT), ("Full", radius[-1]))
        ):
            visible = radius <= upper
            axis.plot(radius[visible], numerical[visible], color="tab:blue", linewidth=1.8,
                      label=f"Charged OS: stationary reconstruction, t/M={time / mass:g}")
            axis.plot(radius[visible], analytical[visible], "--",
                      color="tab:orange", linewidth=1.8, label="Analytical RN")
            axis.set_yscale(scale)
            axis.set_xlim(horizon / mass, min(upper, radius[-1]))
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
    print(f"M={mass:.12g}, Q={charge:.12g}, q/M={np.sqrt(q_squared) / mass:.12g}")
    print(f"RN outer horizon R+/M={horizon / mass:.12g}")
    print("Reconstruction assumes the final snapshot is stationary.")
    crossings = np.flatnonzero(denominator[:-1] * denominator[1:] < 0.0)
    for index in crossings:
        print(f"Numerical denominator changes sign at R/M in "
              f"[{radius[index]:.9f}, {radius[index + 1]:.9f}]")
    for component, numerical, analytical in (
        ("g_RR", g_RR, g_RR_RN),
        ("g_tt", g_tt, g_tt_RN),
    ):
        difference = numerical - analytical
        relative_difference = difference / analytical
        print(f"Outer boundary R/M={radius[-1]:.9f}: "
              f"{component}={numerical[-1]:.9f}, RN={analytical[-1]:.9f}")
        print(f"Outer boundary {component} mismatch: {difference[-1]:+.8e} "
              f"(relative {relative_difference[-1]:+.8e})")
        for target in (3.0, 5.0, 10.0):
            index = int(np.argmin(abs(radius - target)))
            print(f"R/M={radius[index]:.6f}: {component}={numerical[index]:.9f}, "
                  f"RN={analytical[index]:.9f}, "
                  f"relative mismatch={relative_difference[index]:+.8e}")


if __name__ == "__main__":
    main()
