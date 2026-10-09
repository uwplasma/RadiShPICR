"""Compare the final charged-star lapse and chi with static RN, G=c=epsilon_0=1."""

import json
import sys
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DEMO_DIRECTORY = Path(__file__).resolve().parent
INPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/z4c_charged_star_schwarzschild"
OUTPUT_DIRECTORY = INPUT_DIRECTORY / "rn_comparison"

sys.path.insert(0, str(DEMO_DIRECTORY.parent.parent))
from RadiShPICR.ConstraintBasedRelativity.vacuum_conditions import (
    reissner_nordstrom_A,
    reissner_nordstrom_lapse,
    vacuum_rescale_factors,
)

jax.config.update("jax_enable_x64", True)


def main():
    summary = json.loads((INPUT_DIRECTORY / "run_summary.json").read_text())
    snapshot_path = INPUT_DIRECTORY / "metric" / f"metric_step_{summary['final_step']:06d}.npz"

    with np.load(INPUT_DIRECTORY / "initial_data_metadata.npz") as initial:
        M = float(initial["initial_gravitational_mass"])
        Q = float(initial["initial_charge"])
    with np.load(snapshot_path) as snapshot:
        r = snapshot["r"]
        alpha = snapshot["alpha"]
        chi = snapshot["chi"]
        grr = snapshot["conformal_grr"] / chi
        gT = snapshot["conformal_gT"] / chi

    X_r, X_t = map(float, vacuum_rescale_factors(
        np.sqrt(grr[-1]), alpha[-1], r[-1], M, Q,
    ))
    mismatch = (grr[-1] - gT[-1]) / (0.5 * (grr[-1] + gT[-1]))

    # Apply an additional RN normalization to the saved Schwarzschild chart.
    r = X_r * r
    alpha = alpha / X_t
    chi = chi * X_r**2

    # Restrict the analytic curve to the outer static RN branch.
    r_horizon = 0.5 * np.sqrt(M**2 - Q**2 / (4.0 * np.pi))
    r_RN = r[r >= r_horizon]
    alpha_RN = np.asarray(reissner_nordstrom_lapse(r_RN, M, Q))
    chi_RN = np.asarray(reissner_nordstrom_A(r_RN, M, Q))**-2

    print(f"X_r = {X_r:.12g}, X_t = {X_t:.12g}")
    print(f"Boundary metric mismatch (grr-gT)/mean(grr,gT) = {mismatch:.8e}")
    print(f"Boundary lapse - RN = {alpha[-1] - alpha_RN[-1]:.8e}")
    print(f"Boundary chi - RN = {chi[-1] - chi_RN[-1]:.8e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(r / M, alpha, label="Final charged star", color="tab:blue")
    axes[0].plot(r_RN / M, alpha_RN, "--", label="Static RN exterior", color="tab:orange")
    axes[0].set_title("Lapse")
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].legend()

    axes[1].plot(r / M, chi, color="tab:blue")
    axes[1].plot(r_RN / M, chi_RN, "--", color="tab:orange")
    axes[1].set_title("Conformal factor")
    axes[1].set_ylabel(r"$\chi$")

    for ax in axes:
        ax.set_xlabel(r"RN-matched isotropic radius $r/M$")
        ax.set_xlim(0.0, r[-1] / M)
        ax.grid(alpha=0.3)
    fig.suptitle("Final charged star and static RN")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "final_bssn_RN_comparison.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
