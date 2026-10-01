"""Plot the saved final single-puncture state without rerunning the demo.

Run with no arguments to read run_data.pkl beside this script. Use --r-max
to restrict both figures to the inner part of the coordinate-radius grid.
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEMO_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_ROOT = DEMO_DIRECTORY.parent.parent
# The pickle contains a RadiShPICR.Z4C.z4c_metric.Z4C_Metric instance.
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

FIELDS = (
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


def plot_final_bssn(input_path, output_directory, r_max=None):
    """Save radial field profiles and the final lapse-versus-K curve."""
    with open(input_path, "rb") as handle:
        run_data = pickle.load(handle)
    metric = run_data["final_metric"]
    r = np.asarray(metric.r)
    fields = {name: np.asarray(getattr(metric, name)) for name, _ in FIELDS}

    nonfinite = [
        name for name, values in {"r": r, **fields}.items()
        if not np.all(np.isfinite(values))
    ]
    if nonfinite:
        raise ValueError("Nonfinite final fields: " + ", ".join(nonfinite))

    selected = np.ones(r.shape, dtype=bool) if r_max is None else r <= r_max
    if not np.any(selected):
        raise ValueError("The requested radial range contains no grid points.")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True,
                             constrained_layout=True)
    for ax, (name, label) in zip(axes.flat, FIELDS):
        ax.plot(r[selected], fields[name][selected])
        ax.set_title(name)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel(r"Coordinate radius $r$")
    fig.suptitle("Final single-puncture BSSN/Z4C variables")
    radial_path = output_directory / "final_bssn_variables.png"
    fig.savefig(radial_path, dpi=160)
    plt.close(fig)

    # Z4C evolves Kh = K - 2 Theta; reconstruct the physical trace.
    K = fields["Kh"] + 2.0 * fields["theta"]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # Preserve radial order along the curve, even when K is nonmonotonic.
    ax.plot(K[selected], fields["alpha"][selected])
    ax.set_xlabel(r"$K = \widehat{K} + 2\Theta$")
    ax.set_ylabel(r"Lapse $\alpha$")
    ax.set_title("Final single-puncture lapse versus extrinsic curvature")
    ax.grid(alpha=0.3)
    lapse_path = output_directory / "lapse_vs_K.png"
    fig.savefig(lapse_path, dpi=160)
    plt.close(fig)

    print(f"Saved {radial_path}")
    print(f"Saved {lapse_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=DEMO_DIRECTORY / "run_data.pkl")
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIRECTORY)
    parser.add_argument("--r-max", type=float, default=None,
                        help="Maximum coordinate radius included in both figures.")
    args = parser.parse_args()
    plot_final_bssn(args.input, args.output_dir, args.r_max)


if __name__ == "__main__":
    main()
