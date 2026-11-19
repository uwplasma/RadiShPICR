"""Editable settings for the Schwarzschild Proca cloud, in G = c = ε₀ = 1 units."""

from pathlib import Path

MASS = 1.0
MU = 0.1 / MASS
CLOUD_ENERGY = 1.0e-4 * MASS
BOHR = 1.0 / (MASS * MU**2)

R_MAX = 2000.0 * MASS
NUM_CELLS = 40000
CFL = 0.2
EVOLVE_TIME = 1200.0 * MASS
BACKREACTION = True
KAPPA, ETA, NU = 0.1, 2.0, 0.02
DARK_KO = True

SAMPLE_EVERY = 2       # RK steps between probe measurements
SNAPSHOT_EVERY = 500   # RK steps between full radial profiles
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "cloud_evolution"

# Fractions of the recorded probe time span used for the ordinary FFT.
FFT_START_FRACTION = 0.0
FFT_END_FRACTION = 0.9