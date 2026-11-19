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

# Kinetically mixed plasma. PLASMA_FREQUENCY = None keeps the vacuum run.
#
# The species is specified by its plasma frequency rather than by geometrized
# charge and mass: omega_p**2 = q**2 n / (m eps0) is the only O(1) combination,
# while e_geom ~ 1.4e-36 and m_e,geom ~ 6.8e-58 individually are not. Fields are
# rescaled by lambda = q/m, so PLASMA_CHARGE / PLASMA_MASS is alpha_s (-1 for
# the reference electrons) and EPSILON_0 is the rescaled eps0 / lambda**2.
#
# EPSILON_0 is NOT a free normalization while BACKREACTION is on: the plasma
# rest-mass density is eps0 * omega_p**2 / alpha_s**2, so EPSILON_0 is exactly
# the dial setting how strongly the plasma gravitates at fixed omega_p. With
# EPSILON_0 = 1 and PLASMA_FREQUENCY = 0.1 the shell would weigh far more than
# the hole. Pick it from the mass budget first.
PLASMA_FREQUENCY = None
EPSILON_0 = 1.0e-8
SIN_CHI = 0.0          # kinetic mixing; zero decouples the plasma entirely
PLASMA_CHARGE = -1.0   # alpha_s = (q_s/m_s) / lambda
PLASMA_MASS = 1.0
PLASMA_R_MIN = 50.0
PLASMA_R_MAX = 400.0
PARTICLES_PER_CELL = 8
NEUTRALIZE = True      # add the ion species; without it the shell is a monopole

SAMPLE_EVERY = 2       # RK steps between probe measurements
SNAPSHOT_EVERY = 500   # RK steps between full radial profiles
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "cloud_evolution"

# Fractions of the recorded probe time span used for the ordinary FFT.
FFT_START_FRACTION = 0.0
FFT_END_FRACTION = 0.9