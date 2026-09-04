<div align="center">
  <img src="docs/images/logo.png" alt="RadiShPICR Logo" width="200">
</div>

## RadiShPICR ##

RadiShPICR is a spherically symmetric particle in cell code that fuses a purely radial electrostatic 
particle-in-cell method with two formulations of spherically symmetric numerical relativity: Z4C and constraint-based relativity.  The code is written in Python with JAX.

The relativity implementations are localized by formulation.  Constraint-based
radial metric solves and particle timestepping are imported from
`RadiShPICR.ConstraintBasedRelativity`; Z4C metric evolution helpers are
imported from `RadiShPICR.Z4C`.  `RadiShPICR.evolve` remains as a compatibility
import for the constraint-based `step` and `step_rk4` routines.


Features:
- Z4C metric evolution.
- Fully self-consistent constraint-based relativity formulation.
- Particle shape functions for spherical symmetry (first-order and second-order).
- Radial electrostatic field solver.
- Unified Z4C particle, electrostatic, and metric RK4 evolution with runtime
  `EM_on` and `GR_on` switches, stage-centered sources, and electromagnetic
  stress-energy for dynamical spacetimes.
- Explicit metric-only Z4C RK4 evolution for vacuum and prescribed-source
  calculations without particle deposition.

Demos:
- [ ] Single puncture black hole in Z4C.
- [ ] Oppenheimer-Snyder collapse in constraint-based relativity.
- [ ] Charged stellar collapse in constraint-based relativity.
- [X] Relativistic two-stream instability from flat Z4C initial data, with
  fixed-Minkowski and dynamical-GR modes, source-aware Misner--Sharp and
  Kretschmann diagnostics, synchronized metric movies, and signed energy
  composition plots.

STILL UNDER DEVELOPMENT.  The code is not yet ready for production use.

CHECKLIST:
- [ ] Add more tests for the constraint-based relativity implementation.
- [ ] Fix definition of angular position update in constraint-based relativity implementation.
- [X] Finish implementation of single-puncture black hole in Z4C.
- [X] Add radial electrostatic stress-energy to evolved Z4C spacetimes and
  combine fixed-metric and dynamical-metric particle evolution in one RK4
  stepper.
