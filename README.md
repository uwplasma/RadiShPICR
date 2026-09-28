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

Particle shapes now live in `RadiShPICR.particles.shape_factors`; the former
`particles.particle_shapes` module has been removed. Original shape and gather
functions are in `shape_factors.cartesian_shapes`.

```python
from RadiShPICR.particles.shape_factors import (
    particle_deposition_stencil,
    metric_corrected_cic_stencil,
    metric_corrected_quadratic_stencil,
)

indices, even_weights, odd_weights = particle_deposition_stencil(
    particles, metric, inner_open=False,
)
```

The stencil axis comes first and the particle axis second. Weights are
dimensionless; deposition multiplies them by particle mass or charge and
divides by proper shell volume. The production dispatcher preserves ordinary
nearest and linear shapes and uses metric-corrected quadratic TSC. Call
`metric_corrected_cic_stencil(particles, metric, inner_open=False)` explicitly
for corrected CIC. Both corrected APIs select their shape independently of
the species' shape setting and read the existing Z4C metric arrays directly.
They support `jax.jit`, origin parity, and unrenormalized open-inner truncation.
The positivity limiter preserves total raw stencil weight but can leave a
uniform-density residual on steep metrics; the outermost shell is not matched
by the correction recurrence.


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
- Selectable standard Sommerfeld and nonlinear constraint-preserving outer
  boundaries for spherical Z4C evolution; Sommerfeld remains the default.

Demos:
- [ ] Single puncture black hole in Z4C.
- [ ] Oppenheimer-Snyder collapse in constraint-based relativity.
- [ ] Charged stellar collapse in constraint-based relativity.
- [X] Relativistic two-stream instability with fixed-background and dynamical
  Z4C evolution from the same time-symmetric Hamiltonian-constraint solve,
  source-aware Misner--Sharp diagnostics, synchronized metric movies, and
  signed energy composition plots.

The two-stream demo initializes two neutral electron-ion streams at opposite
local velocities.  Co-moving ions suppress the unperturbed charge separation
at the annulus edges; a small electron-only displacement seeds the instability.
Ion rest mass and charge are unchanged, but their streaming kinetic energy and
radial stress enter the initial constraint solve and subsequent evolution.
Both gravity modes solve the same initial metric; `--no-dynamic-gr` holds that
metric fixed.  The local homogeneous growth benchmark includes both electron
and ion susceptibility.  Nonlinear evolution can still generate edge structure.
New runs and the diagnostic renderer default to
`demos/relativistic_two_stream/outputs/z4c_two_stream_neutral_streams`.

STILL UNDER DEVELOPMENT.  The code is not yet ready for production use.

CHECKLIST:
- [ ] Add more tests for the constraint-based relativity implementation.
- [ ] Fix definition of angular position update in constraint-based relativity implementation.
- [X] Finish implementation of single-puncture black hole in Z4C.
- [X] Add radial electrostatic stress-energy to evolved Z4C spacetimes and
  combine fixed-metric and dynamical-metric particle evolution in one RK4
  stepper.
