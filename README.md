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

The shared particle container stores different radial variables in each
formulation: Z4C uses isotropic `r` and covariant `u_r`, while constrained
relativity stores `r_s = A r` and `u_r / A` in the same `r` and `ur` fields.
Conversions belong at formulation boundaries; the container does not convert.

Z4C particle deletion and deposition are configured explicitly:

```python
from RadiShPICR.Z4C import (
    rk4_step, deleting_inner_areal_radius_boundary,
    compute_radial_charge_density, solve_radial_electric_field,
)

# Initialize once. Carry the returned E_r into every subsequent step.
charge_density = compute_radial_charge_density(particles, metric, inner_open=True)
E_r = solve_radial_electric_field(metric, charge_density)

particles, metric, charge_density, E_r = rk4_step(
    particles, metric, dt, E_r=E_r, EM_on=True, GR_on=True,
    particle_boundary=deleting_inner_areal_radius_boundary,
    inner_open=True,
)
```

Pass `inner_open=True` for open-inner deposition, including when the callback
is wrapped. Callback identity no longer selects deposition. `inner_open`,
`EM_on`, `GR_on`, and `zero_shift` remain runtime JAX flags; the callback is a
static argument when compiling the stepper. Ordinary parity deposition remains
the default. Quadratic deposition is metric-corrected, while field gathering
retains ordinary coordinate-space weights.

`E_r` is a required keyword containing the cell-centered covariant electric
field. Supply a zero grid array for EM-off runs. The stepper advances E_r with
the same RK4 tableau as particles and the metric; it never solves Gauss' law.
`compute_radial_current_density(particles, metric, dr_dt, inner_open=False)`
deposits coordinate transport current, `alpha * J_Eulerian^r - beta * rho_q`,
using the stage's coordinate velocity (not stored covariant momentum).
`radial_electric_field_time_derivative` includes the geometric terms from the
actual metric RHS, which vanish on a fixed background.

Direct current deposition does not enforce discrete charge conservation.
`radial_gauss_residual(metric, E_r, charge_density, epsilon_0=1.0)` measures
`epsilon_0 * D_i E^i - rho_q` with the radial differential stencil; its initial
error differs from the finite-volume Gauss initialization error. No cleaning
or field reset is applied when particles are absorbed. The charged-star demo
saves the evolved field and residual, initial/final residual norms, signed
charge removed by weight deletion, and signed charge missing from truncated
particle shapes. Existing fields can remain after their source is deleted.

The OS Z4C runner and movie renderer default to `outputs/z4c_oppenheimer_snyder`
inside their demo directory. Explicit historical run directories remain usable.
Initial data now use a bracketed spatial Heun shooting solve and one lapse
normalization against the requested exterior mass. `--shooting-tolerance`
(default `1e-10`) and `--shooting-max-iterations` (default `64`) replace
`--shooting-iterations`. Unmatched initial data cause an explicit failure.
`integrate_metric_from_origin` exposes a single Heun shot; `calculate_metric`
retains its evolution-time two-shot algorithm and requires `previous_X_t` and
`previous_X_r` by keyword.

The campaign-specific two-stream review script is retained under
`demos/relativistic_two_stream/archive/20260908/`, with its input assumptions
documented there.

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
- [X] Single puncture black hole in Z4C.
- [X] Oppenheimer-Snyder collapse in constraint-based relativity.
- [ ] Charged stellar collapse in constraint-based relativity.
- [ ] Relativistic two-stream instability with fixed-background and dynamical
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
