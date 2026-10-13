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

The charged-star movie renderer includes `z4c_electric_field.mp4`, showing the
saved non-densitized covariant FIDO field `E_r` versus areal radius with fixed
axes. Render only this movie from the existing snapshots with:

```bash
python demos/charged_star_collapse_Z4C_GR/make_collapse_movies.py --movies electric-field
```

Omit `--movies` to render the metric, density, and electric-field movies together.
An optional positional output directory selects a different saved run;
`--radial-limit`, `--frame-stride`, and `--fps` apply to all selected movies.

The charged-star demo also provides a dynamical Reissner-Nordstrom puncture
reference and final-state comparisons:

```bash
python demos/charged_star_collapse_Z4C_GR/run_reissner_nordstrom_puncture.py
python demos/charged_star_collapse_Z4C_GR/make_comparison_plots.py
```

The reference reads the star's initial gravitational mass, physical charge,
permittivity, half-cell grid, and gauge parameters from
`outputs/z4c_charged_star_schwarzschild/initial_data_metadata.npz`. It uses the
analytic single-puncture `u=1` initial data in equations (2.8)-(2.13) of
[arXiv:1311.6483v2](https://arxiv.org/abs/1311.6483v2), with
`q = Q / sqrt(4*pi*epsilon_0)`, `psi^2 = (1+M/(2*r))^2-q^2/(4*r^2)`, and
the OS precollapsed lapse `alpha=psi^-2`.

`electrovac_rk4_step(metric, total_charge, dt, epsilon_0=1.0, zero_shift=0,
puncture_chi_floor=0.0)`
keeps puncture electric flux fixed and recomputes the EM stress at each
projected Z4C RK4 stage. There are no particles or exterior currents. The
covariant field `E_r = Q*sqrt(chi*conformal_grr)/(4*pi*epsilon_0*r^2*conformal_gt)`
changes as the coordinates evolve; freezing that array would not preserve
the source-free spherical Maxwell solution. The existing particle and vacuum
steppers are unchanged.

The default duration is evolution time `t/M=150`, with CFL 0.2 and approximately
500 saved intervals. `--input-directory`, `--output-directory`, `--duration`,
`--cfl`, `--snapshot-count`, `--kappa`, `--eta`, `--nu`, and `--zero-shift`
are available. Existing run directories are never overwritten. The runner
checks every accepted step for finite fields and positive spatial geometry;
on failure it saves the last valid state and marks the run incomplete.
The RN runner enables `--puncture-chi-floor 1e-12` in the first three cells
at every projected RK stage and the final update. This local regularization
changes the discretization while active; the general electrovac API leaves
it disabled unless requested. Use `--puncture-chi-floor 0` to disable it in
the runner. With the saved grid and nu=0.02, the unregularized puncture develops
negative chi before 1M. Setting nu=0 avoids that early crossing but produces
a later instability near 27M, including at half the default time step.
The production reference therefore retains nu=0.02 and records the local
floor explicitly in its metadata and comparison plots.
Bulk diagnostic norms omit three cells at each boundary; snapshots retain
all values. The Gauss diagnostic permits nonzero flux at the puncture.

The comparison script accepts `--star-directory`, `--puncture-directory`,
`--output-directory`, and `--radial-limit` (inner figure extent, default 10M).
It undoes the star's diagnostic coordinate rescaling, labels both actual
time coordinates, and writes full/inner BSSN and physical comparison figures
plus `comparison_summary.json`. The BSSN components remain gauge dependent.
The physical panels compare the normal electric field, electric flux,
Misner-Sharp mass, and flux-corrected RN mass against analytic RN. Charge is
inferred from the field, not surviving particles. The initial mass is never
fitted to the star's measured exterior geometry.

The neutral and charged collapse runners now use consistent particle sources
for their initial constraints. Their output directories have a
`_consistent_initial_data` suffix; historical runs remain readable through
explicit paths. `solve_star_initial_data(particles, metric, total_mass)` solves
the time-symmetric Hamiltonian constraint with the production Z4C stencils and
Gauss operator, returning `(metric, particles, E_r)`. Its input particle radii
are areal radii, and its returned particles use Z4C coordinate radii. It keeps
charge, weights, and areal positions fixed and calibrates a uniform rest-mass
factor to the requested gravitational mass. `solve_charged_star_initial_data`
remains an alias. This cold-star initializer requires zero momenta and a
matter-free outer boundary; it is not an evolution-time mass adjustment.
The constrained OS demo similarly calibrates its initial exterior geometric
mass through `solve_constrained_star_initial_data`, which returns
`(U_state, particles)` in constrained storage.

The constrained shoot still supplies the initial Z4C lapse. Its
`--shooting-tolerance` (default `1e-10`) and `--shooting-max-iterations`
(default `64`) control spatial chart matching. The subsequent Hamiltonian
solve establishes the physical mass, rather than fitting the final metric.

Every constrained Heun shot now iterates a completed map `R = r * A(r)`.
Each particle has one coordinate position and one covariant momentum for
an entire sweep; predictor and corrector apply only their local proper-volume
factors. Both shots in `calculate_metric` use this iteration, preserving the
eight-item state and paired, keyword-only `previous_X_t` / `previous_X_r`
rescaling policy. General evolution never calibrates particle masses.
Point-source helpers require keyword-only `A_values` on the full grid;
a scalar local `A_at_point` cannot determine the particle map.

The completed-map solve uses 0.5 relaxation, at most 100 iterations, and
metric/map tolerance `max(1e-10, 100 * dtype_epsilon)`. Invalid maps and
nonconverged solves return nonfinite states for the existing evolution
rejection checks; host initializers raise an error. Coarse nearest-grid-point
particle configurations can oscillate across a deposition-cell boundary and
fail this consistency requirement; an unconverged shot is never accepted.

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
- Sommerfeld outer boundary conditions for spherical Z4C evolution.

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
