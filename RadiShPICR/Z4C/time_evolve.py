import jax
import jax.numpy as jnp

from RadiShPICR.Z4C.constraint_terms import dGammadt, dthetadt
from RadiShPICR.Z4C.extrinsic_curvature import dArrdt, dAtdt, dKhdt
from RadiShPICR.Z4C.shift_and_lapse import dalphadt, dbetadt
from RadiShPICR.Z4C.spatial_metric import dchidt, dgrrdt, dgtdt
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.energy_momentum_tensor import (
    MatterTerms,
    compute_radial_matter_terms,
    initialize_vacuum_matter_terms,
)
from RadiShPICR.Z4C.electric_field import (
    compute_electrostatic_matter_terms,
    compute_radial_charge_density,
    compute_radial_lorentz_force,
    solve_radial_electric_field,
)
from RadiShPICR.Z4C.geodesic import compute_geodesic_terms
from RadiShPICR.Z4C.utils import (
    trace_free_curvature,
    unit_determinant_conformal_metric,
)


def metric_time_derivatives(metric: Z4C_Metric, matter_terms):
    zeros = jnp.zeros_like(metric.r)
    zero_dr = jnp.zeros_like(metric.dr)

    return Z4C_Metric(
        alpha=dalphadt(metric, matter_terms),
        beta=dbetadt(metric, matter_terms),
        conformal_grr=dgrrdt(metric, matter_terms),
        conformal_gt=dgtdt(metric, matter_terms),
        chi=dchidt(metric, matter_terms),
        Kh=dKhdt(metric, matter_terms),
        Arr=dArrdt(metric, matter_terms),
        At=dAtdt(metric, matter_terms),
        theta=dthetadt(metric, matter_terms),
        Gamma=dGammadt(metric, matter_terms),
        kappa=jnp.zeros_like(metric.kappa),
        eta=jnp.zeros_like(metric.eta),
        nu=jnp.zeros_like(metric.nu),
        r=zeros,
        dr=zero_dr,
    )


def _enforce_algebraic_constraints(metric):
    conformal_grr, conformal_gt = unit_determinant_conformal_metric(
        metric.conformal_grr,
        metric.conformal_gt,
    )
    # Enforce det(conformal_gamma) = conformal_grr * conformal_gt**2 = 1.

    constrained_metric = metric._replace(
        conformal_grr=conformal_grr,
        conformal_gt=conformal_gt,
    )

    Arr, At = trace_free_curvature(
        constrained_metric.Arr,
        constrained_metric.At,
        constrained_metric,
    )
    # Use the projected conformal metric when removing the curvature trace.

    return constrained_metric._replace(Arr=Arr, At=At)


def _add_metric_derivative(metric, derivative, scale):
    new_metric = Z4C_Metric(
        alpha=metric.alpha + scale * derivative.alpha,
        beta=metric.beta + scale * derivative.beta,
        conformal_grr=metric.conformal_grr + scale * derivative.conformal_grr,
        conformal_gt=metric.conformal_gt + scale * derivative.conformal_gt,
        chi=metric.chi + scale * derivative.chi,
        Kh=metric.Kh + scale * derivative.Kh,
        Arr=metric.Arr + scale * derivative.Arr,
        At=metric.At + scale * derivative.At,
        theta=metric.theta + scale * derivative.theta,
        Gamma=metric.Gamma + scale * derivative.Gamma,
        kappa=metric.kappa,
        eta=metric.eta,
        nu=metric.nu,
        r=metric.r,
        dr=metric.dr,
    )
    # Construct the RK stage before imposing the Z4C algebraic constraints.

    return _enforce_algebraic_constraints(new_metric)


def _combine_rk4_derivatives(k1, k2, k3, k4):
    return Z4C_Metric(
        alpha=k1.alpha + 2.0 * k2.alpha + 2.0 * k3.alpha + k4.alpha,
        beta=k1.beta + 2.0 * k2.beta + 2.0 * k3.beta + k4.beta,
        conformal_grr=(
            k1.conformal_grr
            + 2.0 * k2.conformal_grr
            + 2.0 * k3.conformal_grr
            + k4.conformal_grr
        ),
        conformal_gt=(
            k1.conformal_gt
            + 2.0 * k2.conformal_gt
            + 2.0 * k3.conformal_gt
            + k4.conformal_gt
        ),
        chi=k1.chi + 2.0 * k2.chi + 2.0 * k3.chi + k4.chi,
        Kh=k1.Kh + 2.0 * k2.Kh + 2.0 * k3.Kh + k4.Kh,
        Arr=k1.Arr + 2.0 * k2.Arr + 2.0 * k3.Arr + k4.Arr,
        At=k1.At + 2.0 * k2.At + 2.0 * k3.At + k4.At,
        theta=k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta,
        Gamma=k1.Gamma + 2.0 * k2.Gamma + 2.0 * k3.Gamma + k4.Gamma,
        kappa=k1.kappa,
        eta=k1.eta,
        nu=k1.nu,
        r=k1.r,
        dr=k1.dr,
    )


def metric_rk4_step(metric: Z4C_Metric, matter_terms, dt):
    """Advance the Z4C metric with fixed matter terms using classic RK4."""

    metric = _enforce_algebraic_constraints(metric)
    # Every RK right-hand side is evaluated from a constraint-projected metric.

    k1 = metric_time_derivatives(metric, matter_terms)

    metric_k2 = _add_metric_derivative(metric, k1, 0.5 * dt)
    k2 = metric_time_derivatives(metric_k2, matter_terms)

    metric_k3 = _add_metric_derivative(metric, k2, 0.5 * dt)
    k3 = metric_time_derivatives(metric_k3, matter_terms)

    metric_k4 = _add_metric_derivative(metric, k3, dt)
    k4 = metric_time_derivatives(metric_k4, matter_terms)

    weighted_derivative = _combine_rk4_derivatives(k1, k2, k3, k4)

    return _add_metric_derivative(metric, weighted_derivative, dt / 6.0)


def _metric_fields_finite(metric):
    evolved_fields = jnp.stack(
        (
            metric.alpha,
            metric.beta,
            metric.conformal_grr,
            metric.conformal_gt,
            metric.chi,
            metric.Kh,
            metric.Arr,
            metric.At,
            metric.theta,
            metric.Gamma,
        )
    )

    return jnp.all(jnp.isfinite(evolved_fields))


def _copy_particle_state(particles, r, phi, ur, uphi, weight=None):
    if weight is None:
        weight = particles.weight

    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=weight,
        r=r,
        ur=ur,
        phi=phi,
        uphi=uphi,
        shape_mode=particles.shape_mode,
    )


def _zero_metric_derivative(metric):
    return jax.tree.map(jnp.zeros_like, metric)


def _add_matter_terms(particle_matter, field_matter):
    return MatterTerms(
        rho=particle_matter.rho + field_matter.rho,
        Srr=particle_matter.Srr + field_matter.Srr,
        Stt=particle_matter.Stt + field_matter.Stt,
        Sr=particle_matter.Sr + field_matter.Sr,
        St=particle_matter.St + field_matter.St,
    )


def _electrostatic_stage_state(particles, metric, epsilon_0, EM_on):
    def electrostatic_state(_):
        charge_density = compute_radial_charge_density(particles, metric)
        E_r = solve_radial_electric_field(
            metric,
            charge_density,
            epsilon_0=epsilon_0,
        )
        lorentz_force = compute_radial_lorentz_force(particles, metric, E_r)
        field_matter = compute_electrostatic_matter_terms(
            metric,
            E_r,
            epsilon_0=epsilon_0,
        )

        return charge_density, E_r, lorentz_force, field_matter

    def zero_electrostatic_state(_):
        zeros_grid = jnp.zeros_like(metric.r)
        zeros_particles = jnp.zeros_like(particles.r)
        field_matter = initialize_vacuum_matter_terms(metric)

        return zeros_grid, zeros_grid, zeros_particles, field_matter

    return jax.lax.cond(
        EM_on,
        electrostatic_state,
        zero_electrostatic_state,
        operand=None,
    )


def _stage_derivatives(particles, metric, epsilon_0, EM_on, GR_on):
    _, _, lorentz_force, field_matter = _electrostatic_stage_state(
        particles,
        metric,
        epsilon_0,
        EM_on,
    )
    du_r_dt, du_phi_dt, dr_dt, dphi_dt = compute_geodesic_terms(
        particles,
        metric,
    )
    du_r_dt = du_r_dt + lorentz_force

    def dynamical_metric_derivative(_):
        particle_matter = compute_radial_matter_terms(particles, metric)
        matter_terms = _add_matter_terms(particle_matter, field_matter)
        return metric_time_derivatives(metric, matter_terms)

    metric_derivative = jax.lax.cond(
        GR_on,
        dynamical_metric_derivative,
        lambda _: _zero_metric_derivative(metric),
        operand=None,
    )

    return metric_derivative, du_r_dt, du_phi_dt, dr_dt, dphi_dt


def _metric_stage(metric, derivative, scale, GR_on):
    return jax.lax.cond(
        GR_on,
        lambda _: _add_metric_derivative(metric, derivative, scale),
        lambda _: metric,
        operand=None,
    )


def _final_electrostatic_fields(particles, metric, epsilon_0, EM_on):
    def electrostatic_fields(_):
        charge_density = compute_radial_charge_density(particles, metric)
        E_r = solve_radial_electric_field(
            metric,
            charge_density,
            epsilon_0=epsilon_0,
        )
        return charge_density, E_r

    def zero_fields(_):
        zeros = jnp.zeros_like(metric.r)
        return zeros, zeros

    return jax.lax.cond(
        EM_on,
        electrostatic_fields,
        zero_fields,
        operand=None,
    )


def rk4_step(
    particles,
    metric: Z4C_Metric,
    dt,
    *,
    EM_on,
    GR_on,
    epsilon_0=1.0,
    particle_boundary=None,
):
    """Advance particles, electrostatics, and Z4C with one RK4 tableau.

    ``EM_on`` and ``GR_on`` are runtime JAX booleans.  Electromagnetic and
    gravitational sources are recomputed from matching particle and metric
    states at every RK stage.  With ``GR_on=False`` the supplied metric is
    used as a static background without algebraic projection.
    """

    metric = jax.lax.cond(
        GR_on,
        _enforce_algebraic_constraints,
        lambda static_metric: static_metric,
        metric,
    )
    r0, phi0 = particles.get_positions()
    ur0, uphi0 = particles.get_velocities()

    k1 = _stage_derivatives(particles, metric, epsilon_0, EM_on, GR_on)
    k1_metric, k1_du_r_dt, k1_du_phi_dt, k1_dr_dt, k1_dphi_dt = k1

    metric_k2 = _metric_stage(metric, k1_metric, 0.5 * dt, GR_on)
    particles_k2 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k1_dr_dt,
        phi0 + 0.5 * dt * k1_dphi_dt,
        ur0 + 0.5 * dt * k1_du_r_dt,
        uphi0 + 0.5 * dt * k1_du_phi_dt,
    )
    if particle_boundary is not None:
        particles_k2 = particle_boundary(particles_k2)
    k2 = _stage_derivatives(particles_k2, metric_k2, epsilon_0, EM_on, GR_on)
    k2_metric, k2_du_r_dt, k2_du_phi_dt, k2_dr_dt, k2_dphi_dt = k2

    metric_k3 = _metric_stage(metric, k2_metric, 0.5 * dt, GR_on)
    particles_k3 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k2_dr_dt,
        phi0 + 0.5 * dt * k2_dphi_dt,
        ur0 + 0.5 * dt * k2_du_r_dt,
        uphi0 + 0.5 * dt * k2_du_phi_dt,
        weight=particles_k2.weight,
    )
    if particle_boundary is not None:
        particles_k3 = particle_boundary(particles_k3)
    k3 = _stage_derivatives(particles_k3, metric_k3, epsilon_0, EM_on, GR_on)
    k3_metric, k3_du_r_dt, k3_du_phi_dt, k3_dr_dt, k3_dphi_dt = k3

    metric_k4 = _metric_stage(metric, k3_metric, dt, GR_on)
    particles_k4 = _copy_particle_state(
        particles,
        r0 + dt * k3_dr_dt,
        phi0 + dt * k3_dphi_dt,
        ur0 + dt * k3_du_r_dt,
        uphi0 + dt * k3_du_phi_dt,
        weight=particles_k3.weight,
    )
    if particle_boundary is not None:
        particles_k4 = particle_boundary(particles_k4)
    k4 = _stage_derivatives(particles_k4, metric_k4, epsilon_0, EM_on, GR_on)
    k4_metric, k4_du_r_dt, k4_du_phi_dt, k4_dr_dt, k4_dphi_dt = k4

    weighted_metric_derivative = _combine_rk4_derivatives(
        k1_metric,
        k2_metric,
        k3_metric,
        k4_metric,
    )
    final_metric = _metric_stage(
        metric,
        weighted_metric_derivative,
        dt / 6.0,
        GR_on,
    )

    final_particles = _copy_particle_state(
        particles,
        r0 + (dt / 6.0) * (
            k1_dr_dt + 2.0 * k2_dr_dt + 2.0 * k3_dr_dt + k4_dr_dt
        ),
        phi0 + (dt / 6.0) * (
            k1_dphi_dt
            + 2.0 * k2_dphi_dt
            + 2.0 * k3_dphi_dt
            + k4_dphi_dt
        ),
        ur0 + (dt / 6.0) * (
            k1_du_r_dt
            + 2.0 * k2_du_r_dt
            + 2.0 * k3_du_r_dt
            + k4_du_r_dt
        ),
        uphi0 + (dt / 6.0) * (
            k1_du_phi_dt
            + 2.0 * k2_du_phi_dt
            + 2.0 * k3_du_phi_dt
            + k4_du_phi_dt
        ),
        weight=particles_k4.weight,
    )
    if particle_boundary is not None:
        final_particles = particle_boundary(final_particles)

    charge_density, E_r = _final_electrostatic_fields(
        final_particles,
        final_metric,
        epsilon_0,
        EM_on,
    )

    return final_particles, final_metric, charge_density, E_r


def advance_vacuum_steps(metric: Z4C_Metric, dt, num_steps):
    """Advance several exact-vacuum Z4C steps in one compiled scan."""

    def advance_one_step(carry, local_step):
        metric, first_nonfinite_step = carry
        matter_terms = initialize_vacuum_matter_terms(metric)
        metric = metric_rk4_step(
            metric,
            matter_terms,
            dt,
        )
        finite = _metric_fields_finite(metric)
        first_nonfinite_step = jnp.where(
            jnp.logical_and(first_nonfinite_step < 0, jnp.logical_not(finite)),
            local_step.astype(first_nonfinite_step.dtype),
            first_nonfinite_step,
        )

        return (metric, first_nonfinite_step), None

    initial_state = (
        metric,
        jnp.asarray(-1, dtype=jnp.int32),
    )
    (metric, first_nonfinite_step), _ = jax.lax.scan(
        advance_one_step,
        initial_state,
        jnp.arange(num_steps),
    )

    return metric, first_nonfinite_step
