import jax
import jax.numpy as jnp

from RadiShPICR.Z4C.constraint_terms import dGammadt, dthetadt
from RadiShPICR.Z4C.extrinsic_curvature import dArrdt, dAtdt, dKhdt
from RadiShPICR.Z4C.shift_and_lapse import dalphadt, dbetadt
from RadiShPICR.Z4C.spatial_metric import dchidt, dgrrdt, dgtdt
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.energy_momentum_tensor import (
    compute_radial_matter_terms,
    initialize_vacuum_matter_terms,
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


def rk4_step(metric: Z4C_Metric, matter_terms, dt):
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


def advance_vacuum_steps(metric: Z4C_Metric, dt, num_steps):
    """Advance several vacuum steps in one compiled scan."""

    def advance_one_step(carry, local_step):
        metric, first_nonfinite_step = carry
        matter_terms = initialize_vacuum_matter_terms(metric)
        metric = rk4_step(metric, matter_terms, dt)
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


def _copy_particle_state(particles, r, phi, ur, uphi):
    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=r,
        ur=ur,
        phi=phi,
        uphi=uphi,
        shape_mode=particles.shape_mode,
    )


def particles_rk4_step(particles, metric: Z4C_Metric, dt):
    metric = _enforce_algebraic_constraints(metric)
    # Use the same projected metric for the stage-one geodesic and matter terms.

    r0, phi0 = particles.get_positions()
    ur0, uphi0 = particles.get_velocities()

    k1_dvr_dt, k1_duphi_dt, k1_drdt, k1_dphidt = compute_geodesic_terms(particles, metric)
    matter_terms = compute_radial_matter_terms(particles, metric)
    k1_metric = metric_time_derivatives(metric, matter_terms)
    # Stage 1 uses the beginning-of-step particles and metric.

    metric_k2 = _add_metric_derivative(metric, k1_metric, 0.5 * dt)
    particles_k2 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k1_drdt,
        phi0 + 0.5 * dt * k1_dphidt,
        ur0 + 0.5 * dt * k1_dvr_dt,
        uphi0 + 0.5 * dt * k1_duphi_dt,
    )
    k2_dvr_dt, k2_duphi_dt, k2_drdt, k2_dphidt = compute_geodesic_terms(particles_k2, metric_k2)
    matter_terms_k2 = compute_radial_matter_terms(particles_k2, metric_k2)
    k2_metric = metric_time_derivatives(metric_k2, matter_terms_k2)
    # Stage 2 deposits matter from the same half-step particles used by geodesics.

    metric_k3 = _add_metric_derivative(metric, k2_metric, 0.5 * dt)
    particles_k3 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k2_drdt,
        phi0 + 0.5 * dt * k2_dphidt,
        ur0 + 0.5 * dt * k2_dvr_dt,
        uphi0 + 0.5 * dt * k2_duphi_dt,
    )
    k3_dvr_dt, k3_duphi_dt, k3_drdt, k3_dphidt = compute_geodesic_terms(particles_k3, metric_k3)
    matter_terms_k3 = compute_radial_matter_terms(particles_k3, metric_k3)
    k3_metric = metric_time_derivatives(metric_k3, matter_terms_k3)
    # Stage 3 repeats the half-step update, now using k2 particle derivatives.

    metric_k4 = _add_metric_derivative(metric, k3_metric, dt)
    particles_k4 = _copy_particle_state(
        particles,
        r0 + dt * k3_drdt,
        phi0 + dt * k3_dphidt,
        ur0 + dt * k3_dvr_dt,
        uphi0 + dt * k3_duphi_dt,
    )
    k4_dvr_dt, k4_duphi_dt, k4_drdt, k4_dphidt = compute_geodesic_terms(particles_k4, metric_k4)
    matter_terms_k4 = compute_radial_matter_terms(particles_k4, metric_k4)
    k4_metric = metric_time_derivatives(metric_k4, matter_terms_k4)
    # Stage 4 advances both particles and metric by a full dt using k3.

    weighted_derivative = _combine_rk4_derivatives(k1_metric, k2_metric, k3_metric, k4_metric)

    final_metric = _add_metric_derivative(metric, weighted_derivative, dt / 6.0)
    # finish the RK4 step by combining the weighted derivatives and updating the metric

    particles.r = r0 + (dt / 6.0) * (k1_drdt + 2.0 * k2_drdt + 2.0 * k3_drdt + k4_drdt)
    particles.phi = phi0 + (dt / 6.0) * (k1_dphidt + 2.0 * k2_dphidt + 2.0 * k3_dphidt + k4_dphidt)
    particles.ur = ur0 + (dt / 6.0) * (k1_dvr_dt + 2.0 * k2_dvr_dt + 2.0 * k3_dvr_dt + k4_dvr_dt)
    particles.uphi = uphi0 + (dt / 6.0) * (
        k1_duphi_dt + 2.0 * k2_duphi_dt + 2.0 * k3_duphi_dt + k4_duphi_dt
    )
    # update particle positions and velocities using the final metric

    return particles, final_metric
