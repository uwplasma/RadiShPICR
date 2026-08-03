import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.geodesic import compute_geodesic_terms
from RadiShPICR.ConstraintBasedRelativity.lorentz_force import compute_lorentz_terms
from RadiShPICR.ConstraintBasedRelativity.solve_metric import calculate_metric

def step(particles, r_grid, dr, dt):
    drs_dt, dphi_dt, dur_over_A_dt = _particle_derivatives(
        particles,
        r_grid,
        dr,
    )

    particles.r = particles.r + drs_dt * dt
    particles.ur = particles.ur + dur_over_A_dt * dt
    particles.phi = particles.phi + dphi_dt * dt

    return particles


def _copy_particle_state(particles, rs, phi, ur_over_A):
    stage_particles = type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=rs,
        ur=ur_over_A,
        phi=phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )

    return stage_particles


def _particle_derivatives(particles, r_grid, dr, U_state=None):

    if U_state is None:
        U_state = calculate_metric(particles, r_grid, dr)

    dur_dt_EM = compute_lorentz_terms(particles, U_state)
    return compute_geodesic_terms(
        particles,
        U_state,
        dur_dt_EM=dur_dt_EM,
    )


def _step_rk4_particle_update(particles, r_grid, dr, dt, initial_U_state=None):
    rs0 = particles.r
    ur_over_A0 = particles.ur
    phi0 = particles.phi
    uphi0 = particles.uphi

    k1_rs, k1_phi, k1_ur_over_A = _particle_derivatives(
        particles,
        r_grid,
        dr,
        U_state=initial_U_state,
    )

    stage2 = _copy_particle_state(
        particles,
        rs0 + 0.5 * dt * k1_rs,
        phi0 + 0.5 * dt * k1_phi,
        ur_over_A0 + 0.5 * dt * k1_ur_over_A,
    )
    k2_rs, k2_phi, k2_ur_over_A = _particle_derivatives(
        stage2,
        r_grid,
        dr,
    )

    stage3 = _copy_particle_state(
        particles,
        rs0 + 0.5 * dt * k2_rs,
        phi0 + 0.5 * dt * k2_phi,
        ur_over_A0 + 0.5 * dt * k2_ur_over_A,
    )
    k3_rs, k3_phi, k3_ur_over_A = _particle_derivatives(
        stage3,
        r_grid,
        dr,
    )

    stage4 = _copy_particle_state(
        particles,
        rs0 + dt * k3_rs,
        phi0 + dt * k3_phi,
        ur_over_A0 + dt * k3_ur_over_A,
    )
    k4_rs, k4_phi, k4_ur_over_A = _particle_derivatives(
        stage4,
        r_grid,
        dr,
    )

    particles.r = rs0 + (dt / 6.0) * (
        k1_rs + 2.0 * k2_rs + 2.0 * k3_rs + k4_rs
    )
    particles.phi = phi0 + (dt / 6.0) * (
        k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi
    )
    particles.ur = ur_over_A0 + (dt / 6.0) * (
        k1_ur_over_A
        + 2.0 * k2_ur_over_A
        + 2.0 * k3_ur_over_A
        + k4_ur_over_A
    )
    particles.uphi = uphi0

    return particles


def step_rk4(particles, r_grid, dr, dt):
    return _step_rk4_particle_update(particles, r_grid, dr, dt)


def step_rk4_with_metric(particles, U_state, r_grid, dr, dt):
    """Advance particles and return the metric of the completed RK4 state."""

    particles = _step_rk4_particle_update(
        particles,
        r_grid,
        dr,
        dt,
        initial_U_state=U_state,
    )
    U_state_next = calculate_metric(particles, r_grid, dr)

    return particles, U_state_next
