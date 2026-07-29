import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.geodesic import compute_geodesic_terms
from RadiShPICR.ConstraintBasedRelativity.lorentz_force import compute_lorentz_terms
from RadiShPICR.ConstraintBasedRelativity.solve_metric import calculate_metric
from RadiShPICR.ConstraintBasedRelativity.utils import safe_radius


def _freeze_center_particles(particles):
    """Absorb particles that reach the regular center into an inert r = 0 state."""

    center_particles = particles.r <= 0.0

    particles.r = jnp.where(center_particles, 0.0, particles.r)
    particles.ur = jnp.where(center_particles, 0.0, particles.ur)
    particles.uphi = jnp.where(center_particles, 0.0, particles.uphi)

    return particles


def step(particles, r_grid, dr, dt):
    particles = _freeze_center_particles(particles)
    dr_dt, dphi_dt, dur_dt = _particle_derivatives(particles, r_grid, dr)

    r, phi = particles.get_positions()
    ur, uphi = particles.get_velocities()
    center_particles = r <= 0.0

    dr_dt = jnp.where(center_particles, 0.0, dr_dt)
    dur_dt = jnp.where(center_particles, 0.0, dur_dt)

    particles.r = r + dr_dt * dt
    particles.ur = ur + dur_dt * dt
    particles.phi = phi + dphi_dt * dt
    particles.uphi = uphi

    particles = _freeze_center_particles(particles)

    return particles


def _copy_particle_state(particles, r, phi, ur):
    stage_particles = type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=r,
        ur=ur,
        phi=phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )

    return _freeze_center_particles(stage_particles)


def _particle_derivatives(particles, r_grid, dr, U_state=None):
    particles = _freeze_center_particles(particles)
    if U_state is None:
        U_state = calculate_metric(particles, r_grid, dr)
    dr_dt, dur_dt_GR = compute_geodesic_terms(particles, U_state)
    dur_dt_EM = compute_lorentz_terms(particles, U_state)

    r, _ = particles.get_positions()
    _, uphi = particles.get_velocities()
    dphi_dt = uphi / safe_radius(r, 0.5 * dr)
    dur_dt = dur_dt_GR + dur_dt_EM

    center_particles = r <= 0.0

    dr_dt = jnp.where(center_particles, 0.0, dr_dt)
    dphi_dt = jnp.where(center_particles, 0.0, dphi_dt)
    dur_dt = jnp.where(center_particles, 0.0, dur_dt)

    return dr_dt, dphi_dt, dur_dt


def _step_rk4_particle_update(particles, r_grid, dr, dt, initial_U_state=None):
    particles = _freeze_center_particles(particles)

    r0, phi0 = particles.get_positions()
    ur0, uphi0 = particles.get_velocities()

    k1_r, k1_phi, k1_ur = _particle_derivatives(
        particles,
        r_grid,
        dr,
        U_state=initial_U_state,
    )

    stage2 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k1_r,
        phi0 + 0.5 * dt * k1_phi,
        ur0 + 0.5 * dt * k1_ur,
    )
    k2_r, k2_phi, k2_ur = _particle_derivatives(stage2, r_grid, dr)

    stage3 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k2_r,
        phi0 + 0.5 * dt * k2_phi,
        ur0 + 0.5 * dt * k2_ur,
    )
    k3_r, k3_phi, k3_ur = _particle_derivatives(stage3, r_grid, dr)

    stage4 = _copy_particle_state(
        particles,
        r0 + dt * k3_r,
        phi0 + dt * k3_phi,
        ur0 + dt * k3_ur,
    )
    k4_r, k4_phi, k4_ur = _particle_derivatives(stage4, r_grid, dr)

    particles.r = r0 + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    particles.phi = phi0 + (dt / 6.0) * (
        k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi
    )
    particles.ur = ur0 + (dt / 6.0) * (k1_ur + 2.0 * k2_ur + 2.0 * k3_ur + k4_ur)
    particles.uphi = uphi0

    particles = _freeze_center_particles(particles)

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
