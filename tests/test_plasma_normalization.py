"""The plasma-frequency normalization of the particle sources.

These tests pin the relation that lets an electron-scale plasma be specified
without ever forming a geometrized charge or mass:

    omega_p**2 = charges**2 * weight * N / (epsilon_0 * masses * V)

and its two corollaries, the deposited charge density
``epsilon_0 omega_p**2 / (c/m)`` and the rest-mass density
``epsilon_0 omega_p**2 / (c/m)**2``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.initial_data import (
    cold_plasma_shell,
    neutral_plasma_shell,
    plasma_macroparticle_weight,
)
from RadiShPICR.Z4C import (
    compute_radial_charge_density,
    compute_radial_current_density,
    compute_radial_lorentz_force,
)
from RadiShPICR.Z4C.particle_boundaries import deleting_particle_boundary
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume


EPSILON_0 = 1.0e-8
OMEGA_P = 0.3
CHARGE_TO_MASS = -1.0
R_MIN, R_MAX = 4.0, 14.0


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def curved_metric(num_cells=40, dr=0.5):
    """A metric with constant but non-unit conformal factors."""

    r = (jnp.arange(num_cells) + 0.5) * dr
    zeros = jnp.zeros_like(r)
    return Z4C_Metric(
        jnp.ones_like(r), zeros, jnp.full(num_cells, 2.0),
        jnp.full(num_cells, 3.0), jnp.full(num_cells, 0.5),
        zeros, zeros, zeros, zeros, zeros,
        jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0), r, jnp.asarray(dr),
    )


def interior(metric, margin=1.5):
    """Cells whose full shape support lies inside the shell."""

    radius = np.asarray(metric.r)
    return (radius > R_MIN + margin) & (radius < R_MAX - margin)


def shell(metric, particles_per_cell=16, epsilon_0=EPSILON_0,
          plasma_frequency=OMEGA_P, charge=CHARGE_TO_MASS, mass=1.0, ur=0.0):
    return cold_plasma_shell(
        metric, "electrons", plasma_frequency, charge=charge, mass=mass,
        r_min=R_MIN, r_max=R_MAX, particles_per_cell=particles_per_cell,
        epsilon_0=epsilon_0, ur=ur,
    )


def test_weight_inverts_the_cold_linear_response():
    weight = plasma_macroparticle_weight(
        OMEGA_P, charge=-1.0, mass=1.0, epsilon_0=EPSILON_0,
        proper_volume=7.5, particles_per_cell=4,
    )
    np.testing.assert_allclose(
        weight, EPSILON_0 * OMEGA_P**2 * 7.5 / 4.0, rtol=1.0e-15,
    )
    # Reproduces the os_dev two-stream idiom: mass density / ppc per cell.
    mass_density = EPSILON_0 * OMEGA_P**2 / 1.0**2
    np.testing.assert_allclose(weight, mass_density * 7.5 / 4.0, rtol=1.0e-15)


def test_deposited_source_equals_omega_p_squared_on_curved_metric():
    metric = curved_metric()
    particles, _ = shell(metric)
    inside = interior(metric)

    charge_density = np.asarray(compute_radial_charge_density(particles, metric))
    expected = EPSILON_0 * OMEGA_P**2 / CHARGE_TO_MASS
    np.testing.assert_allclose(charge_density[inside], expected, rtol=3.0e-3)

    velocity = 0.25
    current = np.asarray(compute_radial_current_density(
        particles, metric, jnp.full(particles.r.shape, velocity),
    ))
    np.testing.assert_allclose(
        current[inside], expected * velocity, rtol=3.0e-3,
    )


def test_uniform_deposit_converges_at_second_order():
    """The residual bias is the deposition scheme's own dr**2 error.

    Shape weights partition in coordinate space while the deposit is divided
    by proper shell volume, so a radially varying volume leaves an O((dr/r)**2)
    bias. It must converge, not plateau: a plateau would mean the weights
    themselves are inconsistent.
    """

    errors = []
    for num_cells, dr in ((40, 0.5), (80, 0.25), (160, 0.125), (320, 0.0625)):
        metric = curved_metric(num_cells, dr)
        particles, _ = shell(metric)
        deposited = np.asarray(compute_radial_charge_density(particles, metric))
        expected = EPSILON_0 * OMEGA_P**2 / CHARGE_TO_MASS
        errors.append(
            np.max(np.abs(deposited[interior(metric)] / expected - 1.0))
        )
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert np.all(ratios > 3.5), f"not second order: {ratios}"


def test_source_is_invariant_over_thirty_decades_of_epsilon_0():
    """The precision claim.

    With ``charges/masses`` pinned to ``alpha_s``, the physical ``lambda = q/m``
    enters only through ``epsilon_0 = eps0 / lambda**2``. The deposited charge
    then spans the whole electron-scale range while the source term that
    actually drives the fields stays invariant to machine precision.
    """

    metric = curved_metric()
    inside = interior(metric)
    deposited_magnitudes, sources = [], []
    for epsilon_0 in (1.0e-16, 1.0e-12, 1.0e-8, 1.0e-4, 1.0, 1.0e8, 1.0e16):
        particles, _ = shell(metric, epsilon_0=epsilon_0)
        deposited_magnitudes.append(
            float(np.max(np.abs(np.asarray(particles.get_charge()))))
        )
        charge_density = np.asarray(
            compute_radial_charge_density(particles, metric)
        )
        sources.append(charge_density[inside] / epsilon_0)
        # The pusher ratio must stay exactly alpha_s across the whole sweep.
        force = compute_radial_lorentz_force(
            particles, metric, jnp.ones_like(metric.r),
        )
        np.testing.assert_allclose(
            np.asarray(force), CHARGE_TO_MASS, rtol=1.0e-15,
        )

    spread = max(deposited_magnitudes) / min(deposited_magnitudes)
    assert spread > 1.0e30, f"sweep did not exercise the range: {spread:.3e}"
    for source in sources[1:]:
        np.testing.assert_allclose(source, sources[0], rtol=1.0e-13)


def test_rest_mass_density_follows_from_omega_p_and_alpha():
    """Self-gravity is not a free parameter once omega_p and alpha are set."""

    metric = curved_metric()
    particles, diagnostics = shell(metric)
    total_rest_mass = float(jnp.sum(particles.get_mass()))
    expected = EPSILON_0 * OMEGA_P**2 / CHARGE_TO_MASS**2

    np.testing.assert_allclose(
        total_rest_mass / diagnostics["shell_proper_volume"], expected,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        diagnostics["rest_mass_density"], expected, rtol=1.0e-15,
    )


def test_deleting_particles_does_not_redistribute_the_source():
    """The normalization is frozen in the weights, never renormalized.

    A normalization recomputed from the surviving weights would conserve the
    integrated source across a deletion instead of reducing it.
    """

    metric = curved_metric()
    particles, _ = shell(metric)
    velocity = jnp.full(particles.r.shape, 0.25)
    volume = proper_radial_shell_volume(metric)

    before = float(jnp.sum(
        compute_radial_current_density(particles, metric, velocity) * volume
    ))
    doomed = particles.r < 6.0
    removed = float(jnp.sum(jnp.where(
        doomed, particles.get_charge() * velocity, 0.0,
    )))
    survivors = particles.replace(
        r=jnp.where(doomed, -1.0, particles.r),
    )
    after = float(jnp.sum(
        compute_radial_current_density(
            deleting_particle_boundary(survivors), metric, velocity,
        ) * volume
    ))

    assert np.count_nonzero(np.asarray(doomed)) > 0
    np.testing.assert_allclose(after, before - removed, rtol=1.0e-12)


def test_neutral_shell_cancels_charge_but_not_response():
    metric = curved_metric()
    particles, diagnostics = neutral_plasma_shell(
        metric, OMEGA_P, r_min=R_MIN, r_max=R_MAX, particles_per_cell=8,
        epsilon_0=EPSILON_0,
    )
    charge_density = np.asarray(
        compute_radial_charge_density(particles, metric)
    )
    electron_scale = abs(EPSILON_0 * OMEGA_P**2 / CHARGE_TO_MASS)
    np.testing.assert_allclose(
        charge_density, 0.0, atol=1.0e-12 * electron_scale,
    )

    # Co-located opposite charges cancel, but each species still responds with
    # its own charge-to-mass ratio, so the current does not vanish.
    ratio = np.asarray(particles.charges / particles.masses)
    assert np.isclose(ratio.max(), 1.0 / 1836.0)
    assert np.isclose(ratio.min(), -1.0)
    np.testing.assert_allclose(
        diagnostics["pair_plasma_frequency"],
        OMEGA_P * np.sqrt(1.0 + 1.0 / 1836.0), rtol=1.0e-12,
    )


def test_helper_rejects_unphysical_species():
    metric = curved_metric()
    with pytest.raises(ValueError):
        shell(metric, mass=0.0)
    with pytest.raises(ValueError):
        shell(metric, charge=0.0)
    with pytest.raises(ValueError):
        shell(metric, epsilon_0=0.0)
    with pytest.raises(ValueError):
        cold_plasma_shell(
            metric, "e", OMEGA_P, charge=-1.0, mass=1.0, r_min=4.0,
            r_max=1.0e4, particles_per_cell=4,
        )


def test_shell_oscillates_at_the_frequency_it_was_specified_with():
    """End to end: omega_p in, omega_p out, through the Proca coupling.

    With mu = 0 on a flat static background the mixed system closes as

        dE'/dt = -sin_chi omega_p**2 v / alpha_s ,  dv/dt = sin_chi alpha_s E'

    so E' obeys d2E'/dt2 = -(sin_chi omega_p)**2 E'. Starting from E' = 0 and a
    uniform drift v0 gives E'(t) = -(omega_p v0 / alpha_s) sin(omega_p t) at
    sin_chi = 1. Nothing in the setup mentions a charge or a mass, so this is
    the whole chain: weight normalization, deposition, Ampere's law, pusher.
    """

    from functools import partial
    from RadiShPICR.Z4C.time_evolve import rk4_step

    num_cells, dr = 200, 0.25
    radius = (jnp.arange(num_cells) + 0.5) * dr
    zeros, ones = jnp.zeros_like(radius), jnp.ones_like(radius)
    flat = Z4C_Metric(
        ones, zeros, ones, ones, ones, zeros, zeros, zeros, zeros, zeros,
        jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0), radius,
        jnp.asarray(dr),
    )

    epsilon_0, omega_p, drift = 1.0e-6, 0.8, 1.0e-6
    particles, _ = cold_plasma_shell(
        flat, "electrons", omega_p, charge=CHARGE_TO_MASS, mass=1.0,
        r_min=10.0, r_max=40.0, particles_per_cell=16, epsilon_0=epsilon_0,
        ur=drift,
    )
    dt, steps = 0.05, 400
    probe = int(jnp.argmin(jnp.abs(radius - 25.0)))

    @partial(jax.jit, static_argnames=("count",))
    def evolve(species, fields, count):
        def step(state, index):
            species, fields = state
            species, _, _, _, fields = rk4_step(
                species, flat, dt, E_r=zeros, EM_on=False, GR_on=False,
                dark_fields=fields, mu=0.0, sin_chi=1.0, epsilon_0=epsilon_0,
                time=index * dt,
            )
            return (species, fields), fields[0][probe]

        return jax.lax.scan(step, (species, fields), jnp.arange(count))

    _, sampled = evolve(particles, (zeros,) * 3, steps)

    times = (np.arange(steps) + 1) * dt
    measured = np.asarray(sampled)
    analytic = -(omega_p * drift / CHARGE_TO_MASS) * np.sin(omega_p * times)
    amplitude = omega_p * abs(drift / CHARGE_TO_MASS)

    np.testing.assert_allclose(np.max(np.abs(measured)), amplitude, rtol=1.0e-4)
    np.testing.assert_allclose(measured, analytic, atol=1.0e-3 * amplitude)

    crossings = np.flatnonzero(np.diff(np.sign(measured)) != 0)
    assert crossings.size > 4
    interpolated = times[crossings] - measured[crossings] * (
        times[crossings + 1] - times[crossings]
    ) / (measured[crossings + 1] - measured[crossings])
    period = 2.0 * np.mean(np.diff(interpolated))
    np.testing.assert_allclose(2.0 * np.pi / period, omega_p, rtol=1.0e-4)
