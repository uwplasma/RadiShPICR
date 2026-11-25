"""Cold plasma shells specified by their plasma frequency.

In geometrized units an electron-scale plasma has a brutal dynamic range:
``e_geom ~ 1.4e-36 m`` and ``m_e,geom ~ 6.8e-58 m``, so the charge-to-mass
ratio is ~2e21 while the deposited charge is ~1e-36. Only the combination

    omega_p**2 = q**2 n / (m epsilon_0)

is an O(1) code-unit quantity. Carrying ``q`` and ``m`` separately spreads a
physically tame plasma over ~45 decades, which costs relative precision in the
fourth-order stencils and makes ``dark_gauss_residual`` a difference of wildly
mismatched magnitudes.

Rescaling the fields by ``lambda = q/m`` fixes this. The pusher loses its
prefactor, so ``charges/masses`` becomes ``alpha_s = lambda_s / lambda``
(``-1`` for the reference electrons, ``+1/1836`` for protons against an
electron reference), and the current deposited for the Proca source becomes
``omega_p**2 v``. The rescaling is absorbed entirely into the macro-particle
weight, so the deposition, the Gauss solve and the constraint residuals are
untouched and keep their existing meaning.

Linear response fixes the weight. For ``N`` macro-particles of code charge
``c`` and code mass ``m`` in proper volume ``V``,

    omega_p**2 = c**2 weight N / (epsilon_0 m V)
    weight     = epsilon_0 omega_p**2 m V / (c**2 N)

and then, identically,

    charge density    = epsilon_0 omega_p**2 / (c/m)
    rest-mass density = epsilon_0 omega_p**2 / (c/m)**2

The rest-mass density is *not* a free parameter: once ``omega_p``, ``c/m`` and
``epsilon_0`` are chosen, the plasma's self-gravity follows. ``epsilon_0`` is
the dial that sets how strongly the plasma gravitates at fixed ``omega_p``.
"""

import numpy as np

import jax
import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume


def plasma_macroparticle_weight(
    plasma_frequency, charge, mass, epsilon_0, proper_volume,
    particles_per_cell,
):
    """Return the weight realizing ``plasma_frequency`` in ``proper_volume``.

    Inverts the cold linear response ``omega_p**2 = c**2 w N /
    (epsilon_0 m V)``. ``proper_volume`` may be an array, giving one weight
    per shell.
    """

    return (
        epsilon_0 * plasma_frequency**2 * mass * proper_volume
        / (charge**2 * particles_per_cell)
    )


def _validate(plasma_frequency, charge, mass, epsilon_0, particles_per_cell,
              r_min, r_max, outer_radius):
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "Plasma weights span many decades; enable jax_enable_x64 first."
        )
    if epsilon_0 <= 0.0:
        raise ValueError("epsilon_0 must be positive.")
    if plasma_frequency <= 0.0:
        raise ValueError("plasma_frequency must be positive.")
    if charge == 0.0:
        raise ValueError("A plasma species needs a nonzero charge.")
    if mass <= 0.0:
        # compute_radial_lorentz_force masks mass == 0 to zero force, so a
        # massless species would silently never feel the field.
        raise ValueError("mass must be positive.")
    if particles_per_cell < 1:
        raise ValueError("particles_per_cell must be at least one.")
    if not 0.0 <= r_min < r_max:
        raise ValueError("Require 0 <= r_min < r_max.")
    if r_max > outer_radius:
        raise ValueError("r_max lies outside the radial grid.")


def cold_plasma_shell(
    metric, name, plasma_frequency, charge, mass, r_min, r_max,
    particles_per_cell, shape_mode="quadratic", epsilon_0=1.0,
    ur=0.0, uphi=0.0,
):
    """Build a cold shell realizing ``plasma_frequency`` locally.

    ``charge`` and ``mass`` are code units, so ``charge / mass`` is the
    rescaled ``alpha_s`` rather than a geometrized charge-to-mass ratio: use
    ``charge=-1.0, mass=1.0`` for the reference electrons and
    ``charge=+1.0, mass=1836.0`` for protons measured against them.

    Weights are set per shell from that shell's *proper* volume, so the
    realized plasma frequency is uniform even where the metric varies sharply.
    A single global volume would give uniform total charge but a radially
    varying ``omega_p``.

    Returns ``(particles, diagnostics)``. ``omega_p`` is uniform only in the
    interior: the two edge shells lose shape support outward, so analyze a
    strictly interior annulus.
    """

    radius = np.asarray(metric.r)
    _validate(plasma_frequency, charge, mass, epsilon_0, particles_per_cell,
              r_min, r_max, float(radius[-1] + 0.5 * float(metric.dr)))

    dr = float(metric.dr)
    selected = np.flatnonzero((radius >= r_min) & (radius < r_max))
    if selected.size == 0:
        raise ValueError("No grid cell lies in [r_min, r_max).")

    shell_volume = np.asarray(proper_radial_shell_volume(metric))[selected]
    shell_weight = plasma_macroparticle_weight(
        plasma_frequency, charge, mass, epsilon_0, shell_volume,
        particles_per_cell,
    )

    # Sample uniformly in coordinate volume inside each cell, matching the
    # r**3 factor the deposition divides by, so the quiet start is uniform in
    # the same measure rather than merely uniform in r.
    inner = np.maximum(radius[selected] - 0.5 * dr, 0.0)
    outer = radius[selected] + 0.5 * dr
    fraction = (np.arange(particles_per_cell) + 0.5) / particles_per_cell
    positions = np.cbrt(
        inner[:, None] ** 3
        + fraction[None, :] * (outer[:, None] ** 3 - inner[:, None] ** 3)
    ).ravel()
    weights = np.repeat(shell_weight, particles_per_cell)

    particles = particle_species(
        name=name,
        charge=float(charge),
        mass=float(mass),
        weight=jnp.asarray(weights),
        r=jnp.asarray(positions),
        ur=jnp.full(positions.shape, float(ur)),
        phi=jnp.zeros(positions.shape),
        uphi=jnp.full(positions.shape, float(uphi)),
        shape_mode=shape_mode,
        plasma_frequency=float(plasma_frequency),
    )

    charge_to_mass = charge / mass
    diagnostics = {
        "plasma_frequency": float(plasma_frequency),
        "charge_to_mass": float(charge_to_mass),
        "num_particles": int(positions.size),
        "shell_proper_volume": float(shell_volume.sum()),
        "total_rest_mass": float(mass * weights.sum()),
        "total_charge": float(charge * weights.sum()),
        "rest_mass_density": float(
            epsilon_0 * plasma_frequency**2 / charge_to_mass**2
        ),
        "charge_density": float(
            epsilon_0 * plasma_frequency**2 / charge_to_mass
        ),
    }

    return particles, diagnostics


def neutral_plasma_shell(
    metric, plasma_frequency, r_min, r_max, particles_per_cell,
    mass_ratio=1836.0, charge_to_mass=-1.0, shape_mode="quadratic",
    epsilon_0=1.0, name="plasma",
):
    """Build a charge-neutral electron/ion pair sharing one shell.

    Without the neutralizing species the shell carries a large net monopole,
    which dominates any kinetic-mixing signal. Both species occupy the same
    positions, so the deposited charge density cancels cell by cell.

    ``plasma_frequency`` is the *electron* frequency; the pair responds at
    ``omega_p**2 (1 + 1/mass_ratio)``.
    """

    electrons, electron_diagnostics = cold_plasma_shell(
        metric, f"{name}_electrons", plasma_frequency,
        charge=charge_to_mass, mass=1.0, r_min=r_min, r_max=r_max,
        particles_per_cell=particles_per_cell, shape_mode=shape_mode,
        epsilon_0=epsilon_0,
    )
    # Equal and opposite charge at mass_ratio times the mass: the ion weight
    # comes out equal to the electron weight, so the pair cancels exactly.
    ions, ion_diagnostics = cold_plasma_shell(
        metric, f"{name}_ions", plasma_frequency / np.sqrt(mass_ratio),
        charge=-charge_to_mass, mass=mass_ratio, r_min=r_min, r_max=r_max,
        particles_per_cell=particles_per_cell, shape_mode=shape_mode,
        epsilon_0=epsilon_0,
    )
    combined = particle_species(
        name=name,
        charge=jnp.concatenate((
            jnp.full(electrons.r.shape, float(charge_to_mass)),
            jnp.full(ions.r.shape, -float(charge_to_mass)),
        )),
        mass=jnp.concatenate((
            jnp.ones(electrons.r.shape), jnp.full(ions.r.shape, mass_ratio),
        )),
        weight=jnp.concatenate((electrons.weight, ions.weight)),
        r=jnp.concatenate((electrons.r, ions.r)),
        ur=jnp.concatenate((electrons.ur, ions.ur)),
        phi=jnp.concatenate((electrons.phi, ions.phi)),
        uphi=jnp.concatenate((electrons.uphi, ions.uphi)),
        shape_mode=shape_mode,
        plasma_frequency=jnp.concatenate((
            electrons.plasma_frequency, ions.plasma_frequency,
        )),
    )
    diagnostics = {
        "electrons": electron_diagnostics,
        "ions": ion_diagnostics,
        "pair_plasma_frequency": float(
            plasma_frequency * np.sqrt(1.0 + 1.0 / mass_ratio)
        ),
        "num_particles": int(combined.r.size),
    }

    return combined, diagnostics
