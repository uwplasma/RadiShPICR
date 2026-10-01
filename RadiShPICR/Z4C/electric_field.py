import jax.numpy as jnp

from RadiShPICR.particles.shape_factors.common import (
    proper_radial_shell_volume,
)
from RadiShPICR.particles.shape_factors import (
    particle_deposition_stencil,
)
from RadiShPICR.particles.shape_factors.cartesian_shapes import (
    _cell_centered_radial_shape_stencil,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    MatterTerms,
)
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def compute_radial_charge_density(
    particles,
    metric: Z4C_Metric,
    inner_open=False,
):
    """Deposit Eulerian charge density on the cell-centered radial grid.

    The compact stencil discards support beyond the outer grid.  With
    ``inner_open=True`` it also discards, without renormalization, support
    below the instantaneous areal-radius minimum.  Exact charge conservation
    therefore requires every particle's complete shape support to remain in
    the radial domain.
    """

    indices, even_weights, _ = particle_deposition_stencil(
        particles,
        metric,
        inner_open,
    )

    deposited_charge = jnp.zeros_like(metric.r).at[indices].add(
        even_weights * particles.get_charge()[jnp.newaxis, :]
    )

    return deposited_charge / proper_radial_shell_volume(metric)


def _fields_at_radial_faces(field):
    """Interpolate a cell-centered metric field to radial cell faces."""

    interior_faces = 0.5 * (field[:-1] + field[1:])
    return jnp.concatenate((field[:1], interior_faces, field[-1:]))


def solve_radial_electric_field(
    metric: Z4C_Metric,
    charge_density,
    epsilon_0=1.0,
):
    """Solve spherical Gauss law for the covariant radial field ``E_r``.

    Charge is accumulated through proper finite-volume shells.  The enclosed
    charge fixes the physical normal field at every spherical face, with zero
    flux at the origin.  Adjacent face values are averaged back to the Z4C
    cell centers.
    """

    proper_shell_volume = proper_radial_shell_volume(metric)
    shell_charge = charge_density * proper_shell_volume
    enclosed_charge = jnp.concatenate(
        (
            jnp.zeros_like(shell_charge[:1]),
            jnp.cumsum(shell_charge),
        )
    )

    inner_face = jnp.maximum(metric.r[:1] - 0.5 * metric.dr, 0.0)
    face_radius = jnp.concatenate((inner_face, metric.r + 0.5 * metric.dr))
    conformal_grr_face = _fields_at_radial_faces(metric.conformal_grr)
    conformal_gt_face = _fields_at_radial_faces(metric.conformal_gt)
    chi_face = _fields_at_radial_faces(metric.chi)

    physical_face_area = (
        4.0 * jnp.pi * face_radius**2 * conformal_gt_face / chi_face
    )
    nonzero_area = physical_face_area > 0.0
    safe_face_area = jnp.where(nonzero_area, physical_face_area, 1.0)
    normal_electric_field = jnp.where(
        nonzero_area,
        enclosed_charge / (epsilon_0 * safe_face_area),
        0.0,
    )
    covariant_electric_field_face = (
        jnp.sqrt(conformal_grr_face / chi_face) * normal_electric_field
    )

    return 0.5 * (
        covariant_electric_field_face[:-1]
        + covariant_electric_field_face[1:]
    )


def compute_radial_lorentz_force(particles, metric: Z4C_Metric, E_r):
    """Return the electrostatic contribution to ``du_r / dt``.

    Lapse and electric field use ordinary coordinate-space compact weights.
    The lapse has even origin parity and ``E_r`` has odd parity. Quadratic
    charge deposition uses metric-corrected weights, so its weights differ
    from this gather even when both stencils have the same support.
    """

    radial_positions, _ = particles.get_positions()
    indices, even_weights, odd_weights, _ = (
        _cell_centered_radial_shape_stencil(
            radial_positions,
            metric.r,
            metric.dr,
            particles.get_shape(),
        )
    )
    lapse_at_particle = jnp.sum(metric.alpha[indices] * even_weights, axis=0)
    electric_field_at_particle = jnp.sum(E_r[indices] * odd_weights, axis=0)

    charge, mass, weight = jnp.broadcast_arrays(
        particles.charges,
        particles.masses,
        particles.weight,
    )
    active_particle = jnp.logical_and(mass != 0.0, weight != 0.0)
    safe_mass = jnp.where(active_particle, mass, 1.0)
    charge_to_mass = jnp.where(active_particle, charge / safe_mass, 0.0)

    return lapse_at_particle * charge_to_mass * electric_field_at_particle


def compute_electrostatic_matter_terms(
    metric: Z4C_Metric,
    E_r,
    epsilon_0=1.0,
):
    """Return the Eulerian stress-energy of a radial electric field.

    ``E_r`` is the covariant radial field.  A purely radial electrostatic
    field has radial tension, equal tangential pressures, and no Poynting
    momentum density.
    """

    inverse_radial_metric = metric.chi / metric.conformal_grr
    rho = 0.5 * epsilon_0 * inverse_radial_metric * E_r**2
    Srr = -0.5 * epsilon_0 * E_r**2
    Stt = (
        0.5
        * epsilon_0
        * metric.conformal_gt
        / metric.conformal_grr
        * E_r**2
    )
    zeros = jnp.zeros_like(E_r)

    return MatterTerms(
        rho=rho,
        Srr=Srr,
        Stt=Stt,
        Sr=zeros,
        St=zeros,
    )


def electric_field_energy(metric: Z4C_Metric, E_r, epsilon_0=1.0):
    """Return the total electrostatic energy on the radial slice."""

    inverse_radial_metric = metric.chi / metric.conformal_grr
    energy_density = 0.5 * epsilon_0 * inverse_radial_metric * E_r**2

    return jnp.sum(energy_density * proper_radial_shell_volume(metric))
