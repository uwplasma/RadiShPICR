"""Time-symmetric star data with the Z4C particle sources."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.electric_field import (
    compute_radial_charge_density,
    solve_radial_electric_field,
)
from RadiShPICR.Z4C.energy_momentum_tensor import compute_radial_matter_terms


def _radial_laplacian(r, dr):
    """Match the Z4C derivative stencils, omitting the final boundary row."""

    num_cells = r.size
    rows = np.repeat(np.arange(num_cells - 2), 5)
    columns = rows + np.tile(np.arange(-2, 3), num_cells - 2)
    # Even parity on a half-cell-centered grid: -1 -> 0 and -2 -> 1.
    columns = np.where(columns < 0, -columns - 1, columns)
    d1 = np.asarray((1.0, -8.0, 0.0, 8.0, -1.0)) / (12.0 * dr)
    d2 = np.asarray((-1.0, 16.0, -30.0, 16.0, -1.0)) / (12.0 * dr**2)
    values = (d2 + 2.0 * d1 / r[:-2, None]).ravel()

    # The penultimate row uses the same one-sided derivatives as evolution.
    d1_outer = np.asarray((0.0, -1/12, 1/2, -3/2, 5/6, 1/4)) / dr
    d2_outer = np.asarray((1/12, -1/2, 7/6, -1/3, -5/4, 5/6)) / dr**2
    rows = np.concatenate((rows, np.full(6, num_cells - 2)))
    columns = np.concatenate((columns, np.arange(num_cells - 6, num_cells)))
    values = np.concatenate((values, d2_outer + 2.0 * d1_outer / r[-2]))

    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(num_cells - 1, num_cells),
    )


def solve_star_initial_data(
    particles,
    metric,
    total_mass,
    *,
    epsilon_0=1.0,
    inner_open=True,
    tolerance=1.0e-10,
    max_iterations=100,
):
    """Solve charged, conformally flat, time-symmetric data on the Z4C grid.

    Input particles store *areal* radii and must have zero momenta, as returned
    by ``initialize_oppenheimer_snyder_particles``. The supplied metric gives
    the cell-centered grid, lapse, gauge parameters, and initial chi guess.
    Its lapse and gauge parameters are preserved; shift and extrinsic
    curvature start at zero, with unit conformal metric.

    Each nonlinear iteration inverts the completed map ``R = r * psi**2``
    before using the production Z4C matter/charge deposition and Gauss solve.
    The Hamiltonian equation is ``laplacian(psi) = -2*pi*psi**5*rho``. Two
    exterior RN values fix both radial normalization and gravitational mass.
    The additional unknown is a uniform particle rest-mass factor: charges,
    weights, and particle areal radii remain fixed. This chooses initial data
    with the requested mass rather than fitting the evolved black hole.

    Returns ``(metric, particles, E_r)`` with ordinary isotropic-r Z4C particle
    storage. The final two grid cells must be in the matter-free exterior.
    """

    if np.any(np.asarray(particles.ur) != 0.0) or np.any(np.asarray(particles.uphi) != 0.0):
        raise ValueError("Star time-symmetric initial data require zero particle momenta.")

    r = np.asarray(metric.r)
    dr = float(metric.dr)
    num_cells = r.size
    particle_R = np.asarray(particles.r)[np.asarray(particles.weight) != 0.0]
    total_charge = float(jnp.sum(particles.get_charge()))
    charge_radius_squared = total_charge**2 / (4.0 * np.pi * epsilon_0)
    rn_psi = np.sqrt(
        (1.0 + total_mass / (2.0 * r[-2:]))**2
        - charge_radius_squared / (4.0 * r[-2:]**2)
    )

    zeros = jnp.zeros_like(metric.r)
    ones = jnp.ones_like(metric.r)
    metric = metric._replace(
        beta=zeros, conformal_grr=ones, conformal_gt=ones,
        Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
    )

    @jax.jit
    def source_terms(psi):
        trial_metric = metric._replace(chi=psi**-4)
        areal_grid = metric.r * psi**2
        particle_r = jnp.interp(
            particles.r,
            jnp.concatenate((zeros[:1], areal_grid)),
            jnp.concatenate((zeros[:1], metric.r)),
        )
        trial_particles = particle_species(
            particles.name, particles.charges, particles.masses,
            particles.weight, particle_r, particles.ur,
            particles.phi, particles.uphi, particles.shape_mode,
        )
        rho = compute_radial_matter_terms(
            trial_particles, trial_metric, inner_open=inner_open,
        ).rho
        charge_density = compute_radial_charge_density(
            trial_particles, trial_metric, inner_open=inner_open,
        )
        E_r = solve_radial_electric_field(trial_metric, charge_density, epsilon_0)
        rho_EM = 0.5 * epsilon_0 * trial_metric.chi * E_r**2

        return rho, rho_EM, E_r, particle_r

    laplacian = _radial_laplacian(r, dr)
    exterior_values = sparse.csr_matrix(
        (np.ones(2), (np.arange(2), np.arange(num_cells - 2, num_cells))),
        shape=(2, num_cells),
    )
    psi = np.asarray(metric.chi)**-0.25
    mass_factor = 1.0

    for _ in range(max_iterations):
        areal_grid = r * psi**2
        if (not np.all(np.isfinite(psi)) or np.any(psi <= 0.0)
                or np.any(np.diff(areal_grid) <= 0.0)
                or np.any(particle_R > areal_grid[-3])):
            raise RuntimeError("Star initial data require a monotone map and a matter-free outer boundary.")
        rho, rho_EM, _, _ = source_terms(jnp.asarray(psi))
        dust_source = 2.0 * np.pi * psi**5 * np.asarray(rho)
        electric_source = 2.0 * np.pi * psi**5 * np.asarray(rho_EM)

        # The dust factor is solved with psi. Keeping charges independent
        # preserves Q while accounting for electric energy in the target M.
        operator = sparse.bmat(
            [[laplacian, sparse.csr_matrix(dust_source[:-1, None])],
             [exterior_values, sparse.csr_matrix((2, 1))]],
            format="csr",
        )
        rhs = np.concatenate((-electric_source[:-1], rn_psi))
        solution = spsolve(operator, rhs)
        new_psi, new_mass_factor = solution[:-1], solution[-1]
        change = max(np.max(np.abs(new_psi - psi)), abs(new_mass_factor - mass_factor))
        psi = 0.5 * (psi + new_psi)
        mass_factor = 0.5 * (mass_factor + new_mass_factor)
        if change <= tolerance:
            break
    else:
        raise RuntimeError("Star Hamiltonian solve did not converge.")

    if not np.isfinite(mass_factor) or mass_factor <= 0.0:
        raise RuntimeError("Requested stellar mass requires nonpositive or nonfinite rest mass.")

    _, _, E_r, particle_r = source_terms(jnp.asarray(psi))
    metric = metric._replace(chi=jnp.asarray(psi)**-4)
    particles = particle_species(
        particles.name, particles.charges, particles.masses * mass_factor,
        particles.weight, particle_r, particles.ur,
        particles.phi, particles.uphi, particles.shape_mode,
    )

    return metric, particles, E_r


# Compatibility for callers of the original charged-only name.
solve_charged_star_initial_data = solve_star_initial_data
