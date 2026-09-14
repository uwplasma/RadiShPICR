import jax
import jax.numpy as jnp

from RadiShPICR.particles.shape_factors.common import (
    proper_radial_shell_volume,
    proper_radial_shell_quadrature,
)
from RadiShPICR.particles.shape_factors.metric_correct_quadratic import (
    raw_quadratic_stencil,
)
from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.electric_field import (
    compute_radial_charge_density,
    compute_radial_lorentz_force,
    electric_field_energy,
    solve_radial_electric_field,
)
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _metric(
    num_cells=16,
    dr=0.25,
    alpha=1.0,
    conformal_grr=1.0,
    conformal_gt=1.0,
    chi=1.0,
):
    r = (jnp.arange(num_cells) + 0.5) * dr
    zeros = jnp.zeros_like(r)

    return Z4C_Metric(
        alpha=jnp.full_like(r, alpha),
        beta=zeros,
        conformal_grr=jnp.full_like(r, conformal_grr),
        conformal_gt=jnp.full_like(r, conformal_gt),
        chi=jnp.full_like(r, chi),
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Gamma=zeros,
        kappa=jnp.asarray(0.0),
        eta=jnp.asarray(0.0),
        nu=jnp.asarray(0.0),
        r=r,
        dr=jnp.asarray(dr),
    )


def _particles(metric, shape_mode, ur=None):
    radial_positions = jnp.asarray(
        [metric.r[0], metric.r[4], metric.r[9], metric.r[13]]
    )
    if ur is None:
        ur = jnp.zeros_like(radial_positions)

    return particle_species(
        name="charged_particles",
        charge=jnp.asarray([2.0, -1.5, 0.75, -0.5]),
        mass=jnp.ones_like(radial_positions),
        weight=jnp.asarray([0.2, 0.3, 0.4, 0.5]),
        r=radial_positions,
        ur=ur,
        phi=jnp.zeros_like(radial_positions),
        uphi=jnp.zeros_like(radial_positions),
        shape_mode=shape_mode,
    )


def test_charge_deposition_conserves_charge_for_all_particle_shapes():
    metric = _metric()
    proper_shell_volume = proper_radial_shell_volume(metric)

    for shape_mode in ("nearest", "linear", "quadratic"):
        particles = _particles(metric, shape_mode)
        charge_density = compute_radial_charge_density(particles, metric)
        compiled_charge_density = jax.jit(compute_radial_charge_density)(
            particles,
            metric,
        )

        deposited_charge = jnp.sum(charge_density * proper_shell_volume)
        particle_charge = jnp.sum(particles.get_charge())

        assert charge_density.shape == metric.r.shape
        assert jnp.allclose(deposited_charge, particle_charge)
        assert jnp.allclose(compiled_charge_density, charge_density)


def test_nearest_shape_conserves_charge_at_cell_ties_and_respects_origin_parity():
    metric = _metric()
    radial_positions = jnp.asarray(
        [0.0, metric.r[4] + 0.5 * metric.dr]
    )
    particles = particle_species(
        name="nearest_ties",
        charge=jnp.asarray([1.25, -0.5]),
        mass=jnp.ones_like(radial_positions),
        weight=jnp.asarray([0.4, 0.7]),
        r=radial_positions,
        ur=jnp.zeros_like(radial_positions),
        phi=jnp.zeros_like(radial_positions),
        uphi=jnp.zeros_like(radial_positions),
        shape_mode="nearest",
    )

    charge_density = compute_radial_charge_density(particles, metric)
    deposited_charge = jnp.sum(
        charge_density * proper_radial_shell_volume(metric)
    )
    radial_force = compute_radial_lorentz_force(
        particles,
        metric,
        jnp.ones_like(metric.r),
    )

    assert jnp.allclose(deposited_charge, jnp.sum(particles.get_charge()))
    assert radial_force[0] == 0.0
    assert radial_force[1] == particles.charges[1] / particles.masses[1]


def test_open_inner_charge_deposition_discards_the_ghost_shape_share():
    metric = _metric(num_cells=8, dr=1.0)
    particles = particle_species(
        name="open-inner",
        charge=2.0,
        mass=1.0,
        weight=0.4,
        r=jnp.asarray([0.0]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )
    proper_shell_volume = proper_radial_shell_volume(metric)

    parity_density = compute_radial_charge_density(particles, metric)
    open_density = compute_radial_charge_density(particles, metric, True)
    compiled_open_density = jax.jit(compute_radial_charge_density)(
        particles,
        metric,
        True,
    )

    parity_charge = jnp.sum(parity_density * proper_shell_volume)
    open_charge = jnp.sum(open_density * proper_shell_volume)
    raw_indices, corrected_weights = raw_quadratic_stencil(
        particles.r,
        metric,
        True,
    )
    retained_overlap = jnp.sum(
        jnp.where(raw_indices >= 0, corrected_weights, 0.0)
    )

    assert jnp.allclose(parity_charge, jnp.sum(particles.get_charge()))
    assert jnp.allclose(
        open_charge,
        retained_overlap * particles.get_charge()[0],
    )
    assert jnp.allclose(compiled_open_density, open_density)


def test_density_conserving_quadratic_charge_is_uniform_at_origin():
    metric = _metric(num_cells=32, dr=0.2)
    metric = metric._replace(
        conformal_grr=1.0 + 0.04 * metric.r**2,
        conformal_gt=1.0 + 0.03 * metric.r,
        chi=1.0 / (1.0 + 0.02 * metric.r**2),
    )
    expected_charge_density = 0.7
    quadrature_radius, quadrature_volume = proper_radial_shell_quadrature(
        metric
    )
    particles = particle_species(
        name="proper-volume charge quiet start",
        charge=1.0,
        mass=1.0,
        weight=expected_charge_density * quadrature_volume,
        r=quadrature_radius,
        ur=jnp.zeros_like(quadrature_radius),
        phi=jnp.zeros_like(quadrature_radius),
        uphi=jnp.zeros_like(quadrature_radius),
        shape_mode="quadratic",
    )

    for inner_open in (False, True):
        charge_density = compute_radial_charge_density(
            particles,
            metric,
            inner_open,
        )
        compiled_charge_density = jax.jit(compute_radial_charge_density)(
            particles,
            metric,
            inner_open,
        )

        assert jnp.allclose(
            charge_density[:12],
            expected_charge_density,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        assert jnp.allclose(compiled_charge_density, charge_density)


def test_charge_density_is_independent_of_particle_momentum():
    metric = _metric()
    stationary_particles = _particles(metric, "quadratic")
    moving_particles = _particles(
        metric,
        "quadratic",
        ur=jnp.asarray([0.2, -0.5, 1.1, -0.7]),
    )

    stationary_density = compute_radial_charge_density(
        stationary_particles,
        metric,
    )
    moving_density = compute_radial_charge_density(moving_particles, metric)

    assert jnp.allclose(moving_density, stationary_density)


def test_uniform_charge_sphere_matches_analytic_flat_field():
    metric = _metric(num_cells=32, dr=0.125)
    charge_density = jnp.full_like(metric.r, 0.7)
    epsilon_0 = 1.3

    E_r = solve_radial_electric_field(metric, charge_density, epsilon_0)
    compiled_E_r = jax.jit(solve_radial_electric_field)(
        metric,
        charge_density,
        epsilon_0,
    )
    expected_E_r = charge_density * metric.r / (3.0 * epsilon_0)

    assert E_r.shape == metric.r.shape
    assert jnp.allclose(E_r, expected_E_r)
    assert jnp.allclose(compiled_E_r, E_r)
    assert jnp.all(jnp.isfinite(E_r))


def test_uniform_charge_sphere_in_constant_curved_metric():
    metric = _metric(
        num_cells=24,
        dr=0.2,
        conformal_grr=1.4,
        conformal_gt=0.8,
        chi=0.7,
    )
    charge_density = jnp.full_like(metric.r, 0.35)

    E_r = solve_radial_electric_field(metric, charge_density)
    expected_E_r = (
        charge_density
        * metric.conformal_grr
        * metric.r
        / (3.0 * metric.chi)
    )

    assert jnp.allclose(E_r, expected_E_r)


def test_curved_finite_volume_gauss_residual_is_roundoff():
    metric = _metric(num_cells=40, dr=0.1)
    metric = metric._replace(
        conformal_grr=1.0 + 0.04 * metric.r**2,
        conformal_gt=1.0 + 0.03 * metric.r,
        chi=1.0 / (1.0 + 0.02 * metric.r**2),
    )
    charge_density = 0.2 * jnp.cos(0.7 * metric.r) - 0.03 * metric.r
    epsilon_0 = 1.7

    E_r = solve_radial_electric_field(metric, charge_density, epsilon_0)

    # The solver stores the average of adjacent covariant face fields.  Undo
    # that averaging from the regular zero-flux origin face, then evaluate the
    # same physical face flux used by the finite-volume Gauss law.
    covariant_face_field = [0.0]
    for centered_field in E_r:
        covariant_face_field.append(
            2.0 * float(centered_field) - covariant_face_field[-1]
        )
    covariant_face_field = jnp.asarray(covariant_face_field)

    def face_values(field):
        return jnp.concatenate(
            (field[:1], 0.5 * (field[:-1] + field[1:]), field[-1:])
        )

    face_radius = jnp.concatenate(
        (jnp.asarray([0.0]), metric.r + 0.5 * metric.dr)
    )
    conformal_grr_face = face_values(metric.conformal_grr)
    conformal_gt_face = face_values(metric.conformal_gt)
    chi_face = face_values(metric.chi)
    face_area = 4.0 * jnp.pi * face_radius**2 * conformal_gt_face / chi_face
    normal_face_field = covariant_face_field / jnp.sqrt(
        conformal_grr_face / chi_face
    )
    face_flux = epsilon_0 * face_area * normal_face_field

    shell_charge = charge_density * proper_radial_shell_volume(metric)
    gauss_residual = jnp.diff(face_flux) - shell_charge

    assert jnp.max(jnp.abs(gauss_residual)) < 2.0e-12


def test_lorentz_force_uses_odd_field_parity_and_raw_charge_to_mass():
    metric = _metric(num_cells=12, dr=0.5, alpha=0.8)
    E_r = 0.3 * metric.r
    radial_positions = jnp.asarray([metric.r[3], -metric.r[3], 0.0])
    particles = particle_species(
        name="lorentz",
        charge=jnp.asarray([2.0, 2.0, 2.0]),
        mass=jnp.asarray([4.0, 4.0, 4.0]),
        weight=jnp.asarray([0.2, 0.7, 1.1]),
        r=radial_positions,
        ur=jnp.zeros_like(radial_positions),
        phi=jnp.zeros_like(radial_positions),
        uphi=jnp.zeros_like(radial_positions),
        shape_mode="nearest",
    )

    force = compute_radial_lorentz_force(particles, metric, E_r)
    compiled_force = jax.jit(compute_radial_lorentz_force)(
        particles,
        metric,
        E_r,
    )
    positive_force = metric.alpha[3] * 0.5 * E_r[3]
    expected_force = jnp.asarray([positive_force, -positive_force, 0.0])

    assert jnp.allclose(force, expected_force)
    assert jnp.allclose(compiled_force, force)


def test_lorentz_gather_matches_charge_stencil_near_guarded_outer_edge():
    metric = _metric(num_cells=12, dr=0.5)
    E_r = 0.3 * metric.r
    radial_position = metric.r[-2]
    particles = particle_species(
        name="outer_guard",
        charge=jnp.asarray(2.0),
        mass=jnp.asarray(4.0),
        weight=jnp.asarray([1.0]),
        r=jnp.asarray([radial_position]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )

    force = compute_radial_lorentz_force(particles, metric, E_r)

    assert jnp.allclose(force, 0.5 * 0.3 * radial_position)


def test_lorentz_force_masks_zero_mass_and_inactive_particles():
    metric = _metric(num_cells=8, dr=0.5)
    E_r = jnp.ones_like(metric.r)
    particles = particle_species(
        name="inactive",
        charge=jnp.asarray([1.0, 1.0, -2.0]),
        mass=jnp.asarray([0.0, 2.0, 4.0]),
        weight=jnp.asarray([1.0, 0.0, 0.5]),
        r=metric.r[jnp.asarray([1, 2, 3])],
        ur=jnp.zeros(3),
        phi=jnp.zeros(3),
        uphi=jnp.zeros(3),
        shape_mode="linear",
    )

    force = compute_radial_lorentz_force(particles, metric, E_r)

    assert jnp.all(jnp.isfinite(force))
    assert jnp.allclose(force, jnp.asarray([0.0, 0.0, -0.5]))


def test_electric_field_energy_uses_physical_inverse_radial_metric():
    metric = _metric(
        num_cells=10,
        dr=0.2,
        conformal_grr=1.25,
        conformal_gt=0.9,
        chi=0.8,
    )
    E_r = 0.1 + 0.05 * metric.r
    proper_shell_volume = proper_radial_shell_volume(metric)
    epsilon_0 = 1.6
    expected_energy = jnp.sum(
        0.5
        * epsilon_0
        * metric.chi
        / metric.conformal_grr
        * E_r**2
        * proper_shell_volume
    )

    energy = electric_field_energy(metric, E_r, epsilon_0)
    compiled_energy = jax.jit(electric_field_energy)(metric, E_r, epsilon_0)

    assert energy.shape == ()
    assert jnp.allclose(energy, expected_energy)
    assert jnp.allclose(compiled_energy, energy)


def test_electrostatic_matter_terms_have_radial_tension_and_zero_momentum():
    from RadiShPICR.Z4C.electric_field import (
        compute_electrostatic_matter_terms,
    )

    metric = _metric(
        num_cells=10,
        dr=0.2,
        conformal_grr=1.25,
        conformal_gt=0.9,
        chi=0.8,
    )
    E_r = 0.1 + 0.05 * metric.r
    epsilon_0 = 1.6

    matter = jax.jit(compute_electrostatic_matter_terms)(
        metric,
        E_r,
        epsilon_0,
    )
    expected_rho = (
        0.5 * epsilon_0 * metric.chi / metric.conformal_grr * E_r**2
    )
    stress_trace = (
        metric.chi / metric.conformal_grr * matter.Srr
        + 2.0 * metric.chi / metric.conformal_gt * matter.Stt
    )

    assert jnp.allclose(matter.rho, expected_rho)
    assert jnp.allclose(matter.Srr, -0.5 * epsilon_0 * E_r**2)
    assert jnp.allclose(
        matter.Stt,
        0.5
        * epsilon_0
        * metric.conformal_gt
        / metric.conformal_grr
        * E_r**2,
    )
    assert jnp.allclose(stress_trace, matter.rho)
    assert jnp.allclose(matter.Sr, 0.0)
    assert jnp.allclose(matter.St, 0.0)
