import jax.numpy as jnp
import numpy as np

from demos.oppenheimer_snyder_collapse_z4c import (
    run_oppenheimer_snyder_z4c as os_z4c,
)
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.geodesic import (
    compute_geodesic_terms as compute_constrained_geodesic_terms,
)
from RadiShPICR.Z4C.geodesic import (
    compute_geodesic_terms as compute_z4c_geodesic_terms,
    isotropic_particle_state,
    lapse_freezing_particle_state,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    _proper_radial_shell_volume,
    compute_radial_matter_terms,
)
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.time_evolve import metric_time_derivatives
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import interpolate_fields_to_particles


def _flat_metric():
    r = 0.5 * jnp.arange(1, 17)
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)

    return Z4C_Metric(
        alpha=ones,
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=ones,
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Gamma=zeros,
        kappa=zeros,
        eta=zeros,
        nu=zeros,
        r=r,
        dr=r[1] - r[0],
    )


def _particles(metric):
    r = jnp.asarray([1.0, 2.0, 3.0])
    ur = jnp.asarray([0.2, -0.1, 0.05])
    rs, ubar = lapse_freezing_particle_state(r, ur, metric, "nearest")

    return particle_species(
        name="dust",
        charge=0.0,
        mass=1.0,
        weight=jnp.ones_like(r),
        r=rs,
        ur=ubar,
        phi=jnp.zeros_like(r),
        uphi=jnp.zeros_like(r),
        shape_mode="nearest",
    )


def test_schwarzschild_rescaling_preserves_lapse_freezing_particle_state(
    monkeypatch,
):
    metric = _flat_metric()
    particles = _particles(metric)
    X_r = 4.0
    X_t = 2.0

    monkeypatch.setattr(
        os_z4c,
        "schwarzschild_rescale_factors_from_z4c",
        lambda metric, exterior_mass: (X_r, X_t, 0.0),
    )

    diagnostic_metric, diagnostic_particles, _, _, _ = (
        os_z4c.rescale_z4c_to_schwarzschild_coordinates(
            metric,
            particles,
            exterior_mass=1.0,
        )
    )
    diagnostic_r, diagnostic_ur = isotropic_particle_state(
        diagnostic_particles,
        diagnostic_metric,
    )

    assert jnp.allclose(diagnostic_particles.r, particles.r)
    assert jnp.allclose(diagnostic_particles.ur, particles.ur / X_t)
    assert jnp.allclose(diagnostic_r, jnp.asarray([1.0, 2.0, 3.0]) * X_r)
    assert jnp.allclose(diagnostic_ur, jnp.asarray([0.2, -0.1, 0.05]) / X_r)


def test_z4c_snapshot_writes_solver_and_common_comparison_variables(
    monkeypatch,
    tmp_path,
):
    metric = _flat_metric()
    metric = metric._replace(alpha=0.8 * metric.alpha)
    particles = _particles(metric)
    metric_directory = tmp_path / "metric"
    phase_space_directory = tmp_path / "phase_space"
    metric_directory.mkdir()
    phase_space_directory.mkdir()

    monkeypatch.setattr(
        os_z4c,
        "schwarzschild_rescale_factors_from_z4c",
        lambda metric, exterior_mass: (1.0, 1.0, 0.0),
    )

    os_z4c.write_schwarzschild_snapshot(
        metric,
        particles,
        metric_directory,
        phase_space_directory,
        step=0,
        schwarzschild_time=0.0,
    )

    snapshot_path = phase_space_directory / "phase_space_dust_step_000000.npz"
    with np.load(snapshot_path) as snapshot:
        assert {
            "rs",
            "ubar",
            "r",
            "ur",
            "areal_radius",
            "radial_orthonormal_momentum",
        }.issubset(snapshot.files)
        assert str(snapshot["particle_state_variables"]) == "rs_ubar"
        assert str(snapshot["particle_ubar_definition"]) == (
            "alpha*ur*sqrt(chi/conformal_grr)"
        )
        assert np.allclose(snapshot["rs"], particles.r)
        assert np.allclose(snapshot["ubar"], particles.ur)
        assert np.allclose(snapshot["rs"], snapshot["areal_radius"])
        assert np.allclose(snapshot["areal_radius"], snapshot["r"])
        assert np.allclose(
            snapshot["radial_orthonormal_momentum"],
            snapshot["ur"],
        )


def test_reduced_os_initial_state_deposits_total_particle_energy():
    metric, particles, constrained_U_state, _, total_rest_mass = (
        os_z4c.build_initial_state(
            r_max=20.0,
            num_z4c_cells=199,
            particles_per_shell=1,
            shooting_iterations=1,
        )
    )

    matter_terms = compute_radial_matter_terms(particles, metric)
    proper_shell_volume = _proper_radial_shell_volume(metric)
    deposited_energy = jnp.sum(matter_terms.rho * proper_shell_volume)
    particle_energy = jnp.sum(particles.get_mass())

    assert jnp.allclose(particle_energy, total_rest_mass)
    assert jnp.allclose(deposited_energy, particle_energy)
    assert jnp.max(matter_terms.rho) > 1.0e-4

    r_particle, ur = isotropic_particle_state(particles, metric)
    A_at_particle = jnp.interp(
        r_particle,
        constrained_U_state[-1],
        constrained_U_state[0],
    )
    constrained_particles = particle_species(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=particles.r,
        ur=ur / A_at_particle,
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )
    _, _, constrained_dubar_dt = compute_constrained_geodesic_terms(
        constrained_particles,
        constrained_U_state,
    )
    metric_derivative = metric_time_derivatives(metric, matter_terms)
    z4c_dubar_dt, _, _, _ = compute_z4c_geodesic_terms(
        particles,
        metric,
        metric_derivative,
    )
    alpha_p, dalpha_dt_p = interpolate_fields_to_particles(
        jnp.stack((metric.alpha, metric_derivative.alpha)),
        r_particle,
        RadialGrid(metric.r, metric.r, metric.dr, metric.r[-1]),
        shape_mode=particles.get_shape(),
    )
    expected_z4c_dubar_dt = (
        alpha_p * constrained_dubar_dt
        + dalpha_dt_p * particles.ur / alpha_p
    )

    inner_particles = r_particle < metric.r[0]
    relative_discrepancy = jnp.abs(
        (z4c_dubar_dt - expected_z4c_dubar_dt)
        / expected_z4c_dubar_dt
    )

    assert jnp.any(inner_particles)
    assert jnp.max(relative_discrepancy[inner_particles]) < 1.0e-3
