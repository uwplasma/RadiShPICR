import jax.numpy as jnp
import numpy as np

from demos.oppenheimer_snyder_collapse_z4c import (
    run_oppenheimer_snyder_z4c as os_z4c,
)
from RadiShPICR.ConstraintBasedRelativity.geodesic import (
    compute_geodesic_terms as compute_constrained_geodesic_terms,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    _proper_radial_shell_volume,
    compute_radial_matter_terms,
)
from RadiShPICR.Z4C.geodesic import compute_geodesic_terms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles import particle_species


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

    return particle_species(
        name="dust",
        charge=0.0,
        mass=1.0,
        weight=jnp.ones_like(r),
        r=r,
        ur=ur,
        phi=jnp.zeros_like(r),
        uphi=jnp.zeros_like(r),
        shape_mode="nearest",
    )


def test_schwarzschild_rescaling_transforms_standard_particle_state(monkeypatch):
    metric = _flat_metric()
    particles = _particles(metric)
    X_r = 4.0
    X_t = 2.0

    monkeypatch.setattr(
        os_z4c,
        "schwarzschild_rescale_factors_from_z4c",
        lambda metric, exterior_mass: (X_r, X_t, 0.0),
    )

    _, diagnostic_particles, _, _, _ = (
        os_z4c.rescale_z4c_to_schwarzschild_coordinates(
            metric,
            particles,
            exterior_mass=1.0,
        )
    )

    assert jnp.allclose(diagnostic_particles.r, particles.r * X_r)
    assert jnp.allclose(diagnostic_particles.ur, particles.ur / X_r)
    assert jnp.allclose(diagnostic_particles.phi, particles.phi)
    assert jnp.allclose(diagnostic_particles.uphi, particles.uphi)


def test_z4c_snapshot_writes_clean_standard_particle_schema(
    monkeypatch,
    tmp_path,
):
    metric = _flat_metric()
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
            "r",
            "ur",
            "phi",
            "uphi",
            "weight",
            "areal_radius",
            "radial_orthonormal_momentum",
        }.issubset(snapshot.files)
        assert "rs" not in snapshot.files
        assert "ubar" not in snapshot.files
        assert "particle_ubar_definition" not in snapshot.files
        assert str(snapshot["particle_state_variables"]) == "r_ur"
        assert np.allclose(snapshot["r"], particles.r)
        assert np.allclose(snapshot["ur"], particles.ur)
        assert np.allclose(snapshot["areal_radius"], particles.r)
        assert np.allclose(
            snapshot["radial_orthonormal_momentum"],
            particles.ur,
        )


def test_reduced_os_initial_state_uses_standard_particles_and_deposits_energy():
    (
        metric,
        particles,
        constrained_U_state,
        particle_areal_radius,
        total_rest_mass,
    ) = os_z4c.build_initial_state(
        r_max=20.0,
        num_z4c_cells=199,
        particles_per_shell=1,
        shooting_iterations=1,
    )

    matter_terms = compute_radial_matter_terms(particles, metric)
    proper_shell_volume = _proper_radial_shell_volume(metric)
    deposited_energy = jnp.sum(matter_terms.rho * proper_shell_volume)
    particle_energy = jnp.sum(particles.get_mass())

    assert jnp.all(jnp.isfinite(particles.r))
    assert jnp.all(jnp.isfinite(particles.ur))
    assert jnp.all(particles.r >= 0.0)
    assert jnp.allclose(particle_energy, total_rest_mass)
    assert jnp.allclose(deposited_energy, particle_energy)
    assert jnp.max(matter_terms.rho) > 1.0e-4

    A_at_particle = jnp.interp(
        particles.r,
        constrained_U_state[-1],
        constrained_U_state[0],
    )
    constrained_particles = particle_species(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=particle_areal_radius,
        ur=particles.ur / A_at_particle,
        phi=particles.phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )
    _, _, constrained_du_r_over_A_dt = compute_constrained_geodesic_terms(
        constrained_particles,
        constrained_U_state,
    )
    z4c_du_r_dt, _, _, _ = compute_geodesic_terms(particles, metric)
    expected_z4c_du_r_dt = A_at_particle * constrained_du_r_over_A_dt

    inner_particles = particles.r < metric.r[0]
    relative_discrepancy = jnp.abs(
        (z4c_du_r_dt - expected_z4c_du_r_dt)
        / expected_z4c_du_r_dt
    )

    assert jnp.any(inner_particles)
    assert jnp.max(relative_discrepancy[inner_particles]) < 1.0e-3


def test_freefall_time_step_does_not_restrict_vacuum_state():
    metric = _flat_metric()
    particles = _particles(metric)
    particles.weight = jnp.zeros_like(particles.weight)

    assert np.isinf(os_z4c.freefall_collapse_time_step(particles, metric))
