import jax.numpy as jnp
import numpy as np

from demos.oppenheimer_snyder_collapse_z4c import (
    run_oppenheimer_snyder_z4c as os_z4c,
)
from RadiShPICR.Z4C.geodesic import (
    lapse_freezing_particle_state,
    normal_particle_state,
)
from RadiShPICR.Z4C.energy_momentum_tensor import (
    _proper_radial_shell_volume,
    compute_radial_matter_terms,
)
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
    diagnostic_r, diagnostic_ur = normal_particle_state(
        diagnostic_particles,
        diagnostic_metric,
    )

    assert jnp.allclose(diagnostic_particles.r, particles.r / jnp.sqrt(X_r))
    assert jnp.allclose(diagnostic_particles.ur, particles.ur)
    assert jnp.allclose(diagnostic_r, jnp.asarray([1.0, 2.0, 3.0]) * X_r)
    assert jnp.allclose(diagnostic_ur, jnp.asarray([0.2, -0.1, 0.05]) / X_r)


def test_z4c_snapshot_writes_solver_and_common_comparison_variables(
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
            "rs",
            "ubar",
            "r",
            "ur",
            "areal_radius",
            "radial_orthonormal_momentum",
        }.issubset(snapshot.files)
        assert str(snapshot["particle_state_variables"]) == "rs_ubar"
        assert np.allclose(snapshot["rs"], particles.r)
        assert np.allclose(snapshot["ubar"], particles.ur)
        assert np.allclose(snapshot["areal_radius"], snapshot["r"])
        assert np.allclose(
            snapshot["radial_orthonormal_momentum"],
            snapshot["ubar"],
        )


def test_reduced_os_initial_state_deposits_total_particle_energy():
    metric, particles, _, _, total_rest_mass = os_z4c.build_initial_state(
        r_max=20.0,
        num_z4c_cells=199,
        particles_per_shell=1,
        shooting_iterations=1,
    )

    matter_terms = compute_radial_matter_terms(particles, metric)
    proper_shell_volume = _proper_radial_shell_volume(metric)
    deposited_energy = jnp.sum(matter_terms.rho * proper_shell_volume)
    particle_energy = jnp.sum(particles.get_mass())

    assert jnp.allclose(particle_energy, total_rest_mass)
    assert jnp.allclose(deposited_energy, particle_energy)
    assert jnp.max(matter_terms.rho) > 1.0e-4
