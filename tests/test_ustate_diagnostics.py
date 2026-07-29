import numpy as np
import jax.numpy as jnp

from RadiShPICR.diagnostics import write_metric_fields, write_phase_space
from RadiShPICR.particles import particle_species


def test_write_phase_space_uses_current_particle_fields(tmp_path):
    particles = particle_species(
        name="electrons",
        charge=-1.0,
        mass=1.0,
        weight=jnp.asarray([0.25, 0.75]),
        r=jnp.asarray([0.2, 0.4]),
        ur=jnp.asarray([0.1, -0.3]),
        phi=jnp.asarray([0.0, 0.0]),
        uphi=jnp.asarray([0.0, 0.0]),
        shape_mode="nearest",
    )

    output_path = write_phase_space(particles, tmp_path, step=3, time=0.25)
    snapshot = np.load(output_path)

    assert output_path.name == "phase_space_electrons_step_000003.npz"
    assert np.allclose(snapshot["r"], np.array([0.2, 0.4]))
    assert np.allclose(snapshot["ur"], np.array([0.1, -0.3]))
    assert np.allclose(snapshot["weight"], np.array([0.25, 0.75]))
    assert snapshot["step"] == 3
    assert snapshot["time"] == 0.25


def test_write_metric_fields_writes_complete_ustate(tmp_path):
    r_grid = jnp.asarray([0.0, 0.5, 1.0])
    source_terms = (
        jnp.asarray([0.0, 1.0, 0.0]),
        jnp.asarray([0.0, 2.0, 0.0]),
        jnp.asarray([0.0, 3.0, 0.0]),
        jnp.asarray([0.0, 4.0, 0.0]),
    )
    U_state = (
        jnp.asarray([1.0, 1.1, 1.2]),
        jnp.asarray([0.0, -0.1, -0.2]),
        jnp.asarray([1.0, 0.9, 0.8]),
        jnp.asarray([0.0, 0.01, 0.02]),
        jnp.asarray([0.0, 0.03, 0.04]),
        jnp.asarray([0.0, 0.05, 0.06]),
        source_terms,
        r_grid,
    )

    output_path = write_metric_fields(U_state, tmp_path, step=4, time=0.5)
    snapshot = np.load(output_path)

    assert output_path.name == "metric_fields_step_000004.npz"
    for key in (
        "r",
        "A",
        "phi",
        "alpha",
        "Krr",
        "beta_over_r",
        "Er",
        "mass_density",
        "charge_density",
        "Srr",
        "Sr",
    ):
        assert key in snapshot
    assert np.allclose(snapshot["A"], np.array([1.0, 1.1, 1.2]))
    assert np.allclose(snapshot["charge_density"], np.array([0.0, 2.0, 0.0]))
    assert snapshot["step"] == 4
    assert snapshot["time"] == 0.5
