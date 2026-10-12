import numpy as np
import pytest
import jax.numpy as jnp

pytest.importorskip("openpmd_api")

from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.diagnostics.bssn_openpmd import FIELD_NAMES
from RadiShPICR.diagnostics.read_bssn import (
    final_snapshot,
    iter_snapshots,
    read_metadata,
)
from RadiShPICR.Z4C import misner_sharp_mass

from tests.test_bssn_openpmd import (
    CHARGE_DENSITY,
    E_R,
    build_metric,
    build_particles,
)


def write_series(tmp_path, steps=(0, 5, 10), **metadata):
    metric = build_metric()
    dust, _ = build_particles()

    with BSSNFieldSeries(tmp_path, metadata=metadata or None) as fields:
        for index, step in enumerate(steps):
            path = fields.write(
                metric, step, time=0.25 * index, E_r=E_R,
                charge_density=CHARGE_DENSITY, particles=dust,
            )

    return path, metric, dust


def test_snapshots_rebuild_the_metric_field_by_field(tmp_path):
    path, metric, _ = write_series(tmp_path, steps=(0,))

    snapshot = final_snapshot(path)

    for name in FIELD_NAMES:
        assert np.array_equal(
            np.asarray(getattr(snapshot.metric, name)),
            np.asarray(getattr(metric, name)),
        )
    assert np.array_equal(np.asarray(snapshot.metric.r), np.asarray(metric.r))
    # dr comes back from the mesh grid_spacing, not from an attribute.
    assert snapshot.metric.dr == pytest.approx(float(metric.dr))


def test_rebuilt_metric_supports_library_diagnostics(tmp_path):
    path, metric, _ = write_series(tmp_path, steps=(0,))

    snapshot = final_snapshot(path)

    # The point of handing back a real Z4C_Metric: library functions that
    # differentiate the fields give the same answer as on the original.
    assert np.allclose(
        np.asarray(misner_sharp_mass(snapshot.metric)),
        np.asarray(misner_sharp_mass(metric)),
    )
    assert np.allclose(
        np.asarray(snapshot.areal_radius),
        np.asarray(metric.r) * np.sqrt(
            np.asarray(metric.conformal_gt) / np.asarray(metric.chi)
        ),
    )


def test_sources_and_particles_round_trip(tmp_path):
    path, _, dust = write_series(tmp_path, steps=(0,))

    snapshot = final_snapshot(path)

    assert np.array_equal(np.asarray(snapshot.E_r), np.asarray(E_R))
    assert np.array_equal(
        np.asarray(snapshot.charge_density), np.asarray(CHARGE_DENSITY)
    )

    particles = snapshot.species()
    assert particles.name == "dust"
    assert particles.get_shape() == "quadratic"
    assert np.array_equal(np.asarray(particles.r), np.asarray(dust.r))
    assert np.array_equal(np.asarray(particles.ur), np.asarray(dust.ur))
    assert np.array_equal(np.asarray(particles.uphi), np.asarray(dust.uphi))
    # Constant records collapse back to the scalar they were built from, so
    # get_charge() reproduces the original per-particle charges.
    assert np.allclose(
        np.asarray(particles.get_charge()), np.asarray(dust.get_charge())
    )


def test_iteration_order_and_stride(tmp_path):
    path, _, _ = write_series(tmp_path, steps=(0, 5, 10, 15))

    assert [s.step for s in iter_snapshots(path)] == [0, 5, 10, 15]
    assert [s.time for s in iter_snapshots(path)] == [0.0, 0.25, 0.5, 0.75]
    # A strided pass still ends on the final state.
    assert [s.step for s in iter_snapshots(path, stride=2)] == [0, 10, 15]


def test_species_lookup_by_name_and_ambiguity(tmp_path):
    metric = build_metric()
    dust, ions = build_particles()

    with BSSNFieldSeries(tmp_path) as fields:
        path = fields.write(metric, 0, time=0.0, particles=[dust, ions])

    snapshot = final_snapshot(path)
    assert snapshot.species("ions").get_shape() == "linear"
    with pytest.raises(ValueError):
        snapshot.species()
    with pytest.raises(KeyError):
        snapshot.species("missing")


def test_read_metadata_returns_run_attributes(tmp_path):
    path, _, _ = write_series(tmp_path, steps=(0,), total_mass=2.0, completed=True)

    metadata = read_metadata(path)

    assert metadata["total_mass"] == pytest.approx(2.0)
    assert metadata["completed"] is True
    assert metadata["savedCoordinates"] == "evolution"


def test_reading_a_directory_finds_the_default_series(tmp_path):
    write_series(tmp_path, steps=(0,))

    # Callers hand over a run directory, not a file.
    assert final_snapshot(tmp_path).step == 0
    with pytest.raises(FileNotFoundError):
        final_snapshot(tmp_path / "nothing")
