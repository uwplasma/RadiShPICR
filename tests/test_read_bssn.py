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


def test_optional_dark_fields_round_trip_in_rk_order(tmp_path):
    metric = build_metric()
    dark_fields = (metric.r, -2.0 * metric.r, 3.0 + metric.r)
    with BSSNFieldSeries(tmp_path) as series:
        series.write(metric, 0)
        series.write(metric, 1, dark_fields=dark_fields)
    ordinary, proca = iter_snapshots(tmp_path)
    assert ordinary.dark_fields is None
    assert len(ordinary) == len(proca) == 7
    for saved, expected in zip(proca.dark_fields, dark_fields, strict=True):
        assert jnp.array_equal(saved, expected)


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


def test_recorded_plasma_frequency_round_trips(tmp_path):
    """A species specified by its plasma frequency keeps it through the file.

    The normalization itself lives in the weights, so this records provenance
    rather than a source term -- but losing it would leave a reloaded run with
    no way to say which plasma it represents.
    """

    from RadiShPICR.particles import particle_species

    metric = build_metric()
    plasma = particle_species(
        name="plasma", charge=-1.0, mass=1.0,
        weight=jnp.asarray([3.0e-9, 4.0e-9, 5.0e-9]),
        r=jnp.asarray([0.2, 0.4, 0.6]), ur=jnp.zeros(3),
        phi=jnp.zeros(3), uphi=jnp.zeros(3), shape_mode="quadratic",
        plasma_frequency=0.375,
    )
    with BSSNFieldSeries(tmp_path) as fields:
        path = fields.write(metric, 0, time=0.0, particles=plasma)

    restored = final_snapshot(path).species()

    assert np.allclose(np.asarray(restored.plasma_frequency), 0.375)
    # The weights carry the normalization, so they must survive exactly.
    assert np.array_equal(
        np.asarray(restored.weight), np.asarray(plasma.weight)
    )


def test_species_with_no_recorded_frequency_round_trips_as_zero(tmp_path):
    """An ordinary species carries no frequency and must stay that way.

    The reader also tolerates a missing record outright, so series written
    before it existed keep loading; that branch cannot be reached through the
    current writer, which always emits it.
    """

    path, _, dust = write_series(tmp_path, steps=(0,))

    restored = final_snapshot(path).species()

    assert np.allclose(np.asarray(restored.plasma_frequency), 0.0)
    assert np.array_equal(np.asarray(restored.r), np.asarray(dust.r))
