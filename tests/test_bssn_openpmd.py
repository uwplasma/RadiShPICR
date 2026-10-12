import numpy as np
import pytest
import jax.numpy as jnp

io = pytest.importorskip("openpmd_api")

from RadiShPICR.diagnostics import BSSNFieldSeries
from RadiShPICR.diagnostics.bssn_openpmd import FIELD_NAMES, SOURCE_NAMES
from RadiShPICR.Z4C import Z4C_Metric


DR = 0.25
R_GRID = jnp.asarray([0.125, 0.375, 0.625, 0.875])
E_R = jnp.asarray([0.041, 0.123, 0.205, 0.287])
CHARGE_DENSITY = jnp.asarray([3.3, 3.1, 0.0, 0.0])


def build_metric():
    """A four-point half-cell grid with a distinct profile in every field."""

    ones = jnp.ones_like(R_GRID)
    return Z4C_Metric(
        alpha=jnp.asarray([0.90, 0.92, 0.94, 0.96]),
        beta=jnp.asarray([0.01, 0.02, 0.03, 0.04]),
        conformal_grr=jnp.asarray([1.05, 1.04, 1.03, 1.02]),
        conformal_gt=jnp.asarray([1.10, 1.08, 1.06, 1.04]),
        chi=jnp.asarray([0.80, 0.85, 0.90, 0.95]),
        Kh=jnp.asarray([-0.10, -0.08, -0.06, -0.04]),
        Arr=jnp.asarray([0.11, 0.12, 0.13, 0.14]),
        At=jnp.asarray([-0.055, -0.060, -0.065, -0.070]),
        theta=jnp.asarray([1e-4, 2e-4, 3e-4, 4e-4]),
        Gamma=jnp.asarray([0.21, 0.22, 0.23, 0.24]),
        kappa=0.02 * ones,
        eta=2.0 * ones,
        nu=0.02 * ones,
        r=R_GRID,
        dr=DR,
    )


def read_series(path):
    return io.Series(str(path), io.Access.read_only)


def test_series_writes_every_evolved_field_unchanged(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(
            metric, step=0, time=0.0, E_r=E_R, charge_density=CHARGE_DENSITY,
        )
        fields.write(metric, step=10, time=0.5)

    assert output_path.name == "bssn_fields.h5"

    series = read_series(output_path)
    assert sorted(series.iterations) == [0, 10]

    iteration = series.iterations[0]
    expected_records = set(FIELD_NAMES) | set(SOURCE_NAMES) | {"r", "areal_radius"}
    assert set(iteration.meshes) == expected_records

    chunks = {
        name: iteration.meshes[name][io.Mesh.SCALAR].load_chunk()
        for name in expected_records
    }
    series.flush()

    # Written as is: no vacuum rescaling, no Schwarzschild mapping.
    for name in FIELD_NAMES:
        assert np.array_equal(chunks[name], np.asarray(getattr(metric, name)))
    assert np.array_equal(chunks["r"], np.asarray(metric.r))
    assert np.array_equal(chunks["E_r"], np.asarray(E_R))
    assert np.array_equal(chunks["charge_density"], np.asarray(CHARGE_DENSITY))
    series.close()


def test_areal_radius_record_matches_the_z4c_definition(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(metric, step=3, time=0.25)

    series = read_series(output_path)
    areal_radius = series.iterations[3].meshes["areal_radius"][io.Mesh.SCALAR].load_chunk()
    series.flush()

    expected = np.asarray(metric.r) * np.sqrt(
        np.asarray(metric.conformal_gt) / np.asarray(metric.chi)
    )
    assert np.allclose(areal_radius, expected)
    series.close()


def test_mesh_description_records_the_cell_centered_evolution_grid(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(metric, step=0, time=0.0, dt=0.05)

    series = read_series(output_path)
    assert series.get_attribute("savedCoordinates") == "evolution"
    assert series.get_attribute("radialCoordinate") == "areal_radius"

    iteration = series.iterations[0]
    assert iteration.time == pytest.approx(0.0)
    assert iteration.dt == pytest.approx(0.05)

    mesh = iteration.meshes["chi"]
    assert mesh.geometry_string == "other"
    assert mesh.axis_labels == ["r"]
    assert mesh.grid_spacing == [pytest.approx(DR)]
    assert mesh.grid_global_offset == [pytest.approx(0.0)]
    assert mesh.get_attribute("coordinateRecord") == "areal_radius"
    # generate_r_grid samples cell centres, r_i = (i + 1/2) dr.
    assert mesh[io.Mesh.SCALAR].position == [pytest.approx(0.5)]
    series.close()


def test_missing_time_is_written_as_nan(tmp_path):
    metric = build_metric()

    series_writer = BSSNFieldSeries(tmp_path, filename="no_time.h5")
    output_path = series_writer.write(metric, step=0)
    series_writer.close()
    series_writer.close()  # idempotent

    series = read_series(output_path)
    assert np.isnan(series.iterations[0].time)
    series.close()


def test_source_records_are_zero_filled_when_absent(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(metric, step=0, time=0.0)

    series = read_series(output_path)
    meshes = series.iterations[0].meshes
    chunks = {name: meshes[name][io.Mesh.SCALAR].load_chunk() for name in SOURCE_NAMES}
    series.flush()

    # A vacuum run still carries the records, flagged as a zero fill.
    for name in SOURCE_NAMES:
        assert meshes[name].get_attribute("provided") is False
        assert np.array_equal(chunks[name], np.zeros_like(np.asarray(metric.r)))
    series.close()


def test_source_records_are_marked_provided_and_left_unscaled(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(
            metric, step=0, time=0.0, E_r=E_R, charge_density=CHARGE_DENSITY,
        )

    series = read_series(output_path)
    meshes = series.iterations[0].meshes
    electric_field = meshes["E_r"][io.Mesh.SCALAR].load_chunk()
    charge_density = meshes["charge_density"][io.Mesh.SCALAR].load_chunk()
    series.flush()

    for name in SOURCE_NAMES:
        assert meshes[name].get_attribute("provided") is True
        assert meshes[name][io.Mesh.SCALAR].position == [pytest.approx(0.5)]
        assert list(meshes[name][io.Mesh.SCALAR].shape) == [len(R_GRID)]

    # The covariant E_r goes out exactly as handed in; no X_r rescaling.
    assert np.array_equal(electric_field, np.asarray(E_R))
    assert np.array_equal(charge_density, np.asarray(CHARGE_DENSITY))
    series.close()


def test_evolved_fields_carry_no_provided_attribute(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(metric, step=0, time=0.0)

    series = read_series(output_path)
    meshes = series.iterations[0].meshes
    for name in (*FIELD_NAMES, "r", "areal_radius"):
        assert not meshes[name].contains_attribute("provided")
    series.close()


def build_particles():
    """Two species with distinct arrays, so a mix-up between them fails."""

    from RadiShPICR.particles import particle_species

    dust = particle_species(
        name="dust", charge=-1.0e-3, mass=2.0e-3,
        weight=jnp.asarray([1.0, 2.0, 3.0]),
        r=jnp.asarray([0.2, 0.4, 0.6]), ur=jnp.asarray([0.01, -0.02, 0.03]),
        phi=jnp.asarray([0.0, 1.0, 2.0]), uphi=jnp.asarray([0.0, 0.0, 0.1]),
        shape_mode="quadratic",
    )
    ions = particle_species(
        name="ions", charge=5.0e-4, mass=7.0e-3,
        weight=jnp.asarray([4.0, 5.0]),
        r=jnp.asarray([0.3, 0.7]), ur=jnp.asarray([-0.05, 0.06]),
        phi=jnp.asarray([3.0, 4.0]), uphi=jnp.asarray([0.2, 0.3]),
        shape_mode="linear",
    )
    return dust, ions


def test_particles_round_trip_with_shape_mode(tmp_path):
    metric = build_metric()
    dust, ions = build_particles()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(metric, step=0, time=0.0, particles=[dust, ions])

    series = read_series(output_path)
    species = series.iterations[0].particles
    assert sorted(species) == ["dust", "ions"]

    written = species["dust"]
    chunks = {
        "position_r": written["position"]["r"].load_chunk(),
        "position_phi": written["position"]["phi"].load_chunk(),
        "momentum_r": written["momentum"]["r"].load_chunk(),
        "momentum_phi": written["momentum"]["phi"].load_chunk(),
        "weighting": written["weighting"][io.Record_Component.SCALAR].load_chunk(),
        "charge": written["charge"][io.Record_Component.SCALAR].load_chunk(),
        "mass": written["mass"][io.Record_Component.SCALAR].load_chunk(),
        "offset": written["positionOffset"]["r"].load_chunk(),
    }
    series.flush()

    # shapeMode is load-bearing: a reader needs it to re-deposit densities.
    assert written.get_attribute("shapeMode") == "quadratic"
    assert species["ions"].get_attribute("shapeMode") == "linear"

    assert np.array_equal(chunks["position_r"], np.asarray(dust.r))
    assert np.array_equal(chunks["position_phi"], np.asarray(dust.phi))
    assert np.array_equal(chunks["momentum_r"], np.asarray(dust.ur))
    assert np.array_equal(chunks["momentum_phi"], np.asarray(dust.uphi))
    assert np.array_equal(chunks["weighting"], np.asarray(dust.weight))
    # charge and mass are of one physical particle, written as constants.
    assert np.allclose(chunks["charge"], -1.0e-3)
    assert np.allclose(chunks["mass"], 2.0e-3)
    assert np.array_equal(chunks["offset"], np.zeros(3))
    series.close()


def test_iterations_without_particles_carry_none(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(tmp_path) as fields:
        output_path = fields.write(metric, step=0, time=0.0)

    series = read_series(output_path)
    assert list(series.iterations[0].particles) == []
    series.close()


def test_metadata_is_written_as_series_attributes(tmp_path):
    metric = build_metric()

    with BSSNFieldSeries(
        tmp_path, metadata=dict(total_mass=1.0, zero_shift=0, boundary="areal_inner_open")
    ) as fields:
        output_path = fields.write(metric, step=0, time=0.0)
        # set_metadata works after iterations exist, for end-of-run bookkeeping.
        fields.set_metadata(completed=True, final_step=0, skipped=None)

    series = read_series(output_path)
    assert series.get_attribute("total_mass") == pytest.approx(1.0)
    assert series.get_attribute("zero_shift") == 0
    assert series.get_attribute("boundary") == "areal_inner_open"
    assert series.get_attribute("completed") is True
    assert series.get_attribute("final_step") == 0
    assert not series.contains_attribute("skipped")
    series.close()
