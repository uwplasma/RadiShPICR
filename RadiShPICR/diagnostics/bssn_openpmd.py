"""openPMD output of the evolved Z4C/BSSN fields (optional dependency ``openpmd-api``).

One group-based series holding one iteration per written step.  The fields are
the ones the RK stages hold, written under their ``Z4C_Metric`` attribute names
and with their values untouched: no vacuum rescaling, no Schwarzschild or
isotropic post-processing, no regridding.

Layout:
    Scalar meshes ``alpha``, ``beta``, ``conformal_grr``, ``conformal_gt``,
    ``chi``, ``Kh``, ``Arr``, ``At``, ``theta`` and ``Gamma`` on the radial grid,
    plus the two coordinate records ``r`` and ``areal_radius``.

    The electromagnetic sources ``E_r`` and ``charge_density`` are written
    alongside them.  ``E_r`` is the *covariant* radial component
    :math:`E_r = \\gamma_{rr} E^r` that the RK stages carry, not the orthonormal
    one; both sit on the same cell centres as the metric fields.  A vacuum or
    metric-only run has neither, so they are filled with zeros and marked with
    the record attribute ``provided = False``.  Every iteration of every series
    therefore holds the same set of records.

    The radial coordinate of the output is the areal radius
    :math:`R = r\\sqrt{\\tilde\\gamma_T/\\chi}`, which is neither uniform nor
    constant in time, so it cannot be an openPMD mesh axis.  It is written as its
    own record instead, and every mesh is tagged ``geometry = "other"`` with the
    attribute ``coordinateRecord = "areal_radius"`` naming it.  The axis
    description the meshes do carry describes the evolution grid the samples sit
    on: ``axis_labels=["r"]``, ``grid_spacing=[dr]``, ``grid_global_offset=[0]``
    and ``position=[0.5]``, since ``generate_r_grid`` places the samples at cell
    centres, ``r_i = (i + 1/2) dr``.

    Particles, when a run has them, become one openPMD species per
    ``particle_species``: the vector records ``position`` and ``momentum`` with
    the components ``r`` and ``phi``, the scalar ``weighting``, a zero
    ``positionOffset``, and ``charge``/``mass`` of one physical particle.  The
    momentum is the one the pusher advances, the covariant ``(u_r, u_phi)``.
    Each species carries the attribute ``shapeMode``, which a reader needs in
    order to re-deposit densities from the particles.

    Per-particle areal radius and orthonormal momentum are deliberately absent.
    They are a pure function of the species and the metric in the same
    iteration, so they belong to whatever reads the series.

    Everything is geometric units (G = c = 1), so ``unitSI`` is 1 and the
    ``unitDimension`` of every record is empty.  The series records this as the
    attribute ``unitSystem``.

Run metadata:
    The scalars describing a run -- masses, charges, damping parameters, the
    boundary choice, whether it finished -- are series attributes rather than a
    companion file.  Pass them to the constructor and add to them later with
    :meth:`BSSNFieldSeries.set_metadata`.
"""

from pathlib import Path

import numpy as np

__all__ = ["BSSNFieldSeries", "FIELD_NAMES", "SOURCE_NAMES"]

FIELD_NAMES = (
    "alpha",
    "beta",
    "conformal_grr",
    "conformal_gt",
    "chi",
    "Kh",
    "Arr",
    "At",
    "theta",
    "Gamma",
)

SOURCE_NAMES = ("E_r", "charge_density")


def _package_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("RadiShPICR")
    except PackageNotFoundError:
        return "unknown"


class BSSNFieldSeries:
    """Append Z4C/BSSN field iterations to one group-based openPMD series.

    The series stays open across the evolution loop, so use it as a context
    manager (or call :meth:`close` when the run finishes)::

        with BSSNFieldSeries(output_directory / "bssn") as fields:
            for step in range(num_steps):
                particles, metric, charge_density, E_r = rk4_step(...)
                if step % save_every == 0:
                    fields.write(metric, step, time=evolution_time)

    Args:
        output_folder: Directory to hold the series; created if missing.
        filename: Series file name.  The extension picks the backend --
            ``.h5`` (default), ``.bp`` or ``.json``.

    Raises:
        ImportError: If the optional dependency ``openpmd-api`` is missing.
    """

    def __init__(self, output_folder, filename="bssn_fields.h5", metadata=None):
        try:
            import openpmd_api as io
        except ImportError as exc:
            raise ImportError(
                "BSSNFieldSeries needs the optional openpmd-api: "
                "pip install openpmd-api"
            ) from exc

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        self._io = io
        self.path = output_path / filename
        self._series = io.Series(str(self.path), io.Access.create)
        self._series.set_software("RadiShPICR", _package_version())
        self._series.set_attribute("unitSystem", "geometric (G=c=1)")
        self._series.set_attribute("savedCoordinates", "evolution")
        self._series.set_attribute("radialCoordinate", "areal_radius")
        if metadata:
            self.set_metadata(**metadata)

    def set_metadata(self, **entries):
        """Record run metadata as series attributes.

        This stands in for the ``run_summary.json`` and
        ``initial_data_metadata.npz`` files the demos used to write, so it can
        be called before the run to stamp its parameters and again at the end
        for ``completed`` and whatever the run only knows once it is over.
        Values must be the scalar types openPMD stores: ``bool``, ``int``,
        ``float`` or ``str``.
        """

        for name, value in entries.items():
            if isinstance(value, (bool, int, float, str)):
                self._series.set_attribute(name, value)
            elif value is None:
                continue
            else:
                self._series.set_attribute(name, float(value))

    def _store(self, mesh, values, dr, keep, provided=None):
        """Describe one scalar mesh on the radial grid and stage its data.

        ``provided`` marks a record the caller may leave out: ``False`` says the
        values are a zero fill rather than a field the run actually carried.
        """

        io = self._io

        mesh.set_geometry("other")
        mesh.axis_labels = ["r"]
        mesh.grid_spacing = [float(dr)]
        mesh.grid_global_offset = [0.0]
        mesh.grid_unit_SI = 1.0
        mesh.unit_dimension = {}
        mesh.set_attribute("coordinateRecord", "areal_radius")
        if provided is not None:
            mesh.set_attribute("provided", np.bool_(provided))

        # np.array copies, which matters twice: a JAX array converts to a
        # read-only buffer that store_chunk refuses, and the buffer has to stay
        # alive and writeable until the series is flushed, which is what
        # ``keep`` is for.
        data = np.array(values, dtype=np.float64, order="C")
        component = mesh[io.Mesh.SCALAR]
        component.position = [0.5]
        component.reset_dataset(io.Dataset(data.dtype, data.shape))
        component.store_chunk(data)
        component.unit_SI = 1.0
        keep.append(data)

    def _store_component(self, component, values, keep, count=None):
        """Stage one record component, as a constant when ``values`` is scalar."""

        io = self._io
        data = np.array(values, dtype=np.float64, order="C")
        if data.ndim == 0:
            component.reset_dataset(io.Dataset(data.dtype, [int(count)]))
            component.make_constant(float(data))
        else:
            component.reset_dataset(io.Dataset(data.dtype, data.shape))
            component.store_chunk(data)
            keep.append(data)
        component.unit_SI = 1.0

    def _store_species(self, species, particles, keep):
        """Write one ``particle_species`` as an openPMD species."""

        io = self._io
        count = int(np.size(np.asarray(particles.r)))
        species.set_attribute("shapeMode", str(particles.get_shape()))

        vectors = (
            ("position", (particles.r, particles.phi)),
            ("momentum", (particles.ur, particles.uphi)),
        )
        for record_name, components in vectors:
            record = species[record_name]
            record.unit_dimension = {}
            record.set_attribute("macroWeighted", np.uint32(0))
            record.set_attribute(
                "weightingPower", 0.0 if record_name == "position" else 1.0
            )
            for label, values in zip("r phi".split(), components):
                self._store_component(record[label], values, keep, count)

        offset = species["positionOffset"]
        offset.unit_dimension = {}
        offset.set_attribute("macroWeighted", np.uint32(0))
        offset.set_attribute("weightingPower", 0.0)
        for label in ("r", "phi"):
            self._store_component(offset[label], 0.0, keep, count)

        weighting = species["weighting"]
        weighting.unit_dimension = {}
        weighting.set_attribute("macroWeighted", np.uint32(1))
        weighting.set_attribute("weightingPower", 1.0)
        self._store_component(
            weighting[io.Record_Component.SCALAR], particles.weight, keep, count
        )

        # charge and mass of one physical particle, constant when the species
        # was built with scalars.
        for record_name, values in (
            ("charge", particles.charges),
            ("mass", particles.masses),
        ):
            record = species[record_name]
            record.unit_dimension = {}
            record.set_attribute("macroWeighted", np.uint32(0))
            record.set_attribute("weightingPower", 1.0)
            self._store_component(
                record[io.Record_Component.SCALAR], values, keep, count
            )

    def write(
        self,
        metric,
        step,
        time=None,
        dt=None,
        E_r=None,
        charge_density=None,
        particles=None,
    ):
        """Write one iteration holding the current ``Z4C_Metric`` fields.

        Nothing passed in is modified, so this can be called after any timestep
        in a loop.  ``rk4_step`` returns ``(particles, metric, charge_density,
        E_r)``, so the two source arrays come straight off the same call.

        Args:
            metric: A ``Z4C_Metric``; its fields are written as they stand.
            step: Iteration index within the series.
            time: Evolution time of this iteration; ``NaN`` when not given.
            dt: Timestep to record on the iteration.  The snapshot cadence is
                not the timestep, so this is ``0.0`` unless the caller supplies
                the real one.
            E_r: Covariant radial electric field on the metric's cell centers,
                written unscaled.  A vacuum run leaves this out, and the record
                is then zeros marked ``provided = False``.
            charge_density: Eulerian charge density on the same cell centers,
                with the same zero fill when it is left out.
            particles: One ``particle_species`` or a sequence of them, written
                as openPMD species under this iteration.  A run without
                particles leaves this out and the iteration carries none.

        Returns:
            The path of the series being written.
        """

        iteration = self._series.iterations[int(step)]
        iteration.time = np.nan if time is None else float(time)
        iteration.dt = 0.0 if dt is None else float(dt)
        iteration.time_unit_SI = 1.0

        radius = np.asarray(metric.r, dtype=np.float64)
        # The areal radius of the Z4C variables; see _areal_geometry in
        # RadiShPICR.Z4C.curvature_invariants for the canonical definition.
        areal_radius = radius * np.sqrt(
            np.asarray(metric.conformal_gt, dtype=np.float64)
            / np.asarray(metric.chi, dtype=np.float64)
        )

        keep = []
        for name in FIELD_NAMES:
            self._store(iteration.meshes[name], getattr(metric, name), metric.dr, keep)
        self._store(iteration.meshes["r"], radius, metric.dr, keep)
        self._store(iteration.meshes["areal_radius"], areal_radius, metric.dr, keep)

        for name, values in zip(SOURCE_NAMES, (E_r, charge_density)):
            provided = values is not None
            self._store(
                iteration.meshes[name],
                values if provided else np.zeros_like(radius),
                metric.dr,
                keep,
                provided=provided,
            )

        if particles is not None:
            species_list = (
                particles if isinstance(particles, (list, tuple)) else [particles]
            )
            for one in species_list:
                self._store_species(iteration.particles[one.name], one, keep)
        self._series.flush()

        return self.path

    def close(self):
        """Close the series.  Calling this more than once is harmless."""

        if self._series is not None:
            self._series.close()
            self._series = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
        return False
