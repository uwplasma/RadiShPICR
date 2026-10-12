"""openPMD output of the evolved Z4C/BSSN fields (optional dependency ``openpmd-api``).

One group-based series holding one iteration per written step.  The fields are
the ones the RK stages hold, written under their ``Z4C_Metric`` attribute names
and with their values untouched: no vacuum rescaling, no Schwarzschild or
isotropic post-processing, no regridding.

Layout:
    Scalar meshes ``alpha``, ``beta``, ``conformal_grr``, ``conformal_gt``,
    ``chi``, ``Kh``, ``Arr``, ``At``, ``theta`` and ``Gamma`` on the radial grid,
    plus the two coordinate records ``r`` and ``areal_radius``.

    The radial coordinate of the output is the areal radius
    :math:`R = r\\sqrt{\\tilde\\gamma_T/\\chi}`, which is neither uniform nor
    constant in time, so it cannot be an openPMD mesh axis.  It is written as its
    own record instead, and every mesh is tagged ``geometry = "other"`` with the
    attribute ``coordinateRecord = "areal_radius"`` naming it.  The axis
    description the meshes do carry describes the evolution grid the samples sit
    on: ``axis_labels=["r"]``, ``grid_spacing=[dr]``, ``grid_global_offset=[0]``
    and ``position=[0.5]``, since ``generate_r_grid`` places the samples at cell
    centres, ``r_i = (i + 1/2) dr``.

    Everything is geometric units (G = c = 1), so ``unitSI`` is 1 and the
    ``unitDimension`` of every record is empty.  The series records this as the
    attribute ``unitSystem``.
"""

from pathlib import Path

import numpy as np

__all__ = ["BSSNFieldSeries", "FIELD_NAMES"]

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

    def __init__(self, output_folder, filename="bssn_fields.h5"):
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

    def _store(self, mesh, values, dr, keep):
        """Describe one scalar mesh on the radial grid and stage its data."""

        io = self._io

        mesh.set_geometry("other")
        mesh.axis_labels = ["r"]
        mesh.grid_spacing = [float(dr)]
        mesh.grid_global_offset = [0.0]
        mesh.grid_unit_SI = 1.0
        mesh.unit_dimension = {}
        mesh.set_attribute("coordinateRecord", "areal_radius")

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

    def write(self, metric, step, time=None, dt=None):
        """Write one iteration holding the current ``Z4C_Metric`` fields.

        The metric is read, never modified, so this can be called after any
        timestep in a loop.

        Args:
            metric: A ``Z4C_Metric``; its fields are written as they stand.
            step: Iteration index within the series.
            time: Evolution time of this iteration; ``NaN`` when not given.
            dt: Timestep to record on the iteration.  The snapshot cadence is
                not the timestep, so this is ``0.0`` unless the caller supplies
                the real one.

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
