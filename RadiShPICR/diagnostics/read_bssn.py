"""Read back a series written by :mod:`RadiShPICR.diagnostics.bssn_openpmd`.

Snapshots hold raw simulation state, so every diagnostic a plotting script
wants is recomputed from them.  This module hands back the state as the live
objects the library already takes -- a ``Z4C_Metric`` and a
``particle_species`` -- so the existing functions apply directly::

    for snapshot in iter_snapshots(path):
        mass = misner_sharp_mass(snapshot.metric)
        matter = compute_radial_matter_terms(snapshot.particles, snapshot.metric,
                                             inner_open=True)
        residual = radial_gauss_residual(snapshot.metric, snapshot.E_r,
                                         snapshot.charge_density, epsilon_0)

``dr`` comes from the mesh ``grid_spacing`` and the damping parameters from the
series attributes, so the reconstructed metric is complete enough to
differentiate.
"""

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from RadiShPICR.diagnostics.bssn_openpmd import FIELD_NAMES, SOURCE_NAMES
from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

__all__ = ["Snapshot", "iter_snapshots", "read_metadata", "final_snapshot"]

_DAMPING_DEFAULTS = {"kappa": 1.0, "eta": 2.0, "nu": 0.02}


class Snapshot(NamedTuple):
    """One iteration of a series, as live objects."""

    step: int
    time: float
    metric: Z4C_Metric
    E_r: jnp.ndarray
    charge_density: jnp.ndarray
    particles: tuple
    areal_radius: jnp.ndarray

    def species(self, name=None):
        """Return one species by name, or the only one when there is one."""

        if name is None:
            if len(self.particles) != 1:
                raise ValueError(
                    f"Expected exactly one species, found {len(self.particles)}; "
                    "pass a name."
                )
            return self.particles[0]
        for one in self.particles:
            if one.name == name:
                return one
        raise KeyError(f"No species {name!r} in step {self.step}")


def _open(path):
    try:
        import openpmd_api as io
    except ImportError as exc:
        raise ImportError(
            "Reading a BSSN series needs the optional openpmd-api: "
            "pip install openpmd-api"
        ) from exc

    path = Path(path)
    if path.is_dir():
        path = path / "bssn_fields.h5"
    if not path.exists():
        raise FileNotFoundError(f"No openPMD series at {path}")

    return io, io.Series(str(path), io.Access.read_only)


def read_metadata(path):
    """Return the series attributes as a plain dict.

    This is what replaced ``run_summary.json``: the run parameters, and after a
    finished run ``completed`` and the final-step bookkeeping.
    """

    _, series = _open(path)
    metadata = {name: series.get_attribute(name) for name in series.attributes}
    series.close()

    return metadata


def _read_species(io, species, name):
    arrays = {}
    for record, labels in (("position", ("r", "phi")), ("momentum", ("r", "phi"))):
        for label in labels:
            arrays[f"{record}_{label}"] = species[record][label].load_chunk()
    for record in ("weighting", "charge", "mass"):
        arrays[record] = species[record][io.Record_Component.SCALAR].load_chunk()

    return name, arrays, str(species.get_attribute("shapeMode"))


def _build_species(name, arrays, shape_mode):
    # The per-particle charge and mass are written as constant records, so take
    # the single value back rather than carrying an array of copies around.
    def collapse(values):
        values = np.asarray(values)
        return float(values.flat[0]) if values.size and np.ptp(values) == 0 else jnp.asarray(values)

    return particle_species(
        name=name,
        charge=collapse(arrays["charge"]),
        mass=collapse(arrays["mass"]),
        weight=jnp.asarray(arrays["weighting"]),
        r=jnp.asarray(arrays["position_r"]),
        ur=jnp.asarray(arrays["momentum_r"]),
        phi=jnp.asarray(arrays["position_phi"]),
        uphi=jnp.asarray(arrays["momentum_phi"]),
        shape_mode=shape_mode,
    )


def iter_snapshots(path, stride=1, with_particles=True):
    """Yield a :class:`Snapshot` per iteration of the series at ``path``.

    Args:
        path: The series file, or a directory holding ``bssn_fields.h5``.
        stride: Yield every ``stride``-th iteration.  The last one is always
            included, so a strided pass still ends at the final state.
        with_particles: Read the particle species too.  Pass ``False`` to skip
            them when only the fields are needed.
    """

    io, series = _open(path)
    try:
        damping = {
            name: float(series.get_attribute(name))
            if name in series.attributes
            else default
            for name, default in _DAMPING_DEFAULTS.items()
        }

        steps = sorted(series.iterations)
        wanted = steps[::stride]
        if steps and wanted[-1] != steps[-1]:
            wanted.append(steps[-1])

        for step in wanted:
            iteration = series.iterations[step]
            meshes = iteration.meshes
            chunks = {
                name: meshes[name][io.Mesh.SCALAR].load_chunk()
                for name in (*FIELD_NAMES, *SOURCE_NAMES, "r", "areal_radius")
            }
            dr = float(meshes["r"].grid_spacing[0])

            species = []
            if with_particles:
                species = [
                    _read_species(io, iteration.particles[name], name)
                    for name in iteration.particles
                ]

            series.flush()

            ones = jnp.ones_like(jnp.asarray(chunks["r"]))
            metric = Z4C_Metric(
                **{name: jnp.asarray(chunks[name]) for name in FIELD_NAMES},
                kappa=damping["kappa"] * ones,
                eta=damping["eta"] * ones,
                nu=damping["nu"] * ones,
                r=jnp.asarray(chunks["r"]),
                dr=dr,
            )

            yield Snapshot(
                step=int(step),
                time=float(iteration.time),
                metric=metric,
                E_r=jnp.asarray(chunks["E_r"]),
                charge_density=jnp.asarray(chunks["charge_density"]),
                particles=tuple(_build_species(*one) for one in species),
                areal_radius=jnp.asarray(chunks["areal_radius"]),
            )
    finally:
        series.close()


def final_snapshot(path, with_particles=True):
    """Return the last iteration of the series.

    This replaces reading ``run_summary.json["final_step"]`` and rebuilding a
    filename from it -- the series knows its own last iteration.
    """

    last = None
    for snapshot in iter_snapshots(path, with_particles=with_particles):
        last = snapshot
    if last is None:
        raise ValueError(f"Series at {path} holds no iterations")

    return last
