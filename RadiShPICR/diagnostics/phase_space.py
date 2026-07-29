from pathlib import Path

import numpy as np


def write_phase_space(particles, output_folder, step, time=None):
    """Write one radial phase-space snapshot for a particle species.

    The timestepper evolves ``r`` and ``ur`` directly.  This diagnostic writes
    those current particle arrays and the per-particle macroparticle weights
    without changing the particle state, so it can be called after any timestep
    in a loop.
    """

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    species_name = particles.name
    filename = f"phase_space_{species_name}_step_{int(step):06d}.npz"
    snapshot_path = output_path / filename

    radial_positions = np.asarray(particles.r)
    radial_momenta = np.asarray(particles.ur)
    macroparticle_weights = np.asarray(particles.weight)
    output_time = np.nan if time is None else float(time)

    np.savez_compressed(
        snapshot_path,
        r=radial_positions,
        ur=radial_momenta,
        weight=macroparticle_weights,
        step=int(step),
        time=output_time,
        species_name=species_name,
    )

    return snapshot_path
