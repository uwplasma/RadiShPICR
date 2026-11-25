"""Every rebuild of a ``particle_species`` must carry all of its fields.

``_copy_particle_state`` runs once per RK stage and ``_delete_particles`` once
per boundary call, so a field dropped by either vanishes after a single step
rather than failing loudly. Routing the rebuilds through ``replace`` keeps them
total over the leaf set; these tests hold that line for the next field added.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from RadiShPICR.ConstraintBasedRelativity import evolve as constrained_evolve
from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C import time_evolve
from RadiShPICR.Z4C.particle_boundaries import (
    _delete_particles,
    deleting_particle_boundary,
)


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def sentinel_species(num_particles=4):
    """A species whose every leaf carries a distinct recognizable value."""

    return particle_species(
        name="sentinel", charge=-1.0, mass=1.0,
        weight=jnp.full(num_particles, 0.25),
        r=jnp.linspace(1.0, 4.0, num_particles),
        ur=jnp.full(num_particles, 0.1),
        phi=jnp.zeros(num_particles),
        uphi=jnp.full(num_particles, 0.2),
        shape_mode="quadratic",
        plasma_frequency=0.7,
    )


REBUILDS = {
    "z4c_copy_particle_state": lambda p: time_evolve._copy_particle_state(
        p, p.r, p.phi, p.ur, p.uphi,
    ),
    "z4c_delete_particles": lambda p: _delete_particles(
        p, jnp.zeros(p.r.shape, dtype=bool),
    ),
    "z4c_deleting_boundary": deleting_particle_boundary,
    "constrained_copy_particle_state": (
        lambda p: constrained_evolve._copy_particle_state(p, p.r, p.phi, p.ur)
    ),
    "replace_noop": lambda p: p.replace(),
}


@pytest.mark.parametrize("name", sorted(REBUILDS))
def test_rebuild_preserves_every_leaf(name):
    particles = sentinel_species()
    rebuilt = REBUILDS[name](particles)

    assert len(jax.tree.leaves(rebuilt)) == len(jax.tree.leaves(particles))
    assert jax.tree.structure(rebuilt) == jax.tree.structure(particles)
    # The recorded plasma frequency is the field most recently added, and the
    # one a field-by-field rebuild would silently drop.
    np.testing.assert_allclose(
        np.asarray(rebuilt.plasma_frequency),
        np.asarray(particles.plasma_frequency),
    )
    assert rebuilt.name == particles.name
    assert rebuilt.shape_mode == particles.shape_mode
    np.testing.assert_allclose(np.asarray(rebuilt.charges), -1.0)
    np.testing.assert_allclose(np.asarray(rebuilt.masses), 1.0)


def test_tree_structure_matches_across_recorded_and_unrecorded_species():
    """A default species must flatten like a plasma one.

    If the recorded frequency were an optional ``None`` field it would be an
    empty pytree node, so the two would have different treedefs and could not
    be carried through the same ``lax.cond`` branch or ``scan`` body.
    """

    recorded = sentinel_species()
    plain = particle_species(
        name="sentinel", charge=-1.0, mass=1.0, weight=jnp.full(4, 0.25),
        r=jnp.linspace(1.0, 4.0, 4), ur=jnp.full(4, 0.1), phi=jnp.zeros(4),
        uphi=jnp.full(4, 0.2), shape_mode="quadratic",
    )

    assert jax.tree.structure(recorded) == jax.tree.structure(plain)
    np.testing.assert_allclose(np.asarray(plain.plasma_frequency), 0.0)

    selected = jax.lax.cond(
        jnp.asarray(True), lambda _: recorded, lambda _: plain, operand=None,
    )
    np.testing.assert_allclose(np.asarray(selected.plasma_frequency), 0.7)


def test_replace_rejects_unknown_fields():
    with pytest.raises(TypeError):
        sentinel_species().replace(charge=1.0)


def test_replace_round_trips_through_jit():
    particles = sentinel_species()
    rebuilt = jax.jit(lambda p: p.replace(ur=p.ur * 2.0))(particles)

    np.testing.assert_allclose(np.asarray(rebuilt.ur), 0.2)
    np.testing.assert_allclose(np.asarray(rebuilt.plasma_frequency), 0.7)
    assert rebuilt.shape_mode == "quadratic"
