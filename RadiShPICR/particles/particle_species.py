import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class particle_species:
    """Shared particle arrays with formulation-specific radial variables.

    Z4C uses isotropic radius ``r`` and physical covariant momenta ``u_r``
    and ``u_phi``. ConstraintBasedRelativity instead stores areal radius
    ``r_s = A r`` in ``r`` and ``u_r / A`` in ``ur``. The container performs
    no conversion; callers must use the convention of their formulation.

    ``plasma_frequency`` records the local plasma frequency a species was
    built to realize. It is provenance, not a source term: the deposition
    reads ``charges``, ``masses`` and ``weight`` only. Linear response ties
    the four together as

        omega_p**2 = charges**2 * weight * N / (epsilon_0 * masses * V)

    so a species built by ``RadiShPICR.initial_data.cold_plasma_shell``
    satisfies the relation by construction. Zero means "not recorded" and is
    the default; it never changes how a species deposits.
    """

    _DYNAMIC_FIELDS = (
        "charges", "masses", "weight", "r", "ur", "phi", "uphi",
        "plasma_frequency",
    )
    _STATIC_FIELDS = ("name", "shape_mode")

    def __init__(self, name, charge, mass, weight, r, ur, phi, uphi,
                 shape_mode, plasma_frequency=0.0):
        self.name = name
        self.charges = charge
        self.masses = mass
        radial_dtype = jnp.asarray(r).dtype
        particle_weight = jnp.asarray(weight, dtype=radial_dtype)
        self.weight = jnp.broadcast_to(particle_weight, jnp.shape(r))
        self.r = r
        self.ur = ur
        self.phi = phi
        self.uphi = uphi
        self.shape_mode = shape_mode
        # Broadcast like weight so mixed species can carry one omega_p each.
        self.plasma_frequency = jnp.broadcast_to(
            jnp.asarray(plasma_frequency, dtype=radial_dtype), jnp.shape(r),
        )

    def get_positions(self):
        return self.r, self.phi
    
    def get_velocities(self):
        """Return stored momenta: ``(u_r, u_phi)`` or constrained ``(u_r/A, u_phi)``."""

        return self.ur, self.uphi
    
    def get_mass(self):
        return self.masses * self.weight
    
    def get_charge(self):
        return self.charges * self.weight
    
    def get_shape(self):
        return self.shape_mode

    def replace(self, **changes):
        """Return a copy with the named attributes replaced.

        Defined over the pytree leaves rather than the constructor signature,
        so a newly added field is carried through every caller automatically.
        Rebuilding a species attribute by attribute is how fields get silently
        dropped on the evolution hot path.
        """

        unknown = set(changes) - set(self._DYNAMIC_FIELDS + self._STATIC_FIELDS)
        if unknown:
            raise TypeError(
                f"unknown particle_species attributes: {sorted(unknown)}"
            )
        dynamic_fields, static_fields = self.tree_flatten()
        dynamic_fields = tuple(
            changes.get(name, value)
            for name, value in zip(self._DYNAMIC_FIELDS, dynamic_fields)
        )
        static_fields = tuple(
            changes.get(name, value)
            for name, value in zip(self._STATIC_FIELDS, static_fields)
        )

        return self.tree_unflatten(static_fields, dynamic_fields)

    def tree_flatten(self):
        dynamic_fields = tuple(
            getattr(self, name) for name in self._DYNAMIC_FIELDS
        )
        static_fields = tuple(
            getattr(self, name) for name in self._STATIC_FIELDS
        )

        return dynamic_fields, static_fields

    @classmethod
    def tree_unflatten(cls, static_fields, dynamic_fields):
        particles = cls.__new__(cls)
        for name, value in zip(cls._STATIC_FIELDS, static_fields):
            setattr(particles, name, value)
        for name, value in zip(cls._DYNAMIC_FIELDS, dynamic_fields):
            setattr(particles, name, value)

        return particles
