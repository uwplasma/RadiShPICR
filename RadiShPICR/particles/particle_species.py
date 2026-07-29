import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class particle_species:
    def __init__(self, name, charge, mass, weight, r, ur, phi, uphi, shape_mode):
        self.name = name
        self.charges = charge
        self.masses = mass
        particle_weight = jnp.asarray(weight, dtype=jnp.asarray(r).dtype)
        self.weight = jnp.broadcast_to(particle_weight, jnp.shape(r))
        self.r = r
        self.ur = ur
        self.phi = phi
        self.uphi = uphi
        self.shape_mode = shape_mode

    def get_positions(self):
        return self.r, self.phi
    
    def get_velocities(self):
        return self.ur, self.uphi
    
    def get_mass(self):
        return self.masses * self.weight
    
    def get_charge(self):
        return self.charges * self.weight
    
    def get_shape(self):
        return self.shape_mode

    def tree_flatten(self):
        dynamic_fields = (
            jnp.asarray(self.charges),
            jnp.asarray(self.masses),
            jnp.asarray(self.weight),
            self.r,
            self.ur,
            self.phi,
            self.uphi,
        )
        static_fields = (self.name, self.shape_mode)

        return dynamic_fields, static_fields

    @classmethod
    def tree_unflatten(cls, static_fields, dynamic_fields):
        name, shape_mode = static_fields
        charge, mass, weight, r, ur, phi, uphi = dynamic_fields

        return cls(
            name=name,
            charge=charge,
            mass=mass,
            weight=weight,
            r=r,
            ur=ur,
            phi=phi,
            uphi=uphi,
            shape_mode=shape_mode,
        )
