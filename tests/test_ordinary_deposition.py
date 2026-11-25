"""Independent ordinary-shape, parity, and deposited-source checks."""

import unittest

import jax
import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.particles.shape_factors import particle_deposition_stencil
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.energy_momentum_tensor import relativistic_mass_energy_density
from RadiShPICR.Z4C.electric_field import compute_radial_charge_density


def metric_grid():
    r = jnp.arange(16, dtype=float) + 0.5
    ones, zeros = jnp.ones_like(r), jnp.zeros_like(r)
    return Z4C_Metric(
        alpha=ones, beta=zeros, conformal_grr=ones, conformal_gt=ones,
        chi=ones, Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        kappa=zeros, eta=zeros, nu=zeros, r=r, dr=jnp.asarray(1.),
    )


def particles_at(r, shape):
    zeros = jnp.zeros_like(r)
    return particle_species(name="ordinary shapes", charge=2., mass=3.,
        weight=jnp.ones_like(r), r=r, ur=zeros, phi=zeros, uphi=zeros, shape_mode=shape)


def analytic_shape(distance, shape):
    """The 1D NGP/CIC/TSC functions, evaluated independently at every cell."""
    if shape == "nearest":
        return ((distance >= -0.5) & (distance < 0.5)).astype(float)
    q = jnp.abs(distance)
    if shape == "linear":
        return jnp.maximum(1 - q, 0.)
    return jnp.where(q < 0.5, 0.75 - q*q,
                     jnp.where(q < 1.5, 0.5 * (1.5 - q)**2, 0.))


def dense_weights(stencil, num_cells):
    indices, even, odd = stencil
    particles = jnp.broadcast_to(jnp.arange(indices.shape[1]), indices.shape)
    return tuple(jnp.zeros((num_cells, indices.shape[1])).at[indices, particles].add(w)
                 for w in (even, odd))


class OrdinaryDepositionTests(unittest.TestCase):
    def test_known_interior_weights_and_weight_sums(self):
        metric = metric_grid()
        for shape, indices, weights in (
            ("nearest", [2], [1.]),
            ("linear", [2, 3], [.75, .25]),
            ("quadratic", [1, 2, 3], [.03125, .6875, .28125]),
        ):
            stencil = particle_deposition_stencil(particles_at(jnp.array([2.75]), shape), metric)
            self.assertTrue(bool(jnp.array_equal(stencil[0][:, 0], jnp.array(indices))))
            self.assertTrue(bool(jnp.allclose(stencil[1][:, 0], jnp.array(weights), atol=1e-15)))
            particle = particles_at(jnp.linspace(2, 12, 101), shape)
            _, even, odd = particle_deposition_stencil(particle, metric)
            self.assertTrue(bool(jnp.all(even >= 0)))
            self.assertTrue(bool(jnp.allclose(even.sum(axis=0), 1., atol=1e-15)))
            self.assertTrue(bool(jnp.array_equal(even, odd)))

    def test_origin_reflection_changes_only_odd_signs(self):
        metric = metric_grid()
        x = jnp.array([.13, .37, .82, 1.13, 2.72])
        for shape in ("nearest", "linear", "quadratic"):
            direct = analytic_shape((x[None, :] - metric.r[:, None]) / metric.dr, shape)
            reflected = analytic_shape((x[None, :] + metric.r[:, None]) / metric.dr, shape)
            even, odd = dense_weights(particle_deposition_stencil(particles_at(x, shape), metric), 16)
            mirrored_even, mirrored_odd = dense_weights(
                particle_deposition_stencil(particles_at(-x, shape), metric), 16)
            self.assertTrue(bool(jnp.allclose(even, direct + reflected, atol=1e-15)))
            self.assertTrue(bool(jnp.allclose(odd, direct - reflected, atol=1e-15)))
            self.assertTrue(bool(jnp.allclose(even, mirrored_even, atol=1e-15)))
            self.assertTrue(bool(jnp.allclose(odd, -mirrored_odd, atol=1e-15)))
            center_even, center_odd = dense_weights(
                particle_deposition_stencil(particles_at(jnp.array([0.]), shape), metric), 16)
            self.assertAlmostEqual(float(center_even[0, 0]), 1.)
            self.assertTrue(bool(jnp.all(center_odd == 0)))

    def test_open_inner_and_outer_overlap_is_not_renormalized(self):
        metric = metric_grid()
        for boundary in (0, 3):
            # Set a prescribed areal-radius minimum without changing coordinates.
            areal_radius = 1 + (metric.r - metric.r[boundary])**2
            geometry = metric._replace(chi=(metric.r / areal_radius)**2)
            x = jnp.array([-.23, .13, 2.83, 3.23, 15.73, 16.23])
            for shape in ("nearest", "linear", "quadratic"):
                expected = analytic_shape(x[None, :] - metric.r[:, None], shape)
                expected = jnp.where(jnp.arange(16)[:, None] >= boundary, expected, 0.)
                even, odd = dense_weights(particle_deposition_stencil(
                    particles_at(x, shape), geometry, inner_open=True), 16)
                self.assertTrue(bool(jnp.allclose(even, expected, atol=1e-15)))
                self.assertTrue(bool(jnp.array_equal(even, odd)))
                if shape != "nearest":
                    self.assertLess(float(even[:, -1].sum()), 1.)
                    self.assertGreater(float(even[:, -1].sum()), 0.)

    def test_weights_do_not_depend_on_metric_volume_factors(self):
        flat = metric_grid()
        curved = flat._replace(conformal_grr=1 + .04 * flat.r**2,
                               conformal_gt=1 + .03 * flat.r,
                               chi=1 / (1 + .02 * flat.r**2))
        for shape in ("nearest", "linear", "quadratic"):
            particles = particles_at(jnp.array([.13, .82, 2.72, 15.9]), shape)
            for inner_open in (False, True):
                for a, b in zip(particle_deposition_stencil(particles, flat, inner_open),
                                particle_deposition_stencil(particles, curved, inner_open)):
                    self.assertTrue(bool(jnp.array_equal(a, b)))

    def test_quadratic_charge_and_mass_use_ordinary_weights_and_proper_volume(self):
        metric = metric_grid()
        metric = metric._replace(conformal_grr=1 + .04 * metric.r**2,
                                 conformal_gt=1 + .03 * metric.r,
                                 chi=1 / (1 + .02 * metric.r**2))
        x = jnp.array([.13, .37, .82, 1.13, 2.72])
        particles = particles_at(x, "quadratic")
        # Static particles: deposited energy is their rest mass.
        direct = analytic_shape(x[None, :] - metric.r[:, None], "quadratic")
        reflected = analytic_shape(x[None, :] + metric.r[:, None], "quadratic")
        shell_volume = 4 * jnp.pi / 3 * ((metric.r + .5)**3 - (metric.r - .5)**3)
        shell_volume *= jnp.sqrt(metric.conformal_grr) * metric.conformal_gt / metric.chi**1.5
        for inner_open in (False, True):
            weights = direct if inner_open else direct + reflected
            for density_function, amount in ((compute_radial_charge_density, particles.get_charge()),
                                            (relativistic_mass_energy_density, particles.get_mass())):
                expected_shell_amount = (weights * amount[None, :]).sum(axis=1)
                density = density_function(particles, metric, inner_open)
                self.assertTrue(bool(jnp.allclose(density, expected_shell_amount / shell_volume,
                                                 rtol=1e-13, atol=1e-14)))
                self.assertTrue(bool(jnp.allclose(jnp.sum(density * shell_volume),
                                                 jnp.sum(expected_shell_amount), rtol=1e-13)))
                if not inner_open:
                    self.assertTrue(bool(jnp.allclose(jnp.sum(density * shell_volume), jnp.sum(amount))))

    def test_jit_matches_eager_including_empty_particles(self):
        metric = metric_grid()
        compiled = jax.jit(particle_deposition_stencil)
        for shape, width in (("nearest", 1), ("linear", 2), ("quadratic", 3)):
            for x in (jnp.array([]), jnp.array([-.13, 0., .13, 2.72, 16.23])):
                particles = particles_at(x, shape)
                for inner_open in (False, True):
                    eager = particle_deposition_stencil(particles, metric, inner_open)
                    jitted = compiled(particles, metric, inner_open)
                    self.assertEqual(eager[0].shape, (width, x.size))
                    for a, b in zip(eager, jitted):
                        self.assertTrue(bool(jnp.allclose(a, b, atol=1e-15)))

    def test_historical_shape_name_retains_ordinary_fallback(self):
        metric = metric_grid()
        x = jnp.array([.13, .82, 2.72])
        ordinary = particle_deposition_stencil(particles_at(x, "quadratic"), metric)
        legacy = particle_deposition_stencil(
            particles_at(x, "ruyten_density_conserving_quadratic_nonnegative"), metric)
        for a, b in zip(ordinary, legacy):
            self.assertTrue(bool(jnp.array_equal(a, b)))


if __name__ == "__main__":
    unittest.main()
