import unittest
import jax
import jax.numpy as jnp
from RadiShPICR.Z4C.derivatives import upwind_first_derivative, first_derivative
from RadiShPICR.Z4C.dark_photon_field import dark_photon_time_derivatives
from tests.test_z4c_dark_photon_field import flat_metric

class DarkTransportTests(unittest.TestCase):

    def test_shift_derivative_converges_for_both_directions_and_parities(self):
        for parity in (-1, 1):
            errors = []
            for n in (64, 128, 256):
                m = flat_metric(n, 4.0)
                r = m.r
                f = jnp.sin(r) if parity == -1 else jnp.cos(r)
                exact = jnp.cos(r) if parity == -1 else -jnp.sin(r)
                beta = jnp.where(r < 2, 0.2, -0.2)
                actual = upwind_first_derivative(f, m.dr, beta, parity)
                errors.append(jnp.max(jnp.abs(actual - exact)))
                zero = upwind_first_derivative(f, m.dr, jnp.zeros_like(r), parity)
                self.assertTrue(bool(jnp.array_equal(zero, first_derivative(f, m.dr, parity))))
            for coarse, fine in zip(errors[:-1], errors[1:]):
                self.assertGreater(float(coarse / fine), 14.0)
                self.assertLess(float(coarse / fine), 18.0)

    def test_ko_damps_fourier_modes_and_lowers_radial_components(self):
        m = flat_metric(128, 8.0)
        r = m.r
        m = m._replace(nu=jnp.asarray(0.02), conformal_grr=jnp.full_like(r, 4.0))
        zero = jnp.zeros_like(r)
        mr = jax.tree.map(jnp.zeros_like, m)
        k = 7.0
        fields = (4 * jnp.sin(k * r), 8 * jnp.sin(k * r), jnp.cos(k * r))
        base = dark_photon_time_derivatives(m, mr, fields, zero, zero, 0.3)
        damped = jax.jit(dark_photon_time_derivatives)(m, mr, fields, zero, zero, 0.3, dark_ko=jnp.asarray(True))
        rate = -m.nu / m.dr * jnp.sin(k * m.dr / 2) ** 6
        for x, y, f in zip(base, damped, fields):
            self.assertTrue(bool(jnp.allclose((y - x)[:-3], (rate * f)[:-3], rtol=1e-08, atol=1e-12)))
        no_nu = m._replace(nu=jnp.asarray(0.0))
        off = dark_photon_time_derivatives(no_nu, mr, fields, zero, zero, 0.3, dark_ko=True)
        for x, y in zip(base, off):
            self.assertTrue(bool(jnp.array_equal(x, y)))

    def test_upwind_transport_damps_short_waves_for_either_shift_sign(self):
        m = flat_metric(128, 8.0)
        f = jnp.sin(2.5 * jnp.arange(128))
        for beta in (-0.2, 0.2):
            rhs = beta * upwind_first_derivative(f, m.dr, jnp.full_like(f, beta), -1)
            self.assertLess(float(jnp.sum(f[4:-4] * rhs[4:-4])), 0.0)

    def test_rk4_with_upwind_and_ko_has_fourth_order_time_accuracy(self):
        from jax.scipy.linalg import expm
        from RadiShPICR.Z4C.dark_photon_field import dark_field_outer_boundary
        from RadiShPICR.Z4C.time_evolve import rk4_step
        from RadiShPICR.particles import particle_species
        m = flat_metric(16, 1.0)
        m = m._replace(beta=0.1 * m.r, nu=jnp.asarray(0.02))
        z = jnp.zeros_like(m.r)
        mr = jax.tree.map(jnp.zeros_like, m)
        empty = jnp.empty(0)
        particles = particle_species('vacuum', 0.0, 1.0, empty, empty, empty, empty, empty, 'quadratic')
        initial = dark_field_outer_boundary((z, 0.02 * jnp.sin(6 * jnp.pi * m.r), 0.01 * jnp.cos(6 * jnp.pi * m.r)), m)

        def linear_rhs(v):
            f = dark_field_outer_boundary(tuple(v.reshape(3, 16)), m)
            f = dark_photon_time_derivatives(m, mr, f, z, z, 0.3, dark_ko=True)
            return jnp.stack(dark_field_outer_boundary(f, m)).reshape(-1)
        # Matrix exponentiation provides an independent time integration reference.
        L = jax.jacfwd(linear_rhs)(jnp.zeros(48))
        exact = expm(0.16 * L) @ jnp.stack(initial).reshape(-1)
        step = jax.jit(lambda f, h: rk4_step(particles, m, h, E_r=z, EM_on=False, GR_on=False, dark_fields=f, mu=0.3, dark_ko=True)[-1])
        errors = []
        for h in (0.02, 0.01, 0.005):
            f = initial
            for _ in range(round(0.16 / h)):
                f = step(f, h)
            errors.append(jnp.max(jnp.abs(jnp.stack(f).reshape(-1) - exact)))
        for coarse, fine in zip(errors[:-1], errors[1:]):
            self.assertGreater(float(coarse / fine), 13.0)
            self.assertLess(float(coarse / fine), 19.0)

    def test_outgoing_packet_remains_bounded_at_outer_boundary(self):
        from RadiShPICR.Z4C.time_evolve import rk4_step
        from RadiShPICR.particles import particle_species
        m = flat_metric(256, 10.0)._replace(nu=jnp.asarray(0.02))
        r = m.r
        z = jnp.zeros_like(r)
        empty = jnp.empty(0)
        particles = particle_species('vacuum', 0.0, 1.0, empty, empty, empty, empty, empty, 'quadratic')
        # Outgoing massless potential wave: phi = G'/r, A = G'/r - G/r^2.
        G = jnp.exp(-((r - 8.0) / 0.3) ** 2)
        g = -2 * (r - 8.0) / 0.3 ** 2 * G
        initial = (z, g / r - G / r ** 2, g / r)
        norm = lambda f: jnp.sum(r ** 2 * (f[1] ** 2 + f[2] ** 2)) * m.dr

        @jax.jit
        def advance(f):

            def body(_, state):
                f, peak = state
                f = rk4_step(particles, m, 0.2 * m.dr, E_r=z, EM_on=False, GR_on=False, dark_fields=f, mu=0.0, dark_ko=True)[-1]
                return (f, jnp.maximum(peak, norm(f)))
            return jax.lax.fori_loop(0, 640, body, (f, norm(f)))
        final, peak = advance(initial)
        self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.stack(final)))))
        self.assertLess(float(peak), 2 * float(norm(initial)))
        self.assertLess(float(norm(final)), 0.1 * float(norm(initial)))
