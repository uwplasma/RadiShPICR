"""Stage coupling and evolution checks for the spherical dark field."""

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from RadiShPICR.particles import particle_species
from RadiShPICR.Z4C import rk4_step, deleting_particle_boundary
from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
import RadiShPICR.Z4C.time_evolve as time_evolve


def flat_metric(num_cells=16, extent=8.0):
    dr = extent / num_cells
    r = (jnp.arange(num_cells, dtype=jnp.float64) + 0.5) * dr
    zeros, ones = jnp.zeros_like(r), jnp.ones_like(r)
    return Z4C_Metric(
        alpha=ones, beta=zeros, conformal_grr=ones, conformal_gt=ones,
        chi=ones, Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        kappa=jnp.asarray(0.0), eta=jnp.asarray(0.0), nu=jnp.asarray(0.0),
        r=r, dr=jnp.asarray(dr),
    )


def particles(r=(3.0,), ur=(0.0,), charge=1.0, weight=1.0):
    r = jnp.asarray(r, dtype=jnp.float64)
    return particle_species(
        name="test", charge=jnp.asarray(charge), mass=1.0,
        weight=jnp.broadcast_to(jnp.asarray(weight, dtype=jnp.float64), r.shape),
        r=r, ur=jnp.asarray(ur, dtype=jnp.float64), phi=jnp.zeros_like(r),
        uphi=jnp.zeros_like(r), shape_mode="quadratic",
    )


def keep_boundary(dark_fields, metric, time):
    return dark_fields


class DarkPhotonTimeEvolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_x64 = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", cls.previous_x64)

    def assertClose(self, actual, expected, rtol=1.e-11, atol=1.e-13):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=rtol, atol=atol)),
                        msg=f"actual={actual}, expected={expected}")

    def assertTreeClose(self, actual, expected, rtol=1.e-11, atol=1.e-13):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            self.assertClose(actual_leaf, expected_leaf, rtol=rtol, atol=atol)

    def test_omitted_fields_keep_four_returns_and_zero_dark_matches(self):
        metric = flat_metric(8)
        source = particles(ur=(0.1,), charge=0.1)
        E = 0.02 * metric.r
        zeros = jnp.zeros_like(E)
        step = jax.jit(rk4_step)
        for GR_on in (False, True):
            kwargs = dict(E_r=E, EM_on=jnp.asarray(True), GR_on=jnp.asarray(GR_on))
            ordinary = step(source, metric, 1.e-3, **kwargs)
            dark = step(source, metric, 1.e-3, dark_fields=(zeros, zeros, zeros),
                        mu=0.5, sin_chi=0.0, **kwargs)
            self.assertEqual(len(ordinary), 4)
            self.assertEqual(len(dark), 5)
            self.assertTreeClose(ordinary, dark[:4])
            for field in dark[4]:
                self.assertClose(field, zeros, atol=0.0)

    def test_pair_feels_signed_sum_of_fields_including_em_off(self):
        metric = flat_metric()
        source = particles(r=(3.0, 3.0), ur=(0.0, 0.0),
                           charge=(1.0, -1.0), weight=1.e-6)
        E = jnp.full_like(metric.r, 0.03)
        dark = (jnp.full_like(metric.r, 0.05), jnp.zeros_like(E), jnp.zeros_like(E))
        step = jax.jit(rk4_step, static_argnames=("dark_field_boundary",))
        dt = 1.e-5
        for EM_on in (False, True):
            for sin_chi in (-0.2, 0.2):
                updated = step(source, metric, dt, E_r=E,
                               EM_on=jnp.asarray(EM_on), GR_on=False,
                               dark_fields=dark, sin_chi=sin_chi,
                               dark_field_boundary=keep_boundary)
                acceleration = (0.03 * EM_on + sin_chi * 0.05) * jnp.asarray([1.0, -1.0])
                self.assertClose(updated[0].ur / dt, acceleration, rtol=1.e-8)
                self.assertClose(updated[0].ur[0], -updated[0].ur[1])
                self.assertGreater(float(jnp.max(jnp.abs(updated[0].ur))), 0.0)

    def test_boundaries_force_stress_and_current_use_matching_stages(self):
        metric = flat_metric(8)
        source = particles(r=(1.0,))
        zeros = jnp.zeros_like(metric.r)
        times, positions, gathered, stresses, rhs_inputs = [], [], [], [], []
        dark_rhs = time_evolve.dark_photon_time_derivatives
        dark_stress = time_evolve.dark_energy_momentum_tensor

        def boundary(fields, stage_metric, stage_time):
            times.append(stage_time)
            return jnp.full_like(metric.r, 2.0 + stage_time), zeros, zeros

        def motion(stage_particles, stage_metric):
            zero = jnp.zeros_like(stage_particles.r)
            return zero, zero, jnp.ones_like(zero), zero

        def charge(stage_particles, stage_metric, inner_open=False):
            return jnp.full_like(stage_metric.r, 3.0 + stage_particles.r[0])

        def current(stage_particles, stage_metric, dr_dt, inner_open=False):
            positions.append(stage_particles.r[0])
            self.assertClose(dr_dt, 1.0)
            self.assertTrue(inner_open)
            return jnp.full_like(stage_metric.r, stage_particles.r[0])

        def force(stage_particles, stage_metric, E):
            gathered.append(E)
            return jnp.zeros_like(stage_particles.r)

        def stress(stage_metric, fields, mu, epsilon_0=1.0):
            stresses.append(fields[0])
            return dark_stress(stage_metric, fields, mu, epsilon_0)

        def metric_rhs(stage_metric, matter, *, zero_shift=0):
            return jax.tree.map(jnp.zeros_like, stage_metric)._replace(alpha=matter.rho)

        def record_rhs(stage_metric, metric_derivative, fields, rho, J,
                       mu, sin_chi=0.0, epsilon_0=1.0):
            rhs_inputs.append((fields[0], rho, J, metric_derivative.alpha))
            return dark_rhs(stage_metric, metric_derivative, fields, rho, J,
                            mu, sin_chi, epsilon_0)

        with patch.multiple(
            time_evolve,
            compute_geodesic_terms=motion,
            compute_radial_charge_density=charge,
            compute_radial_current_density=current,
            compute_radial_lorentz_force=force,
            compute_radial_matter_terms=lambda p, m, **kw: initialize_vacuum_matter_terms(m),
            dark_energy_momentum_tensor=stress,
            dark_photon_time_derivatives=record_rhs,
            metric_time_derivatives=metric_rhs,
        ), jax.disable_jit():
            result = rk4_step(source, metric, 0.2, E_r=zeros, EM_on=False,
                              GR_on=True, dark_fields=(zeros, zeros, zeros),
                              sin_chi=0.3, time=0.7, inner_open=True,
                              dark_field_boundary=boundary)

        self.assertClose(jnp.asarray(times), jnp.asarray([0.7, 0.8, 0.8, 0.9, 0.9]))
        self.assertClose(jnp.asarray(positions), jnp.asarray([1.0, 1.1, 1.1, 1.2]))
        expected_E = jnp.asarray([2.7, 2.8, 2.8, 2.9])
        self.assertClose(jnp.asarray(gathered), 0.3 * expected_E[:, None])
        self.assertClose(jnp.asarray(stresses), expected_E[:, None])
        self.assertEqual(len(rhs_inputs), 4)
        for i, (E_prime, rho, J, alpha_rhs) in enumerate(rhs_inputs):
            self.assertClose(E_prime, expected_E[i])
            self.assertClose(J, positions[i])
            self.assertClose(rho, 3.0 + positions[i])
            self.assertClose(alpha_rhs, 0.5 * expected_E[i]**2)
        expected_alpha = 1.0 + 0.2 / 6.0 * jnp.dot(
            jnp.asarray([1.0, 2.0, 2.0, 1.0]), 0.5 * expected_E**2,
        )
        self.assertClose(result[1].alpha, expected_alpha)
        self.assertClose(result[4][0], 2.9)
        self.assertClose(result[2], 4.2)

    def test_covariant_electric_field_uses_actual_evolving_metric(self):
        metric = flat_metric(8)
        source = particles(r=(), ur=())
        E = 0.03 * metric.r
        zeros = jnp.zeros_like(E)

        def expanding_metric(stage_metric, matter, *, zero_shift=0):
            return jax.tree.map(jnp.zeros_like, stage_metric)._replace(chi=2.0 * stage_metric.chi)

        with patch.object(time_evolve, "metric_time_derivatives", expanding_metric):
            # This fresh closure prevents mock equations entering another JIT cache.
            step = jax.jit(lambda p, m, dt, gr: rk4_step(
                p, m, dt, E_r=zeros, EM_on=False, GR_on=gr,
                dark_fields=(E, zeros, zeros), dark_field_boundary=keep_boundary,
            ))
            static = step(source, metric, 0.1, jnp.asarray(False))
            evolved = step(source, metric, 0.1, jnp.asarray(True))
        self.assertClose(static[4][0], E)
        self.assertTreeClose(static[1], metric, rtol=0.0, atol=0.0)
        # With K=beta=mu=0 the contravariant E is constant. Covariant E'=a'/a E=-2E.
        rk_minus = 1.0 - 0.2 + 0.2**2 / 2.0 - 0.2**3 / 6.0 + 0.2**4 / 24.0
        rk_plus = 1.0 + 0.2 + 0.2**2 / 2.0 + 0.2**3 / 6.0 + 0.2**4 / 24.0
        self.assertClose(evolved[4][0], E * rk_minus)
        self.assertClose(evolved[1].chi, rk_plus)

    def test_coupled_current_force_oscillator_converges_at_fourth_order(self):
        metric = flat_metric(8)
        zeros, ones = jnp.zeros_like(metric.r), jnp.ones_like(metric.r)
        sin_chi = 0.4

        def motion(stage_particles, stage_metric):
            zero = jnp.zeros_like(stage_particles.r)
            return zero, zero, stage_particles.ur, zero

        def current(stage_particles, stage_metric, dr_dt, inner_open=False):
            return jnp.full_like(stage_metric.r, dr_dt[0])

        def force(stage_particles, stage_metric, E):
            return jnp.full_like(stage_particles.ur, E[0])

        # The prescribed gather/deposit gives u'=E+s E', E_dot=-u,
        # E'_dot=-s u. This checks the coupled RK tableau against an exact mode.
        with patch.multiple(time_evolve, compute_geodesic_terms=motion,
                            compute_radial_current_density=current,
                            compute_radial_lorentz_force=force):
            step = jax.jit(lambda p, E, dark, dt: rk4_step(
                p, metric, dt, E_r=E, EM_on=True, GR_on=False,
                dark_fields=dark, mu=0.0, sin_chi=sin_chi,
                dark_field_boundary=keep_boundary,
            ))
            omega = jnp.sqrt(1.0 + sin_chi**2)
            displacement = sin_chi / omega**2 * (1.0 - jnp.cos(omega))
            exact = jnp.asarray([sin_chi / omega * jnp.sin(omega),
                                 -displacement, 1.0 - sin_chi * displacement,
                                 3.0 + displacement])
            errors = []
            for steps in (5, 10, 20):
                source = particles()
                E, dark = zeros, (ones, zeros, zeros)
                for _ in range(steps):
                    source, _, _, E, dark = step(source, E, dark, 1.0 / steps)
                result = jnp.asarray([source.ur[0], E[0], dark[0][0], source.r[0]])
                errors.append(jnp.max(jnp.abs(result - exact)))
        errors = jnp.asarray(errors)
        ratios = errors[:-1] / errors[1:]
        self.assertTrue(bool(jnp.all((ratios > 14.0) & (ratios < 18.0))),
                        msg=f"coupled RK4 errors={errors}, ratios={ratios}")

    def test_absorption_does_not_erase_dark_field(self):
        metric = flat_metric(8)
        source = particles(r=(-0.1,), ur=(-0.2,))
        vacuum = particles(r=(), ur=())
        E = 0.03 * metric.r
        zeros = jnp.zeros_like(E)
        dark = (E, zeros, zeros)
        step = jax.jit(rk4_step, static_argnames=("particle_boundary", "dark_field_boundary"))
        for dt in (0.0, 0.02):
            kwargs = dict(E_r=zeros, EM_on=False, GR_on=False, dark_fields=dark,
                          sin_chi=0.2, particle_boundary=deleting_particle_boundary,
                          dark_field_boundary=keep_boundary)
            absorbed = step(source, metric, dt, **kwargs)
            empty = step(vacuum, metric, dt, **kwargs)
            self.assertClose(absorbed[0].weight, 0.0, atol=0.0)
            self.assertClose(absorbed[2], 0.0, atol=0.0)
            self.assertClose(absorbed[4][0], E)
            self.assertTreeClose(absorbed[4], empty[4])

    def test_empty_species_runtime_flags_and_eager_jit_agree(self):
        metric = flat_metric(8)
        r = metric.r
        dark = (0.001 * r * jnp.exp(-r**2), 0.0002 * r * jnp.exp(-r**2),
                0.0001 * jnp.exp(-r**2))
        source = particles(r=(), ur=())
        step = jax.jit(rk4_step)
        for EM_on in (False, True):
            for GR_on in (False, True):
                kwargs = dict(E_r=jnp.zeros_like(r), EM_on=jnp.asarray(EM_on),
                              GR_on=jnp.asarray(GR_on), dark_fields=dark, mu=0.5,
                              sin_chi=0.2, zero_shift=jnp.asarray(1))
                with jax.disable_jit():
                    eager = rk4_step(source, metric, 1.e-3, **kwargs)
                compiled = step(source, metric, 1.e-3, **kwargs)
                self.assertTreeClose(compiled, eager)
                self.assertEqual(compiled[0].r.shape, (0,))
                self.assertClose(compiled[2], 0.0, atol=0.0)
                self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in compiled[4]))
                if not GR_on:
                    self.assertTreeClose(compiled[1], metric, rtol=0.0, atol=0.0)

    def test_gr_off_preserves_unprojected_geometry_with_nonzero_shift(self):
        metric = flat_metric(8)
        metric = metric._replace(conformal_grr=2.0 * metric.conformal_grr,
                                 conformal_gt=3.0 * metric.conformal_gt,
                                 beta=0.01 * metric.r)
        zeros = jnp.zeros_like(metric.r)
        E = 0.03 * metric.r
        result = jax.jit(rk4_step)(particles(r=(), ur=()), metric, 0.01,
                                  E_r=zeros, EM_on=False, GR_on=False,
                                  dark_fields=(E, zeros, zeros), mu=0.5)
        self.assertTreeClose(result[1], metric, rtol=0.0, atol=0.0)
        self.assertFalse(bool(jnp.allclose(result[4][0][:-2], E[:-2], rtol=1.e-8, atol=1.e-14)))
        self.assertClose(result[3], zeros, atol=0.0)

    def test_default_outer_boundary_transmits_outgoing_massless_pulse(self):
        metric = flat_metric(128, extent=12.0)
        r = metric.r
        # psi=f(r-t)/r solves the spherical wave equation away from the origin;
        # A=-d_r psi and phi=d_t psi obey the mu=0 potential equations with E=0.
        f = jnp.exp(-((r - 8.0) / 0.5)**2)
        df = -2.0 * (r - 8.0) / 0.5**2 * f
        zeros = jnp.zeros_like(r)
        dark = (zeros, -df / r + f / r**2, -df / r)
        source = particles(r=(), ur=())
        dt, steps = 0.02, 300

        def advance(fields, index):
            result = rk4_step(source, metric, dt, E_r=zeros, EM_on=False,
                              GR_on=False, dark_fields=fields, mu=0.0,
                              time=index * dt)
            return result[4], None

        final, _ = jax.jit(lambda fields: jax.lax.scan(advance, fields, jnp.arange(steps)))(dark)
        # This is a potential-wave norm, not Proca energy (which vanishes here).
        initial_norm = jnp.sum(r**2 * (dark[1]**2 + dark[2]**2))
        final_norm = jnp.sum(r**2 * (final[1]**2 + final[2]**2))
        self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.asarray(final)))))
        self.assertLess(float(final_norm / initial_norm), 0.05)
        self.assertClose(final[0], zeros, atol=0.0)
        self.assertClose(final[2][-1] - final[1][-1], 0.0, atol=1.e-14)


if __name__ == "__main__":
    unittest.main()
