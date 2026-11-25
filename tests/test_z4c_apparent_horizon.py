import unittest

import jax
import jax.numpy as jnp

from RadiShPICR.Z4C import (
    Z4C_Metric,
    find_apparent_horizon,
    misner_sharp_mass,
    outgoing_expansion,
)


def flat_metric(r):
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)
    return Z4C_Metric(
        alpha=ones, beta=zeros, conformal_grr=ones, conformal_gt=ones,
        chi=ones, Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        kappa=zeros, eta=zeros, nu=zeros, r=r, dr=r[1] - r[0],
    )


class TestApparentHorizon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        cls.addClassCleanup(jax.config.update, "jax_enable_x64", previous)

    def test_flat_space_expansion_and_no_horizon(self):
        r = (jnp.arange(64) + 0.5) * 0.125
        metric = flat_metric(r)
        self.assertTrue(jnp.allclose(outgoing_expansion(metric), 2.0 / r,
                                     rtol=0.0, atol=1.0e-12))
        self.assertTrue(jnp.all(jnp.isnan(jnp.asarray(find_apparent_horizon(metric)))))

    def test_physical_angular_curvature_includes_theta(self):
        r = (jnp.arange(64) + 0.5) * 0.125
        metric = flat_metric(r)
        ones = jnp.ones_like(r)
        metric = metric._replace(
            conformal_grr=4.0 * ones, conformal_gt=9.0 * ones,
            At=0.6 * ones, Kh=0.4 * ones, theta=0.3 * ones,
        )
        # R = 3r, sqrt(g_rr) = 2, K = 1, K^theta_theta = 0.4.
        expected = 1.0 / r - 0.8
        self.assertTrue(jnp.allclose(outgoing_expansion(metric), expected,
                                     rtol=0.0, atol=1.0e-12))

    def test_schwarzschild_radius_refinement_and_mass(self):
        errors = []
        for count in (200, 400, 800):
            dr = 4.0 / count
            r = (jnp.arange(count) + 0.5) * dr
            metric = flat_metric(r)._replace(chi=(1.0 + 0.5 / r)**-4)
            horizon_r, horizon_R = find_apparent_horizon(metric)
            errors.append(jnp.abs(horizon_r - 0.5))
            self.assertLess(float(jnp.abs(horizon_r - 0.5)), 3.0 * dr**2)
            self.assertLess(float(jnp.abs(horizon_R - 2.0)), 3.0 * dr**2)
            residual = jnp.interp(horizon_r, r, outgoing_expansion(metric))
            self.assertLess(float(jnp.abs(residual)), 1.0e-9)

            mass_at_horizon = jnp.interp(horizon_r, r, misner_sharp_mass(metric))
            self.assertLess(float(jnp.abs(2.0 * mass_at_horizon / horizon_R - 1.0)),
                            3.0 * dr**2)

        # Linear interpolation of the root is second order even though the
        # derivative stencil is fourth order away from the origin.
        self.assertTrue(jnp.all(jnp.asarray(errors[:-1]) / jnp.asarray(errors[1:]) > 3.5))

    def test_outermost_of_multiple_bracketed_roots(self):
        r = (jnp.arange(64) + 0.5) * 0.125
        metric = flat_metric(r)
        desired = (r - 0.37) * (r - 1.31) * (r - 2.73)
        # Manufacture angular curvature with three outgoing-expansion zeros.
        metric = metric._replace(At=0.5 * (outgoing_expansion(metric) - desired))
        horizon_r, horizon_R = find_apparent_horizon(metric)
        self.assertLess(float(jnp.abs(horizon_r - 2.73)), 0.01)
        self.assertAlmostEqual(float(horizon_r), float(horizon_R), places=10)

    def test_exact_roots_including_domain_endpoints(self):
        r = (jnp.arange(64) + 0.5) * 0.125
        for index in (0, 21, 63):
            with self.subTest(index=index):
                metric = flat_metric(r)
                desired = jnp.ones_like(r).at[index].set(0.0)
                metric = metric._replace(At=0.5 * (outgoing_expansion(metric) - desired))
                horizon_r, horizon_R = find_apparent_horizon(metric)
                self.assertEqual(float(horizon_r), float(r[index]))
                self.assertEqual(float(horizon_R), float(r[index]))

    def test_outermost_means_coordinate_radius_when_areal_radius_folds(self):
        r = (jnp.arange(64) + 0.5) * 0.125
        metric = flat_metric(r)._replace(chi=(1.0 + 0.5 / r)**-4)
        desired = jnp.ones_like(r).at[0].set(0.0).at[16].set(0.0)
        metric = metric._replace(At=0.5 * (outgoing_expansion(metric) - desired))
        horizon_r, horizon_R = find_apparent_horizon(metric)
        R = r / jnp.sqrt(metric.chi)
        self.assertGreater(float(R[0]), float(R[16]))
        self.assertEqual(float(horizon_r), float(r[16]))
        self.assertAlmostEqual(float(horizon_R), float(R[16]), places=12)

    def test_everywhere_negative_expansion_has_no_resolved_root(self):
        r = (jnp.arange(64) + 0.5) * 0.125
        metric = flat_metric(r)
        metric = metric._replace(At=0.5 * (outgoing_expansion(metric) + 1.0))
        self.assertTrue(jnp.all(jnp.isnan(jnp.asarray(find_apparent_horizon(metric)))))

    def test_invalid_geometry_does_not_produce_a_horizon(self):
        r = (jnp.arange(400) + 0.5) * 0.01
        metric = flat_metric(r)._replace(chi=(1.0 + 0.5 / r)**-4)
        for value in (jnp.nan, -1.0, 0.0):
            with self.subTest(value=value):
                invalid = metric._replace(chi=metric.chi.at[-1].set(value))
                self.assertTrue(jnp.all(jnp.isnan(jnp.asarray(find_apparent_horizon(invalid)))))
        invalid = metric._replace(conformal_grr=metric.conformal_grr.at[-1].set(jnp.inf))
        self.assertTrue(jnp.all(jnp.isnan(jnp.asarray(find_apparent_horizon(invalid)))))

    def test_iteration_limit_does_not_return_an_unconverged_radius(self):
        r = (jnp.arange(400) + 0.5) * 0.01
        metric = flat_metric(r)._replace(chi=(1.0 + 0.5 / r)**-4)
        result = find_apparent_horizon(metric, max_iterations=0)
        self.assertTrue(jnp.all(jnp.isnan(jnp.asarray(result))))

    def test_eager_and_jit_agree(self):
        r = (jnp.arange(400) + 0.5) * 0.01
        metric = flat_metric(r)._replace(chi=(1.0 + 0.5 / r)**-4)
        compiled = jax.jit(find_apparent_horizon)(metric)
        with jax.disable_jit():
            eager = find_apparent_horizon(metric)
        self.assertTrue(jnp.allclose(jnp.asarray(compiled), jnp.asarray(eager),
                                     rtol=0.0, atol=1.0e-10))


if __name__ == "__main__":
    unittest.main()
