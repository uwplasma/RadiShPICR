import unittest

import jax
import jax.numpy as jnp

from RadiShPICR.particles.shape_factors.common import proper_radial_shell_volume
from RadiShPICR.Z4C.dark_photon_field import (
    dark_field_energy,
    dark_field_outer_boundary,
    dark_gauss_residual,
    dark_photon_time_derivatives,
)
from RadiShPICR.Z4C.electric_field import compute_electrostatic_matter_terms
from RadiShPICR.Z4C.energy_momentum_tensor import dark_energy_momentum_tensor
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def flat_metric(num_cells=64, radius=10.0):
    dr = jnp.asarray(radius / num_cells)
    r = (jnp.arange(num_cells, dtype=jnp.float64) + 0.5) * dr
    ones = jnp.ones_like(r)
    zeros = jnp.zeros_like(r)

    return Z4C_Metric(
        alpha=ones, beta=zeros, conformal_grr=ones, conformal_gt=ones,
        chi=ones, Kh=zeros, Arr=zeros, At=zeros, theta=zeros, Gamma=zeros,
        kappa=jnp.asarray(0.0), eta=jnp.asarray(0.0), nu=jnp.asarray(0.0),
        r=r, dr=dr,
    )


def spherical_bessel(x):
    j0 = jnp.sinc(x / jnp.pi)
    j1 = jnp.where(
        jnp.abs(x) < 0.1,
        x / 3.0 - x**3 / 30.0 + x**5 / 840.0 - x**7 / 45360.0,
        jnp.sin(x) / x**2 - jnp.cos(x) / x,
    )
    return j0, j1


class DarkPhotonFieldTests(unittest.TestCase):
    def assert_close(self, actual, expected, rtol=2.0e-12, atol=2.0e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=rtol, atol=atol)))

    def test_positive_action_tensor_and_energy(self):
        metric = flat_metric()
        a = 2.0 + 0.02 * metric.r**2
        b = 3.0 + 0.03 * metric.r**2
        metric = metric._replace(conformal_grr=a, conformal_gt=b)
        E_r = 0.2 * metric.r
        A_r = -0.3 * metric.r
        phi = 0.4 + 0.01 * metric.r**2
        mu, epsilon_0 = 0.7, 1.3
        fields = E_r, A_r, phi
        matter = dark_energy_momentum_tensor(metric, fields, mu, epsilon_0)

        self.assert_close(matter.rho, epsilon_0 / 2.0 * (
            E_r**2 / a + mu**2 * (phi**2 + A_r**2 / a)
        ))
        self.assert_close(matter.Srr, epsilon_0 / 2.0 * (
            -E_r**2 + mu**2 * (A_r**2 + a * phi**2)
        ))
        self.assert_close(matter.Stt, epsilon_0 * b / 2.0 * (
            E_r**2 / a + mu**2 * (phi**2 - A_r**2 / a)
        ))
        self.assert_close(matter.Sr, epsilon_0 * mu**2 * phi * A_r / a)
        self.assert_close(matter.St, 0.0)
        self.assertTrue(bool(jnp.all(matter.Sr < 0.0)))
        self.assert_close(
            dark_field_energy(metric, fields, mu, epsilon_0),
            jnp.sum(matter.rho * proper_radial_shell_volume(metric)),
        )

        massless = dark_energy_momentum_tensor(metric, fields, 0.0, epsilon_0)
        ordinary = compute_electrostatic_matter_terms(metric, E_r, epsilon_0)
        for actual, expected in zip(massless, ordinary):
            self.assert_close(actual, expected)

    def test_scalar_field_energy_has_the_lapse_and_proper_volume_factors(self):
        metric = flat_metric(num_cells=32, radius=4.0)
        zero = jnp.zeros_like(metric.r)
        phi, mu, epsilon_0 = 1.3, 0.6, 1.7
        fields = zero, zero, jnp.full_like(metric.r, phi)

        # A spatially constant scalar has constant local mass-energy density.
        # The flat case retains the old value; a constant lapse redshifts it.
        for lapse, spatial_scale in ((1.0, 1.0), (0.7, 4.0)):
            background = metric._replace(
                alpha=jnp.full_like(metric.r, lapse),
                conformal_grr=jnp.full_like(metric.r, spatial_scale),
                conformal_gt=jnp.full_like(metric.r, spatial_scale),
            )
            proper_volume = 4.0 * jnp.pi / 3.0 * 4.0**3 * spatial_scale**1.5
            expected = lapse * 0.5 * epsilon_0 * mu**2 * phi**2 * proper_volume
            self.assert_close(dark_field_energy(background, fields, mu, epsilon_0), expected)
            self.assert_close(dark_field_energy(background, fields, 0.0, epsilon_0), 0.0)

    def test_energy_matches_the_four_dimensional_stress_contraction(self):
        metric = flat_metric(num_cells=32, radius=4.0)
        r, dr = metric.r, metric.dr
        zero = jnp.zeros_like(r)
        alpha = 0.8 + 0.01 * r**2
        a = 2.0 + 0.05 * r**2
        b = 2.0 + 0.02 * r**2
        chi = 0.9 + 0.02 * r**2
        E_r, A_r, phi = 0.2 * r, 0.3 * r, 0.4 + 0.01 * r**2
        fields = E_r, A_r, phi
        epsilon_0 = 1.7
        compiled_energy = jax.jit(dark_field_energy)

        shell_volume = 4.0 * jnp.pi / 3.0 * (
            (r + 0.5 * dr)**3 - (r - 0.5 * dr)**3
        ) * jnp.sqrt(a) * b

        for shift_rate in (-0.03, 0.03):
            beta = shift_rate * r
            background = metric._replace(
                alpha=alpha, beta=beta, conformal_grr=a * chi,
                conformal_gt=b * chi, chi=chi,
            )

            # Construct g_mu_nu, F_mu_nu, and A_mu at theta = pi/2.
            # This reference uses the covariant Proca stress tensor directly,
            # without the production Eulerian rho and S^r decomposition.
            g = jnp.zeros((r.size, 4, 4))
            g = g.at[:, 0, 0].set(-alpha**2 + a * beta**2)
            g = g.at[:, 0, 1].set(a * beta)
            g = g.at[:, 1, 0].set(a * beta)
            g = g.at[:, 1, 1].set(a)
            g = g.at[:, 2, 2].set(b * r**2)
            g = g.at[:, 3, 3].set(b * r**2)
            g_inverse = jnp.linalg.inv(g)
            F = jnp.zeros_like(g)
            F = F.at[:, 0, 1].set(-alpha * E_r)
            F = F.at[:, 1, 0].set(alpha * E_r)
            A = jnp.stack((beta * A_r - alpha * phi, A_r, zero, zero), axis=1)
            F_raised = g_inverse @ F @ g_inverse
            F_squared = jnp.sum(F * F_raised, axis=(1, 2))
            A_squared = jnp.einsum("ni,nij,nj->n", A, g_inverse, A)

            for mu in (0.0, 0.6):
                stress = epsilon_0 * (
                    F @ g_inverse @ jnp.swapaxes(F, 1, 2)
                    - 0.25 * g * F_squared[:, None, None]
                    + mu**2 * (
                        A[:, :, None] * A[:, None, :]
                        - 0.5 * g * A_squared[:, None, None]
                    )
                )
                # Contract T_mu_nu n^mu (partial_t)^nu, including n^r=-beta/alpha.
                energy_density = (stress[:, 0, 0] - beta * stress[:, 1, 0]) / alpha
                expected = jnp.sum(energy_density * shell_volume)
                eager = dark_field_energy(background, fields, mu, epsilon_0)
                compiled = compiled_energy(background, fields, mu, epsilon_0)
                self.assert_close(eager, expected)
                self.assert_close(compiled, expected)

    def test_current_conversion_and_actual_metric_rate_with_shift(self):
        metric = flat_metric()
        r = metric.r
        a = 2.0 + 0.1 * r**2
        b = 3.0 + 0.05 * r**2
        alpha = 0.8 + 0.01 * r**2
        beta = 0.03 * r
        metric = metric._replace(
            conformal_grr=a, conformal_gt=b, alpha=alpha, beta=beta,
            Kh=jnp.full_like(r, 0.12), theta=jnp.full_like(r, 0.03),
        )
        metric_rhs = jax.tree.map(jnp.zeros_like, metric)._replace(
            conformal_grr=0.04 * a, chi=-0.02 * metric.chi,
        )
        E_up = 0.3 * r
        A_up = 0.4 * r
        phi = 0.2 + 0.02 * r**2
        charge = 0.1 + 0.03 * r**2
        current = 0.07 * r
        mu, sin_chi, epsilon_0 = 0.6, -0.2, 1.7
        fields = a * E_up, a * A_up, phi
        rhs = dark_photon_time_derivatives(
            metric, metric_rhs, fields, charge, current, mu, sin_chi, epsilon_0,
        )

        E_up_dt = (
            alpha * 0.18 * E_up
            - sin_chi * (current + beta * charge) / epsilon_0
            + alpha * mu**2 * A_up
        )
        alpha_phi_dr = 0.036 * r + 0.0008 * r**3
        # Differentiate the stored covector A_r = 0.8*r + 0.04*r**3
        # analytically, independently of raising and lowering in the RHS.
        A_r_dt = (
            beta * (0.8 + 0.12 * r**2) + 0.03 * fields[1]
            - alpha * fields[0] - alpha_phi_dr
        )
        divergence_alpha_A = (
            0.32 + 0.012 * r**2
            + (2.0 / r + 0.1 * r / a + 0.1 * r / b) * alpha * A_up
        )
        phi_dt = beta * 0.04 * r - divergence_alpha_A + alpha * 0.18 * phi

        self.assert_close(rhs[0], a * E_up_dt + 0.06 * a * E_up)
        self.assert_close(rhs[1], A_r_dt)
        self.assert_close(rhs[2], phi_dt)
        compiled = jax.jit(dark_photon_time_derivatives)(
            metric, metric_rhs, fields, charge, current, mu, sin_chi, epsilon_0,
        )
        for actual, expected in zip(compiled, rhs):
            self.assert_close(actual, expected)

    def test_covariant_potential_does_not_acquire_an_expansion_term(self):
        metric = flat_metric()
        r = metric.r
        zero = jnp.zeros_like(r)
        H = 0.1
        metric = metric._replace(Kh=jnp.full_like(r, -3.0 * H))
        zero_rhs = jax.tree.map(jnp.zeros_like, metric)
        expanding_rhs = zero_rhs._replace(chi=jnp.full_like(r, -2.0 * H))

        # For gamma_ij = exp(2Ht) delta_ij, A_r has no geometric source:
        # beta = E_r = phi = 0 implies d_t A_r = 0, even though gamma evolves.
        for metric_rhs in (zero_rhs, expanding_rhs):
            rhs = dark_photon_time_derivatives(
                metric, metric_rhs, (zero, r, zero), zero, zero, 0.5,
            )
            self.assert_close(rhs[1], zero)

    def test_covariant_potential_has_the_covector_shift_sign(self):
        metric = flat_metric()
        r = metric.r
        zero = jnp.zeros_like(r)
        H = 0.1
        metric = metric._replace(
            beta=H * r, Kh=jnp.full_like(r, 3.0 * H),
        )
        zero_rhs = jax.tree.map(jnp.zeros_like, metric)
        changing_rhs = zero_rhs._replace(chi=jnp.full_like(r, -0.2))

        # The stationary metric has K_ij = H gamma_ij. Its covector Lie
        # derivative is beta*d_r(r) + r*d_r(beta) = 2H*r, not zero.
        for metric_rhs in (zero_rhs, changing_rhs):
            rhs = dark_photon_time_derivatives(
                metric, metric_rhs, (zero, r, zero), zero, zero, 0.5,
            )
            self.assert_close(rhs[1], 2.0 * H * r)

    def test_separate_conformal_factors_have_fourth_order_rhs_and_gauss(self):
        rhs_errors, gauss_errors = [], []
        mu, shift_rate = 0.5, 0.03
        for num_cells in (64, 128):
            metric = flat_metric(num_cells, radius=2.0)
            r = metric.r
            zero = jnp.zeros_like(r)
            grr = 1.0 + 0.1 * r**2
            gt = 1.0 + 0.05 * r**2
            chi = 1.0 + 0.2 * r**2
            a = grr / chi
            beta = shift_rate * r

            # Analytic derivatives of the physical volume and radial metric.
            # All three stored conformal factors vary independently.
            dln_a_dr = 0.2 * r / grr - 0.4 * r / chi
            volume_rate = 2.0 / r + 0.1 * r / grr + 0.1 * r / gt - 0.6 * r / chi
            K = shift_rate + beta * volume_rate
            metric = metric._replace(
                conformal_grr=grr, conformal_gt=gt, chi=chi, beta=beta, Kh=K,
            )
            zero_rhs = jax.tree.map(jnp.zeros_like, metric)
            E_up, A_up, phi = jnp.sin(r), jnp.sin(2.0 * r), jnp.cos(r)
            fields = a * E_up, a * A_up, phi

            # These fields are smooth manufactured data on a stationary
            # background. Compare each RHS with its continuum derivative.
            dA_r_dr = a * (2.0 * jnp.cos(2.0 * r) + dln_a_dr * A_up)
            exact_rhs = (
                a * (beta * jnp.cos(r) - shift_rate * E_up + K * E_up + mu**2 * A_up),
                beta * dA_r_dr + shift_rate * fields[1] - fields[0] + jnp.sin(r),
                -beta * jnp.sin(r) - 2.0 * jnp.cos(2.0 * r) - volume_rate * A_up + K * phi,
            )
            exact_gauss = jnp.cos(r) + volume_rate * E_up + mu**2 * phi
            rhs = dark_photon_time_derivatives(
                metric, zero_rhs, fields, zero, zero, mu,
            )
            gauss = dark_gauss_residual(metric, fields, zero, mu)

            # Keep the origin cells; exclude only the two one-sided outer
            # stencils when measuring the interior convergence rate.
            rhs_errors.append(jnp.max(jnp.abs(
                jnp.stack(rhs)[:, :-2] - jnp.stack(exact_rhs)[:, :-2]
            ), axis=1))
            gauss_errors.append(jnp.max(jnp.abs(gauss[:-2] - exact_gauss[:-2])))

        ratios = rhs_errors[0] / rhs_errors[1]
        self.assertTrue(bool(jnp.all((ratios > 14.0) & (ratios < 18.0))),
                        msg=f"curved RHS refinement ratios={ratios}")
        self.assertGreater(float(gauss_errors[0] / gauss_errors[1]), 14.0)

    def test_cylindrical_puncture_divergence_includes_the_first_cell(self):
        metric = flat_metric()
        r = metric.r
        zero = jnp.zeros_like(r)
        shift_rate = 0.1
        metric = metric._replace(chi=r**2, beta=shift_rate * r)
        zero_rhs = jax.tree.map(jnp.zeros_like, metric)
        fields = 0.3 * r, 0.4 * r, jnp.full_like(r, 0.7)
        mu, epsilon_0 = 0.5, 1.7

        # gamma_rr = gamma_theta_theta/r**2 = r**(-2) gives cylindrical
        # proper volume dV proportional to dr/r. Thus div(c*r**3)=2*c*r**2.
        # chi itself is regular and its parity derivative is exact at cell 0.
        rhs = dark_photon_time_derivatives(
            metric, zero_rhs, fields, zero, zero, mu, epsilon_0=epsilon_0,
        )
        gauss = dark_gauss_residual(
            metric, fields, zero, mu, epsilon_0=epsilon_0,
        )
        self.assert_close(rhs[0], 2.0 * shift_rate * fields[0] + mu**2 * fields[1])
        self.assert_close(rhs[1], 2.0 * shift_rate * fields[1] - fields[0])
        self.assert_close(rhs[2], -0.8 * r**2)
        self.assert_close(gauss, epsilon_0 * (0.6 * r**2 + mu**2 * fields[2]))

    def test_regular_origin_and_proca_mass_signs(self):
        metric = flat_metric()
        zero_rhs = jax.tree.map(jnp.zeros_like, metric)
        r = metric.r
        zero = jnp.zeros_like(r)
        fields = 0.3 * r, 0.4 * r, jnp.full_like(r, 0.7)
        mu = 0.5
        rhs = dark_photon_time_derivatives(
            metric, zero_rhs, fields, zero, zero, mu,
        )

        self.assert_close(rhs[0], mu**2 * fields[1])
        self.assert_close(rhs[1], -fields[0])
        self.assert_close(rhs[2], -1.2)
        # Include the first cell: parity must provide the regular origin
        # limit div(c*r) = 3*c without a separate origin exclusion.
        residual = dark_gauss_residual(metric, fields, zero, mu)
        self.assert_close(residual, 0.9 + mu**2 * 0.7)

    def test_gauss_current_normalization_in_curved_geometry(self):
        metric = flat_metric()
        r = metric.r
        a = 2.0 + 0.1 * r**2
        b = 3.0 + 0.05 * r**2
        metric = metric._replace(conformal_grr=a, conformal_gt=b)
        phi = 0.2 + 0.03 * r**2
        E_up = 0.4 * r
        mu, sin_chi, epsilon_0 = 0.5, 0.3, 1.7
        divergence_E = 0.4 + (2.0 / r + 0.1 * r / a + 0.1 * r / b) * E_up
        charge = epsilon_0 * (divergence_E + mu**2 * phi) / sin_chi
        fields = a * E_up, jnp.zeros_like(r), phi

        self.assert_close(
            dark_gauss_residual(metric, fields, charge, mu, sin_chi, epsilon_0),
            0.0,
        )

    def test_spherical_mode_fourth_order_rhs_and_gauss(self):
        rhs_errors, gauss_errors = [], []
        mu, k, E0, time = 0.5, 1.0, 1.0e-3, 0.4
        # Proca dispersion: this mode also annihilates D_i E'^i + mu**2 phi'.
        omega = jnp.sqrt(k**2 + mu**2)
        for num_cells in (128, 256):
            metric = flat_metric(num_cells)
            zero = jnp.zeros_like(metric.r)
            zero_rhs = jax.tree.map(jnp.zeros_like, metric)
            j0, j1 = spherical_bessel(k * metric.r)
            fields = (
                E0 * j1 * jnp.cos(omega * time),
                -omega * E0 / mu**2 * j1 * jnp.sin(omega * time),
                -k * E0 / mu**2 * j0 * jnp.cos(omega * time),
            )
            exact_rhs = (
                -omega * E0 * j1 * jnp.sin(omega * time),
                -omega**2 * E0 / mu**2 * j1 * jnp.cos(omega * time),
                k * omega * E0 / mu**2 * j0 * jnp.sin(omega * time),
            )
            rhs = dark_photon_time_derivatives(
                metric, zero_rhs, fields, zero, zero, mu,
            )
            # Exclude only the two analytically prescribed outer cells.
            # The origin-adjacent cells remain in these maximum norms.
            rhs_errors.append(jnp.max(jnp.abs(
                jnp.stack(rhs)[:, :-2] - jnp.stack(exact_rhs)[:, :-2]
            )))
            gauss_errors.append(jnp.max(jnp.abs(
                dark_gauss_residual(metric, fields, zero, mu)[:-2]
            )))
        self.assertGreater(float(rhs_errors[0] / rhs_errors[1]), 14.0)
        self.assertGreater(float(gauss_errors[0] / gauss_errors[1]), 14.0)

    def test_outer_boundary_preserves_outgoing_pulse(self):
        metric = flat_metric()
        r = metric.r
        metric = metric._replace(conformal_grr=jnp.full_like(r, 4.0))
        pulse = jnp.exp(-((r - 9.0) / 0.8)**2)
        outgoing = pulse, 2.0 * pulse, pulse
        bounded = dark_field_outer_boundary(outgoing, metric)
        for actual, expected in zip(bounded, outgoing):
            self.assert_close(actual, expected)

        incoming = pulse, -2.0 * pulse, pulse
        bounded = dark_field_outer_boundary(incoming, metric)
        self.assert_close(bounded[0], pulse)
        self.assert_close(bounded[1][-1], 0.0)
        self.assert_close(bounded[2][-1], 0.0)
        for actual, expected in zip(bounded, incoming):
            self.assert_close(actual[:-1], expected[:-1])

    def test_outer_boundary_characteristic_speeds_and_jit(self):
        metric = flat_metric()
        fields = (0.2 * metric.r, 0.3 * metric.r, jnp.ones_like(metric.r))
        for beta, all_inward in ((2.0, True), (-2.0, False)):
            shifted = metric._replace(beta=jnp.full_like(metric.r, beta))
            bounded = dark_field_outer_boundary(fields, shifted)
            compiled = jax.jit(dark_field_outer_boundary)(fields, shifted)
            for original, actual, jit_actual in zip(fields, bounded, compiled):
                self.assert_close(actual[:-1], original[:-1])
                self.assert_close(actual[-1], 0.0 if all_inward else original[-1])
                self.assert_close(jit_actual, actual)


if __name__ == "__main__":
    unittest.main()
