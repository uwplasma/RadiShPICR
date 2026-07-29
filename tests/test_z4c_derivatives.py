import jax.numpy as jnp

from RadiShPICR.Z4C.derivatives import first_derivative
from RadiShPICR.Z4C.derivatives import second_derivative
from RadiShPICR.Z4C.derivatives import sixth_derivative


def test_outer_boundary_derivatives_match_pointwise_stencils():
    field = jnp.asarray([
        0.4, -1.2, 2.1, 0.7, -0.3, 1.8, 2.5, -0.9, 1.1, 3.2, -2.4, 0.6,
    ])
    dr = 0.17

    dfield_dr = first_derivative(field, dr, parity=1)
    d2field_dr2 = second_derivative(field, dr, parity=1)
    d6field_dr6 = sixth_derivative(field, dr, parity=1)

    expected_dfield_dr = jnp.asarray([
        (
            -1/12 * field[-5]
            + 1/2 * field[-4]
            - 3/2 * field[-3]
            + 5/6 * field[-2]
            + 1/4 * field[-1]
        ) / dr,
        (
            1/4 * field[-5]
            - 4/3 * field[-4]
            + 3 * field[-3]
            - 4 * field[-2]
            + 25/12 * field[-1]
        ) / dr,
    ])
    expected_d2field_dr2 = jnp.asarray([
        (
            1/12 * field[-6]
            - 1/2 * field[-5]
            + 7/6 * field[-4]
            - 1/3 * field[-3]
            - 5/4 * field[-2]
            + 5/6 * field[-1]
        ) / dr**2,
        (
            -5/6 * field[-6]
            + 61/12 * field[-5]
            - 13 * field[-4]
            + 107/6 * field[-3]
            - 77/6 * field[-2]
            + 15/4 * field[-1]
        ) / dr**2,
    ])
    expected_d6field_dr6 = jnp.asarray([
        (
            -field[-8]
            + 8 * field[-7]
            - 27 * field[-6]
            + 50 * field[-5]
            - 55 * field[-4]
            + 36 * field[-3]
            - 13 * field[-2]
            + 2 * field[-1]
        ) / dr**6,
        (
            -2 * field[-8]
            + 15 * field[-7]
            - 48 * field[-6]
            + 85 * field[-5]
            - 90 * field[-4]
            + 57 * field[-3]
            - 20 * field[-2]
            + 3 * field[-1]
        ) / dr**6,
        (
            -3 * field[-8]
            + 22 * field[-7]
            - 69 * field[-6]
            + 120 * field[-5]
            - 125 * field[-4]
            + 78 * field[-3]
            - 27 * field[-2]
            + 4 * field[-1]
        ) / dr**6,
    ])

    assert jnp.allclose(dfield_dr[-2:], expected_dfield_dr)
    assert jnp.allclose(d2field_dr2[-2:], expected_d2field_dr2)
    assert jnp.allclose(d6field_dr6[-3:], expected_d6field_dr6)


def test_centered_interior_derivatives_are_unchanged():
    field = jnp.sin(jnp.linspace(0.2, 2.7, 16)) + 0.1 * jnp.arange(16) ** 2
    dr = 0.2

    expected_dfield_dr = (
        -field[4:] + 8 * field[3:-1] - 8 * field[1:-3] + field[:-4]
    ) / (12 * dr)
    expected_d2field_dr2 = (
        -field[4:]
        + 16 * field[3:-1]
        - 30 * field[2:-2]
        + 16 * field[1:-3]
        - field[:-4]
    ) / (12 * dr**2)
    expected_d6field_dr6 = (
        field[6:]
        - 6 * field[5:-1]
        + 15 * field[4:-2]
        - 20 * field[3:-3]
        + 15 * field[2:-4]
        - 6 * field[1:-5]
        + field[:-6]
    ) / dr**6

    assert jnp.allclose(first_derivative(field, dr, parity=1)[2:-2], expected_dfield_dr)
    assert jnp.allclose(second_derivative(field, dr, parity=1)[2:-2], expected_d2field_dr2)
    assert jnp.allclose(sixth_derivative(field, dr, parity=1)[3:-3], expected_d6field_dr6)


def test_origin_derivatives_keep_even_and_odd_parity_reflection():
    field = jnp.asarray([0.7, -0.2, 1.3, 2.0, -0.6, 0.9, 1.5, -1.1])
    dr = 0.25

    for parity in (1, -1):
        expected_dfield_dr = (
            -field[2]
            + 8 * field[1]
            - 8 * parity * field[0]
            + parity * field[1]
        ) / (12 * dr)
        expected_d2field_dr2 = (
            -field[2]
            + 16 * field[1]
            - 30 * field[0]
            + 16 * parity * field[0]
            - parity * field[1]
        ) / (12 * dr**2)
        expected_d6field_dr6 = (
            field[3]
            - 6 * field[2]
            + 15 * field[1]
            - 20 * field[0]
            + 15 * parity * field[0]
            - 6 * parity * field[1]
            + parity * field[2]
        ) / dr**6

        assert jnp.allclose(first_derivative(field, dr, parity=parity)[0], expected_dfield_dr)
        assert jnp.allclose(second_derivative(field, dr, parity=parity)[0], expected_d2field_dr2)
        assert jnp.allclose(sixth_derivative(field, dr, parity=parity)[0], expected_d6field_dr6)
