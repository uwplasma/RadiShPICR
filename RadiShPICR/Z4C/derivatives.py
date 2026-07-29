import jax.numpy as jnp
import jax

@jax.jit
def first_derivative(field, dr, parity=-1):
    # compute finite difference derivative
    # reduce to 4th order finite difference in the interior, and use forward/backward difference at the boundaries

    Nr = field.shape[0]
    # get the number of grid points

    field_ = jnp.zeros(shape=(Nr+2), dtype=field.dtype) # initialize dummy array
    field_ = field_.at[2:].set(field) # add field

    field_ = field_.at[0].set(field_[3] * parity)  # set the first point using parity
    field_ = field_.at[1].set(field_[2] * parity)  # set the second point using parity

    derivative  = (-jnp.roll(field_, -2) + 8 * jnp.roll(field_, -1) - 8 * jnp.roll(field_, 1) + jnp.roll(field_, 2)) / (12 * dr)
    # define the fourth order finite difference for the first derivative


    #### Boundary conditions ####
    right_boundary_derivative = (
        1/4    * field[-5]
        - 4/3 * field[-4]
        + 3   * field[-3]
        - 4   * field[-2]
        + 25/12 * field[-1]
    ) / dr
    # define the one-sided finite difference for the first derivative at the right boundary

    second_to_right_derivative = (
        -1/12 * field[-5]
        + 1/2 * field[-4]
        - 3/2 * field[-3]
        + 5/6 * field[-2]
        + 1/4 * field[-1]
    ) / dr
    # define the one-sided finite difference at the second point from the right boundary

    derivative = derivative.at[-1].set(right_boundary_derivative)
    derivative = derivative.at[-2].set(second_to_right_derivative)
    # set the last two points to the one-sided finite difference values to avoid boundary issues


    return derivative[2:]

@jax.jit
def second_derivative(field, dr, parity=-1):
    # compute finite difference derivative
    # assume total periodic domain for now

    Nr = field.shape[0]
    # get the number of grid points


    field_ = jnp.zeros(shape=(Nr+2), dtype=field.dtype) # initialize dummy array
    field_ = field_.at[2:].set(field) # add field

    field_ = field_.at[0].set(field_[3] * parity)  # set the first point using parity
    field_ = field_.at[1].set(field_[2] * parity)  # set the second point using parity

    derivative = (-jnp.roll(field_, -2) + 16 * jnp.roll(field_, -1) - 30 * field_ + 16 * jnp.roll(field_, 1) - jnp.roll(field_, 2)) / (12 * dr ** 2)
    # define the fourth order finite difference for the second derivative


    ## Boundary conditions ##
    second_to_right_derivative = (
        1/12 * field[-6]
        - 1/2 * field[-5]
        + 7/6 * field[-4]
        - 1/3 * field[-3]
        - 5/4 * field[-2]
        + 5/6 * field[-1]
    ) / (dr ** 2)
    # define the one-sided finite difference at the second point from the right boundary

    right_boundary_derivative = (
        -5/6 * field[-6]
        + 61/12 * field[-5]
        - 13 * field[-4]
        + 107/6 * field[-3]
        - 77/6 * field[-2]
        + 15/4 * field[-1]
    ) / (dr ** 2)
    # define the one-sided finite difference for the second derivative at the right boundary

    derivative = derivative.at[-1].set(right_boundary_derivative)
    derivative = derivative.at[-2].set(second_to_right_derivative)
    # set the last two points to the one-sided finite difference values to avoid boundary issues

    return derivative[2:]

@jax.jit
def sixth_derivative(field, dr, parity=-1):
    # compute 6th order finite difference derivative
    # assume total periodic domain for now

    Nr = field.shape[0]
    # get the number of grid points

    field_ = jnp.zeros(shape=(Nr+3), dtype=field.dtype) # initialize dummy array
    field_ = field_.at[3:].set(field) # add field
    
    field_ = field_.at[0].set(field_[5] * parity)  # set the first point using parity
    field_ = field_.at[1].set(field_[4] * parity)  # set the second point using parity
    field_ = field_.at[2].set(field_[3] * parity)  # set the third point using parity

    derivative =  (jnp.roll(field_, -3) - 6 * jnp.roll(field_, -2) + 15 * jnp.roll(field_, -1) - 20 * field_ + 15 * jnp.roll(field_, 1) - 6 * jnp.roll(field_, 2) + jnp.roll(field_, 3)) / (dr ** 6)
    # define the finite difference for the 6th derivative

    right_boundary_derivative = (
        -3 * field[-8]
        + 22 * field[-7]
        - 69 * field[-6]
        + 120 * field[-5]
        - 125 * field[-4]
        + 78 * field[-3]
        - 27 * field[-2]
        + 4 * field[-1]
    ) / (dr ** 6)
    # define the one-sided finite difference for the 6th derivative at the right boundary

    second_to_right_derivative = (
        -2 * field[-8]
        + 15 * field[-7]
        - 48 * field[-6]
        + 85 * field[-5]
        - 90 * field[-4]
        + 57 * field[-3]
        - 20 * field[-2]
        + 3 * field[-1]
    ) / (dr ** 6)
    # define the one-sided finite difference at the second point from the right boundary

    third_to_right_derivative = (
        -field[-8]
        + 8 * field[-7]
        - 27 * field[-6]
        + 50 * field[-5]
        - 55 * field[-4]
        + 36 * field[-3]
        - 13 * field[-2]
        + 2 * field[-1]
    ) / (dr ** 6)
    # define the one-sided finite difference at the third point from the right boundary


    derivative = derivative.at[-1].set(right_boundary_derivative)
    derivative = derivative.at[-2].set(second_to_right_derivative)
    derivative = derivative.at[-3].set(third_to_right_derivative)
    # set the last three points to the one-sided finite difference values to avoid boundary issues

    return derivative[3:]
