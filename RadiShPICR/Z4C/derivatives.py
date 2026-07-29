import jax.numpy as jnp
import jax

@jax.jit
def first_derivative(field, dr, parity=-1):
    # compute finite difference derivative
    # reduce to 4th order finite difference in the interior, and use forward/backward difference at the boundaries

    Nr = field.shape[0]
    # get the number of grid points

    field_ = jnp.zeros(shape=(Nr+2)) # initialize dummy array
    field_ = field_.at[2:].set(field) # add field

    field_ = field_.at[0].set(field_[3] * parity)  # set the first point using parity
    field_ = field_.at[1].set(field_[2] * parity)  # set the second point using parity

    derivative  = (-jnp.roll(field_, -2) + 8 * jnp.roll(field_, -1) - 8 * jnp.roll(field_, 1) + jnp.roll(field_, 2)) / (12 * dr)
    # define the fourth order finite difference for the first derivative


    #### Boundary conditions ####
    one_side_derivative = 1/4 * jnp.roll(field_, -4)
    one_side_derivative -= 4/3 * jnp.roll(field_, -3) 
    one_side_derivative += 3 * jnp.roll(field_, -2) 
    one_side_derivative -= 4 * jnp.roll(field_, -1) 
    one_side_derivative += 25/12 * field_
    one_side_derivative = one_side_derivative / dr
    # define the one-sided finite difference for the first derivative at the right boundary

    lop_side_derivative = -1/12 * jnp.roll(field_, -3)
    lop_side_derivative += 1/2 * jnp.roll(field_, -2)
    lop_side_derivative += -3/2 * jnp.roll(field_, -1)
    lop_side_derivative += 5/6 * field_
    lop_side_derivative += 1/4 * jnp.roll(field_, 1)
    lop_side_derivative = lop_side_derivative / dr
    # define the one-sided finite difference for the first derivative at the left boundary

    derivative = derivative.at[-1].set(one_side_derivative[-1])
    derivative = derivative.at[-2].set(lop_side_derivative[-2])
    # set the last two points to the one-sided finite difference values to avoid boundary issues


    return derivative[2:]

@jax.jit
def second_derivative(field, dr, parity=-1):
    # compute finite difference derivative
    # assume total periodic domain for now

    Nr = field.shape[0]
    # get the number of grid points


    field_ = jnp.zeros(shape=(Nr+2)) # initialize dummy array
    field_ = field_.at[2:].set(field) # add field

    field_ = field_.at[0].set(field_[3] * parity)  # set the first point using parity
    field_ = field_.at[1].set(field_[2] * parity)  # set the second point using parity

    derivative = (-jnp.roll(field_, -2) + 16 * jnp.roll(field_, -1) - 30 * field_ + 16 * jnp.roll(field_, 1) - jnp.roll(field_, 2)) / (12 * dr ** 2)
    # define the fourth order finite difference for the second derivative


    ## Boundary conditions ##
    lop_side_derivative = 1/12 * jnp.roll(field_, -4)
    lop_side_derivative -= 1/2 * jnp.roll(field_, -3)
    lop_side_derivative += 7/6 * jnp.roll(field_, -2)
    lop_side_derivative -= 1/3 * jnp.roll(field_, -1)
    lop_side_derivative -= 5/4 * field_
    lop_side_derivative += 5/6 * jnp.roll(field_, 1)
    lop_side_derivative = lop_side_derivative / (dr ** 2)
    # define the one-sided finite difference for the second derivative at the left boundary

    one_side_derivative = -5/6   * jnp.roll(field_, -5)
    one_side_derivative += 61/12 * jnp.roll(field_, -4)
    one_side_derivative -= 13    * jnp.roll(field_, -3)
    one_side_derivative += 107/6 * jnp.roll(field_, -2)
    one_side_derivative -= 77/6  * jnp.roll(field_, -1)
    one_side_derivative += 15/4  * field_
    one_side_derivative = one_side_derivative / (dr ** 2)
    # define the one-sided finite difference for the second derivative at the right boundary

    derivative = derivative.at[-1].set(one_side_derivative[-1])
    derivative = derivative.at[-2].set(lop_side_derivative[-2])
    # set the last two points to the one-sided finite difference values to avoid boundary issues

    return derivative[2:]

@jax.jit
def sixth_derivative(field, dr, parity=-1):
    # compute 6th order finite difference derivative
    # assume total periodic domain for now

    Nr = field.shape[0]
    # get the number of grid points

    field_ = jnp.zeros(shape=(Nr+3)) # initialize dummy array
    field_ = field_.at[3:].set(field) # add field
    
    field_ = field_.at[0].set(field_[5] * parity)  # set the first point using parity
    field_ = field_.at[1].set(field_[4] * parity)  # set the second point using parity
    field_ = field_.at[2].set(field_[3] * parity)  # set the third point using parity

    derivative =  (jnp.roll(field_, -3) - 6 * jnp.roll(field_, -2) + 15 * jnp.roll(field_, -1) - 20 * field_ + 15 * jnp.roll(field_, 1) - 6 * jnp.roll(field_, 2) + jnp.roll(field_, 3)) / (dr ** 6)
    # define the finite difference for the 6th derivative

    one_side_derivative = -3   * jnp.roll(field_, -7)
    one_side_derivative += 22  * jnp.roll(field_, -6)
    one_side_derivative -= 69  * jnp.roll(field_, -5)
    one_side_derivative += 120 * jnp.roll(field_, -4)
    one_side_derivative -= 125 * jnp.roll(field_, -3)
    one_side_derivative += 78  * jnp.roll(field_, -2)
    one_side_derivative -= 27  * jnp.roll(field_, -1)
    one_side_derivative += 4   * field_
    one_side_derivative = one_side_derivative / (dr ** 6)
    # define the one-sided finite difference for the 6th derivative at the right boundary

    lop_side_derivative = -2   * jnp.roll(field_, -6)
    lop_side_derivative += 15  * jnp.roll(field_, -5)
    lop_side_derivative -= 48  * jnp.roll(field_, -4)
    lop_side_derivative += 85  * jnp.roll(field_, -3)
    lop_side_derivative -= 90  * jnp.roll(field_, -2)
    lop_side_derivative += 57  * jnp.roll(field_, -1)
    lop_side_derivative -= 20  * field_
    lop_side_derivative += 3   * jnp.roll(field_, 1)
    lop_side_derivative = lop_side_derivative / (dr ** 6)
    # define the one-sided finite difference for the 6th derivative at the left boundary

    more_lop_side_derivative = -1   * jnp.roll(field_, -5)
    more_lop_side_derivative += 8   * jnp.roll(field_, -4)
    more_lop_side_derivative -= 27  * jnp.roll(field_, -3)
    more_lop_side_derivative += 50  * jnp.roll(field_, -2)
    more_lop_side_derivative -= 55  * jnp.roll(field_, -1)
    more_lop_side_derivative += 36  * field_
    more_lop_side_derivative -= 13  * jnp.roll(field_, 1)
    more_lop_side_derivative += 2   * jnp.roll(field_, 2)
    more_lop_side_derivative = more_lop_side_derivative / (dr ** 6)
    # define the one-sided finite difference for the 6th derivative at the second


    derivative = derivative.at[-1].set(one_side_derivative[-1])
    derivative = derivative.at[-2].set(more_lop_side_derivative[-2])
    derivative = derivative.at[-3].set(lop_side_derivative[-3])
    # set the last three points to the one-sided finite difference values to avoid boundary issues

    return derivative[3:]
