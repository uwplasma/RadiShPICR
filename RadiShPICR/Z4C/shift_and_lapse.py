import jax.numpy as jnp
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative, sixth_derivative

def dalphadt(metric: Z4C_Metric, matter_terms):
    alpha = metric.alpha
    grr   = metric.conformal_grr
    beta = metric.beta
    Kh = metric.Kh
    nu = metric.nu

    dalphadr = first_derivative(alpha, metric.dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods
 
    # dalphadt = -2 * alpha * Kh + beta * dalphadr;
    # original equation from mathmatica notebook

    dalphadt = -2 * alpha * Kh
    dalphadt += beta * dalphadr
    # compute the time derivative of alpha using the Z4C evolution equations

    dalphadt += nu / 64 * (sixth_derivative(alpha, metric.dr, parity=1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of alpha


    # SOMMERFELD BOUNDARY CONDITION FOR ALPHA AT OUTER BOUNDARY

    lapse_speed = -beta[-1] + jnp.sqrt(2 * alpha[-1] ) / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dalphadt = dalphadt.at[-1].set(  - lapse_speed * (  dalphadr[-1]    +   (alpha[-1] - 1) / metric.r[-1] )  )
    # set the time derivative of alpha at the outer boundary using the Sommerfeld boundary condition

    return dalphadt


def dbetadt(metric: Z4C_Metric, matter_terms):
    beta = metric.beta
    Gamma = metric.Gamma
    nu = metric.nu
    # unpack the metric and matter terms

    dbetadr = first_derivative(beta, metric.dr, parity=-1)
    # compute derivatives of the metric functions using finite difference methods

    # dbetadt = -\[Eta] beta +  5/2 * Gamma + beta * dbetadr;
    # original equation from mathmatica notebook

    dbetadt = -metric.eta * beta
    dbetadt += (5 / 2) * Gamma
    dbetadt += beta * dbetadr
    # compute the time derivative of beta using the Z4C evolution equations

    dbetadt += nu / 64 * (sixth_derivative(beta, metric.dr, parity=-1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of beta


    # SOMMERFELD BOUNDARY CONDITION FOR BETA AT OUTER BOUNDARY

    shift_speed = -beta[-1] * jnp.sqrt(5/2)
    # compute the speed of light at the outer boundary using the lapse and shift

    dbetadt = dbetadt.at[-1].set(  - shift_speed * (  dbetadr[-1]    +   beta[-1] / metric.r[-1] )  )
    # set the time derivative of beta at the outer boundary using the Sommerfeld boundary condition



    return dbetadt
