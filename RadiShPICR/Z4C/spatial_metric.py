import jax.numpy as jnp
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative, sixth_derivative


def dchidt(metric: Z4C_Metric, matter_terms):
    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    Kh = metric.Kh
    chi = metric.chi
    grr = metric.conformal_grr
    theta = metric.theta
    nu   = metric.nu
    # unpack the metric and matter terms

    dchidr = first_derivative(chi, metric.dr, parity=1)
    dbetadr = first_derivative(beta, metric.dr, parity=-1)
    # compute derivatives of the metric functions using finite difference methods


    # d\[Chi]dt = 
    #   4/3 alpha * theta* chi + 2/3 alpha * Kh * chi  - (4 beta * chi )/(
    #    3 r) - 2/3 chi *  dbetadr + beta * dchidr ;
    # original equation from mathmatica notebook



    dchidt = -2.0 * chi * dbetadr / 3.0
    dchidt += beta * dchidr
    dchidt += 2.0 * alpha * chi * Kh / 3.0
    dchidt += -4.0 * beta * chi / (3.0 * metric.r)
    dchidt += 4.0 * alpha * chi * theta / 3.0
    # compute the time derivative of chi using the Z4C evolution equations

    dchidt += nu / 64 * (sixth_derivative(chi, metric.dr, parity=1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of chi


    # SOMMERFELD BOUNDARY CONDITION FOR CHI AT OUTER BOUNDARY
    dchidr = first_derivative(chi, metric.dr, parity=1)

    speed_of_light = -beta[-1] + alpha[-1] / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dchidt = dchidt.at[-1].set(  - speed_of_light * (  dchidr[-1]    +   chi[-1] / metric.r[-1] )  )
    # set the time derivative of chi at the outer boundary using the Sommerfeld boundary condition

    return dchidt


def dgrrdt(metric: Z4C_Metric, matter_terms):
    alpha = metric.alpha
    beta = metric.beta
    Arr  = metric.Arr
    grr = metric.conformal_grr
    nu  = metric.nu
    # unpack the metric and matter terms

    dbetadr = first_derivative(beta, metric.dr, parity=-1)
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods


    # dgrrdt = -2 Arr * alpha + beta * dgrrdr + (
    #    4 grr * (-beta + r * dbetadr ))/(3 r);
    # original equation from mathmatica notebook

    dgrrdt = -2.0 * alpha * Arr
    dgrrdt += beta * dgrrdr
    dgrrdt += 4.0 * grr * ( -beta + dbetadr * metric.r) / (3.0 * metric.r)
    # compute the time derivative of grr using the Z4C evolution equations

    dgrrdt += nu / 64 * (sixth_derivative(grr, metric.dr, parity=1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of grr


    # SOMMERFELD BOUNDARY CONDITION FOR GRR AT OUTER BOUNDARY
    dgrrdr = first_derivative(grr, metric.dr, parity=1)

    speed_of_light = -beta[-1] + alpha[-1] / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dgrrdt = dgrrdt.at[-1].set(  - speed_of_light * (  dgrrdr[-1]    +   grr[-1] / metric.r[-1] )  )
    # set the time derivative of grr at the outer boundary using the Sommerfeld boundary condition


    return dgrrdt

def dgtdt(metric: Z4C_Metric, matter_terms):
    alpha = metric.alpha
    beta = metric.beta
    At   = metric.At
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    nu  = metric.nu
    # unpack the metric and matter terms

    dbetadr = first_derivative(beta, metric.dr, parity=-1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods


    # dgTdt = -2 At * alpha + beta * dgtdr + (
    #    2 gt * (beta - r * dbetadr ))/(3 r);
    # original equation from mathmatica notebook

    dgtdt = -2.0 * alpha * At
    dgtdt += beta * dgtdr
    dgtdt += (2.0 * gt * (beta - dbetadr * metric.r)) / (3.0 * metric.r)
    # compute the time derivative of gt using the Z4C evolution equations

    dgtdt += nu / 64 * (sixth_derivative(gt, metric.dr, parity=1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of gt



    # SOMMERFELD BOUNDARY CONDITION FOR GT AT OUTER BOUNDARY
    dgtdr = first_derivative(gt, metric.dr, parity=1)

    speed_of_light = -beta[-1] + alpha[-1] / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dgtdt = dgtdt.at[-1].set(  - speed_of_light * (  dgtdr[-1]    +   gt[-1] / metric.r[-1] )  )
    # set the time derivative of gt at the outer boundary using the Sommerfeld boundary condition

    return dgtdt
