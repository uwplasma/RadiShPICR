import jax.numpy as jnp
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative, sixth_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

CHI_FLOOR_VALUE = 1e-12



def dthetadt(metric: Z4C_Metric, matter_terms):
    
    Arr = metric.Arr
    At = metric.At
    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Kh = metric.Kh
    chi = metric.chi
    kappa = metric.kappa
    theta = metric.theta
    Gamma = metric.Gamma
    nu    = metric.nu
    # unpack the metric and matter terms

    dchidr = first_derivative(chi, metric.dr, parity=1)
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    dGammadr = first_derivative(Gamma, metric.dr, parity=-1)
    dthetadr = first_derivative(theta, metric.dr, parity=1)
    d2chidr2 = second_derivative(chi, metric.dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods

    rho = matter_terms.rho
    # unpack the matter terms

    # dThetadt = ((At )^2 alpha )/(gt )^2 
    # + (2 Arr * At * alpha )/(grr * gt ) 
    # - 2 \[Kappa]1 * alpha * theta 
    #  +  4/3 alpha * (theta)^2 
    # + 4/3 alpha * theta* Kh 
    # + 1/3 alpha * Kh^2 
    # - 8 \[Pi] alpha *rho
    # -((2 alpha * chi  )/(r^2 grr )) 
    # - (2 grr * alpha * chi  )/(r^2 (gt )^2) 
    # + (4 alpha * chi  )/(r^2 gt ) 
    # + (grr * alpha * Gamma * chi  )/(r gt ) 
    # + (alpha * chi *  dgrrdr )/(r (grr )^2) 
    # - (3 alpha * chi *  dgrrdr )/(2 r grr * gt ) 
    # + (alpha * Gamma * chi *  dgrrdr )/(2 grr ) 
    # + (2 alpha * chi *  dgtdr )/(r * (gt )^2) 
    # - (3 alpha * chi *  dgtdr )/(r * grr * gt )
    #  + (alpha * chi *  dgrrdr * dgtdr )/(2 ((grr )^2) * gt )
    #  - (3 alpha * chi *  (dgtdr)^2)/(4 grr * (gt)^2) 
    # + beta * dthetadr 
    # + 1/2 alpha * chi *  dGammadr 
    # + (2 alpha * dchidr)/(r grr) 
    # - (alpha * dgrrdr * dchidr )/(2 (grr )^2)
    #  + (alpha * dgtdr * dchidr)/(grr * gt ) 
    # - (5 alpha * (dchidr )^2)/(4 grr * chi  )
    #  + (alpha * d2chidr2)/grr;
    #   original equation from mathmatica notebook

    dthetadt = (At ** 2 * alpha) / (gt ** 2 )
    dthetadt += (2 * Arr * At * alpha) / (grr * gt)
    dthetadt += -2 * kappa * alpha * theta
    dthetadt += (4 / 3) * alpha * (theta ** 2)
    dthetadt += (4 / 3) * alpha * theta * Kh
    dthetadt += (1 / 3) * alpha * (Kh ** 2)
    dthetadt += -8 * jnp.pi * alpha * rho
    dthetadt += -((2 * alpha * chi) / (metric.r ** 2 * grr))
    dthetadt += -(2 * grr * alpha * chi) / (metric.r ** 2 * (gt ** 2))
    dthetadt += (4 * alpha * chi) / (metric.r ** 2 * gt)
    dthetadt += (grr * alpha * Gamma * chi) / (metric.r * gt)
    dthetadt += (alpha * chi * dgrrdr) / (metric.r * (grr ** 2))
    dthetadt += -(3 * alpha * chi * dgrrdr) / (2 * metric.r * grr * gt)
    dthetadt += (alpha * Gamma * chi * dgrrdr) / (2 * grr)
    dthetadt += (2 * alpha * chi * dgtdr) / (metric.r * (gt ** 2))
    dthetadt += -(3 * alpha * chi * dgtdr) / (metric.r * grr * gt)
    dthetadt += (alpha * chi * dgrrdr * dgtdr) / (2 * (grr ** 2) * gt)
    dthetadt += -(3 * alpha * chi * (dgtdr ** 2)) / (4 * grr * (gt ** 2))
    dthetadt += beta * dthetadr
    dthetadt += (1 / 2) * alpha * chi * dGammadr
    dthetadt += (2 * alpha * dchidr) / (metric.r * grr)
    dthetadt += -(alpha * dgrrdr * dchidr) / (2 * (grr ** 2))
    dthetadt += (alpha * dgtdr * dchidr) / (grr * gt)
    dthetadt += -(5 * alpha * (dchidr ** 2)) / (4 * grr * jnp.maximum(chi, CHI_FLOOR_VALUE))
    dthetadt += (alpha * d2chidr2) / grr
    # compute the time derivative of theta using the Z4C evolution equations

    dthetadt += nu / 64 * (sixth_derivative(theta, metric.dr, parity=1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of theta


    # SOMMERFELD BOUNDARY CONDITION FOR THETA AT OUTER BOUNDARY
    dthetadr = first_derivative(theta, metric.dr, parity=1)

    speed_of_light = -beta[-1] + alpha[-1] / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dthetadt = dthetadt.at[-1].set(  - speed_of_light * (  dthetadr[-1]    +   theta[-1] / metric.r[-1] )  )
    # set the time derivative of theta at the outer boundary using the Sommerfeld boundary condition
    


    return dthetadt


def dGammadt(metric: Z4C_Metric, matter_terms):
    Arr = metric.Arr
    At = metric.At
    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Kh = metric.Kh
    chi = metric.chi
    kappa = metric.kappa
    theta = metric.theta
    Gamma = metric.Gamma
    nu    = metric.nu
    # unpack the metric and matter terms

    dchidr = first_derivative(chi, metric.dr, parity=1)
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    dGammadr = first_derivative(Gamma, metric.dr, parity=-1)
    dalphadr = first_derivative(alpha, metric.dr, parity=1)
    dbetadr = first_derivative(beta, metric.dr, parity=-1)
    dthetadr = first_derivative(theta, metric.dr, parity=1)
    dKhdr = first_derivative(Kh, metric.dr, parity=1)
    d2betadr2 = second_derivative(beta, metric.dr, parity=-1)
    # compute derivatives of the metric functions using finite difference methods

    # dGammadt = -((2 \[Kappa]1 * alpha )/(r grr )) + (4 At * alpha )/(
    #    r (gt )^2) + (2 \[Kappa]1 alpha )/(r * gt ) - (4 At * alpha )/(
    #    r grr * gt ) - (10 beta )/(3 (r^2) grr ) + (2 beta )/(
    #    3 (r^2) gt ) - \[Kappa]1 alpha * Gamma - (16 \[Pi] Sr* alpha )/
    #    chi  + (Arr * alpha * dgrrdr )/(grr )^3 + (\[Kappa]1 alpha * 
    #     dgrrdr )/(grr )^2 + (4 beta * dgrrdr )/(3 r (grr )^2) - (
    #    2 At * alpha * dgtdr )/(grr * (gt )^2) - (
    #    2 Arr * dalphadr)/(grr )^2 - (2 alpha * dthetadr)/(3 grr ) - (
    #    4 alpha * dKhdr)/(3 grr ) + (4 dbetadr )/(3 r grr ) + (
    #    4 dbetadr )/(3 r gt ) - (dgrrdr * dbetadr )/(3 (grr )^2) + 
    #    beta * dGammadr - (3 Arr * alpha * dchidr )/((grr )^2 chi ) + (
    #    4 d2betadr2)/(3 grr );
    #  original equation from mathmatica notebook

    dGammadt = -((2 * kappa * alpha) / (metric.r * grr))
    dGammadt += (4 * At * alpha) / (metric.r * (gt ** 2))
    dGammadt += (2 * kappa * alpha) / (metric.r * gt)
    dGammadt += -(4 * At * alpha) / (metric.r * grr * gt)
    dGammadt += -(10 * beta) / (3 * (metric.r ** 2) * grr)
    dGammadt += (2 * beta) / (3 * (metric.r ** 2) * gt)
    dGammadt += -kappa * alpha * Gamma
    dGammadt += -(16 * jnp.pi * matter_terms.Sr * alpha) / jnp.maximum(chi, CHI_FLOOR_VALUE)
    dGammadt += (Arr * alpha * dgrrdr) / (grr ** 3)
    dGammadt += (kappa * alpha * dgrrdr) / (grr ** 2)
    dGammadt += (4 * beta * dgrrdr) / (3 * metric.r * (grr ** 2))
    dGammadt += -(2 * At * alpha * dgtdr) / (grr * (gt ** 2))
    dGammadt += -(2 * Arr * dalphadr) / (grr ** 2)
    dGammadt += -(2 * alpha * dthetadr) / (3 * grr)
    dGammadt += -(4 * alpha * dKhdr) / (3 * grr)
    dGammadt += (4 * dbetadr) / (3 * metric.r * grr)
    dGammadt += (4 * dbetadr) / (3 * metric.r * gt)
    dGammadt += -(dgrrdr * dbetadr) / (3 * (grr ** 2))
    dGammadt += beta * dGammadr
    dGammadt += -(3 * Arr * alpha * dchidr) / ((grr ** 2) * jnp.maximum(chi, CHI_FLOOR_VALUE))
    dGammadt += (4 * d2betadr2) / (3 * grr)
    # compute the time derivative of Gamma using the Z4C evolution equations

    dGammadt += nu / 64 * (sixth_derivative(Gamma, metric.dr, parity=-1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of Gamma


    # SOMMERFELD BOUNDARY CONDITION FOR GAMMA AT OUTER BOUNDARY

    shift_speed = -beta[-1] * jnp.sqrt(5/2)
    # compute the speed of light at the outer boundary using the lapse and shift

    dGammadt = dGammadt.at[-1].set(  - shift_speed * (  dGammadr[-1]    +   Gamma[-1] / metric.r[-1] )  )
    # set the time derivative of Gamma at the outer boundary using the Sommerfeld boundary condition

    return dGammadt