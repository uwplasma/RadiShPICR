import jax.numpy as jnp
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative, sixth_derivative

CHI_FLOOR_VALUE = 1e-12


def dKhdt(metric: Z4C_Metric, matter_terms):
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
    nu    = metric.nu
    # unpack the metric and matter terms

    dalphadr = first_derivative(alpha, metric.dr, parity=1)
    dchidr = first_derivative(chi, metric.dr, parity=1)
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    dKhdr = first_derivative(Kh, metric.dr, parity=1)
    d2alphadr2 = second_derivative(alpha, metric.dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods

    St = matter_terms.St
    Srr = matter_terms.Srr
    rho = matter_terms.rho
    # unpack the matter terms


    #     dKhdt = ((Arr )^2 * alpha )/(grr )^2 + (
    #    2 ((At )^2) alpha )/(gt )^2 + \[Kappa]1 * alpha * theta + 
    #    4/3 alpha * (theta)^2 + 4/3 alpha * theta* Kh + 1/3 alpha * Kh^2 + 
    #    4 \[Pi] alpha *rho +  (4 \[Pi] * alpha *Srr * chi )/ grr  + (
    #    8 \[Pi] * alpha *St * chi )/gt - (2 chi *  dalphadr)/(r * grr) + (
    #    chi *  dgrrdr * dalphadr)/(2 (grr)^2) - (chi *  dgtdr * dalphadr)/(
    #    grr * gt)  +  beta * dKhdr + (dalphadr * dchidr)/(2 grr) - (
    #    chi * d2alphadr2)/grr;
    # original expression from mathematica

    dKhdt = (Arr ** 2 * alpha) / (grr ** 2)
    dKhdt += (2 * (At ** 2) * alpha) / (gt ** 2)
    dKhdt += kappa * alpha * theta
    dKhdt += (4 / 3) * alpha * (theta ** 2)
    dKhdt += (4 / 3) * alpha * theta * Kh
    dKhdt += (1 / 3) * alpha * (Kh ** 2)
    dKhdt += 4 * jnp.pi * alpha * rho
    dKhdt += (4 * jnp.pi * alpha * Srr * chi) / grr
    dKhdt += (8 * jnp.pi * alpha * St * chi) / gt
    dKhdt += -(2 * chi * dalphadr) / (metric.r * grr)
    dKhdt += (chi * dgrrdr * dalphadr) / (2 * (grr ** 2))
    dKhdt += -(chi * dgtdr * dalphadr) / (grr * gt)
    dKhdt += beta * dKhdr
    dKhdt += (dalphadr * dchidr) / (2 * grr)
    dKhdt += -(chi * d2alphadr2) / grr
    # compute the time derivative of Kh using the Z4C evolution equations

    dKhdt += nu / 64 * (sixth_derivative(Kh, metric.dr, parity=1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of Kh


    # SOMMERFELD BOUNDARY CONDITION FOR ALPHA AT OUTER BOUNDARY

    lapse_speed = -beta[-1] + jnp.sqrt(2 * alpha[-1] ) / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dKhdt = dKhdt.at[-1].set(  - lapse_speed * (  dKhdr[-1]    +   Kh[-1] / metric.r[-1] )  )
    # set the time derivative of Kh at the outer boundary using the Sommerfeld boundary condition


    return dKhdt



def dArrdt(metric: Z4C_Metric, matter_terms):
    Arr = metric.Arr
    At = metric.At
    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Kh = metric.Kh
    chi = metric.chi
    theta = metric.theta
    nu    = metric.nu
    # unpack the metric and matter terms

    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    dArrdr = first_derivative(Arr, metric.dr, parity=1)
    dAtdr = first_derivative(At, metric.dr, parity=1)
    dalphadr = first_derivative(alpha, metric.dr, parity=1)
    dchidr = first_derivative(chi, metric.dr, parity=1)
    d2chidr2 = second_derivative(chi, metric.dr, parity=1)
    d2grrdr2 = second_derivative(grr, metric.dr, parity=1)
    d2gtdr2 = second_derivative(gt, metric.dr, parity=1)
    dbetadr = first_derivative(beta, metric.dr, parity=-1)
    dGammadr = first_derivative(metric.Gamma, metric.dr, parity=-1)
    d2alphadr2 = second_derivative(alpha, metric.dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods

    St = matter_terms.St
    Srr = matter_terms.Srr
    # unpack the matter terms

    # dArrdt = -((2 (Arr)^2 * alpha)/(3 grr))
    #  - (4 (At )^2 grr * alpha)/(3 (gt )^2) 
    #  + (2 Arr * At * alpha)/gt 
    # + 4/3 Arr * alpha * theta 
    # - (4 At * grr * alpha * theta)/(3 gt) 
    # + 2/3 Arr * alpha * Kh
    #  - (2 At * grr * alpha * Kh)/(3 gt)
    #  - (8 Arr * beta)/(9 r)
    #  + ( 8 At * grr * beta)/(9 r * gt)
    #  - (2 alpha * chi * )/(3 r^2) 
    # + ( 4 ((grr)^2) * alpha * chi  )/(3 r^2 *(gt)^2)
    #  - (2 grr * alpha * chi  )/(3 (r^2) * gt )
    #  - 16/3 \[Pi] alpha *Srr * chi
    #  + (16 \[Pi] grr * alpha *St * chi )/(3 gt)
    #  - (2 ((grr )^2) * alpha * Gamma * chi )/(3 r * gt) 
    # + 2/3 beta * dArrdr 
    # - (2 grr * beta *dAtdr)/(3 gt) 
    # - ( 2 At * beta * dgrrdr)/(3 gt)
    #  + (alpha * chi *  dgrrdr)/( 3 r * grr) 
    # - (alpha * chi *  dgrrdr )/(r * gt)
    #  +  2/3 alpha * Gamma * chi * dgrrdr 
    # +((alpha * chi *  (dgrrdr)^2)/(3 (grr)^2)) 
    # + (2 At * grr * beta * dgtdr)/(3 (gt )^2) 
    # + (2 grr * alpha * chi *  dgtdr)/(3 r (gt)^2) 
    # + ( alpha * chi *  dgrrdr * dgtdr)/(6 grr * gt)
    #  - ( alpha * chi *  (dgtdr)^2)/(3 (gt)^2)
    #  + (2 chi *  dalphadr)/( 3 r) 
    # + (chi *  dgrrdr * dalphadr)/(3 grr)
    #  + ( chi *  dgtdr * dalphadr)/(3 gt)
    #  +  8/9 Arr * dbetadr 
    # -((8 At * grr * dbetadr)/(9 gt))
    #  +  2/3 grr * alpha * chi *  dGammadr
    #  - (alpha * dchidr)/(3 r)
    #  - (  alpha * dgrrdr * dchidr)/(6 grr)
    #  - (alpha * dgtdr * dchidr)/( 6 gt)
    #  - 2/3 dalphadr * dchidr 
    # -((alpha * (dchidr )^2)/(6 chi))
    #  - ( alpha * chi *  d2grrdr2)/(3 grr)
    #  + (alpha * chi *  d2gtdr2)/( 3 gt)
    #  - 2/3 chi * d2alphadr2 
    # + 1/3 alpha * d2chidr2;

    # original expression from mathematica


    dArrdt = -((2 * (Arr ** 2) * alpha) / (3 * grr))
    dArrdt += -(4 * (At ** 2) * grr * alpha) / (3 * (gt ** 2))
    dArrdt += (2 * Arr * At * alpha) / gt
    dArrdt += (4 / 3) * Arr * alpha * theta
    dArrdt += -(4 * At * grr * alpha * theta) / (3 * gt)
    dArrdt += (2 / 3) * Arr * alpha * Kh
    dArrdt += -(2 * At * grr * alpha * Kh) / (3 * gt)
    dArrdt += -(8 * Arr * beta) / (9 * metric.r)
    dArrdt += (8 * At * grr * beta) / (9 * metric.r * gt)
    dArrdt += -(2 * alpha * chi) / (3 * metric.r ** 2)
    dArrdt += (4 * (grr ** 2) * alpha * chi) / (3 * metric.r ** 2 * (gt ** 2))
    dArrdt += -(2 * grr * alpha * chi) / (3 * (metric.r ** 2) * gt)
    dArrdt += -(16 / 3) * jnp.pi * alpha * Srr * chi
    dArrdt += (16 * jnp.pi * grr * alpha * St * chi) / (3 * gt)
    dArrdt += -(2 * (grr ** 2) * alpha * metric.Gamma * chi) / (3 * metric.r * gt)
    dArrdt += (2 / 3) * beta * dArrdr
    dArrdt += -(2 * grr * beta * dAtdr) / (3 * gt)
    dArrdt += -(2 * At * beta * dgrrdr) / (3 * gt)
    dArrdt += (alpha * chi * dgrrdr) / (3 * metric.r * grr)
    dArrdt += -(alpha * chi * dgrrdr) / (metric.r * gt)
    dArrdt += (2 / 3) * alpha * metric.Gamma * chi * dgrrdr
    dArrdt += (alpha * chi * (dgrrdr ** 2)) / (3 * (grr ** 2))
    dArrdt += (2 * At * grr * beta * dgtdr) / (3 * (gt ** 2))
    dArrdt += (2 * grr * alpha * chi * dgtdr) / (3 * metric.r * (gt ** 2))
    dArrdt += (alpha * chi * dgrrdr * dgtdr) / (6 * grr * gt)
    dArrdt += -(alpha * chi * (dgtdr ** 2)) / (3 * (gt ** 2))
    dArrdt += (2 * chi * dalphadr) / (3 * metric.r)
    dArrdt += (chi * dgrrdr * dalphadr) / (3 * grr)
    dArrdt += (chi * dgtdr * dalphadr) / (3 * gt)
    dArrdt += (8 / 9) * Arr * dbetadr
    dArrdt += -(8 * At * grr * dbetadr) / (9 * gt)
    dArrdt += (2 / 3) * grr * alpha * chi * dGammadr
    dArrdt += -(alpha * dchidr) / (3 * metric.r)
    dArrdt += -(alpha * dgrrdr * dchidr) / (6 * grr)
    dArrdt += -(alpha * dgtdr * dchidr) / (6 * gt)
    dArrdt += -(2 / 3) * dalphadr * dchidr
    dArrdt += -(alpha * (dchidr ** 2)) / (6 * jnp.maximum(chi, CHI_FLOOR_VALUE))
    dArrdt += -(alpha * chi * d2grrdr2) / (3 * grr)
    dArrdt += (alpha * chi * d2gtdr2) / (3 * gt)
    dArrdt += -(2 / 3) * chi * d2alphadr2
    dArrdt += (1 / 3) * alpha * d2chidr2
    # compute the time derivative of Arr using the Z4C evolution equations

    dArrdt += nu / 64 * (sixth_derivative(Arr, metric.dr, parity=1)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of Arr


    # SOMMERFELD BOUNDARY CONDITION FOR THETA AT OUTER BOUNDARY

    speed_of_light = -beta[-1] + alpha[-1] / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dArrdt = dArrdt.at[-1].set(  - speed_of_light * (  dArrdr[-1]    +   Arr[-1] / metric.r[-1] )  )
    # set the time derivative of theta at the outer boundary using the Sommerfeld boundary condition
    

    return dArrdt



def dAtdt(metric: Z4C_Metric, matter_terms):
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

    dalphadr = first_derivative(alpha, metric.dr, parity=1)
    dchidr = first_derivative(chi, metric.dr, parity=1)
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    d2alphadr2 = second_derivative(alpha, metric.dr, parity=1)
    d2chidr2 = second_derivative(chi, metric.dr, parity=1)
    d2grrdr2 = second_derivative(grr, metric.dr, parity=1)
    d2gtdr2 = second_derivative(gt, metric.dr, parity=1)
    dbetadr = first_derivative(beta, metric.dr, parity=-1)
    dGammadr = first_derivative(Gamma, metric.dr, parity=-1)
    dAtdr = first_derivative(At, metric.dr, parity=1)
    dArrdr = first_derivative(Arr, metric.dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods

    St = matter_terms.St
    Srr = matter_terms.Srr
    # unpack the matter terms


    #     dATdt = (Arr * At * alpha)/(3 grr) 
    # - ((Arr)^2 gt * alpha)/(3 (grr)^2)
    #  + 2/3 At * alpha * theta -
    #  (2 Arr * gt * alpha * theta)/(3 grr)
    #  + 1/3 At * alpha * Kh 
    # - (Arr * gt * alpha * Kh)/(3 grr ) 
    # + (2 At * beta )/(9 r)
    #  - (2 Arr * gt * beta )/(9 r * grr )
    #  + (alpha * chi  )/(3 r^2)
    #  - (2 grr * alpha * chi  )/(3 (r^2) * gt ) 
    # + (gt * alpha * chi  )/(3 (r^2) * grr )
    #  + (8 \[Pi] gt * alpha *Srr * chi  )/(3 grr ) 
    # - 8/3 \[Pi] alpha *St * chi  
    # + (grr * alpha * Gamma * chi  )/(3 r) 
    # - (gt * beta * dArrdr)/(3 grr ) 
    # + 1/3 beta *dAtdr 
    # + (Arr * gt * beta * dgrrdr )/(3 (grr )^2) 
    # + (alpha * chi *  dgrrdr )/(2 r * grr ) 
    # - (gt * alpha * chi *  dgrrdr )/(6 r * (grr )^2) 
    # - (gt * alpha * Gamma * chi *  dgrrdr )/(3 grr ) 
    # - (gt * alpha * chi *  (dgrrdr )^2)/(6 (grr )^3)
    #  - (Arr * beta * dgtdr )/(3 grr ) 
    # - (alpha * chi *  dgtdr )/(3 r *gt ) 
    # - (alpha * chi *  dgrrdr * dgtdr )/(12 (grr )^2)
    #  + ( alpha * chi *  (dgtdr )^2)/(6 grr * gt )
    #  - (gt * chi *  dalphadr)/(3 r * grr ) 
    # - (gt * chi *  dgrrdr * dalphadr)/(6 (grr )^2) 
    # - (chi *  dgtdr * dalphadr)/(6 grr ) 
    # - 2/9 At * dbetadr 
    # + (2 Arr * gt * dbetadr )/(9 grr ) 
    # - 1/3 gt * alpha * chi *  dGammadr 
    # + (gt * alpha * dchidr )/(6 r grr ) 
    # + (gt * alpha * dgrrdr * dchidr )/(12 (grr )^2)
    #  + (alpha * dgtdr * dchidr )/(12 grr ) 
    # + (gt * dalphadr * dchidr )/(3 grr ) 
    # + (gt * alpha * (dchidr)^2)/(12 grr * chi  ) 
    # + (gt * alpha * chi *  d2grrdr2)/(6 (grr )^2) 
    # - (alpha * chi *  d2gtdr2)/(6 grr ) 
    # + (gt * chi * d2alphadr2)/(3 grr ) 
    # - (gt * alpha * d2chidr2)/(6 grr );

    dAtdt = (Arr * At * alpha) / (3 * grr)
    dAtdt += -((Arr ** 2) * gt * alpha) / (3 * (grr ** 2))
    dAtdt += (2 / 3) * At * alpha * theta
    dAtdt += -(2 * Arr * gt * alpha * theta) / (3 * grr)
    dAtdt += (1 / 3) * At * alpha * Kh
    dAtdt += -(Arr * gt * alpha * Kh) / (3 * grr)
    dAtdt += (2 * At * beta ) / (9 * metric.r)
    dAtdt += -(2 * Arr * gt * beta ) / (9 * metric.r * grr)
    dAtdt += (alpha * chi) / (3 * metric.r ** 2)
    dAtdt += -(2 * grr * alpha * chi) / (3 * (metric.r ** 2) * gt)
    dAtdt += (gt * alpha * chi) / (3 * (metric.r ** 2) * grr)
    dAtdt += (8 * jnp.pi * gt * alpha * Srr * chi) / (3 * grr)
    dAtdt += -(8 / 3) * jnp.pi * alpha * St * chi
    dAtdt += (grr * alpha * Gamma * chi) / (3 * metric.r)
    dAtdt += -(gt * beta * dArrdr) / (3 * grr)
    dAtdt += (1 / 3) * beta * dAtdr
    dAtdt += (Arr * gt * beta * dgrrdr) / (3 * (grr ** 2))
    dAtdt += (alpha * chi * dgrrdr) / (2 * metric.r * grr)
    dAtdt += -(gt * alpha * chi * dgrrdr) / (6 * metric.r * (grr ** 2))
    dAtdt += -(gt * alpha * Gamma * chi * dgrrdr) / (3 * grr)
    dAtdt += -(gt * alpha * chi * (dgrrdr ** 2)) / (6 * (grr ** 3))
    dAtdt += -(Arr * beta * dgtdr) / (3 * grr)
    dAtdt += -(alpha * chi * dgtdr) / (3 * metric.r * gt)
    dAtdt += -(alpha * chi * dgrrdr * dgtdr) / (12 * (grr ** 2))
    dAtdt += (alpha * chi * (dgtdr ** 2)) / (6 * grr * gt)
    dAtdt += -(gt * chi * dalphadr) / (3 * metric.r * grr)
    dAtdt += -(gt * chi * dgrrdr * dalphadr) / (6 * (grr ** 2))
    dAtdt += -(chi * dgtdr * dalphadr) / (6 * grr)
    dAtdt += -(2 / 9) * At * dbetadr
    dAtdt += (2 * Arr * gt * dbetadr) / (9 * grr)
    dAtdt += -(1 / 3) * gt * alpha * chi * dGammadr
    dAtdt += (gt * alpha * dchidr) / (6 * metric.r * grr)
    dAtdt += (gt * alpha * dgrrdr * dchidr) / (12 * (grr ** 2))
    dAtdt += (alpha * dgtdr * dchidr) / (12 * grr)
    dAtdt += (gt * dalphadr * dchidr) / (3 * grr)
    dAtdt += (gt * alpha * (dchidr ** 2)) / (12 * grr * jnp.maximum(chi, CHI_FLOOR_VALUE))
    dAtdt += (gt * alpha * chi * d2grrdr2) / (6 * (grr ** 2))
    dAtdt += -(alpha * chi * d2gtdr2) / (6 * grr)
    dAtdt += (gt * chi * d2alphadr2) / (3 * grr)
    dAtdt += -(gt * alpha * d2chidr2) / (6 * grr)
    # compute the time derivative of At using the Z4C evolution equations

    dAtdt += nu / 64 * (sixth_derivative(At, metric.dr, parity=1)) * (metric.dr ** 5)   


    # SOMMERFELD BOUNDARY CONDITION FOR THETA AT OUTER BOUNDARY

    speed_of_light = -beta[-1] + alpha[-1] / jnp.sqrt(grr[-1])
    # compute the speed of light at the outer boundary using the lapse and shift

    dAtdt = dAtdt.at[-1].set(  - speed_of_light * (  dAtdr[-1]    +   At[-1] / metric.r[-1] )  )
    # set the time derivative of At at the outer boundary using the Sommerfeld boundary condition


    return dAtdt
