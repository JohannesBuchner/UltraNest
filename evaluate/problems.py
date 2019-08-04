import numpy as np
from math import gamma, pi

def nsphere_volume(radius, ndim):
    return pi**(ndim/2.) / gamma(ndim/2. + 1) * radius**ndim

def transform(x): return x

def loglike_gauss(x):
    """ gaussian problem (circles) """
    return -0.5 * ((x - 0.5)**2).sum()

def volume_gauss(loglike, ndim):
    """ compute volume enclosed at loglike threshold """
    sqr_radius = -2 * loglike
    radius = sqr_radius**0.5
    if radius >= 0.5:
        # the volume is still touching the unit cube
        return np.nan
    
    # compute volume of a n-sphere
    return nsphere_volume(radius, ndim)

def generate_asymgauss_problem(ndim):
    asym_sigma = 0.1 / (1 + 4*np.arange(ndim))
    asym_sigma_max = asym_sigma.max()
    def loglike_asymgauss(x):
        """ assymmetric gaussian problem"""
        return -0.5 * (((x - 0.5)/asym_sigma)**2).sum()

    def volume_asymgauss(loglike, ndim):
        """ compute volume enclosed at loglike threshold """
        sqr_radius = -2 * loglike
        radius = sqr_radius**0.5
        # assume that all of that is in the asym_sigma_max direction
        # how far would that be?
        if radius * asym_sigma_max >= 0.5:
            # the volume is still touching the unit cube
            return np.nan
        
        # compute volume of a n-sphere
        return nsphere_volume(radius, ndim) * np.product(asym_sigma / asym_sigma_max)
    return loglike_asymgauss, volume_asymgauss


def loglike_pyramid(x): 
    """ hyper-pyramid problem (squares) """
    return -np.abs(x - 0.5).max()**0.01

def volume_pyramid(loglike, ndim):
    """ compute volume enclosed at loglike threshold """
    sidelength = (-loglike)**100
    return sidelength**ndim

def loglike_multigauss(x):
    """ two-peaked gaussian problem """
    a = -0.5 * (((x - 0.4)/0.01)**2).sum()
    b = -0.5 * (((x - 0.6)/0.01)**2).sum()
    return np.logaddexp(a, b)

def volume_multigauss(loglike, ndim):
    """ compute volume enclosed at loglike threshold """
    sqr_radius = -2 * loglike
    radius = sqr_radius**0.5 * 0.01
    if radius >= 0.5:
        # the volume is still touching the unit cube
        return np.nan
    if radius >= (0.2**2 * ndim)**0.5:
        # the two peaks are still touching each other
        return np.nan
    
    # compute volume of a n-sphere
    return nsphere_volume(radius, ndim)

def loglike_shell(x):
    """ gaussian shell, tilted """
    # square distance from center
    r = ((x - 0.5)**2).sum()
    # gaussian shell centered at 0.5, radius 0.4, thickness 0.004
    L1 = -0.5 * ((r - 0.4**2) / 0.004)**2
    return L1

def volume_shell(loglike, ndim):
    """ compute volume enclosed at loglike threshold """
    sqr_deviation = -2 * loglike * (0.004)**2
    # how far are we from the center of the shell?
    deviation = sqr_deviation**0.5
    
    if deviation >= 0.1:
        # the volume is still touching the unit cube
        return np.nan
    
    # so 0.4 +- deviation is the current shell
    outer_volume = nsphere_volume(0.4 + deviation, ndim)

    if deviation >= 0.4:
        # all of the enclosed volume is contained
        inner_volume = 0
    else:
        inner_volume = nsphere_volume(0.4 - deviation, ndim)
    
    volume = outer_volume - inner_volume
    return volume

def __loglike_asymshell(x):
    """ gaussian shell, tilted """
    # square distance from center
    r = ((x - 0.5)**2).sum()
    # gaussian shell centered at 0.5, radius 0.4, thickness 0.004
    L1 = -0.5 * (r - 0.4**2)**2 / 0.004**2
    if (r - 0.4**2) > 0.04:
        return L1
    # inside the gaussian is truncated to be flat
    # except we add a tilt:
    
    L1 = -0.5 * 0.04**2 / 0.004**2
    # tilt:
    d = (x - 0.5).sum()
    L2 = 100 * d
    return L1 + L2

def get_problem(problemname, ndim):
    if problemname == 'circgauss':
        return loglike_gauss, volume_gauss
    elif problemname == 'asymgauss':
        return generate_asymgauss_problem(ndim)
    elif problemname == 'pyramid':
        return loglike_pyramid, volume_pyramid
    elif problemname == 'multigauss':
        return loglike_multigauss, volume_multigauss
    elif problemname == 'shell':
        return loglike_shell, volume_shell
    
    raise Exception("Problem '%d' unknown" % problemname)



