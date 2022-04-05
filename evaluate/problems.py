import numpy as np
from math import gamma, pi, exp

def random_vector(ndim, length=1):
    v = np.random.normal(size=ndim)
    return v * length / (v**2).sum()**0.5

def random_point_in_sphere(ndim, radius=1):
    return random_vector(ndim, radius * np.random.uniform()**(1. / ndim))

def nsphere_volume(radius, ndim):
    return pi**(ndim/2.) / gamma(ndim/2. + 1) * radius**ndim

def gradient_to_center(x, ctr=0.5):
    """ return normalised vector pointing to center """
    v = ctr - x
    v /= (v**2).sum()**0.5
    return v

def transform(x): return x

def loglike_gauss(x):
    """ gaussian problem (circles) """
    return -0.5 * ((x - 0.5)**2).sum()

gradient_gauss = gradient_to_center

def volume_gauss(loglike, ndim):
    """ compute volume enclosed at loglike threshold """
    sqr_radius = -2 * loglike
    radius = sqr_radius**0.5
    if radius >= 0.5:
        # the volume is still touching the unit cube
        return np.nan
    
    # compute volume of a n-sphere
    return nsphere_volume(radius, ndim)

def warmup_gauss(ndim):
    return 0.5 + random_point_in_sphere(ndim, radius = 0.4)

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
    
    gradient_asymgauss = gradient_to_center
    
    def warmup_asymgauss(ndim):
        return 0.5 + random_point_in_sphere(ndim, radius = asym_sigma)
    
    return loglike_asymgauss, gradient_asymgauss, volume_asymgauss, warmup_asymgauss


def generate_corrgauss_problem(ndim, gamma=0.95):
    mean = np.zeros(ndim)
    M = np.ones((ndim, ndim)) * gamma
    np.fill_diagonal(M, 1)
    Minv = np.linalg.inv(M)
    Mdet = np.linalg.det(M)
    center = np.zeros(ndim)

    loglike_asymgauss, gradient_asymgauss, volume_asymgauss, warmup_asymgauss = generate_asymgauss_problem(ndim)
    
    from ultranest.mlfriends import AffineLayer
    
    layer = AffineLayer(center, M, Minv)
    
    def warmup_corrgauss(ndim):
        # the gaussian is defined in our aux coordinate system:
        y = warmup_asymgauss(ndim)
        # so transform to these
        return layer.transform(y - 0.5) + 0.5
    
    def loglike_corrgauss(x):
        """  gaussian problem """
        # transform back to aux coordinate system, where gaussian is nice
        y = layer.untransform(x - 0.5) + 0.5
        return loglike_asymgauss(y)

    def volume_corrgauss(loglike, ndim):
        # volume is defined in aux coordinate system
        # we hope that no intersection with unit cube happens
        return volume_asymgauss(loglike, ndim) / Mdet

    def gradient_corrgauss(x):
        y = layer.untransform(x - 0.5) + 0.5
        return gradient_to_center(y)
    
    return loglike_corrgauss, gradient_corrgauss, volume_corrgauss, warmup_corrgauss


def loglike_pyramid(x): 
    """ hyper-pyramid problem (squares) """
    return -np.abs(x - 0.5).max()**0.01

def gradient_pyramid(x):
    j = np.argmax(np.abs(x - 0.5))
    v = np.zeros(len(x))
    v[j] = -1 if x[j] > 0.5 else 1
    return v

def volume_pyramid(loglike, ndim):
    """ compute volume enclosed at loglike threshold """
    sidelength = (-loglike)**100
    return sidelength**ndim

def warmup_pyramid(ndim):
    return np.random.uniform(0.4, 0.6, size=ndim)

def loglike_multigauss(x):
    """ two-peaked gaussian problem """
    a = -0.5 * (((x - 0.4)/0.01)**2).sum()
    b = -0.5 * (((x - 0.6)/0.01)**2).sum()
    return np.logaddexp(a, b)

def gradient_multigauss(x, plot=False):
    va = gradient_to_center(x, ctr=0.4)
    vb = gradient_to_center(x, ctr=0.6)
    logwa = -0.5 * (((x - 0.4)/0.01)**2).sum()
    logwb = -0.5 * (((x - 0.6)/0.01)**2).sum()
    logwmax = max(logwa, logwb)
    wa = exp(logwa - logwmax)
    wb = exp(logwb - logwmax)
    
    v = va * wa + vb * wb
    # normalise
    v /= (v**2).sum()**0.5
    return v

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

def warmup_multigauss(ndim):
    if np.random.uniform() < 0.5:
        ctr = 0.4
    else:
        ctr = 0.6
    return ctr + random_point_in_sphere(ndim, radius = 0.04)

def loglike_shell(x):
    """ gaussian shell, tilted """
    # square distance from center
    r = ((x - 0.5)**2).sum()
    # gaussian shell centered at 0.5, radius 0.4, thickness 0.004
    L1 = -0.5 * ((r - 0.4**2) / 0.004)**2
    return L1

def gradient_shell(x):
    r = ((x - 0.5)**2).sum()
    # second term gives the vector pointing to the center
    # third term is positive if r > 0.4, negative otherwise
    # v = -4 * (x - 0.5) * ((r - 0.4))**3
    # v /= (v**2).sum()**0.5
    
    # simplified:
    v = gradient_to_center(x)
    if r < 0.4:
        # point outwards if inside
        v *= -1
    
    return v

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

def warmup_shell(ndim):
    radius = 0.1
    ctr = 0.5
    # choose radial distance inside shell
    length = 0.4 + np.random.uniform(-radius, radius)
    
    # choose direction
    x = ctr + random_vector(ndim, length=length)
    return x

def get_problem(problemname, ndim):
    if problemname == 'circgauss':
        return loglike_gauss, gradient_gauss, volume_gauss, warmup_gauss
    elif problemname == 'asymgauss':
        return generate_asymgauss_problem(ndim)
    elif problemname == 'corrgauss':
        return generate_corrgauss_problem(ndim)
    elif problemname == 'pyramid':
        return loglike_pyramid, gradient_pyramid, volume_pyramid, warmup_pyramid
    elif problemname == 'multigauss':
        return loglike_multigauss, gradient_multigauss, volume_multigauss, warmup_multigauss
    elif problemname == 'shell':
        return loglike_shell, gradient_shell, volume_shell, warmup_shell
    
    raise Exception("Problem '%s' unknown" % problemname)
