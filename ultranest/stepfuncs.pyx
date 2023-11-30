# cython: language_level=3,annotate=True,profile=True,fast_fail=True,warning_errors=True
"""
Efficient helper functions for vectorized step-samplers
-------------------------------------------------------

"""

import numpy as np
cimport numpy as np
from numpy import nan as np_nan
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _within_unit_cube(
    np.float_t [:, :] u, 
    np.uint8_t [:] acceptable, 
):
    cdef size_t popsize = u.shape[0]
    cdef size_t ndim = u.shape[1]
    cdef size_t i, j

    for i in range(popsize):
        for j in range(ndim):
            if not 0.0 < u[i,j] < 1.0:
                acceptable[i] = 0
                break


def within_unit_cube(u):
    """whether all fields are between 0 and 1, for each row

    Parameters
    ----------
    u: np.array((npoints, ndim), dtype=float):
        points

    Returns
    ---------
    within: np.array(npoints, dtype=bool):
        for each point, whether it is within the unit cube
    """
    acceptable = np.ones(u.shape[0], dtype=bool)
    _within_unit_cube(u, acceptable)
    return acceptable


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _evolve_prepare(
    np.ndarray[np.uint8_t, ndim=1] searching_left, 
    np.ndarray[np.uint8_t, ndim=1] searching_right,
    np.ndarray[np.uint8_t, ndim=1] search_right,
    np.ndarray[np.uint8_t, ndim=1] bisecting
):
    # define three mutually exclusive states: 
    # stepping out to the left, to the right, bisecting on the slice
    cdef size_t n = searching_left.shape[0]
    cdef size_t i
    for i in range(n):
        search_right[i] = not searching_left[i] and searching_right[i]
        bisecting[i] = not (searching_left[i] or searching_right[i])


def evolve_prepare(searching_left, searching_right):
    """Get auxiliary slice sampler state selectors.

    Vectorized computation for multiple (`nwalkers`) walkers.

    Parameters
    ----------
    searching_left: np.array(nwalkers, dtype=bool)
        whether stepping out in the negative direction
    searching_right: np.array(nwalkers, dtype=bool)
        whether stepping out in the positive direction

    Returns
    -------
    search_right: np.array(nwalkers, dtype=bool):
        if searching right and not left
    bisecting: np.array(nwalkers, dtype=bool):
        if not searching right nor left any more
    """
    search_right = np.empty_like(searching_left)
    bisecting = np.empty_like(searching_left)
    _evolve_prepare(searching_left, searching_right, search_right, bisecting)
    return search_right, bisecting


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef evolve_update(
    np.ndarray[np.uint8_t, ndim=1] acceptable, 
    np.ndarray[np.float_t, ndim=1] Lnew, 
    np.float_t Lmin, 
    np.ndarray[np.uint8_t, ndim=1] search_right, 
    np.ndarray[np.uint8_t, ndim=1] bisecting, 
    np.float_t[:] currentt,
    np.float_t[:] current_left,
    np.float_t[:] current_right,
    np.uint8_t[:] searching_left,
    np.uint8_t[:] searching_right,
    np.uint8_t[:] success
):
    """Update the state of each walker.

    This uses the robust logic of slice sampling, 
    with stepping out by doubling.

    Parameters
    ----------
    acceptable: np.array(nwalkers, dtype=bool)
        whether a likelihood evaluation was made. If false, rejected because out of contour.
    Lnew: np.array(acceptable.sum(), dtype=bool)
        likelihood value of proposed point
    Lmin: float
        current log-likelihood threshold
    search_right: np.array(nwalkers, dtype=bool)
        whether stepping out in the positive direction
    bisecting: np.array(nwalkers, dtype=bool)
        whether bisecting. If neither search_right nor bisecting, then 
    currentt: np.array(nwalkers)
        proposed coordinate on the slice
    current_left: np.array(nwalkers)
        current slice negative end
    current_right: np.array(nwalkers)
        current slice positive end
    searching_left: np.array(nwalkers, dtype=bool)
        whether stepping out in the negative direction
    searching_right: np.array(nwalkers, dtype=bool)
        whether stepping out in the positive direction
    success: np.array(nwalkers, dtype=bool)
        whether the walker accepts the point.

    Notes
    -----
    Writes to `currentt`, `current_left`, `current_right`, 
    `searching_left`, `searching_right`, `success`.
    """
    cdef size_t popsize = acceptable.shape[0]
    cdef size_t j = 0
    cdef size_t i
    cdef float my_nan = np_nan
    
    for k in range(popsize):
        if acceptable[k]:
            if Lnew[j] > Lmin:
                success[k] = 1
            j += 1

    for i in prange(popsize, nogil=True):
        # handle cases based on the result:
        # 1) step out further, if still accepting
        if success[i] != 0:
            if searching_left[i]:
                current_left[i] *= 2
            elif search_right[i]:
                current_right[i] *= 2
        # 2) done stepping out, if rejected
        else:
            if searching_left[i]:
                searching_left[i] = 0
            elif search_right[i]:
                searching_right[i] = 0
        # bisecting, rejected or not acceptable
        if bisecting[i]:
            if currentt[i] < 0:
                # bisect shrink left:
                current_left[i] = currentt[i]
            else:
                current_right[i] = currentt[i]
            # bisect accepted: start new slice and new generation there
            if success[i] != 0:
                currentt[i] = my_nan
        else:
            success[i] = 0

# precompute to avoid slow allocations.
pnew_empty = np.empty((0,1))
Lnew_empty = np.empty(0)

def evolve(
    transform, loglike, Lmin, 
    currentu, currentL, currentt, currentv,
    current_left, current_right, searching_left, searching_right
):
    """Evolve each slice sampling walker.

    Parameters
    ----------
    transform: function
        prior transform function
    loglike: function
        loglikelihood function
    Lmin: float
        current log-likelihood threshold
    currentu: np.array((nwalkers, ndim))
        slice starting point (where currentt=0)
    currentL: np.array(nwalkers)
        current loglikelihood
    currentt: np.array(nwalkers)
        proposed coordinate on the slice
    currentv: np.array((nwalkers, ndim))
        slice direction vector
    current_left: np.array(nwalkers)
        current slice negative end
    current_right: np.array(nwalkers)
        current slice positive end
    searching_left: np.array(nwalkers, dtype=bool)
        whether stepping out in the negative direction
    searching_right: np.array(nwalkers, dtype=bool)
        whether stepping out in the positive direction

    Returns
    -------
    currentt: np.array(nwalkers)
        as above
    currentv: np.array((nwalkers, ndim))
        as above
    current_left: np.array(nwalkers)
        as above
    current_right: np.array(nwalkers)
        as above
    searching_left: np.array(nwalkers, dtype=bool)
        as above
    searching_right: np.array(nwalkers, dtype=bool)
        as above
    success: np.array(nwalkers, dtype=bool)
        whether the walker accepts the point.
    unew: np.array((success.sum(), ndim))
        coordinates of accepted points
    pnew: np.array((success.sum(), nparams))
        transformed coordinates of accepted points
    Lnew: np.array(success.sum())
        log-likelihoods of accepted points
    nc: int
        number of points for which the log-likelihood function was called.

    This function writes in-place to 
    `currentt`, `currentv`, `current_left`, `current_right`, `searching_left`, 
    `searching_right` and `currentu`, but also returns these.
    """
    search_right, bisecting = evolve_prepare(searching_left, searching_right)

    unew = currentu
    unew[searching_left,:] = currentu[searching_left,:] + currentv[searching_left,:] * current_left[searching_left].reshape((-1,1))
    unew[search_right,:] = currentu[search_right,:] + currentv[search_right,:] * current_right[search_right].reshape((-1,1))
    currentt[bisecting] = np.random.uniform(current_left[bisecting], current_right[bisecting])
    unew[bisecting,:] = currentu[bisecting,:] + currentv[bisecting,:] * currentt[bisecting].reshape((-1,1))

    acceptable = within_unit_cube(unew)

    nc = 0
    if acceptable.any():
        pnew = transform(unew[acceptable,:])
        Lnew = loglike(pnew)
        nc += len(pnew)
    else:
        pnew = pnew_empty
        Lnew = Lnew_empty

    success = np.zeros_like(searching_left)
    evolve_update(
        acceptable, Lnew, Lmin, search_right, bisecting, currentt,
        current_left, current_right, searching_left, searching_right,
        success
    )

    return (
        (
        currentt, currentv,
        current_left, current_right, searching_left, searching_right), 
        (success, unew[success,:], pnew[success[acceptable],:], Lnew[success[acceptable]]), 
        nc
    )


def step_back(Lmin, allL, generation, currentt, log=False):
    """Revert walkers which have wandered astray.

    Revert until all previous steps have likelihoods allL above Lmin.
    Updates currentt, generation and allL, in-place.

    Parameters
    ----------
    Lmin: float
        current loglikelihood threshold
    allL: np.array((nwalkers, ngenerations))
        loglikelihoods of the chain. NaN where not evaluated yet.
    generation: np.array(nwalkers, dtype=int)
        how many iterations each walker has completed.
    currentt: np.array(nwalkers)
        current slice coordinate
    log: bool
        whether to print when steps are reverted


    """
    # step back where step was excluded by Lmin increase
    # delete from the back until all are good:
    max_width = generation.max() + 1
    below_threshold = allL[:,:max_width] < Lmin
    problematic_parent = np.any(below_threshold, axis=1)
    if not problematic_parent.any():
        return
    parent_i, = np.where(problematic_parent)
    below_threshold_parent = below_threshold[parent_i,:]
    # first, all of them (because we already identified them)
    problematic = np.ones(len(parent_i), dtype=bool)
    step = 0

    while True:
        step += 1
        ii, = np.where(problematic)
        i = parent_i[problematic]
        g = generation[i]
        generation[i] -= 1
        currentt[i] = np_nan
        allL[i,g] = np_nan
        below_threshold_parent[problematic, g] = False
        if log:
            print("resetting %d%%" % (problematic.meancount_good_generations() * 100), 'by', step, 'steps', 'to', g)

        del problematic
        problematic = np.any(below_threshold_parent, axis=1)
        if not problematic.any():
            break


cdef _fill_directions(
    np.ndarray[np.float_t, ndim=2] v,
    np.ndarray[np.int_t, ndim=1] indices,
    float scale
):
    cdef size_t nsamples = v.shape[0]
    cdef size_t i
    for i in range(nsamples):
        v[i, indices[i]] = scale


def generate_cube_oriented_direction(ui, region, scale=1):
    """Draw a unit direction vector in direction of a random unit cube axes.

    Parameters
    ----------
    ui: np.array((npoints, ndim), dtype=float)
        starting points (not used)
    region:
        not used
    scale: float
        length of returned vector

    Returns
    ---------
    v: np.array((npoints, ndim), dtype=float)
        Random axis vectors of length `scale`, one for each starting point.
    """
    nsamples, ndim = ui.shape
    v = np.zeros((nsamples, ndim))
    # choose axis
    j = np.random.randint(ndim, size=nsamples)
    _fill_directions(v, j, scale)
    return v


def generate_cube_oriented_direction_scaled(ui, region, scale=1):
    """Draw a unit direction vector in direction of a random unit cube axes.
    Scale by the live point min-max range.

    Parameters
    ----------
    ui: np.array((npoints, ndim), dtype=float)
        starting points (not used)
    region:
        not used
    scale: float
        length of returned vector

    Returns
    ---------
    v: np.array((npoints, ndim), dtype=float)
        Random axis vectors of length `scale`, one for each starting point.
    """
    nsamples, ndim = ui.shape
    v = np.zeros((nsamples, ndim))
    scales = region.u.std(axis=0)
    # choose axis
    j = np.random.randint(ndim, size=nsamples)
    _fill_directions(v, j, scale)
    v *= scales[j].reshape((-1, 1))
    return v

def generate_random_direction(ui, region, scale=1):
    """Draw uniform direction vector in unit cube space of length `scale`.

    Parameters
    -----------
    ui: np.array((npoints, ndim), dtype=float)
        starting points (not used)
    region: MLFriends object
        current region (not used)
    scale: float
        length of direction vector
    
    Returns
    --------
    v: array
        new direction vector
    """
    del region
    nsamples, ndim = ui.shape
    v = np.random.normal(size=(nsamples, ndim))
    v *= scale / np.linalg.norm(v, axis=1).reshape((nsamples, 1))
    return v


def generate_region_oriented_direction(ui, region, scale=1):
    """Draw a random direction vector in direction of one of the `region` axes.

    If given, the vector length is `scale`.
    If not, the vector length in transformed space is `tscale`.

    Parameters
    -----------
    ui: np.array((npoints, ndim), dtype=float)
        starting points (not used)
    region: MLFriends object
        current region
    scale: float
        length of direction vector in t-space

    Returns
    --------
    v: array
        new direction vector (in u-space)
    """
    nsamples, ndim = ui.shape
    # choose axis in transformed space:
    j = np.random.randint(ndim, size=nsamples)
    v = region.transformLayer.axes[j] * scale
    return v


def generate_region_random_direction(ui, region, scale=1):
    """Draw a direction vector in a random direction of the region.

    The vector length is `scale` (in unit cube space).

    Parameters
    -----------
    ui: np.array((npoints, ndim), dtype=float)
        starting points (not used)
    region: MLFriends object
        current region
    scale: float:
        length of direction vector (in t-space)
    
    Returns
    --------
    v: array
        new direction vector
    """
    nsamples, ndim = ui.shape
    # choose axis in transformed space:
    v1 = np.random.normal(size=(nsamples, ndim))
    v1 *= scale / np.linalg.norm(v1, axis=1).reshape((nsamples, 1))
    v = np.einsum('ij,kj->ki', region.transformLayer.axes, v1)
    return v

def generate_differential_direction(ui, region, scale=1):
    """Sample a vector using the difference between two randomly selected live points.

    Parameters
    -----------
    ui: np.array((npoints, ndim), dtype=float)
        starting point
    region: MLFriends object
        current region
    scale: float:
        length of direction vector (in t-space)

    Returns
    --------
    v: array
        new direction vector
    """
    nsamples, ndim = ui.shape
    nlive, ndim = region.u.shape
    # choose pair
    i = np.random.randint(nlive, size=nsamples)
    i2 = np.random.randint(nlive - 1, size=nsamples)
    i2[i2 >= i] += 1

    # compute difference vector
    v = (region.u[i,:] - region.u[i2,:]) * scale
    return v



def generate_mixture_random_direction(ui, region, scale=1):
    """Sample randomly uniformly from two proposals.

    Randomly applies either :py:func:`generate_differential_direction`,
    which transports far, or :py:func:`generate_region_oriented_direction`,
    which is stiffer.

    Best method according to https://arxiv.org/abs/2211.09426

    Parameters
    -----------
    ui: np.array((npoints, ndim), dtype=float)
        starting point
    region: MLFriends object
        current region
    scale: float:
        length of direction vector (in t-space)

    Returns
    --------
    v: array
        new direction vector
    """
    nsamples, ndim = ui.shape
    v_DE = generate_differential_direction(ui, region, scale=scale)
    v_axis = generate_region_oriented_direction(ui, region, scale=scale)
    return np.where(np.random.uniform(size=nsamples).reshape((-1, 1)) < 0.5, v_DE, v_axis)
