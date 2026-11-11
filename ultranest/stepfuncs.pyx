# cython: language_level=3,annotate=True,profile=True,fast_fail=True,warning_errors=True
"""
Efficient helper functions for vectorized step-samplers
-------------------------------------------------------

"""

import numpy as np
cimport numpy as np
np.import_array()
from numpy import nan as np_nan
cimport cython
from cython.parallel import prange
from libc.math cimport log, fmax



ctypedef np.int64_t decl_int_t
int_dtype = np.int64


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
    np.ndarray[decl_int_t, ndim=1] indices,
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
    j = np.random.randint(ndim, size=nsamples, dtype=int_dtype)
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
    j = np.random.randint(ndim, size=nsamples, dtype=int_dtype)
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
    j = np.random.randint(ndim, size=nsamples, dtype=int_dtype)
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
    i = np.random.randint(nlive, size=nsamples, dtype=int_dtype)
    i2 = np.random.randint(nlive - 1, size=nsamples, dtype=int_dtype)
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple update_vectorised_slice_sampler(
    np.ndarray[np.float_t, ndim=1] t,
    np.ndarray[np.float_t, ndim=1] tleft,
    np.ndarray[np.float_t, ndim=1] tright,
    np.ndarray[np.float_t, ndim=1] proposed_L,
    np.ndarray[np.float_t, ndim=2] proposed_u,
    np.ndarray[np.float_t, ndim=2] proposed_p,
    np.ndarray[decl_int_t, ndim=1] worker_running,
    np.ndarray[decl_int_t, ndim=1] status,
    np.float_t Likelihood_threshold,
    np.float_t shrink_factor,
    np.ndarray[np.float_t, ndim=2] allu,
    np.ndarray[np.float_t, ndim=1] allL,
    np.ndarray[np.float_t, ndim=2] allp,
    int popsize
):
    """Update the slice sampler state of each walker in the populations.

    Parameters
    -----------
    t: array
        proposed slice coordinate
    tleft: array
        current slice negative end
    tright: array
        current slice positive end
    proposed_L: array
        log-likelihood of proposed point
    proposed_u: array
        proposed point in unit cube space
    proposed_p: array
        proposed point in transformed space
    worker_running: array
        index of the point associated with each worker
    status: array
        integer status of the point
    Likelihood_threshold: float
        current log-likelihood threshold
    shrink_factor: float
        factor by which to shrink the slice
    allu: array
        Accepted points in unit cube space
    allL: array
        log-likelihoods of accepted points
    allp: array
        Accepted points in transformed space
    popsize: int
        number of points

    Returns
    --------
    tleft: array
        updated current slice negative end
    tright: array
        updated current slice positive end
    worker_running: array
        updated index of the point associated with each worker
    status: array
        updated integer status of the point
    allu: array
        updated accepted points in unit cube space
    allL: array
        updated log-likelihoods of accepted points
    allp: array
        updated accepted points in transformed space
    discarded: int
        Point where the likelihood was evaluated but was not taken into account.
    """
                            
    cdef int j, k
    cdef discarded = 0
    for l in range(popsize):
        if t[l] > tright[worker_running[l]] or t[l] < tleft[worker_running[l]]:
            if proposed_L[l]>Likelihood_threshold:
                discarded+=1
            continue
        if 0 < t[l] < tright[worker_running[l]]:
            tright[worker_running[l]] = t[l]/shrink_factor
        if 0 > t[l] > tleft[worker_running[l]]:
            tleft[worker_running[l]] = t[l]/shrink_factor
        if proposed_L[l] > Likelihood_threshold and status[worker_running[l]] == 0:
            status[worker_running[l]] = 1
            allu[worker_running[l], :] = proposed_u[l, :]
            allL[worker_running[l]] = proposed_L[l]
            allp[worker_running[l], :] = proposed_p[l, :]

    j = 0
    while j < popsize and (status == 0).any():
        for k in range(popsize):
            if status[k] == 0 and j < popsize:
                worker_running[j] = k
                j += 1

    return (tleft, tright, worker_running, status, allu, allL, allp,discarded)


ctypedef np.float64_t float_t
#ctypedef np.int32_t decl_int_t

# Define the integer dtype used elsewhere in the code
#int_dtype = np.int32


cdef class LinearInterpolator:
    """Fast Cython linear interpolator replacing scipy.interpolate.interp1d"""
    
    cdef float_t[:] x_data
    cdef float_t[:] y_data
    cdef double fill_left
    cdef double fill_right
    cdef int n_points
    
    def __init__(self, np.ndarray x, np.ndarray y, double fill_left=0.0, double fill_right=1.0):
        """
        Initialize linear interpolator.
        
        Parameters:
        -----------
        x : ndarray, shape (n,)
            X coordinates (must be sorted)
        y : ndarray, shape (n,)
            Y coordinates
        fill_left : float
            Value to return for x < x[0]
        fill_right : float
            Value to return for x > x[-1]
        """
        # Convert to contiguous float64 arrays and store as memoryviews
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        
        self.x_data = x_arr
        self.y_data = y_arr
        self.fill_left = fill_left
        self.fill_right = fill_right
        self.n_points = len(x)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double evaluate(self, double x_val) noexcept:
        """
        Fast linear interpolation for a single point.
        This is a cdef function, so it's called without Python overhead.
        """
        cdef:
            float_t[:] x_data = self.x_data
            float_t[:] y_data = self.y_data
            int idx, n = self.n_points
            double x0, x1, y0, y1, t
        
        # Check boundaries
        if x_val < x_data[0]:
            return self.fill_left
        if x_val > x_data[n - 1]:
            return self.fill_right
        
        # Binary search for the interval
        idx = self._binary_search(x_val)
        
        # Linear interpolation
        x0 = <double>x_data[idx]
        x1 = <double>x_data[idx + 1]
        y0 = <double>y_data[idx]
        y1 = <double>y_data[idx + 1]
        
        t = (x_val - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _binary_search(self, double x_val) noexcept:
        """Binary search to find the interval containing x_val"""
        cdef:
            float_t[:] x_data = self.x_data
            int left = 0, right = self.n_points - 1, mid
        
        while left < right:
            mid = (left + right) >> 1
            if x_data[mid] < x_val:
                left = mid + 1
            else:
                right = mid
        
        return left - 1
    
    def __call__(self, double x_val):
        """Allow interpolator to be called like a function"""
        return self.evaluate(x_val)


cdef class QuantileDistribution:
    """Cython-optimized quantile-based distribution functions using memoryviews."""
    
    cdef int popsize
    cdef float_t[:, ::1] quantiles  # C-contiguous memoryview
    cdef float_t[::1] quantiles_points
    cdef list cdf_fns
    cdef list ppf_fns
    cdef float_t[:, ::1] pdf_slopes  # C-contiguous memoryview
    
    def __init__(self, np.ndarray quantiles, np.ndarray quantiles_points, int popsize):
        """
        Initialize the distribution.
        
        Parameters:
        -----------
        quantiles : ndarray of shape (n_quantiles, popsize)
            Quantile values for each worker
        quantiles_points : ndarray of shape (n_quantiles,)
            Percentile points (0-100)
        popsize : int
            Number of workers
        """
        # Convert to C-contiguous float64 arrays
        quantiles_arr = np.ascontiguousarray(quantiles, dtype=np.float64)
        quantiles_points_arr = np.ascontiguousarray(quantiles_points, dtype=np.float64)
        
        # Store as memoryviews
        self.quantiles = quantiles_arr
        self.quantiles_points = quantiles_points_arr
        self.popsize = popsize
        
        # Pre-compute interpolation functions
        self._build_cdf_ppf_functions()
        self._build_pdf_slopes()
    
    cdef _build_cdf_ppf_functions(self):
        """Build CDF and PPF linear interpolation functions."""
        self.cdf_fns = []
        self.ppf_fns = []
        
        cdef:
            int w
            LinearInterpolator cdf_fn, ppf_fn
            np.ndarray quantiles_norm_py
        
        # Convert memoryview to numpy array for mathematical operations
        quantiles_points_py = np.asarray(self.quantiles_points)
        quantiles_norm_py = np.ascontiguousarray(
            quantiles_points_py / 100.0, dtype=np.float64
        )
        
        for w in range(self.popsize):
            # Extract column as a numpy array (necessary for LinearInterpolator)
            quantiles_col = np.asarray(self.quantiles[:, w])
            
            # CDF: maps quantile values -> probabilities
            cdf_fn = LinearInterpolator(
                quantiles_col,
                quantiles_norm_py,
                fill_left=0.0,
                fill_right=1.0
            )
            self.cdf_fns.append(cdf_fn)
            
            # PPF (inverse CDF): maps probabilities -> quantile values
            ppf_fn = LinearInterpolator(
                quantiles_norm_py,
                quantiles_col,
                fill_left=<double>self.quantiles[0, w],
                fill_right=<double>self.quantiles[-1, w]
            )
            self.ppf_fns.append(ppf_fn)
    
    cdef _build_pdf_slopes(self):
        """Pre-compute PDF slopes for efficient log-pdf calculation."""
        cdef:
            np.ndarray quantiles_py = np.asarray(self.quantiles)
            np.ndarray quantiles_points_py = np.asarray(self.quantiles_points)
            np.ndarray dq_arr
            np.ndarray dq_vals_arr
            np.ndarray slopes_arr

        """
        # Perform numpy operations on numpy arrays
        dq_arr = np.diff(quantiles_points_py / 100.0)
        dq_vals_arr = np.diff(quantiles_py, axis=0)
        
        # Avoid division by zero
        dq_vals_arr = np.maximum(dq_vals_arr, 1e-12)
        
        # Compute slopes: dq / dq_vals
        slopes_arr = dq_arr[:, None] / dq_vals_arr
        
        # Store as C-contiguous array (will be automatically converted to memoryview)
        self.pdf_slopes = np.ascontiguousarray(slopes_arr, dtype=np.float64)
        """
        p = quantiles_points_py / 100.0

        mass = np.diff(p)
        width = np.diff(quantiles_py, axis=0)
        width = np.maximum(width, 1e-12)  # Avoid division by zero

        norm = mass.sum(axis=0)
        
        slopes_arr = mass[:, None] / width

        slopes_arr /= norm
        self.pdf_slopes = np.ascontiguousarray(slopes_arr, dtype=np.float64)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cdf(self, np.ndarray x_points, np.ndarray worker):
        """
        Evaluate CDF at given points for each worker.
        
        Parameters:
        -----------
        x_points : ndarray of shape (n,)
            Points at which to evaluate CDF
        worker : ndarray of shape (n,), dtype int
            Worker indices
        
        Returns:
        --------
        cdf_values : ndarray of shape (n,)
            CDF values
        """
        x_points = np.ascontiguousarray(x_points, dtype=np.float64)
        worker = np.ascontiguousarray(worker, dtype=int_dtype)
        
        cdef:
            float_t[::1] x_view = x_points
            decl_int_t[::1] worker_view = worker
            int n = len(x_points)
            np.ndarray[float_t, ndim=1] out = np.empty(n, dtype=np.float64)
            int i, w
            LinearInterpolator cdf_fn
        
        for i in range(n):
            w = <int>worker_view[i]
            cdf_fn = self.cdf_fns[w]
            out[i] = <float_t>cdf_fn.evaluate(<double>x_view[i])
        
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ppf(self, np.ndarray q_points, np.ndarray worker):
        """
        Evaluate PPF (inverse CDF) at given points for each worker.
        
        Parameters:
        -----------
        q_points : ndarray of shape (n,)
            Quantile points (probabilities)
        worker : ndarray of shape (n,), dtype int
            Worker indices
        
        Returns:
        --------
        ppf_values : ndarray of shape (n,)
            PPF values
        """
        q_points = np.ascontiguousarray(q_points, dtype=np.float64)
        worker = np.ascontiguousarray(worker, dtype=int_dtype)
        
        cdef:
            float_t[::1] q_view = q_points
            decl_int_t[::1] worker_view = worker
            int n = len(q_points)
            np.ndarray[float_t, ndim=1] out = np.empty(n, dtype=np.float64)
            int i, w
            LinearInterpolator ppf_fn
        
        for i in range(n):
            w = <int>worker_view[i]
            ppf_fn = self.ppf_fns[w]
            out[i] = <float_t>ppf_fn.evaluate(<double>q_view[i])
        
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def logpdf(self, np.ndarray x, np.ndarray worker):
        """
        Evaluate log-PDF at given points for each worker.

        Matches scipy.stats.rv_histogram:
          - constant density within each quantile interval
          - logpdf = -inf outside the support
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        worker = np.ascontiguousarray(worker, dtype=int_dtype)

        cdef:
            float_t[::1] x_view = x
            decl_int_t[::1] worker_view = worker
            float_t[:, ::1] quantiles = self.quantiles
            float_t[:, ::1] pdf_slopes = self.pdf_slopes
            int W = x.shape[0]
            int n_quantiles = quantiles.shape[0]
            np.ndarray[float_t, ndim=1] out = np.empty(W, dtype=np.float64)

            int i, w, idx
            double xi, slope

        for i in range(W):
            w = <int>worker_view[i]
            xi = <double>x_view[i]

            # largest j such that quantiles[j,w] <= xi
            idx = self._search_interval_cdef(
                quantiles,
                w,
                xi,
                n_quantiles
            )

            # Outside support -> pdf = 0 -> logpdf = -inf
            if idx < 0 or idx >= n_quantiles - 1:
                out[i] = -np.inf
                continue

            slope = <double>pdf_slopes[idx, w]

            if slope <= 0.0:
                out[i] = -np.inf
            else:
                out[i] = log(slope)

        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _search_interval_cdef(
            self,
            float_t[:, ::1] quantiles,
            int w,
            double xval,
            int n_quantiles) noexcept:
        """
        Binary search to find the interval containing xval for worker w.
        Operates directly on memoryviews for maximum speed.
        """
        cdef:
            int left = 0, right = n_quantiles, mid
        
        while left < right:
            mid = (left + right) >> 1
            if quantiles[mid, w] <= xval:
                left = mid + 1
            else:
                right = mid
        
        return left - 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def process_workers(
    decl_int_t popsize,
    decl_int_t npoints,
    np.ndarray[decl_int_t, ndim=1] status,
    np.ndarray[decl_int_t, ndim=1] worker_running,
    np.ndarray[np.float_t, ndim=2] limit_worker,
    np.ndarray[np.float_t, ndim=1] scale_worker,
    np.ndarray[np.float_t, ndim=2] limit,
    np.ndarray[np.float_t, ndim=2] v,
    np.ndarray[np.float_t, ndim=1] end_scale,
    np.float_t scale,
    np.float_t sign):
    """Cython function to process workers for stepping (C-level loops)"""
    cdef int j = 0
    cdef int k
    cdef int n_scale=1
    
    # Assign workers

    while j<popsize and (status==0).any():
        for k in range(npoints):
            if status[k]==0 and j<popsize:
                worker_running[j] = k
                for dim in range(limit_worker.shape[1]):
                    limit_worker[j, dim] = limit[k, dim] +sign * n_scale * scale * v[k, dim]
                scale_worker[j] = n_scale+end_scale[k]
                j += 1
        n_scale += 1
        
    return (worker_running, scale_worker, limit_worker)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def update_limits_NS(
    decl_int_t popsize,
    np.ndarray[decl_int_t, ndim=1] status,
    np.ndarray[decl_int_t, ndim=1] worker_running,
    np.ndarray[np.float_t, ndim=1] scale_worker,
    np.ndarray[np.float_t, ndim=1] LogLimit,
    np.float_t Lmin,
    np.ndarray[decl_int_t, ndim=1] max_n,
    np.ndarray[np.float_t, ndim=2] limit,
    np.ndarray[np.float_t, ndim=2] limit_worker,
    np.ndarray[np.float_t, ndim=1] t_unitcube,
    np.ndarray[np.float_t, ndim=1] end_scale
):
    """Cython function to update limits (C-level loops)"""
    cdef int l, worker_idx
    cdef int dims = limit.shape[1]
    
    for l in range(popsize):
        worker_idx = worker_running[l]
        # Check if scale exceeds maximum
        if scale_worker[l] > max_n[worker_idx] and status[worker_idx] == 0:
            status[worker_idx] = 1
            #for dim in range(dims):
            #    limit[worker_idx, dim] = t_unitcube[worker_idx]
            end_scale[worker_idx] = np.abs(t_unitcube[worker_idx])
        # Check likelihood threshold
        if LogLimit[l] > Lmin  and status[worker_idx] == 0:
            for dim in range(dims):
                limit[worker_idx, dim] = limit_worker[l, dim]
            #end_scale[worker_idx] = scale_worker[l]
            end_scale[worker_idx] += 1 # should be in order
        if LogLimit[l] < Lmin and status[worker_idx] == 0:
            status[worker_idx] = 1
            if max_n[worker_idx] >end_scale[worker_idx] + 1:
                end_scale[worker_idx] += 1
            else:
                end_scale[worker_idx] = np.abs(t_unitcube[worker_idx])

            #end_scale[worker_idx] += scale_worker[l]
    return (status, limit, end_scale)



