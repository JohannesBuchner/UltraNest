"""Utility functions for logging and statistics."""

from __future__ import print_function, division
import logging
import sys
import os
import numpy as np
from numpy import pi
import errno


def create_logger(module_name, log_dir=None, level=logging.INFO):
    """
    Set up the logging channel `module_name`.

    Append to ``debug.log`` in `log_dir` (if not ``None``).
    Write to stdout with output level `level`.

    If logging handlers are already registered for this module,
    no new handlers are registered.

    Parameters
    ----------
    module_name: str
        logger module
    log_dir: str
        directory to write debug.log file into
    level: logging level
        which level (and above) to log to stdout.

    Returns
    -------
    logger:
        logger instance
    """
    logger = logging.getLogger(str(module_name))
    first_logger = logger.handlers == []
    if log_dir is not None and first_logger:
        # create file handler which logs even debug messages
        handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
        msgformat = '%(asctime)s [{}] [%(levelname)s] %(message)s'
        formatter = logging.Formatter(
            msgformat.format(module_name), datefmt='%H:%M:%S')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if first_logger:
        logger.setLevel(logging.DEBUG)
        # if it is new, register to write to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('[{}] %(message)s'.format(module_name))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(logging.NullHandler())
    return logger


def _makedirs(name):
    """python2-compatible makedir."""
    # for Python2 compatibility:
    try:
        os.makedirs(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
    # Python 3:
    # os.makedirs(name, exist_ok=True)


def make_run_dir(log_dir, run_num=None, append_run_num=True, max_run_num=10000):
    """Generate a new numbered directory for this run to store output.

    Parameters
    ----------
    log_dir: str
        base path
    run_num: int
        folder to add to path, such as prefix/run1/
    append_run_num: bool
        If true, set run_num to next unused number
    max_run_num: int
        Maximum number of automatic run subfolders

    Returns
    -------
    folderpath: dict
        dictionary of folder paths for different purposes.
        Keys are "run_dir" (the path), "info", "results", "chains", "plots".

    """
    _makedirs(log_dir)

    if run_num is None or run_num == '':
        # loop over existing folders (or files) of the form log_dir/runX
        # to find next available run_num (up to the hardcoded maximum of 1000)
        for run_num in range(1, max_run_num):
            if os.path.exists(os.path.join(log_dir, 'run%s' % run_num)):
                continue
            else:
                break
        else:
            raise ValueError("log directory '%s' already contains maximum number of run subdirectories (%d)" % (log_dir, max_run_num))
    if append_run_num:
        run_dir = os.path.join(log_dir, 'run%s' % run_num)
    else:
        run_dir = log_dir
    if not os.path.isdir(run_dir):
        print('Creating directory for new run %s' % run_dir)
        _makedirs(run_dir)
    if not os.path.isdir(os.path.join(run_dir, 'info')):
        _makedirs(os.path.join(run_dir, 'info'))
        _makedirs(os.path.join(run_dir, 'results'))
        _makedirs(os.path.join(run_dir, 'chains'))
        _makedirs(os.path.join(run_dir, 'extra'))
        _makedirs(os.path.join(run_dir, 'plots'))

    return {'run_dir': run_dir,
            'info': os.path.join(run_dir, 'info'),
            'results': os.path.join(run_dir, 'results'),
            'chains': os.path.join(run_dir, 'chains'),
            'extra': os.path.join(run_dir, 'extra'),
            'plots': os.path.join(run_dir, 'plots')
            }


def vectorize(function):
    """Vectorize likelihood or prior_transform function."""
    def vectorized(args):
        """Vectorized version of function."""
        return np.asarray([function(arg) for arg in args])

    # give a user-friendly name to the vectorized version of the function
    # getattr works around methods, which do not have __name__
    vectorized.__name__ = getattr(function, '__name__', vectorized.__name__)
    return vectorized


"""Square root of a small number."""
SQRTEPS = (float(np.finfo(float).eps))**0.5


def resample_equal(samples, weights, rstate=None):
    """Resample the samples so that the final samples all have equal weight.

    Each input sample appears in the output array either
    `floor(weights[i] * N)` or `ceil(weights[i] * N)` times, with
    `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray`
        Unequally weight samples returned by the nested sampling algorithm.
        Shape is (N, ...), with N the number of samples.
    weights : `~numpy.ndarray`
        Weight of each sample. Shape is (N,).
    rstate : `~numpy.random.RandomState`
        random number generator. If not provided, numpy.random is used.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray`
        Samples with equal weights, same shape as input samples.

    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> nestle.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in
    `this PDF <http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf>`_.
    Another way to sample according to weights would be::

        N = len(weights)
        new_samples = samples[np.random.choice(N, size=N, p=weights)]

    However, the method used in this function is less "noisy".

    """
    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("weights do not sum to 1 (%g)" % np.sum(weights))

    if rstate is None:
        rstate = np.random

    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (rstate.random() + np.arange(N)) / N

    idx = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    rstate.shuffle(idx)
    return samples[idx]


def listify(*args):
    """
    Concatenate args, which are (made to be) lists.

    Parameters
    ----------
    args: iterable
        Lists to concatenate.

    Returns
    -------
    list:
        Concatenation of the lists in args.
    """
    out = []
    for a in args:
        out += list(a)
    return out


def quantile(x, q, weights=None):
    """Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """
    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def vol_prefactor(n):
    """Volume constant for an `n`-dimensional sphere.

    for `n` even:  $$    (2pi)^(n    /2) / (2 * 4 * ... * n)$$
    for `n` odd :  $$2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)$$

    Parameters
    ----------
    n: int
        Dimensionality

    Returns
    -------
    Volume: float

    """
    if n % 2 == 0:
        f = 1.
        i = 2
    else:
        f = 2.
        i = 3

    while i <= n:
        f *= 2. / i * pi
        i += 2

    return f


def is_affine_transform(a, b):
    """
    Check if one points *a* and *b* are related by an affine transform.

    The implementation currently returns False for rotations.

    Parameters
    ----------
    a: array
        transformed points
    b: array
        original points

    Returns
    -------
    is_affine: bool

    """
    n, da = a.shape
    nb, db = b.shape
    assert n == nb
    assert db >= da

    n = (n // 2) * 2
    a1 = a[0:n:2]
    a2 = a[1:n:2]
    b1 = b[0:n:2,:da]
    b2 = b[1:n:2,:da]
    slopes = (b2 - b1) / (a2 - a1)
    if not np.allclose(slopes, slopes[0]):
        return False
    offsets = b1 - slopes * a1
    if not np.allclose(offsets, offsets[0]):
        return False
    return True


def normalised_kendall_tau_distance(values1, values2, i=None, j=None):
    """
    Normalised Kendall tau distance between two equally sized arrays.

    see https://en.wikipedia.org/wiki/Kendall_tau_distance

    You can optionally pass precomputed indices::

        i, j = np.meshgrid(np.arange(N), np.arange(N))

    Parameters
    ----------
    values1: array of ints
        ranks
    values2: array of ints
        other ranks (same length as values1)
    i: array of ints
        2d indices selecting values1
    j: array of ints
        2d indices selecting values2

    Returns
    -------
    distance: float

    """
    N = len(values1)
    assert len(values2) == N, "Both lists have to be of equal length"
    if i is None or j is None:
        i, j = np.meshgrid(np.arange(N), np.arange(N))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (N * (N - 1))


def _merge_transform_loglike_gradient_function(transform, loglike, gradient):
    def transform_loglike_gradient(u):
        """Combine transform, likelihood and gradient function."""
        p = transform(u.reshape((1, -1)))
        return p[0], loglike(p)[0], gradient(u)
    return transform_loglike_gradient


def verify_gradient(ndim, transform, loglike, gradient, verbose=False, combination=False):
    """
    Check with numerical differentiation if gradient function is plausibly correct.

    Raises AssertError if not fulfilled.
    All functions are vectorized.

    Parameters
    ----------
    ndim : int
        dimensionality
    transform : function
        transform unit cube parameters to physical parameters, vectorized
    loglike : function
        loglikelihood function, vectorized
    gradient : function
        computes gradient of loglike to unit cube parameters.
        Takes a single point and returns a single vector.
    verbose : bool
        whether to show intermediate test results
    combination : bool
        if true, the gradient function should return a tuple of:
        (transformed parameters, loglikelihood, gradient) for a
        given unit cube point.
    """
    if combination:
        transform_loglike_gradient = gradient
    else:
        transform_loglike_gradient = _merge_transform_loglike_gradient_function(transform, loglike, gradient)

    eps = 1e-6
    N = 10
    for i in range(N):
        u = np.random.uniform(2 * eps, 1 - 2 * eps, size=(1, ndim))
        theta = transform(u)
        if verbose:
            print("---")
            print()
            print("starting at:", u, ", theta=", theta)
        Lref = loglike(theta)[0]
        if verbose:
            print("Lref=", Lref)
        p, L, grad = transform_loglike_gradient(u[0,:])
        assert np.allclose(p, theta), (p, theta)
        if verbose:
            print("gradient function gave: L=", L, "grad=", grad)
        assert np.allclose(L, Lref), (L, Lref)
        # walk so that L increases by 10
        step = eps * grad / (grad**2).sum()**0.5
        uprime = u + step
        thetaprime = transform(uprime)
        if verbose:
            print("new position:", uprime, ", theta=", thetaprime)
        Lprime = loglike(thetaprime)[0]
        if verbose:
            print("L=", Lprime)
        # going a step of eps in the prior, should be a step in L by:
        Lexpected = Lref + np.dot(step, grad)
        if verbose:
            print("expectation was L=", Lexpected, ", given", Lref, grad, eps)
        assert np.allclose(Lprime, Lexpected, atol=0.1 / ndim), \
            (u, uprime, theta, thetaprime, grad, eps * grad / L, L, Lprime, Lexpected)

def distributed_work_chunk_size(num_total_tasks, mpi_rank, mpi_size):
    """
    Computes the number of tasks for process number `mpi_rank`, so that
    `num_total_tasks` tasks are spread uniformly among `mpi_size` processes.

    Parameters
    ----------
    num_total_tasks : int
        total number of tasks to be split
    mpi_rank : int
        process id
    mpi_size : int
        total number of processes
    """
    return (num_total_tasks + mpi_size - 1 - mpi_rank) // mpi_size


def submasks(mask, *masks):
    """
    Get indices for an array, so that
    array[indices] is equivalent to a[mask][mask1][mask2].

    Parameters
    ----------
    mask : np.array(dtype=bool)
        selection of some array
    masks : list of np.array(dtype=bool)
        each further mask is a subselection

    Returns
    -------
    indices : np.array(dtype=int)
        indices which select the subselection in the original array

    """
    indices, = np.where(mask)
    for othermask in masks:
        indices = indices[othermask]
    return indices
