"""Utility functions for logging and statistics."""

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

    If logging handlers are already registered, no new handlers are
    registered.
    """
    logger = logging.getLogger(str(module_name))
    first_logger = logger.handlers == []
    if log_dir is not None and first_logger:
        # create file handler which logs even debug messages
        handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
        formatter = logging.Formatter('%(asctime)s [{}] [%(levelname)s] %(message)s'.format(module_name),
            datefmt='%H:%M:%S')
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

    return logger


def make_run_dir(log_dir, run_num=None, append_run_num=True):
    """Generate a new numbered directory for this run to store output.

    Parameters
    ----------
    log_dir: str
        base path
    run_num: int
        folder to add to path, such as prefix/1/
    append_run_num: bool
        If true, set run_num to next unused number

    Returns
    -------
    dictionary of folder paths for different purposes.
    Keys are "run_dir" (the path), "info", "results", "chains", "plots".

    """
    def makedirs(name):
        # for Python2 compatibility:
        try:
            os.makedirs(name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # Python 3:
        # os.makedirs(name, exist_ok=True)

    makedirs(log_dir)

    if run_num is None or run_num == '':
        run_num = (sum(os.path.isdir(os.path.join(log_dir,i))
                      for i in os.listdir(log_dir)) + 1)
    if append_run_num:
        run_dir = os.path.join(log_dir, 'run%s' % run_num)
    else:
        run_dir = log_dir
    if not os.path.isdir(run_dir):
        print('Creating directory for new run %s' % run_dir)
        makedirs(run_dir)
    if not os.path.isdir(os.path.join(run_dir, 'info')):
        makedirs(os.path.join(run_dir, 'info'))
        makedirs(os.path.join(run_dir, 'results'))
        makedirs(os.path.join(run_dir, 'chains'))
        makedirs(os.path.join(run_dir, 'extra'))
        makedirs(os.path.join(run_dir, 'plots'))

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
        return np.asarray([function(arg) for arg in args])

    vectorized.__name__ = function.__name__
    return vectorized


"""Square root of a small number."""
SQRTEPS = (float(np.finfo(np.float64).eps))**0.5


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

    idx = np.zeros(N, dtype=np.int)
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
    """
    if n % 2 == 0:
        f = 1.
        i = 2
        while i <= n:
            f *= (2. / i * pi)
            i += 2
    else:
        f = 2.
        i = 3
        while i <= n:
            f *= (2. / i * pi)
            i += 2

    return f


def verify_gradient(ndim, transform, loglike, gradient, verbose=False, combination=False):
    """
    check with numerical differentiation if the gradient function
    is plausibly correct. Raises AssertError otherwise.

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
        def transform_loglike_gradient(u):
            p = transform(u.reshape((1, -1)))
            return p[0], loglike(p)[0], gradient(u)
    
    eps = 1e-6
    N = 10
    for i in range(N):
        u = np.random.uniform(2*eps, 1-2*eps, size=(1, ndim))
        theta = transform(u)
        if verbose:
            print("---")
            print()
            print("starting at:", u, ", theta=", theta)
        Lref = loglike(theta)[0]
        if verbose: print("Lref=", Lref)
        p, L, grad = transform_loglike_gradient(u[0,:])
        assert np.allclose(p, theta), (p, theta)
        if verbose: print("gradient function gave: L=", L, "grad=", grad)
        assert np.allclose(L, Lref), (L, Lref)
        #step = grad / L
        # walk so that L increases by 10
        step = eps * grad / (grad**2).sum()**0.5
        uprime = u + step
        thetaprime = transform(uprime)
        if verbose: print("new position:", uprime, ", theta=", thetaprime)
        Lprime = loglike(thetaprime)[0]
        if verbose: print("L=", Lprime)
        # going a step of eps in the prior, should be a step in L by:
        #Lexpected = Lref + ((grad / L)**2).sum()**0.5 * eps
        Lexpected = Lref + np.dot(step, grad)
        if verbose: print("expectation was L=", Lexpected, ", given", Lref, grad, eps)
        assert np.allclose(Lprime, Lexpected, atol=0.1 / ndim), (u, uprime, theta, thetaprime, grad, eps*grad/L, L, Lprime, Lexpected)
    
