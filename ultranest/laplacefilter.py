"""
Laplace Approximation Likelihood Filter
---------------------------------------

"""

import numpy as np


def laplacify(samples, loglikes, center):
    """Fit a laplace approximation.

    Builds a quadratic approximation of samples,
    with least squares.
    This serves as a Laplace approximation to the
    log-likelihood.

    Parameters
    ----------
    samples: array
        location of points. each row is a point, each column a dimension.
    loglikes: vector
        value of the `samples`
    center: vector
        vector that will be subtracted from `samples`,
        indicating the centre of the quadratic approximation.

    Returns
    -------
    offset: float
        constant, zeroth order coefficient. Peak log-likelihood.
    linterms: vector
        linear, first-order coefficients.
    matrix: array
        quadratic, second-order coefficients.
    residuals: vector
        for each entry in `samples`, the square difference between
        the quadratic approximation and the `loglikes`.

    """
    N, dim = samples.shape
    delta = samples - center.reshape((1, -1))
    # make cross products:
    indices = np.triu_indices(dim)
    all_mixed_terms = (delta.reshape((N, 1, dim)) * delta.reshape((N, dim, 1)))
    mixed_terms = all_mixed_terms[:, indices[0], indices[1]].reshape((N, -1))

    # Now we say:
    # coeff0 + dot(coefflin, delta) + dot(coeffmixed, mixed_terms) = loglikes

    # print("shapes:", delta.shape, mixed_terms.shape, N, loglikes.shape, indices)
    terms = np.hstack((np.ones((N, 1)), delta * -0.5, mixed_terms * -0.5))

    # print("unknown terms:", terms.shape)
    coeffs, residuals, rank, s = np.linalg.lstsq(terms, loglikes, rcond=None)
    # print("results:", coeffs.shape, terms.shape, loglikes.shape, coeffs, residuals, rank, s)
    if rank != terms.shape[1]:
        raise ValueError("Could not construct laplace approximation: Not full rank")
    offset = coeffs[0]
    linterms = coeffs[1:1 + dim]
    matrix = np.zeros((dim, dim))
    matrix[indices] = coeffs[1 + dim:]
    return offset, linterms, matrix, residuals


def train_approximator(samples, loglikes):
    """Fit a laplace approximation.

    Builds a numerically stable quadratic approximation of samples,
    with least squares. This serves as a Laplace approximation to the
    log-likelihood.

    Parameters
    ----------
    samples: array
        location of points. each row is a point, each column a dimension.
    loglikes: vector
        value of the `samples`

    Returns
    -------
    residuals: vector
        for each entry in `samples`, the square difference between
        the quadratic approximation and the `loglikes`.
    laplace_parameters: tuple
        return value of :py:func:`laplacify`
    """
    center_guess = samples.mean(axis=0)
    offset1, linterms1, matrix1, residuals1 = laplacify(samples, loglikes, center_guess)
    realcenter = np.linalg.solve(matrix1, -linterms1) + center_guess
    offset, linterms, matrix, residuals = laplacify(samples, loglikes, realcenter)
    laplace_parameters = (realcenter, offset, linterms, matrix)
    return residuals, laplace_parameters


def apply_approximator(samples, laplace_parameters):
    """Predict with an existing laplace approximation.

    Parameters
    ----------
    samples: array
        location of points to predict. each row is a point, each column a dimension.
    laplace_parameters: tuple
        return value of :py:func:`laplacify`

    Returns
    -------
    loglikes: vector
        value of the `samples`
    """
    center, offset, linterms, matrix = laplace_parameters
    N, dim = samples.shape
    delta = samples - center.reshape((1, -1))
    lintermsum = - 0.5 * np.einsum('ji,i->j', delta, linterms)
    mixedtermsum = - 0.5 * np.einsum('ij,jk,ik->i', delta, matrix, delta)
    return offset + lintermsum + mixedtermsum


class LaplaceApproximationFilter:
    """Filter nested sampling proposals based on a Laplace approximation of the live points.

    This filter should be used together with a ReactiveNestedSampler.
    It replaces the loglikelihood of the sampler with
    a filtered version, which only calls the original loglikelihood
    when necessary.

    A Laplace approximation of the live points and their log-likelihoods
    is built and evaluated. The maximum absolute prediction error is stored.
    For newly proposed points, the learned approximation is used to predict the
    expected log-likelihood. If this falls below the current
    nested sampling likelihood threshold, considering also the prediction error,
    the proposed point is skipped. If it could be near or above the current
    nested sampling likelihood threshold, the original log-likelihood function
    is called.
    """

    def __init__(self, sampler, safety, verbose=True):
        """Initialise.

        Parameters
        ----------
        sampler: ReactiveNestedSampler
            sampler to use. The loglikelihood and callback of this sampler is overwritten,
            and its Lmin attribute read.
        safety: float
            The rejection threshold is Lpred + Lerr * safety > Lmin,
            where Lmin is the current nested sampling loglikelihood threshold,
            Lpred is the predicted loglikelihood, and Lerr
            is the maximum prediction error over the training sample
            (live point set at some iteration).
            Larger safety thus permits more samples to see the original likelihood.
        verbose: bool
            whether to print out filter rates and Lerr diagnostics.

        """
        self.sampler = sampler
        self.orig_loglike = self.sampler.loglike
        self.sampler.loglike = self.loglike
        self.orig_viz_callback = None
        self.laplace_parameters = None
        self.maxerr = None
        self.approximator_lastLmin = -np.inf
        self.ncalls_skipped = 0
        self.safety = safety
        self.verbose = verbose

    def update_approximator(self, u, p, L):
        """Learn the Laplace approximation.

        Parameters
        ----------
        u: array
            live point samples in untransformed u-space. Not used.
        p: array
            live point samples in transformed p-space. used.
        L: vector
            log-likelihoods of the live points
        """
        samples = p
        resid, laplace_parameters = train_approximator(samples, L)
        self.laplace_parameters = laplace_parameters
        # assume the worst case:
        self.maxerr = resid.max()**0.5
        if self.verbose:
            print("quadratic log-likelihood approximation residual: %.2f   " % self.maxerr)

    def loglike(self, params):
        """Compute log-likelihood, filtered.

        Parameters
        ----------
        params: array
            proposed points

        Returns
        -------
        L: vector
            log-likelihoods of the params
        """
        if self.laplace_parameters is None:
            return self.orig_loglike(params)

        Lerr = self.maxerr
        Lpredict = apply_approximator(params, self.laplace_parameters)

        # check whether the likelihood can plausibly be above the threshold
        mask_outside = Lpredict + Lerr * self.safety < self.sampler.Lmin
        self.ncalls_skipped += mask_outside.sum()
        if self.verbose:
            print("fraction of proposals skipped: %.2f%%" % (100 * mask_outside.mean()))
        if not mask_outside.all():
            # some have a chance:
            Lpredict[~mask_outside] = self.orig_loglike(params[~mask_outside,:])
        Lpredict[mask_outside] -= Lerr
        return Lpredict

    def viz_callback(self, points, *args, **kwargs):
        """Call for visualisation, updates Laplace approximation.

        Parameters
        ----------
        points: array
            information about the live point set.
        *args: tuple
            additional arguments, passed on to original viz_callback.
        **kwargs: dict
            additional arguments, passed on to original viz_callback.
        """
        # train on the regular callbacks
        try:
            self.update_approximator(points['u'], points['p'], points['logl'])
        except ValueError:
            pass

        if self.orig_viz_callback:
            self.orig_viz_callback(points=points, *args, **kwargs)
