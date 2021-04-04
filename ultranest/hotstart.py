"""Hot start helper functions."""

import numpy as np
import scipy.stats

def get_auxiliary_problem(loglike, transform, ctr, invcov, enlargement_factor, df=1):
    """Return a new loglike and transform based on an auxiliary distribution.

    Given a likelihood and prior transform, and information about
    the (expected) posterior peak, generates a auxiliary 
    likelihood and prior transform that is identical but
    requires fewer nested sampling iterations.

    This is achieved by deforming the prior space, and undoing that
    transformation by correction weights in the likelihood.

    The auxiliary distribution used for transformation/weighting is
    a d-dimensional Student-t distribution.

    Usage::

        aux_loglikelihood, aux_aftertransform = get_auxiliary_problem(loglike, transform, ctr, invcov, enlargement_factor, df=1)
        aux_sampler = ReactiveNestedSampler(parameters, aux_loglikelihood)
        aux_results = aux_sampler.run()
        posterior_samples = [aux_aftertransform(sample) for sample in aux_results['samples']]

    Parameters
    ------------
    loglike: function
        original likelihood function
    transform: function
        original prior transform function
    ctr: array
        Posterior center (in u-space).
    invcov: array
        Covariance of the posterior (in u-space).
    enlargement_factor: float
        Factor by which the scale of the auxiliary distribution is enlarged
        in all dimensions.

        For Gaussian-like posteriors, sqrt(ndim) seems to work,
        Heavier tailed or non-elliptical distributions may need larger factors.
    df: float
        Number of degrees of freedom of the auxiliary student-t distribution.
        The default is recommended. For truly gaussian posteriors,
        the student-t can be made more gaussian (by df>=30) for accelation.

    Returns:
    ---------
    aux_loglike: function
        auxiliary loglikelihood function.
    aux_aftertransform: function
        auxiliary transform function.
        Takes d u-space coordinates, and returns d + 1 p-space parameters.
        The first d return coordinates are identical to what ``transform`` would return.
        The final coordinate is the correction weight.
    """

    ndim, = ctr.shape
    assert invcov.shape == (ndim, ndim)
    assert df >= 1, ('Degrees of freedom must be above 1', df)

    l, v = np.linalg.eigh(invcov)
    rotation_matrix = np.dot(v, enlargement_factor * np.diag(1. / np.sqrt(l)))
    
    rv_auxiliary1d = scipy.stats.t(df)

    def aux_rotator(coords):
        return ctr + np.dot(coords, rotation_matrix)

    def aux_loglikelihood(u):
        # get uniform gauss/t distributed values:
        coords = rv_auxiliary1d.ppf(u)
        # rotate & stretch; transform into physical parameters
        x = aux_rotator(coords)
        # avoid outside regions
        if not (x > 0).all() or not (x < 1).all():
            return -1e300
        # undo the effect of the auxiliary distribution
        l = rv_auxiliary1d.logpdf(coords).sum()
        return loglike(transform(x)) - l
    
    def aux_aftertransform(u):
        return transform(aux_rotator(rv_auxiliary1d.ppf(u)))

    return aux_loglikelihood, aux_aftertransform

def get_extended_auxiliary_problem(loglike, transform, ctr, invcov, enlargement_factor, df=1):
    """Return a new loglike and transform based on an auxiliary distribution.

    Given a likelihood and prior transform, and information about
    the (expected) posterior peak, generates a auxiliary 
    likelihood and prior transform that is identical but
    requires fewer nested sampling iterations.

    This is achieved by deforming the prior space, and undoing that
    transformation by correction weights in the likelihood.

    The auxiliary distribution used for transformation/weighting is
    a d-dimensional Student-t distribution.

    Parameters
    ------------
    loglike: function
        original likelihood function
    transform: function
        original prior transform function
    ctr: array
        Posterior center (in u-space).
    invcov: array
        Covariance of the posterior (in u-space).
    enlargement_factor: float
        Factor by which the scale of the auxiliary distribution is enlarged
        in all dimensions.

        For Gaussian-like posteriors, sqrt(ndim) seems to work,
        Heavier tailed or non-elliptical distributions may need larger factors.
    df: float
        Number of degrees of freedom of the auxiliary student-t distribution.
        The default is recommended. For truly gaussian posteriors,
        the student-t can be made more gaussian (by df>=30) for accelation.

    Returns:
    ---------
    aux_loglike: function
        auxiliary loglikelihood function. Takes d + 1 parameters (see below).
        The likelihood is the same as loglike, but adds weights.
    aux_transform: function
        auxiliary transform function.
        Takes d u-space coordinates, and returns d + 1 p-space parameters.
        The first d return coordinates are identical to what ``transform`` would return.
        The final coordinate is the correction weight.
    """
    ndim, = ctr.shape
    assert invcov.shape == (ndim, ndim)
    assert df >= 1, ('Degrees of freedom must be above 1', df)
    
    l, v = np.linalg.eigh(invcov)
    rotation_matrix = np.dot(v, enlargement_factor * np.diag(1. / np.sqrt(l)))

    rv_auxiliary1d = scipy.stats.t(df)
    weight_ref = rv_auxiliary1d.logpdf(0) * ndim
    
    def aux_transform(u):
        # get uniform gauss/t distributed values:
        coords = rv_auxiliary1d.ppf(u)
        # rotate & stretch; transform into physical parameters
        x = ctr + np.dot(rotation_matrix, coords)
        # avoid outside regions
        if (x > 0).all() and (x < 1).all():
            weight = -rv_auxiliary1d.logpdf(coords).sum() + weight_ref
        else:
            weight = -1e101
            x = u * 0 + 0.5
        # add weight as a additional parameter
        return np.append(transform(x), weight)

    def aux_loglikelihood(x):
        x_actual = x[:-1]
        weight = x[-1]
        if -1e100 < weight < 1e100:
            return loglike(x_actual) + weight - weight_ref
        else:
            return -1e300

    return aux_loglikelihood, aux_transform
