"""Hot start helper functions."""

import numpy as np
import scipy.stats
from .utils import vectorize, resample_equal


def get_extended_auxiliary_problem(
    loglike, transform, usamples, weights,
    enlargement_factor, df=1, vectorized=False
):
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
    usamples: array
        Untransformed posterior samples (in u-space).
    weights: array
        Weights for the usamples
    enlargement_factor: float
        Factor by which the scale of the auxiliary distribution is enlarged
        in all dimensions.

        For Gaussian-like posteriors, sqrt(ndim) seems to work,
        Heavier tailed or non-elliptical distributions may need larger factors.
    df: float
        Number of degrees of freedom of the auxiliary student-t distribution.
        The default is recommended. For truly gaussian posteriors,
        the student-t can be made more gaussian (by df>=30) for accelation.
    vectorized: bool
        Whether the likelihood and transform functions are vectorized

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
    assert df > 1, ('Degrees of freedom must exceed 1', df)
    
    # Transform to a unit gaussian auxiliary space (g-space)
    isamplesg = scipy.stats.norm.ppf(usamples)
    # get parameters of auxiliary transform in this space
    
    # gctr = np.mean(isamplesg, axis=0)
    # gcov = np.cov(isamplesg, rowvar=0)
    QW = isamplesg * weights
    gctr = np.sum(QW, axis=0) / np.sum(weights)
    gcov = QW.T.dot(QW) / weights.T.dot(weights)
    
    ginvcov = np.linalg.inv(gcov)
    
    # build transform
    l, v = np.linalg.eigh(ginvcov)
    rotation_matrix = np.dot(v, enlargement_factor * np.diag(1. / np.sqrt(l)))

    rv_auxiliary1d = scipy.stats.t(df)
    
    if vectorized:
        def combine_with_weights(p, w):
            return np.hstack((p, w.reshape((-1,1))))
    else:
        def combine_with_weights(p, w):
            return np.append(p, w)

    def aux_transform(u):
        # get uniform gauss/t distributed values in g-space:
        coords = rv_auxiliary1d.ppf(u)
        # rotate & stretch; transform into g-space
        g = gctr + np.dot(coords, rotation_matrix)
        # since our proposal above is with the auxiliary distribution,
        # instead of the prior, a importance weight to adjust
        # the likelihood is needed
        # this is the ratio of the density of the aux dist
        # to the unit normal gaussian
        logweight_aux = rv_auxiliary1d.logpdf(coords).sum()
        logweight_unit = scipy.stats.norm.logpdf(g).sum()
        logweight = logweight_unit - logweight_aux
        
        # transform back to u space
        u = scipy.stats.norm.cdf(g)
        # transform to p space with user transform
        return combine_with_weights(transform(u), logweight)

    if vectorize:
        def aux_loglikelihood(x):
            x_actual = x[:,:-1]
            logweight = x[:,-1]
            aux_like = np.where(
                np.logical_and(-1e100 < logweight, logweight < 1e100),
                loglike(x_actual) + logweight,
                -1e300
            )
            return aux_like
    else:
        def aux_loglikelihood(x):
            x_actual = x[:-1]
            logweight = x[-1]
            if -1e100 < logweight < 1e100:
                return loglike(x_actual) + logweight
            else:
                return -1e300

    return aux_loglikelihood, aux_transform


def reuse_samples(
    param_names, loglike, points, logl, logw=None,
    logz=0.0, logzerr=0.0, upoints=None,
    batchsize=128, vectorized=False, log_weight_threshold=-10,
    **kwargs
):
    """
    Reweight existing nested sampling run onto a new loglikelihood.

    Parameters
    ------------
    param_names: list of strings
        Names of the parameters
    loglike: function
        New likelihood function
    points: np.array of shape (npoints, ndim)
        Equally weighted (unless logw is passed) posterior points
    logl: np.array(npoints)
        Previously likelihood values of points
    logw: np.array(npoints)
        Log-weights of existing points.
    logz: float
        Previous evidence / marginal likelihood value.
    logzerr: float
        Previous evidence / marginal likelihood uncertainty.
    upoints: np.array of shape (npoints, ndim)
        Posterior points before transformation.
    vectorized: bool
        Whether loglike function is vectorized
    batchsize: int
        Number of points simultaneously passed to vectorized loglike function
    log_weight_threshold: float
        Lowest log-weight to consider

    Returns:
    ---------
    results: dict
        All information of the run. Important keys:
        Number of nested sampling iterations (niter),
        Evidence estimate (logz),
        Effective Sample Size (ess),
        weighted samples (weighted_samples),
        equally weighted samples (samples),
        best-fit point information (maximum_likelihood),
        posterior summaries (posterior).
    """
    if not vectorized:
        loglike = vectorize(loglike)

    Npoints, ndim = points.shape
    if logw is None:
        # assume equally distributed if no weights given
        logw = np.zeros(Npoints) - np.log(Npoints)
    logl_new = np.zeros(Npoints) - np.inf
    logw_new = np.zeros(Npoints) - np.inf
    assert logl.shape == (Npoints,)
    assert logw.shape == (Npoints,)

    # process points, highest weight first:
    indices = np.argsort(logl + logw)[::-1]
    ncall = 0
    for i in range(int(np.ceil(Npoints / batchsize))):
        batch = indices[i * batchsize:(i + 1) * batchsize]
        logl_new[batch] = loglike(points[batch,:])
        logw_new[batch] = logw[batch] + logl_new[batch]
        ncall += len(batch)
        if (logw_new[batch] < np.nanmax(logw_new) - np.log(Npoints) + log_weight_threshold).all():
            print("skipping", i)
            break

    logw_new0 = logw_new.max()
    w = np.exp(logw_new - logw_new0)
    print("weights:", w)
    logz_new = np.log(w.sum()) + logw_new0
    w /= w.sum()
    ess = len(w) / (1.0 + ((len(w) * w - 1)**2).sum() / len(w))

    integral_uncertainty_estimator = (((w - 1 / Npoints)**2).sum() / (Npoints - 1))**0.5
    logzerr_new = np.log(1 + integral_uncertainty_estimator)
    logzerr_new_total = (logzerr_new**2 + logzerr**2)**0.5

    samples = resample_equal(points, w)
    information_gain_bits = []
    for i in range(ndim):
        H, _ = np.histogram(points[:,i], weights=w, density=True, bins=np.linspace(0, 1, 40))
        information_gain_bits.append(float((np.log2(1 / ((H + 0.001) * 40)) / 40).sum()))

    j = logl_new.argmax()
    return dict(
        ncall=ncall,
        niter=Npoints,
        logz=logz_new, logzerr=logzerr_new_total,
        ess=ess,
        posterior=dict(
            mean=samples.mean(axis=0).tolist(),
            stdev=samples.std(axis=0).tolist(),
            median=np.percentile(samples, 50, axis=0).tolist(),
            errlo=np.percentile(samples, 15.8655, axis=0).tolist(),
            errup=np.percentile(samples, 84.1345, axis=0).tolist(),
            information_gain_bits=information_gain_bits,
        ),
        weighted_samples=dict(
            upoints=upoints, points=points, weights=w, logw=logw,
            logl=logl_new),
        samples=samples,
        maximum_likelihood=dict(
            logl=logl_new[j],
            point=points[j,:].tolist(),
            point_untransformed=upoints[j,:].tolist() if upoints is not None else None,
        ),
        param_names=param_names,
    )
