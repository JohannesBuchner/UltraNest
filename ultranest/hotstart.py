"""Hot start helper functions."""

import numpy as np
import scipy.stats
from .utils import vectorize, resample_equal


def get_extended_auxiliary_problem(
    loglike, transform, usamples, weights,
    enlargement_factor, df, vectorized=False
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
    aux_transform: function
        auxiliary transform function.
        Takes d u-space coordinates, and returns d + 1 p-space parameters.
        The first d return coordinates are identical to what ``transform`` would return.
        The final coordinate is the correction weight.
    aux_loglike: function
        auxiliary loglikelihood function. Takes d + 1 parameters (see above).
        The likelihood is the same as loglike, but adds weights.
    """
    assert df > 1, ('Degrees of freedom must exceed 1', df)
    
    assert np.isfinite(usamples).all(), usamples
    assert (usamples < 1).all(), usamples
    assert (usamples > 0).all(), usamples

    nsamples, x_dim = usamples.shape
    assert weights.ndim == 1, ('expected a 1d array for weights, but got:', weights.shape)
    assert len(usamples) == len(weights), ('expected usamples to have the same length as weights', usamples.shape, weights.shape)
    assert np.isfinite(weights).all(), weights
    assert np.all(weights >= 0), weights

    # Transform to a unit gaussian auxiliary space (g-space)
    isamplesg = scipy.stats.norm.ppf(usamples)
    # get parameters of auxiliary transform in this space
    assert np.isfinite(isamplesg).all(), isamplesg

    # remove extremely low weight points, for numerical stability
    mask = weights > 1e-10 * weights.mean()
    print("g samples: ", isamplesg[mask,:])
    print("training samples: ", (transform if vectorized else vectorize(transform))(usamples[mask,:]))
    assert mask.sum() > x_dim + 1, ("too few points with non-negligible weight for dimensionality", mask.sum(), x_dim)
    gctr = np.average(isamplesg[mask,:], weights=weights[mask], axis=0)
    gcov = np.cov(isamplesg[mask, :], aweights=weights[mask], rowvar=0, ddof=0)
    print("  ctr: ", gctr)
    print("  std: ", np.diag(gcov)**0.5)
    assert gctr.shape == (x_dim,), (gctr.shape, x_dim)
    assert gcov.shape == (x_dim, x_dim), (gcov.shape, x_dim)
    assert np.isfinite(gctr).all()
    assert np.isfinite(gcov).all()

    ginvcov = np.linalg.inv(gcov)
    assert np.isfinite(gctr).all()
    assert np.isfinite(ginvcov).all()

    # build transform
    l, v = np.linalg.eigh(ginvcov)
    rotation_matrix = np.dot(v, enlargement_factor * np.diag(1. / np.sqrt(l)))
    sign, rotation_logdet = np.linalg.slogdet(rotation_matrix)
    print("rotation_matrix:", rotation_matrix)
    print("rotation det", sign, rotation_logdet)
    assert np.isfinite(rotation_matrix).all(), (v, enlargement_factor, np.diag(1. / np.sqrt(l)), rotation_matrix)

    rv_auxiliary1d = scipy.stats.t(df)
    
    if vectorized:
        def combine_with_weights(p, w):
            return np.hstack((p, w.reshape((-1,1))))
        sumaxis = 1
    else:
        combine_with_weights = np.append
        sumaxis = None

    def aux_transform(uprime):
        # get uniform gauss/t distributed values in g-space:
        assert (uprime < 1).all(), uprime
        assert (uprime > 0).all(), uprime
        coords = rv_auxiliary1d.ppf(uprime)
        # rotate & stretch; transform into g-space
        g = gctr + np.dot(coords, rotation_matrix)
        assert np.isfinite(g).all(), g
        # since our proposal above is with the auxiliary distribution,
        # instead of the prior, a importance weight to adjust
        # the likelihood is needed
        # this is the ratio of the density of the aux dist
        # to the unit normal gaussian

        # TODO: is coords or uprime or g needed here?
        logweight_aux = rv_auxiliary1d.logpdf(coords).sum(axis=sumaxis)
        logweight_unit = scipy.stats.norm.logpdf(g).sum(axis=sumaxis)
        logweight = logweight_unit - logweight_aux
        
        # transform back to u space
        # print(u, " -> u to g ->", g, "with coords", coords)
        u = scipy.stats.norm.cdf(g)
        assert np.isfinite(u).all(), u
        # avoid borders of the guessing space, where cdf=1 or 0 is hit
        outside = ~np.logical_and(u < 1, u > 0).all(axis=sumaxis)
        np.clip(u, 1e-16, 1 - 1e-16, out=u)
        logweight = np.where(outside, -1e300, logweight)
        if np.any(outside): print("had some points outside", u, g)
        assert (u < 1).all(), u
        assert (u > 0).all(), u
        # transform to p space with user transform
        return combine_with_weights(transform(u), logweight)

    if vectorized:
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
                return loglike(x_actual) + logweight + rotation_logdet
            else:
                print("very low weight", x)
                return -1e300

    return aux_transform, aux_loglikelihood

class BrokenDistribution():
    """
    three-part uniform distribution.

    Flat with 1e-10 of the probability from 0 to lo
    Flat with 1e-10 of the probability from hi to 1
    Flat with 1-2e-10 of the probability from lo to hi
    """
    def __init__(self, lo, hi, edgep):
        self.lo = np.asarray(lo)
        self.hi = np.asarray(hi)
        self.edgep = float(edgep)
        self.logpdf_left = np.log(edgep / self.lo)
        self.logpdf_middle = np.log((1 - 2 * edgep) / (self.hi - self.lo))
        self.logpdf_high = np.log(edgep / (1 - self.hi))

    def logpdf(self, v):
        v = np.asarray(v)
        return np.where(
            v < self.lo, self.logpdf_left, 
            np.where(
                v > self.hi, self.logpdf_high, self.logpdf_middle)
        )

    def ppf(self, c):
        c = np.asarray(c)
        lo = self.lo
        hi = self.hi
        edgep = self.edgep
        return np.where(
            c < edgep,
            c / edgep * lo,
            np.where(
                c > 1 - edgep,
                1 - (1 - c) / edgep * (1 - hi),
                (c - edgep) / (1 - 2 * edgep) * (hi - lo) + lo
            )
        )

    def cdf(self, v):
        v = np.asarray(v)
        lo = self.lo
        hi = self.hi
        edgep = self.edgep
        return np.where(
            v < lo,
            v / lo * self.edgep,
            np.where(
                v > hi,
                1 - (1 - v) / (1 - hi) * edgep,
                (v - lo) / (hi - lo) * (1 - 2 * edgep) + edgep
            )
        )


def get_extended_auxiliary_problem_simple(
    loglike, transform, usamples, suppression_probability=1e-3, vectorized=False
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
    suppression_probability: float
        Probability assigned outside auxiliary uniform distribution
    vectorized: bool
        Whether the likelihood and transform functions are vectorized

    Returns:
    ---------
    aux_transform: function
        auxiliary transform function.
        Takes d u-space coordinates, and returns d + 1 p-space parameters.
        The first d return coordinates are identical to what ``transform`` would return.
        The final coordinate is the correction weight.
    aux_loglike: function
        auxiliary loglikelihood function. Takes d + 1 parameters (see above).
        The likelihood is the same as loglike, but adds weights.
    """
    assert np.isfinite(usamples).all(), usamples
    assert (usamples < 1).all(), usamples
    assert (usamples > 0).all(), usamples

    nsamples, x_dim = usamples.shape

    # Transform to a unit gaussian auxiliary space (g-space)
    ulo = usamples.min(axis=0)
    uhi = usamples.max(axis=0)
    aux_dist = BrokenDistribution(ulo, uhi, suppression_probability)
    print("defining broken dist with:", ulo, uhi)
    
    if vectorized:
        def combine_with_weights(p, w):
            return np.hstack((p, w.reshape((-1,1))))
        sumaxis = 1
    else:
        combine_with_weights = np.append
        sumaxis = None

    def aux_transform(uprime):
        # get uniform distributed values in u-space:
        u = aux_dist.ppf(uprime)
        assert np.isfinite(u).all(), u
        assert (u < 1).all(), u
        assert (u > 0).all(), u
        # since our proposal above is with the auxiliary distribution,
        # instead of the prior, a importance weight to adjust
        # the likelihood is needed
        # this is the ratio of the density of the aux dist
        # to the unit normal gaussian
        logweight = -aux_dist.logpdf(u).sum(axis=sumaxis)
        return combine_with_weights(transform(u), logweight)

    if vectorized:
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
                print("very low weight", x)
                return -1e300

    return aux_transform, aux_loglikelihood


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
