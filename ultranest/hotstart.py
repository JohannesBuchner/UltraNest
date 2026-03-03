# noqa: D400 D205
"""
Warm start
----------

Helper functions for deforming the parameter space to enable
a more efficient sampling.

Based on ideas from Petrosyan & Handley (2022, https://arxiv.org/abs/2212.01760).

"""

import numpy as np

from .utils import resample_equal, vectorize

from scipy.interpolate import PchipInterpolator, make_interp_spline
from scipy.integrate import trapezoid
from getdist import MCSamples


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

    Returns
    ---------
    aux_loglike: function
        auxiliary loglikelihood function.
    aux_aftertransform: function
        auxiliary transform function.
        Takes d u-space coordinates, and returns d + 1 p-space parameters.
        The first d return coordinates are identical to what ``transform`` would return.
        The final coordinate is the correction weight.
    """
    import scipy.stats
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
        loglike_total = rv_auxiliary1d.logpdf(coords).sum()
        return loglike(transform(x)) - loglike_total

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

    Returns
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
    import scipy.stats
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


def get_extended_auxiliary_independent_problem(loglike, transform, ctr, err, df=1):
    """Return a new loglike and transform based on an auxiliary distribution.

    Given a likelihood and prior transform, and information about
    the (expected) posterior peak, generates a auxiliary
    likelihood and prior transform that is identical but
    requires fewer nested sampling iterations.

    This is achieved by deforming the prior space, and undoing that
    transformation by correction weights in the likelihood.

    The auxiliary distribution used for transformation/weighting is
    a independent Student-t distribution for each parameter.

    Usage::

        aux_loglikelihood, aux_transform = get_auxiliary_problem(loglike, transform, ctr, invcov, enlargement_factor, df=1)
        aux_sampler = ReactiveNestedSampler(parameters, aux_loglikelihood, transform=aux_transform, derived_param_names=['logweight'])
        aux_results = aux_sampler.run()
        posterior_samples = aux_results['samples'][:,-1]

    Parameters
    ------------
    loglike: function
        original likelihood function
    transform: function
        original prior transform function
    ctr: array
        Posterior center (in u-space).
    err: array
        Standard deviation around the posterior center (in u-space).
    df: float
        Number of degrees of freedom of the auxiliary student-t distribution.
        The default is recommended. For truly gaussian posteriors,
        the student-t can be made more gaussian (by df>=30) for accelation.

    Returns
    ---------
    aux_loglike: function
        auxiliary loglikelihood function.
    aux_transform: function
        auxiliary transform function.
        Takes d u-space coordinates, and returns d + 1 p-space parameters.
        The first d return coordinates are identical to what ``transform`` would return.
        The final coordinate is the log of the correction weight.
    """
    import scipy.stats
    ndim, = np.shape(ctr)
    assert np.shape(err) == (ndim,)
    assert df >= 1, ('Degrees of freedom must be above 1', df)

    rv_aux = scipy.stats.t(df, ctr, err)
    # handle the case where the aux distribution extends beyond the unit cube
    aux_lo = rv_aux.cdf(0)
    aux_hi = rv_aux.cdf(1)
    aux_w = aux_hi - aux_lo
    weight_ref = rv_aux.logpdf(ctr).sum()

    def aux_transform(u):
        # get uniform gauss/t distributed values:
        x = rv_aux.ppf(u * aux_w + aux_lo)
        weight = -rv_aux.logpdf(x).sum() + weight_ref
        return np.append(transform(x), weight)

    def aux_loglikelihood(x):
        x_actual = x[:-1]
        weight = x[-1]
        if -1e100 < weight < 1e100:
            return loglike(x_actual) + weight - weight_ref
        else:
            return -1e300

    return aux_loglikelihood, aux_transform


def compute_quantile_intervals(steps, upoints, uweights):
    """Compute lower and upper axis quantiles.

    Parameters
    ------------
    steps: array
        list of quantiles q to compute.
    upoints: array
        samples, with dimensions (N, d)
    uweights: array
        sample weights

    Returns
    ---------
    ulo: array
        list of lower quantiles (at q), one entry for each dimension d.
    uhi: array
        list of upper quantiles (at 1-q), one entry for each dimension d.
    """
    ndim = upoints.shape[1]
    nboxes = len(steps)
    ulos = np.empty((nboxes + 1, ndim))
    uhis = np.empty((nboxes + 1, ndim))
    for j, pthresh in enumerate(steps):
        for i, ui in enumerate(upoints.transpose()):
            order = np.argsort(ui)
            c = np.cumsum(uweights[order])
            usel = ui[order][np.logical_and(c >= pthresh, c <= 1 - pthresh)]
            ulos[j,i] = usel.min()
            uhis[j,i] = usel.max()
    ulos[-1] = 0
    uhis[-1] = 1
    return ulos, uhis


def compute_quantile_intervals_refined(steps, upoints, uweights, logsteps_max=20):
    """Compute lower and upper axis quantiles.

    Parameters
    ------------
    steps: array
        list of quantiles q to compute, with dimensions
    upoints: array
        samples, with dimensions (N, d)
    uweights: array
        sample weights. N entries.
    logsteps_max: int
        number of intermediate steps to inject between largest quantiles interval and full unit cube

    Returns
    ---------
    ulo: array
        list of lower quantiles (at `q`), of shape (M, d), one entry per quantile and dimension d.
    uhi: array
        list of upper quantiles (at 1-`q`), of shape (M, d), one entry per quantile and dimension d.
    uinterpspace: array
        list of steps (length of `steps` plus `logsteps_max` long)
    """
    nboxes = len(steps)
    ulos_orig, uhis_orig = compute_quantile_intervals(steps, upoints, uweights)
    assert len(ulos_orig) == nboxes + 1
    assert len(uhis_orig) == nboxes + 1

    smallest_axis_width = np.min(uhis_orig[-2,:] - ulos_orig[-2,:])
    logsteps = min(logsteps_max, int(np.ceil(-np.log10(max(1e-100, smallest_axis_width)))))

    weights = np.logspace(-logsteps, 0, logsteps + 1).reshape((-1, 1))
    # print("logspace:", weights, logsteps)
    assert len(weights) == logsteps + 1, (weights.shape, logsteps)
    # print("quantiles:", ulos_orig, uhis_orig)
    ulos_new = ulos_orig[nboxes - 1, :].reshape((1, -1)) * (1 - weights) + 0 * weights
    uhis_new = uhis_orig[nboxes - 1, :].reshape((1, -1)) * (1 - weights) + 1 * weights

    # print("additional quantiles:", ulos_new, uhis_new)

    ulos = np.vstack((ulos_orig[:-1,:], ulos_new))
    uhis = np.vstack((uhis_orig[:-1,:], uhis_new))
    # print("combined quantiles:", ulos, uhis)
    assert (ulos[-1,:] == 0).all()
    assert (uhis[-1,:] == 1).all()

    uinterpspace = np.ones(nboxes + logsteps + 1)
    uinterpspace[:nboxes + 1] = np.linspace(0, 1, nboxes + 1)
    assert 0 < uinterpspace[nboxes - 1] < 1, uinterpspace[nboxes]
    uinterpspace[nboxes:] = np.linspace(uinterpspace[nboxes - 1], 1, logsteps + 2)[1:]

    return ulos, uhis, uinterpspace

def conservative_prior_guess(all_x, all_p, single_dim_fraction):
    """ Given a function p:[0, 1] -> R+, 
    represented by some samples (`all_p`) corresponding
    to the points `all_x`, return another normalized function
    which is the sum of p and a constant, weighed by `single_dim_fraction`
    and `(1-single_dim_fraction)` respectively.
    """
    
    integral = trapezoid(all_p, x=all_x)
    
    return all_p * single_dim_fraction + np.ones_like(all_p) * (1 - single_dim_fraction) * integral

def get_aux_splines(upoints, uweights, volume_contribution=0.5, beta=1.):
    """ Compute spline representations of the auxiliary transforms.
    
    upoints: shape (n_points, dim)
    uweights: shape (n_points,)
    volume_contribution: fraction of total prior mass which will 
        be taken up by the "guess" distribution represented by upoints
        (default: 0.5)
    beta: temperature for the prior guess (default: 1)
    
    """

    
    n_points, dim = upoints.shape
    assert uweights.shape == (n_points, )
    single_dim_fraction = volume_contribution**(1/dim)
    
    mcsamples = MCSamples(
        samples=upoints, 
        weights=uweights, 
        names=[str(i) for i in range(upoints.shape[1])], 
        ranges={str(i):[0, 1] for i in range(upoints.shape[1])}
    )
    
    densities, inverse_cumulatives = [], []
    
    for i in range(dim):
        density_interior = mcsamples.get1DDensity(str(i))
        
        interior_x = density_interior.x
        
        all_x = np.sort(np.unique(np.concatenate((
            np.linspace(0, 1, num=1024),
            interior_x
        ))))
        
        all_p = np.maximum(density_interior(all_x), 0)
        
        all_p_thresholded = conservative_prior_guess(all_x, all_p**beta, single_dim_fraction)
        
        interpolator = PchipInterpolator(all_x, all_p_thresholded)
        cumulative = interpolator.antiderivative()
        integral = cumulative(1) - cumulative(0)
        cumulative_points = cumulative(all_x) / integral
        
        unique_cumulative, unique_idx = np.unique(cumulative_points, return_index=True)
        
        inverse_cumulative = make_interp_spline(unique_cumulative, all_x[unique_idx], k=1)
        inverse_cumulatives.append(inverse_cumulative)

        cumulative = make_interp_spline(all_x, cumulative_points, k=1)
        densities.append(cumulative.derivative())

    return densities, inverse_cumulatives


def get_auxiliary_contbox_parameterization(
    param_names, loglike, transform, upoints, uweights=None, vectorized=False,
    volume_contribution=0.5, beta=1.,
):
    """Return a new loglike and transform based on an auxiliary distribution.

    Given a likelihood and prior transform, and information about
    the (expected) posterior peak, generates a auxiliary
    likelihood and prior transform that is identical but
    requires fewer nested sampling iterations.

    This is achieved by deforming the prior space, and undoing that
    transformation by correction weights in the likelihood.
    A additional parameter, "aux_logweight", is added at the end,
    which contains the correction weight.
    If this parameter is negative for most of the sampling, 
    it means that the samples are being taken from the "expanded" region, 
    therefore the hotstart is working as expected.
    If it is positive, the auxiliary prior guess was likely mis-specified;
    the sampler may still give correct results, but it will take 
    as long as a regular nested sampling run (or longer).

    Parameters
    ------------
    param_names: list
        parameter names
    loglike: function
        original likelihood function
    transform: function
        original prior transform function
    upoints: array
        Posterior samples (in u-space).
    uweights: array
        Weights of samples (needs to sum of 1)
    vectorized: bool
        whether the loglike & transform functions are vectorized
    volume_contribution: float
        fraction of total prior mass which will 
        be taken up by the "guess" distribution (default: 0.5), 
        should be between 0 and 1
    beta: float
        inverse temperature for the prior guess (default: 1).
        The guess will be rescaled as prior**beta: higher beta 
        will make the prior guess more informative and peaked,
        lower beta will make it flatter and more conservative.

    Returns
    ---------
    aux_param_names: list
        new parameter names (`param_names`) plus additional 'aux_logweight'
    aux_loglike: function
        auxiliary loglikelihood function.
    aux_transform: function
        auxiliary transform function.
        Takes d u-space coordinates, and returns d + 1 p-space parameters.
        The first d return coordinates are identical to what ``transform`` would return.
        The final coordinate is the log of the correction weight.
    vectorized: bool
        whether the returned functions are vectorized

    Usage
    ------
    ::

        aux_loglikelihood, aux_transform = get_auxiliary_contbox_parameterization(
            loglike, transform, auxiliary_usamples)
        aux_sampler = ReactiveNestedSampler(parameters, aux_loglikelihood, transform=aux_transform, derived_param_names=['logweight'])
        aux_results = aux_sampler.run()
        posterior_samples = aux_results['samples'][:,-1]

    """
    upoints = np.asarray(upoints)
    assert upoints.ndim == 2, ('expected 2d array for upoints, got shape: %s' % upoints.shape)
    mask = np.logical_and(upoints > 0, upoints < 1).all(axis=1)
    assert np.all(mask), (
        'upoints must be between 0 and 1, have:', upoints[~mask,:])

    nsamples, ndim = upoints.shape

    if uweights is None:
        uweights = np.ones(nsamples) / nsamples
    assert uweights.shape == (nsamples,)

    aux_param_names = param_names + ['aux_logweight']

    densities, inv_cumulatives = get_aux_splines(upoints, uweights, volume_contribution, beta)

    def aux_transform(u):
        ndim2, = u.shape
        assert ndim2 == ndim + 1
        umod = np.empty(ndim)
        log_aux_volume_factors = 0
        for i in range(ndim):
            umod[i] = inv_cumulatives[i](u[i])
            log_aux_volume_factors -= np.log(densities[i](umod[i]))
        return np.append(transform(umod), log_aux_volume_factors)

    def aux_transform_vectorized(u):
        nsamples, ndim2 = u.shape
        assert ndim2 == ndim + 1
        umod = np.empty((nsamples, ndim2 - 1))
        log_aux_volume_factors = np.zeros((nsamples, 1))
        for i in range(ndim):
            umod[:, i] = inv_cumulatives[i](u[:,i])
            log_aux_volume_factors[:,0] -= np.log(densities[i](umod[:, i]))
        return np.hstack((transform(umod), log_aux_volume_factors))

    def aux_loglikelihood(x):
        x_actual = x[:-1]
        logl = loglike(x_actual)
        aux_logweight = x[-1]
        # downweight if we are in the auxiliary distribution
        return logl + aux_logweight

    def aux_loglikelihood_vectorized(x):
        x_actual = x[:,:-1]
        logl = loglike(x_actual)
        aux_logweight = x[:,-1]
        # downweight if we are in the auxiliary distribution
        return logl + aux_logweight

    if vectorized:
        return aux_param_names, aux_loglikelihood_vectorized, aux_transform_vectorized, vectorized
    else:
        return aux_param_names, aux_loglikelihood, aux_transform, vectorized


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

    Returns
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
