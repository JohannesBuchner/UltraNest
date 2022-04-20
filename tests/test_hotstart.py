from __future__ import print_function, division
import numpy as np
import scipy.stats
import os
from ultranest import ReactiveNestedSampler, HotReactiveNestedSampler
from ultranest.hotstart import reuse_samples, get_extended_auxiliary_problem, get_extended_auxiliary_problem_simple, BrokenDistribution

rng_data = np.random.RandomState(42)
Ndata = 10000
mean_true = -9900.0
sigma_true = 0.1
y = rng_data.normal(mean_true, sigma_true, size=Ndata)

param_names = ['mean', 'scatter']

def prior_transform(x):
    z = np.empty_like(x)
    z[0] = x[0] * 20000 - 10000
    z[1] = x[1] * 8 - 4
    return z

def log_likelihood(params):
    mean, logsigma = params
    sigma = 10**logsigma
    loglikes = scipy.stats.norm.logpdf(y, mean, sigma)
    loglikes[~(loglikes>-1e300)] = -1e300
    return loglikes.sum()

def log_likelihood_without_last(params):
    mean, logsigma = params
    sigma = 10**logsigma
    loglikes = scipy.stats.norm.logpdf(y[:-1], mean, sigma)
    loglikes[~(loglikes>-1e300)] = -1e300
    return loglikes.sum()

log_likelihood_without_last = log_likelihood

def gauss_log_likelihood_vectorized(params):
    return -0.5 * (((params-0.1) / 1e-4)**2).sum(axis=1)

def gauss_log_likelihood(params):
    return -0.5 * (((params-0.1) / 1e-4)**2).sum()

def gauss_prior_transform(cube):
    return cube * 2 - 1


"""
param_names = ['x1', 'x2', 'x3']

prior_transform = gauss_prior_transform
log_likelihood = gauss_log_likelihood
log_likelihood_without_last = log_likelihood
"""

def test_aux_problem():
    np.random.seed(1)
    ndim = 20
    usamples = np.random.multivariate_normal(np.zeros(ndim) + 0.55, np.diag((np.ones(ndim) * 5e-5))**2, size=10000)
    weights = np.ones(len(usamples))

    aux_transform, aux_loglike = get_extended_auxiliary_problem(
        gauss_log_likelihood_vectorized, gauss_prior_transform, usamples, weights,
        enlargement_factor=1.0, df=200, vectorized=True)

    num_test_samples = 310
    u = np.random.uniform(size=(num_test_samples, ndim))
    print("Verifying that the proposal proposes close to maximum likelihood")
    p = aux_transform(u)
    print(p)
    assert p.shape == (num_test_samples, ndim + 1), (p.shape, num_test_samples, ndim+1, u.shape)
    aux_logweight = p[:,-1]
    pmeans = np.mean(p, axis=0)
    pstds = np.std(p, axis=0)
    assert pmeans.shape == (ndim+1,), (pmeans.shape, ndim)
    assert 0.08 < pmeans[0] < 0.12, pmeans
    assert 0.08 < pmeans[-2] < 0.12, pmeans
    assert 8e-5 < pstds[0] < 12e-5, pstds
    assert 8e-5 < pstds[-2] < 12e-5, pstds
    print("Checking weights:")
    print(aux_logweight)
    assert (np.abs(aux_logweight) < 50).all(), aux_logweight

    print("Checking modified likelihood")
    L = aux_loglike(p)
    assert L.shape == (num_test_samples,), (L.shape, num_test_samples)
    print(L)
    assert (L > -100).all(), L

    print("Checking integral")
    importance_Z = np.exp(L).mean()
    print(importance_Z, -0.5 * np.log(2 * np.pi) - ndim * np.log(1e-4))
    assert 0.5 < importance_Z < 2, importance_Z

    print("Checking non-vectorized functionality")
    aux_transform, aux_loglike = get_extended_auxiliary_problem(
        gauss_log_likelihood, gauss_prior_transform, usamples, weights,
        enlargement_factor=1.0, df=200, vectorized=False)
    u = np.random.uniform(size=ndim)
    p = aux_transform(u)
    assert p.shape == (ndim + 1,), (p.shape, ndim+1, u.shape)
    L = aux_loglike(p)
    assert np.shape(L) == (), (np.shape(L))


def test_brokendist():
    dist = BrokenDistribution(0.1, 0.8, 0.01)
    for i in range(100):
        u = np.random.uniform()
        c = dist.ppf(u)
        um = dist.cdf(c)
        np.testing.assert_allclose(um, u)
    
    np.testing.assert_allclose(dist.logpdf(0.05), np.log(0.01/0.1))
    np.testing.assert_allclose(dist.logpdf(0.85), np.log(0.01/0.2))
    np.testing.assert_allclose(dist.logpdf(0.5), np.log(0.98/0.7))

    dist = BrokenDistribution([0.2, 0.2], [0.9, 0.9], 0.01)
    for i in range(100):
        u = np.random.uniform(size=2)
        c = dist.ppf(u)
        um = dist.cdf(c)
        np.testing.assert_allclose(um, u)
    
    np.testing.assert_allclose(dist.logpdf([0.15]*2), np.log(0.01/0.2))
    np.testing.assert_allclose(dist.logpdf([0.95]*2), np.log(0.01/0.1))
    np.testing.assert_allclose(dist.logpdf([0.5]*2), np.log(0.98/0.7))


def test_aux_problem_simple():
    np.random.seed(2)
    ndim = 4
    usamples = np.random.multivariate_normal(np.zeros(ndim) + 0.55, np.diag((np.ones(ndim) * 5e-5))**2, size=10000)

    aux_transform, aux_loglike = get_extended_auxiliary_problem_simple(
        gauss_log_likelihood_vectorized, gauss_prior_transform, usamples,
        vectorized=True)

    num_test_samples = 3100
    u = np.random.uniform(size=(num_test_samples, ndim))
    print("Verifying that the proposal proposes close to maximum likelihood")
    p = aux_transform(u)
    print(p)
    assert p.shape == (num_test_samples, ndim + 1), (p.shape, num_test_samples, ndim+1, u.shape)
    aux_logweight = p[:,-1]
    pmeans = np.mean(p, axis=0)
    pstds = np.std(p, axis=0)
    assert pmeans.shape == (ndim+1,), (pmeans.shape, ndim)
    assert 0.08 < pmeans[0] < 0.12, pmeans
    assert 0.08 < pmeans[-2] < 0.12, pmeans
    assert 1e-2 < pstds[0] < 4e-2, pstds
    assert 1e-2 < pstds[-2] < 4e-2, pstds
    print("Checking weights:")
    print(aux_logweight)
    assert (np.abs(aux_logweight) < 200).all(), aux_logweight

    print("Checking modified likelihood")
    L = aux_loglike(p)
    assert L.shape == (num_test_samples,), (L.shape, num_test_samples)
    print(L.min(), L.max(), L)
    assert (L > -1000).mean() > 0.99, L

    print("Checking integral")
    importance_lnZ = np.log(np.exp(L).mean()) - 0.5 * np.log(2 * np.pi * 1e-4**2) * ndim
    print(importance_lnZ)
    assert -ndim < importance_lnZ < ndim, importance_lnZ

    print("Checking non-vectorized functionality")
    aux_transform, aux_loglike = get_extended_auxiliary_problem_simple(
        gauss_log_likelihood, gauss_prior_transform, usamples,
        vectorized=False)
    u = np.random.uniform(size=ndim)
    p = aux_transform(u)
    assert p.shape == (ndim + 1,), (p.shape, ndim+1, u.shape)
    L = aux_loglike(p)
    assert np.shape(L) == (), (np.shape(L))


def test_reuse_SLOW():
    np.random.seed(1)
    ref_sampler = ReactiveNestedSampler(
        param_names, log_likelihood, transform=prior_transform,
    )
    ref_results = ref_sampler.run(frac_remain=0.5, viz_callback=None)
    ref_sampler.print_results()

    print("RECYCLING:")
    print("ref:", ref_results)
    rec_results = reuse_samples(param_names, log_likelihood, **ref_results['weighted_samples'], **ref_results)
    #assert rec_results['ncall'] < ref_results['ncall'] / 4, (ref_results['ncall'], rec_results['ncall'])
    assert np.abs(ref_results['posterior']['mean'][0] - rec_results['posterior']['mean'][0]) < 0.5, (ref_results['posterior'], rec_results['posterior'])
    assert np.abs(ref_results['posterior']['mean'][1] - rec_results['posterior']['mean'][1]) < 0.05, (ref_results['posterior'], rec_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][0] / rec_results['posterior']['stdev'][0]) < 1.2, (ref_results['posterior'], rec_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][1] / rec_results['posterior']['stdev'][1]) < 1.2, (ref_results['posterior'], rec_results['posterior'])
    assert np.abs(ref_results['logzerr'] - rec_results['logzerr']) < 0.5, (ref_results['logzerr'], rec_results['logzerr'])
    print("rec:", rec_results)
    del rec_results
    logls = np.array([log_likelihood(s) for s in ref_results['samples']])
    rec_results2 = reuse_samples(param_names, log_likelihood, points=ref_results['samples'], logl=logls)
    print("rec2:", rec_results2)
    assert rec_results2['ncall'] == len(logls), (ref_results['ncall'], rec_results2['ncall'])
    assert np.abs(ref_results['posterior']['mean'][0] - rec_results2['posterior']['mean'][0]) < 0.5, (ref_results['posterior'], rec_results2['posterior'])
    assert np.abs(ref_results['posterior']['mean'][1] - rec_results2['posterior']['mean'][1]) < 0.05, (ref_results['posterior'], rec_results2['posterior'])
    assert 0.5 < (ref_results['posterior']['stdev'][0] / rec_results2['posterior']['stdev'][0]) < 1.5, (ref_results['posterior'], rec_results2['posterior'])
    assert 0.5 < (ref_results['posterior']['stdev'][1] / rec_results2['posterior']['stdev'][1]) < 1.5, (ref_results['posterior'], rec_results2['posterior'])


def test_hotstart_SLOW():
    seed = 10
    np.random.seed(seed)
    run_args = dict(frac_remain=0.5, viz_callback=None, max_num_improvement_loops=0)
    
    if os.path.exists('test-hotstart-ref/chains/weighted_post_untransformed.txt'):
        os.unlink('test-hotstart-ref/chains/weighted_post_untransformed.txt')
    if os.path.exists('test-hotstart-ref/results/points.hdf5'):
        os.unlink('test-hotstart-ref/results/points.hdf5')
    print()
    print("HOTSTART: PARTIAL REFERENCE RUN ==========")
    print()
    ref_sampler = ReactiveNestedSampler(
        param_names, log_likelihood, transform=prior_transform,
        log_dir='test-hotstart-ref', resume='overwrite',
    )
    # emulate a keyboard-interrupt or similar:
    ref_partial_results = ref_sampler.run(max_iters=7000, **run_args)
    ref_sampler.print_results()
    if os.path.exists('test-hotstart-ref/chains/weighted_post_untransformed.txt'):
        os.unlink('test-hotstart-ref/chains/weighted_post_untransformed.txt')
    assert os.path.exists('test-hotstart-ref/results/points.hdf5')
    print(ref_partial_results['ncall'])

    print()
    print("HOTSTART: PARTIAL RESUME ==========")
    print()
    np.random.seed(seed+1)
    hot_sampler = HotReactiveNestedSampler(
        log_dir_old='test-hotstart-ref', log_dir_new='test-hotstart-new',
        param_names=param_names, loglike=log_likelihood_without_last, transform=prior_transform,
        resume='overwrite', backoff_factor=10.0)
    p = hot_sampler.transform(np.random.uniform(size=(2000, len(param_names))))
    print("proposal guesses:", p)
    print("  means :", p.mean(axis=0))
    print("  stdevs:", p.std(axis=0))
    livepoints_resume_results = hot_sampler.run(**run_args)
    hot_sampler.print_results()
    print(livepoints_resume_results['ncall'])
    
    print()
    print("HOTSTART: COMPLETE REFERENCE RUN ==========")
    print()
    np.random.seed(seed+2)
    ref_results = ref_sampler.run(**run_args)
    ref_sampler.print_results()
    assert os.path.exists('test-hotstart-ref/chains/weighted_post_untransformed.txt')
    assert os.path.exists('test-hotstart-ref/results/points.hdf5')
    print(ref_results['ncall'])

    np.random.seed(seed+3)
    print()
    print("HOTSTART: COMPLETE RESUME ==========")
    print()
    hot_sampler = HotReactiveNestedSampler(
        log_dir_old='test-hotstart-ref', log_dir_new='test-hotstart-new',
        param_names=param_names, loglike=log_likelihood_without_last, transform=prior_transform,
        resume='overwrite')
    similar_resume_results = hot_sampler.run(**run_args)
    hot_sampler.print_results()
    
    print(similar_resume_results['ncall'])

"""
    assert aux_results['ncall'] < ref_results['ncall'] / 4, (ref_results['ncall'], aux_results['ncall'])
    assert np.abs(ref_results['posterior']['mean'][0] - aux_results['posterior']['mean'][0]) < 0.5, (ref_results['posterior'], aux_results['posterior'])
    assert np.abs(ref_results['posterior']['mean'][1] - aux_results['posterior']['mean'][1]) < 0.05, (ref_results['posterior'], aux_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][0] / aux_results['posterior']['stdev'][0]) < 1.2, (ref_results['posterior'], aux_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][1] / aux_results['posterior']['stdev'][1]) < 1.2, (ref_results['posterior'], aux_results['posterior'])
    assert np.abs(ref_results['logzerr'] - aux_results['logzerr']) < 0.5, (ref_results['logzerr'], aux_results['logzerr'])
"""

if __name__ == '__main__':
    test_hotstart_SLOW()
