from __future__ import print_function, division
import numpy as np
import scipy.stats
from numpy import log10
from ultranest import ReactiveNestedSampler
from ultranest.hotstart import reuse_samples, get_extended_auxiliary_problem

rng_data = np.random.RandomState(42)
Ndata = 100
mean_true = 42.0
sigma_true = 0.1
y = rng_data.normal(mean_true, sigma_true, size=Ndata)

parameters = ['mean', 'scatter']

def prior_transform(x):
    z = np.empty_like(x)
    z[0] = x[0] * 2000 - 1000
    z[1] = 10**(x[1] * 4 - 2)
    return z

def log_likelihood(params):
    mean, sigma = params
    return scipy.stats.norm.logpdf(y, mean, sigma).sum()

def test_hotstart_SLOW():
    np.random.seed(2)
    ctr = np.array([(42.0 + 1000) / 2000, (log10(0.1) + 2) / 4])
    cov = np.diag([0.01 / 2000, (log10(0.1) + 2) / 4 - (log10(0.09) + 2) / 4])**2
    invcov = np.linalg.inv(cov)

    Lguess = log_likelihood(prior_transform(np.random.uniform(size=len(parameters))))
    Lctr = log_likelihood(prior_transform(ctr))
    print(Lguess, Lctr)
    assert Lguess < Lctr - 100, (Lguess, Lctr)
    
    aux_log_likelihood, aux_transform = get_extended_auxiliary_problem(
            log_likelihood, prior_transform, ctr, invcov, 
            enlargement_factor=len(parameters)**0.5, df=20)
    
    proposals = np.array([aux_transform(np.random.uniform(size=len(parameters))) for i in range(40)])
    valid = proposals[:,2] > -1e100
    assert valid.sum() > 0.9, valid.sum()
    proposals = proposals[valid,:]
    print("proposals:", proposals, valid.sum())
    assert (np.abs(proposals[:,0] - 42) < 2).mean() > 0.9, proposals
    assert (np.abs(log10(proposals[:,1] / 0.1)) < 0.5).mean() > 0.9, proposals
    Lproposed = np.array([log_likelihood(p[:-1]) for p in proposals])
    assert np.mean(Lproposed > Lctr - 10) > 0.5, (Lproposed, Lctr)

    aux_sampler = ReactiveNestedSampler(
        parameters, aux_log_likelihood, transform=aux_transform,
        derived_param_names=['aux_logweight'],
    )
    aux_results = aux_sampler.run(frac_remain=0.5, viz_callback=None)
    aux_sampler.print_results()

    ref_sampler = ReactiveNestedSampler(
        parameters, log_likelihood, transform=prior_transform,
    )
    ref_results = ref_sampler.run(frac_remain=0.5, viz_callback=None)
    ref_sampler.print_results()
    
    assert aux_results['ncall'] < ref_results['ncall'] / 4, (ref_results['ncall'], aux_results['ncall'])
    assert np.abs(ref_results['posterior']['mean'][0] - aux_results['posterior']['mean'][0]) < 0.5, (ref_results['posterior'], aux_results['posterior'])
    assert np.abs(ref_results['posterior']['mean'][1] - aux_results['posterior']['mean'][1]) < 0.05, (ref_results['posterior'], aux_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][0] / aux_results['posterior']['stdev'][0]) < 1.2, (ref_results['posterior'], aux_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][1] / aux_results['posterior']['stdev'][1]) < 1.2, (ref_results['posterior'], aux_results['posterior'])
    assert np.abs(ref_results['logzerr'] - aux_results['logzerr']) < 0.5, (ref_results['logzerr'], aux_results['logzerr'])

    print("RECYCLING:")
    print("ref:", ref_results)
    rec_results = reuse_samples(parameters, log_likelihood, **ref_results['weighted_samples'], **ref_results)
    #assert rec_results['ncall'] < ref_results['ncall'] / 4, (ref_results['ncall'], rec_results['ncall'])
    assert np.abs(ref_results['posterior']['mean'][0] - rec_results['posterior']['mean'][0]) < 0.5, (ref_results['posterior'], rec_results['posterior'])
    assert np.abs(ref_results['posterior']['mean'][1] - rec_results['posterior']['mean'][1]) < 0.05, (ref_results['posterior'], rec_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][0] / rec_results['posterior']['stdev'][0]) < 1.2, (ref_results['posterior'], rec_results['posterior'])
    assert 0.8 < (ref_results['posterior']['stdev'][1] / rec_results['posterior']['stdev'][1]) < 1.2, (ref_results['posterior'], rec_results['posterior'])
    assert np.abs(ref_results['logzerr'] - rec_results['logzerr']) < 0.5, (ref_results['logzerr'], rec_results['logzerr'])
    print("rec:", rec_results)
    del rec_results
    logls = np.array([log_likelihood(s) for s in ref_results['samples']])
    rec_results2 = reuse_samples(parameters, log_likelihood, points=ref_results['samples'], logl=logls)
    print("rec2:", rec_results2)
    assert rec_results2['ncall'] == len(logls), (ref_results['ncall'], rec_results2['ncall'])
    assert np.abs(ref_results['posterior']['mean'][0] - rec_results2['posterior']['mean'][0]) < 0.5, (ref_results['posterior'], rec_results2['posterior'])
    assert np.abs(ref_results['posterior']['mean'][1] - rec_results2['posterior']['mean'][1]) < 0.05, (ref_results['posterior'], rec_results2['posterior'])
    assert 0.5 < (ref_results['posterior']['stdev'][0] / rec_results2['posterior']['stdev'][0]) < 1.5, (ref_results['posterior'], rec_results2['posterior'])
    assert 0.5 < (ref_results['posterior']['stdev'][1] / rec_results2['posterior']['stdev'][1]) < 1.5, (ref_results['posterior'], rec_results2['posterior'])

if __name__ == '__main__':
    test_hotstart()
