import numpy as np
import pytest
from ultranest import ReactiveNestedSampler
from ultranest.snowball import snowball


def make_simple_sampler():
    paramnames = ['x', 'y']

    def loglike(z):
        a = -0.5 * (((z - 0.5) / 0.1) ** 2).sum(axis=1)
        return a

    def transform(x):
        return x

    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, vectorized=True)
    return sampler


def test_snowball_returns_correct_keys():
    np.random.seed(42)
    Kmax = 50
    Kmin = 20
    sampler = make_simple_sampler()
    result = snowball(sampler, Kmin=Kmin, Kmax=Kmax, Kfactor=1.5)
    assert 'K' in result
    assert 'logz' in result
    assert 'logzerr' in result
    assert len(result['K']) == len(result['logz'])
    assert len(result['K']) == len(result['logzerr'])
    K = result['K']
    assert len(K) >= 2
    for i in range(1, len(K)):
        assert K[i] > K[i - 1], (K[i], K[i - 1])
    assert result['K'][0] == Kmin
    for k in result['K']:
        assert k < Kmax, k
    for lz in result['logz']:
        assert np.isfinite(lz), lz
    for lze in result['logzerr']:
        assert lze >= 0, lze
    assert sampler.stepsampler is not None
    logzs = result['logz']
    logzerrs = result['logzerr']
    for i in range(len(logzs) - 1):
        diff = abs(logzs[i] - logzs[i + 1])
        combined_err = logzerrs[i] + logzerrs[i + 1]
        assert diff < 5 * combined_err + 2.0, (diff, combined_err, logzs)
