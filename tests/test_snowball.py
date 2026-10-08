import numpy as np
import pytest
import os
import tempfile
import shutil
from ultranest.snowball import SnowballingNestedSampler, _make_plot


def make_snowballing_sampler(log_dir=None):
    paramnames = ['x']

    def loglike(z):
        return -0.5 * (((z - 0.5) / 0.5) ** 2).sum(axis=1)

    def transform(x):
        return x

    sampler = SnowballingNestedSampler(
        paramnames,
        loglike,
        transform=transform,
        vectorized=True,
        log_dir=log_dir,
    )
    return sampler


def test_make_plot():
    folder = tempfile.mkdtemp()
    try:
        K_arr = [20, 30, 45]
        logz_arr = [-1.0, -1.1, -1.05]
        logzerr_arr = [0.1, 0.08, 0.07]
        _make_plot(folder, K_arr, logz_arr, logzerr_arr)
        plot_path = os.path.join(folder, 'snowballing.pdf')
        assert os.path.exists(plot_path), "Plot file should exist after _make_plot call"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_make_plot_none_dir():
    K_arr = [20, 30]
    logz_arr = [-1.0, -1.1]
    logzerr_arr = [0.1, 0.08]
    _make_plot(None, K_arr, logz_arr, logzerr_arr)


def test_snowball_returns():
    np.random.seed(42)
    Kmax = 50
    Kmin = 20
    folder = tempfile.mkdtemp()
    try:
        sampler = make_snowballing_sampler(log_dir=folder)
        result = sampler.run(Kmin=Kmin, Kmax=Kmax, Kfactor=1.5)
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
        assert sampler.sampler.stepsampler is not None
        logzs = result['logz']
        logzerrs = result['logzerr']
        for i in range(len(logzs) - 1):
            diff = abs(logzs[i] - logzs[i + 1])
            combined_err = logzerrs[i] + logzerrs[i + 1]
            assert diff < 5 * combined_err + 2.0, (diff, combined_err, logzs)

        plot_path = os.path.join(folder, 'plots', 'snowballing.pdf')
        assert os.path.exists(plot_path), "Plot file should exist after run with log_dir"

        assert sampler.history is result
        assert sampler.results is not None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_snowballing_nested_sampler_stepsampler_configured():
    folder = tempfile.mkdtemp()
    try:
        sampler = make_snowballing_sampler(log_dir=folder)
        assert sampler.sampler.stepsampler is not None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_snowballing_nested_sampler_custom_nsteps():
    folder = tempfile.mkdtemp()
    try:
        paramnames = ['x', 'y']

        def loglike(z):
            return -0.5 * (((z - 0.5) / 0.5) ** 2).sum(axis=1)

        def transform(x):
            return x

        sampler = SnowballingNestedSampler(
            paramnames,
            loglike,
            transform=transform,
            vectorized=True,
            log_dir=folder,
            nsteps=5,
        )
        assert sampler.sampler.stepsampler is not None
        assert sampler.sampler.stepsampler.nsteps == 5
    finally:
        shutil.rmtree(folder, ignore_errors=True)
