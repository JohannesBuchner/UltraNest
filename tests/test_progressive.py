import numpy as np
import shutil
import tempfile
from ultranest.progressive import ProgressiveNestedSampler, ReactiveNestedSampler

def loglike_vectorized(z):
    return -0.5 * (z**2).sum(axis=1)
def transform(x):
    return 10. * x - 5.
def loglike(z):
    return -0.5 * (z**2).sum()
paramnames = ['Hinz', 'Kunz', 'Franz']
lnZ_expected = -0.5 * 1**2 * np.pi * len(paramnames)

def test_progressive_run():
    #np.random.seed(1)
    sampler = ProgressiveNestedSampler(paramnames, loglike, transform=transform)
    r = sampler.run()
    assert r['samples'].shape[1] == len(paramnames)
    sampler.plot()
    sampler.print_results()
    assert abs(lnZ_expected - sampler.results['logz']) < 1
def test_progressive_run_warm():
    #np.random.seed(1)
    sampler = ProgressiveNestedSampler(paramnames, loglike, transform=transform)
    sampler.warmup()
    r = sampler.run()
    assert r['samples'].shape[1] == len(paramnames)
    sampler.plot()
    sampler.print_results()
    assert abs(lnZ_expected - sampler.results['logz']) < 1
def test_progressive_run_vectorized():
    #np.random.seed(1)
    sampler = ProgressiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)
    r = sampler.run()
    assert r['samples'].shape[1] == len(paramnames)
    sampler.print_results()
    assert abs(lnZ_expected - sampler.results['logz']) < 1
def test_progressive_run_vectorized_warm():
    np.random.seed(1)
    sampler = ProgressiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)
    sampler.warmup()
    r = sampler.run()
    assert r['samples'].shape[1] == len(paramnames)
    sampler.print_results()
    assert abs(lnZ_expected - sampler.results['logz']) < 1
def test_progressive_run_resume():
    np.random.seed(1)
    folder = tempfile.mkdtemp()
    try:
        print("running warmup only...")
        sampler = ProgressiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True, log_dir=folder)
        sampler.warmup()
        sampler.sampler.pointstore.close()
        print("running one iteration...")
        sampler = ProgressiveNestedSampler(paramnames, loglike, transform=transform, log_dir=folder)
        sampler.warmup()
        sampler.run(max_iter=1)
        sampler.print_results()
        sampler.sampler.pointstore.close()
        print("running rest ...")
        sampler = ProgressiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True, log_dir=folder)
        sampler.warmup()
        sampler.run()
        sampler.sampler.pointstore.close()
        sampler.print_results()
        assert abs(lnZ_expected - sampler.results['logz']) < 1
    finally:
        shutil.rmtree(folder, ignore_errors=True)

if __name__ == '__main__':
    test_progressive_run_vectorized_warm()
    #test_progressive_run_resume()
    #test_progressive_run()
    #test_progressive_run_warm()
    #test_progressive_run_vectorized()
