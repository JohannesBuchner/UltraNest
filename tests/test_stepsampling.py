import numpy as np
import os
import matplotlib.pyplot as plt
from mininest.mlfriends import ScalingLayer, AffineLayer, MLFriends
from mininest import ReactiveNestedSampler
from mininest.stepsampler import DESampler, RegionMHSampler, CubeMHSampler, CubeSliceSampler, RegionSliceSampler

#here = os.path.dirname(__file__)

def loglike(z):
    a = np.array([-0.5 * sum([((xi - 0.7 + i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    b = np.array([-0.5 * sum([((xi - 0.3 - i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    return np.logaddexp(a, b)

def transform(x):
    return x # * 10. - 5.

paramnames = ['param%d' % i for i in range(6)]
#paramnames = ['param%d' % i for i in range(40)]

def test_stepsampler_cubemh(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, min_num_live_points=400)
    sampler.stepsampler = CubeMHSampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def test_stepsampler_regionmh(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, min_num_live_points=400)
    sampler.stepsampler = RegionMHSampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def test_stepsampler_de(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, min_num_live_points=400)
    sampler.stepsampler = DESampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50)
    sampler.print_results()
    #sampler.plot()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def test_stepsampler_cubeslice(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, min_num_live_points=400)
    sampler.stepsampler = CubeSliceSampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def test_stepsampler_regionslice(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, min_num_live_points=400)
    sampler.stepsampler = RegionSliceSampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

if __name__ == '__main__':
    #test_stepsampler_cubemh(plot=True)
    #test_stepsampler_regionmh(plot=True)
    #test_stepsampler_de(plot=True)
    #test_stepsampler_cubeslice(plot=True)
    test_stepsampler_regionslice(plot=True)
