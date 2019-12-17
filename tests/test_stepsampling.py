import numpy as np
from ultranest.mlfriends import ScalingLayer, AffineLayer, MLFriends
from ultranest import ReactiveNestedSampler
from ultranest.stepsampler import RegionMHSampler, CubeMHSampler, CubeSliceSampler, RegionSliceSampler
from ultranest.pathsampler import SamplingPathStepSampler
from numpy.testing import assert_allclose

#here = os.path.dirname(__file__)

def loglike(z):
    a = np.array([-0.5 * sum([((xi - 0.7 + i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    b = np.array([-0.5 * sum([((xi - 0.3 - i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    return np.logaddexp(a, b)

def transform(x):
    return x # * 10. - 5.

paramnames = ['param%d' % i for i in range(3)]
#paramnames = ['param%d' % i for i in range(40)]

def test_stepsampler_cubemh(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, vectorized=True)
    sampler.stepsampler = CubeMHSampler(nsteps=4 * len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1, a.sum()
    assert b.sum() > 1, b.sum()

def test_stepsampler_regionmh(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, vectorized=True)
    sampler.stepsampler = RegionMHSampler(nsteps=4 * len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1, a
    assert b.sum() > 1, b

def test_stepsampler_cubeslice(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, vectorized=True)
    sampler.stepsampler = CubeSliceSampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def test_stepsampler_regionslice(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, vectorized=True)
    sampler.stepsampler = RegionSliceSampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def make_region(ndim, us=None):
    if us is None:
        us = np.random.uniform(size=(1000, ndim))
    
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=1.0)
    return region


def test_stepsampler(plot=False):
    np.random.seed(1)
    region = make_region(len(paramnames))
    Ls = loglike(region.u)
    
    stepsampler = CubeMHSampler(nsteps=len(paramnames))
    while True:
        u1, p1, L1, nc = stepsampler.__next__(region, -1e100, region.u, Ls, transform, loglike)
        if u1 is not None:
            break
    assert L1 > -1e100
    print(u1, L1)
    while True:
        u2, p2, L2, nc = stepsampler.__next__(region, -1e100, region.u, Ls, transform, loglike)
        if u2 is not None:
            break
    assert L2 > -1e100
    print(u2, L2)
    assert np.all(u1 != u2)
    assert np.all(L1 != L2)

def test_stepsampler_adapt_when_stuck(plot=False):
    # check that a stuck sampler can free itself
    np.random.seed(1)
    us = np.random.normal(0.7, 0.001, size=(1000, len(paramnames)))
    region = make_region(len(paramnames), us=us)
    Ls = loglike(us)
    Lmin = Ls.min()

    print('CubeMHSampler')
    stepsampler = CubeMHSampler(nsteps=1, region_filter=True)
    np.random.seed(23)
    old_scale = stepsampler.scale
    for i in range(1000):
        if i > 100:
            assert False, i
        unew, pnew, Lnew, nc = stepsampler.__next__(region, Lmin, us, Ls, transform, loglike, ndraw=10)
        if unew is not None:
            break
    
    new_scale = stepsampler.scale
    assert new_scale != old_scale
    assert new_scale < 0.01, new_scale
    
    print('CubeSliceSampler')
    stepsampler = CubeSliceSampler(nsteps=1, region_filter=True)
    np.random.seed(23)
    old_scale = stepsampler.scale
    for j in range(100):
        for i in range(1000):
            if i > 100:
                assert False, i
            unew, pnew, Lnew, nc = stepsampler.__next__(region, Lmin, us, Ls, transform, loglike, ndraw=10)
            if unew is not None:
                break
    
    new_scale = stepsampler.scale
    assert new_scale != old_scale
    assert new_scale < 0.01, new_scale

def test_stepsampler_regionmh_adapt(plot=False):
    np.random.seed(1)
    region = make_region(len(paramnames))
    Ls = loglike(region.u)
    try:
        RegionMHSampler(nsteps=len(paramnames), adaptive_nsteps='Hello')
        assert False, 'expected error'
    except ValueError:
        pass
    
    for sampler_class in RegionMHSampler, CubeMHSampler, CubeSliceSampler, RegionSliceSampler: 
        for adaptation in False, 'move-distance', 'proposal-total-distances', 'proposal-summed-distances':
            print()
            stepsampler = sampler_class(nsteps=len(paramnames), adaptive_nsteps=adaptation)
            print(stepsampler)
            stepsampler.region_changed(Ls, region)
            np.random.seed(23)
            old_scale = stepsampler.scale
            for i in range(5):
                while True:
                    unew, pnew, Lnew, nc = stepsampler.__next__(region, -1e100, region.u, Ls, transform, loglike)
                    if unew is not None:
                        break
            new_scale = stepsampler.scale
            assert new_scale != old_scale
            
            if adaptation:
                assert stepsampler.nsteps != len(paramnames)
            else:
                assert stepsampler.nsteps == len(paramnames)

def test_pathsampler():
    stepper = SamplingPathStepSampler(nresets=1, nsteps=4, log=True)
    #stepper.scale = 0.01
    origscale = stepper.scale
    Lmin = -1.0
    us = 0.5 + np.zeros((100, 2))
    Ls = np.zeros(100)
    region = make_region(2)
    def transform(x): return x
    def loglike(x): return 0*x[:,0]
    def gradient(x):
        j = np.argmax(np.abs(x - 0.5))
        v = np.zeros(len(x))
        v[j] = -1 if x[j] > 0.5 else 1
        return v
    
    def nocall(x):
        assert False
    
    stepper.generate_direction = lambda ui, region, scale: np.array([0.01, 0.01])
    stepper.set_gradient(nocall)
    assert stepper.iresets == 1
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert (stepper.naccepts, stepper.nrejects) == (1, 0), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert (stepper.naccepts, stepper.nrejects) == (2, 0), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert (stepper.naccepts, stepper.nrejects) == (3, 0), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert_allclose(x, [0.54, 0.54])
    # check that path was reset
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert origscale < stepper.scale, (origscale, stepper.scale)
    origscale = stepper.scale

    print()
    print("make reflect")
    print()
    stepper.set_gradient(gradient)
    def loglike(x): return np.where(x[:,0] < 0.505, 0.0, -100)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert (stepper.naccepts, stepper.nrejects) == (1, 0), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert (stepper.naccepts, stepper.nrejects) == (2, 0), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert (stepper.naccepts, stepper.nrejects) == (3, 0), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert_allclose(x, [0.47, 0.55])
    assert origscale < stepper.scale, (origscale, stepper.scale)
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)

    print()
    print("make stuck")
    print()
    # make stuck
    origscale = stepper.scale
    def loglike(x): return -100 + 0*x[:,0]
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert stepper.nstuck == 0
    assert (stepper.naccepts, stepper.nrejects) == (0, 1), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert (stepper.naccepts, stepper.nrejects) == (0, 2), (stepper.naccepts, stepper.nrejects)
    assert stepper.nstuck == 0
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert x is None, x
    assert stepper.nstuck == 1
    assert (stepper.naccepts, stepper.nrejects) == (0, 3), (stepper.naccepts, stepper.nrejects)
    x, v, L, nc = stepper.__next__(region, Lmin, us, Ls, transform, loglike)
    assert_allclose(x, [0.50, 0.50])
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert origscale > stepper.scale, (origscale, stepper.scale, "should shrink scale")

if __name__ == '__main__':
    #test_stepsampler_cubemh(plot=True)
    #test_stepsampler_regionmh(plot=False)
    #test_stepsampler_de(plot=False)
    #test_stepsampler_cubeslice(plot=True)
    #test_stepsampler_regionslice(plot=True)
    pass
