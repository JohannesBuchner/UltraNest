import numpy as np
from ultranest.mlfriends import ScalingLayer, AffineLayer, MLFriends
from ultranest import ReactiveNestedSampler
from ultranest.stepsampler import RegionMHSampler, CubeMHSampler, CubeSliceSampler, RegionSliceSampler, AHARMSampler
from ultranest.stepsampler import generate_region_random_direction, ellipsoid_bracket, crop_bracket_at_unit_cube
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

def assert_point_touches_ellipsoid(ucurrent, v, t, ellipsoid_center, ellipsoid_invcov, enlarge):
    unext = ucurrent + v * t
    d = unext - ellipsoid_center
    r = np.einsum('j,jk,k->', d, ellipsoid_invcov, d)
    assert np.isclose(r, enlarge), (ucurrent, t, r, enlarge)

def test_ellipsoid_bracket(plot=False):
    for seed in range(20):
        print("seed:", seed)
        np.random.seed(seed)
        if seed % 2 == 0:
            us = np.random.normal(size=(2**np.random.randint(3, 10), 2))
            us /= ((us**2).sum(axis=1)**0.5).reshape((-1, 1))
            us = us * 0.1 + 0.5
        else:
            us = np.random.uniform(size=(2**np.random.randint(3, 10), 2))
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(us[:,0], us[:,1], 'o ', ms=2)
        
        transformLayer = ScalingLayer()
        region = MLFriends(us, transformLayer)
        try:
            region.maxradiussq, region.enlarge = region.compute_enlargement()
            region.create_ellipsoid()
        except ValueError:
            continue

        print(region.ellipsoid_center)
        print(region.enlarge)
        print(region.ellipsoid_cov)
        print(region.ellipsoid_invcov)
        print(region.ellipsoid_axes)
        print(region.ellipsoid_inv_axes)

        ucurrent = np.array([2**0.5*0.1/2+0.5, 2**0.5*0.1/2+0.5])
        ucurrent = np.array([0.4, 0.525])
        v = np.array([1., 0])
        if plot: plt.plot(ucurrent[0], ucurrent[1], 'o')
        print("from", ucurrent, "in direction", v)
        left, right = ellipsoid_bracket(ucurrent, v, region.ellipsoid_center, region.ellipsoid_inv_axes, region.enlarge)
        uleft = ucurrent + v * left
        uright = ucurrent + v * right

        if plot: 
            plt.plot([uleft[0], uright[0]], [uleft[1], uright[1]], 'x-')
            
            plt.savefig('test_ellipsoid_bracket.pdf', bbox_inches='tight')
            plt.close()
        print("ellipsoid bracket:", left, right)
        assert left <= 0, left
        assert right >= 0, right
        
        assert_point_touches_ellipsoid(ucurrent, v, left, region.ellipsoid_center, region.ellipsoid_invcov, region.enlarge)
        assert_point_touches_ellipsoid(ucurrent, v, right, region.ellipsoid_center, region.ellipsoid_invcov, region.enlarge)

def test_crop_bracket(plot=False):
    ucurrent = np.array([0.39676747, 0.53881673])
    v = np.array([-0.79619985, -0.60503372])
    ellipsoid_center = np.array([0.23556461, 0.49899689])
    ellipsoid_inv_axes = np.array([[-3.28755896,  0.70136518], [ 1.33333377,  1.72933397]])
    enlarge = 26.66439694551674
    ellipsoid_invcov = np.array([[11.29995701, -3.17051875], [-3.17051875,  4.76837493]])
    #enlarge = 1.0
    #ellipsoid_inv_axes = np.array([[1.0, 0.], [0., 1]])
    

    eleft, eright = ellipsoid_bracket(ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)
    if plot:
        us = np.random.uniform(-2, +2, size=(10000, 2))
        d = us - ellipsoid_center
        r = np.einsum('ij,jk,ik->i', d, ellipsoid_invcov, d)
        mask_inside = r <= enlarge
        
        import matplotlib.pyplot as plt
        plt.plot(us[mask_inside,0], us[mask_inside,1], '+', ms=2)
        plt.plot(ucurrent[0], ucurrent[1], 'o ', ms=2)
        plt.plot([ucurrent[0] + eleft * v[0], ucurrent[0] + eright * v[0]],
            [ucurrent[1] + eleft * v[1], ucurrent[1] + eright * v[1]],
            '-s', ms=8)
        plt.savefig('test_crop_bracket.pdf', bbox_inches='tight')
    print("left:", eleft, ucurrent + v * eleft)
    assert eleft <= 0, eleft
    assert_point_touches_ellipsoid(ucurrent, v, eleft, ellipsoid_center, ellipsoid_invcov, enlarge)
    print("right:", eright, ucurrent + v * eright)
    assert eright >= 0, eright
    assert_point_touches_ellipsoid(ucurrent, v, eright, ellipsoid_center, ellipsoid_invcov, enlarge)

    left, right, cropleft, cropright = crop_bracket_at_unit_cube(ucurrent, v, eleft, eright)
    if plot:
        plt.plot([ucurrent[0] + left * v[0], ucurrent[0] + right * v[0]],
            [ucurrent[1] + left * v[1], ucurrent[1] + right * v[1]],
            's--', ms=8)
        plt.savefig('test_crop_bracket.pdf', bbox_inches='tight')
        plt.close()
    assert cropleft
    assert cropright
    assert (ucurrent + v * left <= 1).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)
    assert (ucurrent + v * right <= 1).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)
    assert (ucurrent + v * left >= 0).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)
    assert (ucurrent + v * right >= 0).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)

def test_aharm_sampler():
    def loglike(theta):
        return -0.5 * (((theta - 0.5)/0.01)**2).sum(axis=1)
    def transform(x):
        return x

    seed = 1
    Nlive = 10
    np.random.seed(seed)
    us = np.random.uniform(size=(Nlive, 2))
    Ls = loglike(us)
    Lmin = Ls.min()
    transformLayer = ScalingLayer()
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement()
    region.create_ellipsoid()
    assert region.inside(us).all()
    nsteps = 10
    sampler = AHARMSampler(nsteps=nsteps, region_filter=True)

    nfunccalls = 0
    ncalls = 0
    while True:
        u, p, L, nc = sampler.__next__(region, Lmin, us, Ls, transform, loglike)
        nfunccalls += 1
        ncalls += nc
        if u is not None:
            break
        if nfunccalls > 100 + nsteps:
            assert False, ('infinite loop?', seed, nsteps, Nlive)
    print("done in %d function calls, %d likelihood evals" % (nfunccalls, ncalls))
    

def run_aharm_sampler():
    for seed in [733] + list(range(10)):
        print()
        print("SEED=%d" % seed)
        print()
        np.random.seed(seed)
        nsteps = max(1, int(10**np.random.uniform(0, 3)))
        Nlive = int(10**np.random.uniform(1.5, 3))
        print("Nlive=%d nsteps=%d" % (Nlive, nsteps))
        sampler = AHARMSampler(nsteps, adaptive_nsteps=False, region_filter=False)
        us = np.random.uniform(0.6, 0.8, size=(4000, 2))
        Ls = loglike(us)
        i = np.argsort(Ls)[-Nlive:]
        us = us[i,:]
        Ls = Ls[i]
        Lmin = Ls.min()
        
        transformLayer = ScalingLayer()
        region = MLFriends(us, transformLayer)
        region.maxradiussq, region.enlarge = region.compute_enlargement()
        region.create_ellipsoid()
        nfunccalls = 0
        ncalls = 0
        while True:
            u, p, L, nc = sampler.__next__(region, Lmin, us, Ls, transform, loglike)
            nfunccalls += 1
            ncalls += nc
            if u is not None:
                break
            if nfunccalls > 100 + nsteps:
                assert False, ('infinite loop?', seed, nsteps, Nlive)
        print("done in %d function calls, %d likelihood evals" % (nfunccalls, ncalls))


if __name__ == '__main__':
    #test_stepsampler_cubemh(plot=True)
    #test_stepsampler_regionmh(plot=False)
    #test_stepsampler_de(plot=False)
    #test_stepsampler_cubeslice(plot=True)
    #test_stepsampler_regionslice(plot=True)
    run_aharm_sampler()
    #test_ellipsoid_bracket()
    #test_crop_bracket(plot=True)
