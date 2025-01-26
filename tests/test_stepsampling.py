import numpy as np
import os
import pytest
import tempfile

from ultranest.mlfriends import ScalingLayer, AffineLayer, MLFriends
from ultranest import ReactiveNestedSampler
from ultranest.stepsampler import RegionMHSampler, CubeMHSampler, SliceSampler, CubeSliceSampler, RegionSliceSampler, SpeedVariableRegionSliceSampler, RegionBallSliceSampler, SpeedVariableGenerator
from ultranest.stepsampler import ellipsoid_bracket, crop_bracket_at_unit_cube, _inside_region
from ultranest.stepsampler import generate_random_direction, generate_cube_oriented_direction
from ultranest.stepsampler import SequentialDirectionGenerator, OrthogonalDirectionGenerator, SequentialRegionDirectionGenerator
from ultranest.stepsampler import generate_region_random_direction, generate_region_oriented_direction, generate_cube_oriented_differential_direction
from ultranest.stepsampler import generate_differential_direction, generate_partial_differential_direction, generate_mixture_random_direction

from ultranest.pathsampler import SamplingPathStepSampler
from ultranest.stepsampler import select_random_livepoint, IslandPopulationRandomLivepointSelector
from numpy.testing import assert_allclose

#here = os.path.dirname(__file__)

def loglike_vectorized(z):
    a = np.array([-0.5 * sum([((xi - 0.7 + i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    b = np.array([-0.5 * sum([((xi - 0.3 - i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    return np.logaddexp(a, b)

def loglike(x):
    a = -0.5 * sum([((xi - 0.7 + i*0.001)/0.1)**2 for i, xi in enumerate(x)])
    b = -0.5 * sum([((xi - 0.3 - i*0.001)/0.1)**2 for i, xi in enumerate(x)])
    return np.logaddexp(a, b)

def transform(x):
    return x # * 10. - 5.

def transform1(x):
    return x**2

paramnames = ['param%d' % i for i in range(3)]
#paramnames = ['param%d' % i for i in range(40)]

def test_stepsampler_cubemh(plot=False):
    np.random.seed(1)
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform1, vectorized=True)
    sampler.stepsampler = CubeMHSampler(nsteps=4 * len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1, a.sum()
    assert b.sum() > 1, b.sum()

    # check that diagnostics fail
    print("mean jump distance:", sampler.stepsampler.mean_jump_distance)
    print("far enough fraction:", sampler.stepsampler.far_enough_fraction)
    assert sampler.stepsampler.mean_jump_distance < 1.0, sampler.stepsampler.mean_jump_distance
    assert sampler.stepsampler.far_enough_fraction < 0.5, sampler.stepsampler.far_enough_fraction

    print("Diagnostic print:")
    sampler.stepsampler.print_diagnostic()

def test_stepsampler_regionmh(plot=False):
    np.random.seed(2)
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)
    sampler.stepsampler = RegionMHSampler(nsteps=4 * len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1, a
    assert b.sum() > 1, b

def test_direction_proposals():
    ndim = 10
    np.random.seed(12)
    region = make_region(ndim)
    ui = region.u[0]

    scale = np.random.uniform()
    vcube = generate_cube_oriented_direction(ui, region, scale)
    assert (vcube != 0).sum() == 1, vcube
    assert np.linalg.norm(vcube) == scale, vcube

    vcubede = generate_cube_oriented_differential_direction(ui, region, scale)
    assert (vcubede != 0).sum() == 1, vcubede
    assert np.linalg.norm(vcubede) > 0, vcubede

    vharm = generate_random_direction(ui, region, scale)
    assert (vharm != 0).all(), vharm

    vde = generate_differential_direction(ui, region, scale)
    assert (vde != 0).all(), vde

    vregionslice = generate_region_oriented_direction(ui, region, scale)
    assert (vregionslice != 0).all(), vregionslice

    vmix = generate_mixture_random_direction(ui, region, scale)
    assert (vmix != 0).all(), vmix

    vregionharm = generate_region_random_direction(ui, region, scale)
    assert (vregionharm != 0).all(), vregionharm

    direction_generator = SequentialDirectionGenerator()
    for i in range(ndim * 2):
        vdir = direction_generator(ui, region, scale)
        assert (vdir != 0).sum() == 1, vdir
        assert np.abs(vdir[i % ndim]) > 0, vdir

    region_direction_generator = SequentialRegionDirectionGenerator()
    for i in range(ndim * 2):
        vdirharm = region_direction_generator(ui, region, scale)
        assert (vdirharm != 0).all(), vdirharm

    vpartialde = generate_partial_differential_direction(ui, region, scale)
    assert (vpartialde != 0).sum() > 1, vpartialde
    assert (vpartialde != 0).sum() < ndim, vpartialde

    # test that applying OrthogonalDirectionGenerator to SequentialDirectionGenerator has no effect
    ortho_direction_generator = OrthogonalDirectionGenerator(SequentialDirectionGenerator())
    for i in range(ndim * 2):
        vdir = ortho_direction_generator(ui, region, scale)
        assert (vdir != 0).sum() == 1, vdir
        assert np.abs(vdir[i % ndim]) > 0, vdir

def test_inside_region():
    ndim = 10
    np.random.seed(12)
    region = make_region(ndim, us = np.random.uniform(0.5, 0.51, size=(400, ndim)))
    i = np.random.randint(400)
    ui = region.u[i]
    assert _inside_region(region, ui, ui)
    # corner case where a new point is close to a old case, but both are somehow outside the region
    unew, uold = np.random.uniform(0.4, 0.401, size=(2, ndim))
    assert _inside_region(region, unew, uold)

def test_stepsampler_cubeslice(plot=False):
    np.random.seed(3)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform1)
    sampler.stepsampler = CubeSliceSampler(nsteps=len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def test_stepsampler_regionslice(plot=False):
    np.random.seed(4)
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform)
    sampler.stepsampler = RegionSliceSampler(nsteps=2 + len(paramnames))
    r = sampler.run(log_interval=50, min_num_live_points=400)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

    # check that diagnostics pass
    print("mean jump distance:", sampler.stepsampler.mean_jump_distance)
    print("far enough fraction:", sampler.stepsampler.far_enough_fraction)
    assert sampler.stepsampler.mean_jump_distance > 1.0, sampler.stepsampler.mean_jump_distance
    assert sampler.stepsampler.far_enough_fraction > 0.6, sampler.stepsampler.far_enough_fraction

    print("Diagnostic print:")
    sampler.stepsampler.print_diagnostic()

def test_SpeedVariableGenerator():
    np.random.seed(4)
    ndims = [3, 10]
    matrices = [
        np.array([[True, True, True], [False, True, True], [False, False, True]]),
        [Ellipsis, slice(1,None), slice(2,4)]
    ]
    for matrix, ndim in zip(matrices, ndims):
        direction_generator = SpeedVariableGenerator(matrix, generate_direction=generate_random_direction)
        for i in range(10):
            u0 = np.random.uniform(size=ndim)
            for mask_varying in matrix:
                mask = np.zeros(ndim, dtype=bool)
                mask[mask_varying] = True
                print("starting at u0", u0)
                print("varying:", mask_varying, mask)
                v = direction_generator(u0, None)
                print("direction:", v)
                assert_allclose(v[~mask], 0)
                u1 = u0 + np.random.uniform() * v
                print("new point:", u1)
                assert_allclose(u1[~mask], u0[~mask])


def test_stepsampler_variable_speed_SLOW(plot=False):
    matrices = [
        np.array([[True, True, True], [False, True, True], [False, False, True]]),
        [Ellipsis, slice(1,None), slice(2,4)]
    ]
    for i, matrix in enumerate(matrices):
        np.random.seed(42 + i)
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform)
        sampler.stepsampler = SpeedVariableRegionSliceSampler(matrix)
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
    np.random.seed(6)
    region = make_region(len(paramnames))
    Ls = loglike_vectorized(region.u)

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
    np.random.seed(7)
    us = np.random.normal(0.7, 0.0001, size=(1000, len(paramnames)))
    region = make_region(len(paramnames), us=us)
    Ls = loglike_vectorized(us)
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
    assert new_scale < 0.01, (new_scale, unew)

    print('CubeSliceSampler')
    stepsampler = SliceSampler(nsteps=1, region_filter=True, generate_direction=generate_cube_oriented_direction)
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
    assert new_scale < 0.01, (new_scale, unew)

def test_stepsampler_adapt(plot=True):
    np.random.seed(8)
    region = make_region(len(paramnames))
    Ls = loglike_vectorized(region.u)
    with pytest.raises(ValueError):
        RegionMHSampler(nsteps=len(paramnames), adaptive_nsteps='Hello')

    with tempfile.TemporaryDirectory() as tempdir:
        for sampler_class in RegionMHSampler, CubeMHSampler, CubeSliceSampler, RegionSliceSampler:
            for adaptation in False, 'move-distance', 'move-distance-midway', 'proposal-total-distances', 'proposal-summed-distances':
                print()
                if sampler_class in (CubeMHSampler, CubeSliceSampler):
                    logfilename = os.path.join(tempdir, 'test-stepsampler-%s.log' % adaptation)
                    log = open(logfilename, 'w')
                else:
                    logfilename = None
                    log = False
                stepsampler = sampler_class(nsteps=len(paramnames), adaptive_nsteps=adaptation, log=log)
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

                if logfilename:
                    print(np.loadtxt(logfilename).shape)
                    log_nentries, log_ncolumns = np.loadtxt(logfilename).shape
                    assert log_nentries == 5
                    assert log_ncolumns == (1 + 4 * len(unew) + 7)

                if adaptation == 'move-distance' and sampler_class == RegionSliceSampler and plot:
                    # test plotting
                    prefix = os.path.join(tempdir, 'test-stepsampler')
                    assert not os.path.exists(prefix + '-plot.pdf')
                    stepsampler.plot(prefix + '-plot.pdf')
                    assert os.path.exists(prefix + '-plot.pdf')

                    assert not os.path.exists(prefix + '-plot-jumps.pdf')
                    stepsampler.plot_jump_diagnostic_histogram(prefix + '-plot-jumps.pdf')
                    assert os.path.exists(prefix + '-plot-jumps.pdf')

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

    left, right, cropleft, cropright = crop_bracket_at_unit_cube(ucurrent, -v, eleft, eright)
    if plot:
        plt.plot([ucurrent[0] - left * v[0], ucurrent[0] - right * v[0]],
            [ucurrent[1] - left * v[1], ucurrent[1] - right * v[1]],
            's--', ms=8)
        plt.savefig('test_crop_bracket_negative.pdf', bbox_inches='tight')
        plt.close()
    assert cropleft
    assert cropright
    assert (ucurrent - v * left <= 1).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)
    assert (ucurrent - v * right <= 1).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)
    assert (ucurrent - v * left >= 0).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)
    assert (ucurrent - v * right >= 0).all(), (ucurrent, v, ellipsoid_center, ellipsoid_inv_axes, enlarge)

def test_random_point_selector():
    np.random.seed(41)
    K = 10
    ndim = 2
    i1 = np.random.randint(0, K)
    i2 = np.random.randint(0, K)
    i3 = np.random.randint(0, K)
    us = np.random.normal(size=(K, ndim))
    Ls = np.random.normal(size=K)
    Lmin = Ls.min()
    np.random.seed(41)
    j1 = select_random_livepoint(us, Ls, Lmin)
    j2 = select_random_livepoint(us, Ls, Lmin)
    j3 = select_random_livepoint(us, Ls, Lmin)
    assert i1 == j1, (i1, j1)
    assert i2 == j2, (i2, j2)
    assert i3 == j3, (i3, j3)


def test_island_point_selector():
    K = 10
    ndim = 2
    self_selector = IslandPopulationRandomLivepointSelector(1)
    selector = IslandPopulationRandomLivepointSelector(5)
    imbalanced_selector = IslandPopulationRandomLivepointSelector(9)
    for i in range(100):
        us = np.random.normal(size=(K, ndim))
        Ls = np.random.normal(size=K)
        Lmin = Ls.min()
        j1 = np.argmin(Ls)
        j2 = selector(us, Ls, Lmin)
        assert j1 == self_selector(us, Ls, Lmin)
        if j1 >= 5:
            assert j2 >= 5, (j1, j2)
        if j1 < 5:
            assert j2 < 5, (j1, j2)
        if j1 == 9:
            assert j1 == imbalanced_selector(us, Ls, Lmin)
        else:
            assert imbalanced_selector(us, Ls, Lmin) < 9

    np.random.seed(421)
    leaked = False
    selector = IslandPopulationRandomLivepointSelector(5, 0.1)
    for i in range(100):
        j1 = np.argmin(Ls)
        j2 = selector(us, Ls, Lmin)
        if j1 >= 5 and j2 >= 5 or j1 < 5 and j2 < 5:
            pass
        else:
            # leak, as expected
            leaked = True
            break
    assert leaked


if __name__ == '__main__':
    #test_stepsampler_cubemh(plot=True)
    #test_stepsampler_regionmh(plot=False)
    #test_stepsampler_de(plot=False)
    #test_stepsampler_cubeslice(plot=True)
    #test_stepsampler_regionslice(plot=True)
    test_ellipsoid_bracket()
    #test_crop_bracket(plot=True)
