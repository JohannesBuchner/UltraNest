import os
import tempfile
import numpy as np

from ultranest import ReactiveNestedSampler
from ultranest.mlfriends import AffineLayer, ScalingLayer, MLFriends, RobustEllipsoidRegion, SimpleRegion
from ultranest.popstepsampler import PopulationSliceSampler, PopulationRandomWalkSampler
from ultranest.popstepsampler import generate_cube_oriented_direction, generate_random_direction, generate_cube_oriented_direction_scaled
from ultranest.popstepsampler import generate_region_oriented_direction, generate_region_random_direction

def make_region(ndim, us=None, nlive=400):
    if us is None:
        us = np.random.uniform(size=(nlive, ndim))
    
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=1.0)
    return region

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

paramnames = ['param%d' % i for i in range(3)]

def test_stepsampler_cubeslice(plot=False):
    np.random.seed(3)
    nsteps = np.random.randint(10, 50)
    popsize = np.random.randint(1, 20)
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)

    sampler.stepsampler = PopulationSliceSampler(
        popsize=popsize, nsteps=nsteps, 
        generate_direction=generate_cube_oriented_direction,
        log=True,
    )
    r = sampler.run(viz_callback=None, log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

    with tempfile.TemporaryDirectory() as tempdir:
        prefix = os.path.join(tempdir, 'test-stepsampler')
        sampler.stepsampler.plot(prefix + '-plot.pdf')
        assert os.path.exists('test-popstepsampler-plot.pdf')
        sampler.stepsampler.plot_jump_diagnostic_histogram(prefix + '-plot-jumps.pdf')
        assert os.path.exists(prefix + '-plot-jumps.pdf')
        sampler.stepsampler.print_diagnostic()
        print(sampler.stepsampler)
        print(sampler.stepsampler.status)

def test_stepsampler_cubegausswalk(plot=False):
    np.random.seed(2)
    nsteps = np.random.randint(10, 50)
    popsize = np.random.randint(1, 20)
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)

    sampler.stepsampler = PopulationRandomWalkSampler(
        popsize=popsize, nsteps=nsteps, 
        generate_direction=generate_cube_oriented_direction,
        scale=0.1, log=True,
    )
    r = sampler.run(viz_callback=None, log_interval=50, max_iters=200, max_num_improvement_loops=0)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

    with tempfile.TemporaryDirectory() as tempdir:
        prefix = os.path.join(tempdir, 'test-stepsampler')
        sampler.stepsampler.plot(prefix + '-plot.pdf')
        assert os.path.exists('test-popstepsampler-plot.pdf')
        sampler.stepsampler.plot_jump_diagnostic_histogram(prefix + '-plot-jumps.pdf')
        assert os.path.exists(prefix + '-plot-jumps.pdf')
        sampler.stepsampler.print_diagnostic()
        print(sampler.stepsampler)

def test_direction_proposals():
    proposals = [generate_cube_oriented_direction, generate_random_direction, 
        generate_region_oriented_direction, generate_region_random_direction]

    points = np.random.uniform(size=(100, 10))
    minvol = 1.0

    scale = 1. # np.random.uniform()
    for layer in AffineLayer, ScalingLayer:
        transformLayer = layer()
        transformLayer.optimize(points, points)
        for region_class in MLFriends, RobustEllipsoidRegion, SimpleRegion:
            region = region_class(points, transformLayer)
            r, f = region.compute_enlargement(minvol=minvol, nbootstraps=30)
            region.maxradiussq = r
            region.enlarge = f
            region.create_ellipsoid(minvol=minvol)

            for prop in proposals:
                print("test of proposal:", prop, "with region:", region_class, "layer:", layer)
                directions = prop(points, region, scale=scale)
                assert directions.shape == points.shape, (directions.shape, points.shape)
                #assert np.allclose(norms, scale), (norms, scale)


def test_direction_proposal_values():
    ndim = 10
    np.random.seed(12)
    region = make_region(ndim, nlive=400)
    ui = region.u[::2]
    
    scale = np.random.uniform()
    vcube = generate_cube_oriented_direction(ui, region, scale)
    assert vcube.shape == ui.shape
    assert vcube.sum(axis=1).shape == (len(ui),)
    assert ((vcube != 0).sum(axis=1) == 1).all(), vcube
    assert np.allclose(np.linalg.norm(vcube, axis=1), scale), (vcube, np.linalg.norm(vcube, axis=1), scale)

    vharm = generate_random_direction(ui, region, scale)
    assert (vharm != 0).all(), vharm
    vregionslice = generate_region_oriented_direction(ui, region, scale)
    assert (vregionslice != 0).all(), vregionslice
    vregionharm = generate_region_random_direction(ui, region, scale)
    assert (vregionharm != 0).all(), vregionharm
    vcubestd = generate_cube_oriented_direction_scaled(ui, region, scale)
    assert vcubestd.shape == ui.shape
    assert vcubestd.sum(axis=1).shape == (len(ui),)
    assert ((vcubestd != 0).sum(axis=1) == 1).all(), vcubestd


if __name__ == '__main__':
    test_stepsampler_cubegausswalk()
    test_direction_proposals()
