import numpy as np
from ultranest import ReactiveNestedSampler
from ultranest.popstepsampler import PopulationSliceSampler, generate_cube_oriented_direction, \
    generate_random_direction, generate_region_oriented_direction, generate_region_random_direction

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
    )
    r = sampler.run(viz_callback=None, log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

from ultranest.mlfriends import update_clusters, AffineLayer, ScalingLayer, MLFriends, RobustEllipsoidRegion, SimpleRegion

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
                norms = np.linalg.norm(directions, axis=1)
                #print(norms[0], directions[0])
                assert directions.shape == points.shape, (directions.shape, points.shape)
                #assert np.allclose(norms, scale), (norms, scale)

if __name__ == '__main__':
    test_direction_proposals()
