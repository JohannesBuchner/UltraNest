import numpy as np
import os
import matplotlib.pyplot as plt
from mininest.mlfriends import ScalingLayer, AffineLayer, MLFriends

here = os.path.dirname(__file__)

def test_region_sampling_affine(plot=False):
    np.random.seed(1)
    points = np.random.uniform(size=(1000, 2))
    points[:,1] *= 0.5
    
    transformLayer = AffineLayer(wrapped_dims=[])
    transformLayer.optimize(points, points)
    region = MLFriends(points, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    print("enlargement factor:", region.enlarge, 1 / region.enlarge)
    region.create_ellipsoid()
    nclusters = transformLayer.nclusters
    assert nclusters == 1
    if plot:
        plt.plot(points[:,0], points[:,1], 'x ')
        for method in region.sampling_methods:
            points, nc = method(nsamples=400)
            plt.plot(points[:,0], points[:,1], 'o ', label=str(method.__name__))
        plt.legend(loc='best')
        plt.savefig('test_regionsampling.pdf', bbox_inches='tight')
        plt.close()

    for method in region.sampling_methods:
        points, nc = method(nsamples=4000)
        lo1, lo2 = points.min(axis=0)
        hi1, hi2 = points.max(axis=0)
        assert 0 <= lo1 < 0.1, (method.__name__, points, lo1, hi1, lo2, hi2)
        assert 0 <= lo2 < 0.1, (method.__name__, points, lo1, hi1, lo2, hi2)
        assert 0.95 < hi1 <= 1, (method.__name__, points, lo1, hi1, lo2, hi2)
        assert 0.45 <= hi2 < 0.55, (method.__name__, points, lo1, hi1, lo2, hi2)


def test_region_sampling_scaling(plot=False):
    np.random.seed(1)
    points = np.random.uniform(0.2, 0.5, size=(1000, 2))
    points[:,1] *= 0.1
    
    transformLayer = ScalingLayer(wrapped_dims=[])
    transformLayer.optimize(points, points)
    region = MLFriends(points, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    print("enlargement factor:", region.enlarge, 1 / region.enlarge)
    region.create_ellipsoid()
    nclusters = transformLayer.nclusters
    assert nclusters == 1
    if plot:
        plt.plot(points[:,0], points[:,1], 'x ')
        for method in region.sampling_methods:
            points, nc = method(nsamples=400)
            plt.plot(points[:,0], points[:,1], 'o ', label=str(method.__name__))
        plt.legend(loc='best')
        plt.savefig('test_regionsampling_affine.pdf', bbox_inches='tight')
        plt.close()

    for method in region.sampling_methods:
        points, nc = method(nsamples=4000)
        lo1, lo2 = points.min(axis=0)
        hi1, hi2 = points.max(axis=0)
        assert 0.15 < lo1 < 0.25, (method.__name__, points, lo1, hi1, lo2, hi2)
        assert 0.015 < lo2 < 0.025, (method.__name__, points, lo1, hi1, lo2, hi2)
        assert 0.45 < hi1 < 0.55, (method.__name__, points, lo1, hi1, lo2, hi2)
        assert 0.045 < hi2 < 0.055, (method.__name__, points, lo1, hi1, lo2, hi2)


if __name__ == '__main__':
    test_region_sampling_affine(plot=True)
    test_region_sampling_scaling(plot=True)
