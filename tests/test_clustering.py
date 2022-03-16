from __future__ import print_function, division
import numpy as np
import os
import matplotlib.pyplot as plt
from ultranest.utils import create_logger
from ultranest import ReactiveNestedSampler
from ultranest.mlfriends import MLFriends, AffineLayer

here = os.path.dirname(__file__)


def test_clustering():
    from ultranest.mlfriends import update_clusters
    for i in range(5):
        np.random.seed(i * 100)

        points = np.random.uniform(size=(100,2))

        nclusters, clusteridxs, overlapped_points = update_clusters(points, points, 0.1**2)

        for i in np.unique(clusteridxs):
            x, y = points[clusteridxs == i].transpose()
            plt.scatter(x, y)
        plt.savefig('testclustering_0p1.pdf', bbox_inches='tight')
        plt.close()
        assert 1 < nclusters < 30

        nclusters, clusteridxs, overlapped_points = update_clusters(points, points, 0.2**2)

        for i in np.unique(clusteridxs):
            x, y = points[clusteridxs == i].transpose()
            plt.scatter(x, y)
        plt.savefig('testclustering_0p2.pdf', bbox_inches='tight')
        plt.close()
        assert 1 <= nclusters < 2


def test_clusteringcase():
    from ultranest.mlfriends import update_clusters
    here = os.path.dirname(__file__)
    points = np.loadtxt(os.path.join(here, "clusters2.txt"))
    maxr = np.loadtxt(os.path.join(here, "clusters2_radius.txt"))
    # transformLayer = ScalingLayer()
    # transformLayer.optimize(points)
    # region = MLFriends(points, transformLayer)
    # maxr = region.compute_maxradiussq(nbootstraps=30)
    print('maxradius:', maxr)
    nclusters, clusteridxs, overlapped_points = update_clusters(points, points, maxr)

    plt.title('nclusters: %d' % nclusters)
    for i in np.unique(clusteridxs):
        x, y = points[clusteridxs == i].transpose()
        plt.scatter(x, y)
    plt.savefig('testclustering_2.pdf', bbox_inches='tight')
    plt.close()


def test_clusteringcase_eggbox():
    from ultranest.mlfriends import update_clusters, ScalingLayer, MLFriends
    points = np.loadtxt(os.path.join(here, "eggboxregion.txt"))
    transformLayer = ScalingLayer()
    transformLayer.optimize(points, points)
    region = MLFriends(points, transformLayer)
    maxr = region.compute_maxradiussq(nbootstraps=30)
    assert 1e-10 < maxr < 5e-10
    print('maxradius:', maxr)
    nclusters, clusteridxs, overlapped_points = update_clusters(points, points, maxr)
    # plt.title('nclusters: %d' % nclusters)
    # for i in np.unique(clusteridxs):
    #    x, y = points[clusteridxs == i].transpose()
    #    plt.scatter(x, y)
    # plt.savefig('testclustering_eggbox.pdf', bbox_inches='tight')
    # plt.close()
    assert 14 < nclusters < 20, nclusters


class MockIntegrator(ReactiveNestedSampler):
    def __init__(self):
        self.use_mpi = False
        self.mpi_size = 1
        self.mpi_rank = 0
        self.region = None
        self.transformLayer = None
        self.wrapped_axes = []
        self.log = True
        self.logger = create_logger("mock")
        self.region_class = MLFriends
        self.transform_layer_class = AffineLayer


def test_overclustering_eggbox_txt():
    from ultranest.mlfriends import update_clusters, ScalingLayer, MLFriends
    np.random.seed(1)
    for i in [20, 23, 24, 27, 49]:
        print()
        print("==== TEST CASE %d =====================" % i)
        print()
        points = np.loadtxt(os.path.join(here, "overclustered_u_%d.txt" % i))

        for k in range(3):
            transformLayer = ScalingLayer(wrapped_dims=[])
            transformLayer.optimize(points, points)
            region = MLFriends(points, transformLayer)
            maxr = region.compute_maxradiussq(nbootstraps=30)
            region.maxradiussq = maxr
            nclusters = transformLayer.nclusters

            print("manual: r=%e nc=%d" % (region.maxradiussq, nclusters))
            # assert 1e-10 < maxr < 5e-10
            nclusters, clusteridxs, overlapped_points = update_clusters(points, points, maxr)
            print("reclustered: nc=%d" % (nclusters))

        if False:
            plt.title('nclusters: %d' % nclusters)
            for k in np.unique(clusteridxs):
                x, y = points[clusteridxs == k].transpose()
                plt.scatter(x, y)
            plt.savefig('testoverclustering_eggbox_%d.pdf' % i, bbox_inches='tight')
            plt.close()
        assert 14 < nclusters < 20, (nclusters, i)

        for j in range(3):
            nclusters, clusteridxs, overlapped_points = update_clusters(points, points, maxr)
            assert 14 < nclusters < 20, (nclusters, i)


def test_overclustering_eggbox_update(plot=False):
    np.random.seed(1)
    for i in [20, 23, 24, 27, 42]:
        print()
        print("==== TEST CASE %d =====================" % i)
        print()
        mock = MockIntegrator()
        print("loading...")
        data = np.load(os.path.join(here, "overclustered_%d.npz" % i))
        print("loading... done")

        nsamples, mock.x_dim = data['u0'].shape
        noverlap = 0
        for i, u1 in enumerate(data['u']):
            assert len((u1 == data['u0']).all(axis=1)) == nsamples
            noverlap += (u1 == data['u0']).all(axis=1).sum()
        print("u0:%d -> u:%d : %d points are common" % (nsamples, nsamples, noverlap))

        mock._update_region(data['u0'], data['u0'])
        nclusters = mock.transformLayer.nclusters
        print("initialised with: r=%e nc=%d" % (mock.region.maxradiussq, nclusters))
        smallest_cluster = min(
            (mock.transformLayer.clusterids == i).sum()
            for i in np.unique(mock.transformLayer.clusterids))
        if smallest_cluster == 1:
            print("found lonely points")

        print(" --- intermediate tests how create_new reacts ---")
        nextTransformLayer = mock.transformLayer.create_new(data['u0'], mock.region.maxradiussq)
        print("updated to (with same data): r=%e nc=%d" % (mock.region.maxradiussq, nclusters))
        smallest_cluster = min((nextTransformLayer.clusterids == i).sum() for i in np.unique(nextTransformLayer.clusterids))
        assert smallest_cluster > 1, ("found lonely points", i, nclusters, np.unique(mock.transformLayer.clusterids, return_counts=True))

        nextTransformLayer = mock.transformLayer.create_new(data['u'], mock.region.maxradiussq)
        nclusters = nextTransformLayer.nclusters
        print("updated to (with new data): r=%e nc=%d" % (mock.region.maxradiussq, nclusters))
        smallest_cluster = min((nextTransformLayer.clusterids == i).sum() for i in np.unique(nextTransformLayer.clusterids))
        if smallest_cluster > 1:
            # this happens because mock.region.maxradiussq is not valid anymore
            # when nlive changes
            print("found lonely points", i, nclusters, np.unique(mock.transformLayer.clusterids, return_counts=True))

        if plot:
            for xi0, yi0, xi, yi in zip(data['u0'][:,0], data['u0'][:,1], data['u'][:,0], data['u'][:,1]):
                plt.plot([xi0, xi], [yi0, yi], 'x-', ms=2)

            plt.savefig('testoverclustering_eggbox_%d_diff.pdf' % i, bbox_inches='tight')
            plt.close()
        print(" --- end ---")

        if len(data['u']) < nsamples or True:
            # maxradius has to be invalidated if live points change
            print("setting maxradiussq to None")
            mock.region.maxradiussq = None

        updated = mock._update_region(data['u'], data['u'])
        nclusters = mock.transformLayer.nclusters
        print("transitioned to : r=%e nc=%d %s" % (mock.region.maxradiussq, nclusters, updated))
        smallest_cluster = min((mock.transformLayer.clusterids == i).sum() for i in np.unique(mock.transformLayer.clusterids))
        if smallest_cluster == 1:
            print("found lonely points")
        for k in np.unique(mock.transformLayer.clusterids):
            x, y = mock.region.u[mock.transformLayer.clusterids == k].transpose()
            print('cluster %d/%d: %d points @ %.5f +- %.5f , %.5f +- %.5f' % (k, nclusters, len(x), x.mean(), x.std(), y.mean(), y.std()))
        if plot:
            plt.title('nclusters: %d' % nclusters)
            for k in np.unique(mock.transformLayer.clusterids):
                x, y = mock.region.u[mock.transformLayer.clusterids == k].transpose()
                plt.scatter(x, y, s=2)

            plt.savefig('testoverclustering_eggbox_%d.pdf' % i, bbox_inches='tight')
            plt.close()
        assert 14 < nclusters < 20, (nclusters, i)
        assert smallest_cluster > 1, (i, nclusters, np.unique(mock.transformLayer.clusterids, return_counts=True))


if __name__ == '__main__':
    # test_clustering()
    # test_clusteringcase()
    test_overclustering_eggbox_update(plot=True)
