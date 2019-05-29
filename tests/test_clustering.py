from mininest.mlfriends import update_clusters
import numpy as np
import matplotlib.pyplot as plt

def test_clustering():
	for i in range(5):
		np.random.seed(i*100)
		
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
	points = np.loadtxt("clusters2.txt")
	maxr = np.loadtxt("clusters2_radius.txt")
	from mininest.mlfriends import update_clusters
	#transformLayer = ScalingLayer()
	#transformLayer.optimize(points)
	#region = MLFriends(points, transformLayer)
	#maxr = region.compute_maxradiussq(nbootstraps=30)
	print('maxradius:', maxr)
	nclusters, clusteridxs, overlapped_points = update_clusters(points, points, maxr)
	plt.title('nclusters: %d' % nclusters)
	for i in np.unique(clusteridxs):
		x, y = points[clusteridxs == i].transpose()
		plt.scatter(x, y)
	plt.savefig('testclustering_2.pdf', bbox_inches='tight')
	plt.close()


if __name__ == '__main__':
	#test_clustering()
	test_clusteringcase()
