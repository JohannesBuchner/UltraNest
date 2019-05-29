from mininest.mlfriends import MLFriends, ScalingLayer
import numpy as np
import time
import matplotlib.pyplot as plt

def test_transform():
	np.random.seed(1)
	corrs = np.arange(-1, 1, 0.1)
	corrs *= 0.999
	for corr in corrs:
		for scaleratio in [1, 0.001]:
			covmatrix = np.array([[1., corr], [corr, 1.]])
			points = np.random.multivariate_normal(np.zeros(2), covmatrix, size=1000) 
			print(corr, scaleratio, covmatrix.flatten(), points.shape)
			points[:,0] = points[:,0] * 0.01 * scaleratio + 0.5
			points[:,1] = points[:,1] * 0.01 + 0.5
			
			layer = ScalingLayer()
			layer.optimize(points, points)
			tpoints = layer.transform(points)
			assert tpoints.shape == points.shape, (tpoints.shape, points.shape)
			points2 = layer.untransform(tpoints)
			assert tpoints.shape == points2.shape, (tpoints.shape, points2.shape)
			
			assert (points2 == points).all(), (points, tpoints, points2)

			# transform a single point
			points = points[0]
			tpoints = layer.transform(points)
			assert tpoints.shape == points.shape, (tpoints.shape, points.shape)
			points2 = layer.untransform(tpoints)
			assert tpoints.shape == points2.shape, (tpoints.shape, points2.shape)
			
			assert (points2 == points).all(), (points, tpoints, points2)

def test_wrap(plot=False):
	np.random.seed(1)
	for Npoints in 10, 100, 1000:
		for wrapids in [[], [0], [1], [0,1]]:
			print("Npoints=%d wrapped_dims=%s" % (Npoints, wrapids))
			#wrapids = np.array(wrapids)
			points = np.random.normal(0.5, 0.01, size=(Npoints, 2))
			for wrapi in wrapids:
				points[:,wrapi] = np.fmod(points[:,wrapi] + 0.5, 1)
			
			assert (points > 0).all(), points
			assert (points < 1).all(), points
			layer = ScalingLayer(wrapped_dims=wrapids)
			layer.optimize(points, points)
			tpoints = layer.transform(points)
			assert tpoints.shape == points.shape, (tpoints.shape, points.shape)
			points2 = layer.untransform(tpoints)
			assert tpoints.shape == points2.shape, (tpoints.shape, points2.shape)
			
			if plot:
				plt.subplot(1, 2, 1)
				plt.scatter(points[:,0], points[:,1])
				plt.scatter(points2[:,0], points2[:,1], marker='x')
				plt.subplot(1, 2, 2)
				plt.scatter(tpoints[:,0], tpoints[:,1])
				plt.savefig("testtransform_%d_wrap%d.pdf" % (Npoints, len(wrapids)), bbox_inches='tight')
				plt.close()
			
			
			assert np.allclose(points2, points), (points, tpoints, points2)


	
if __name__ == '__main__':
	test_wrap(plot=True)
	test_transform()
