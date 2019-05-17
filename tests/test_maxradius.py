from mininest.mlfriends import MLFriends, ScalingLayer
import numpy as np
import time
import matplotlib.pyplot as plt

def test_maxradius():
	print(" ndim |  npts | duration")
	for ndim in 2, 4, 8, 16, 32, 64:
		plotpoints = []
		np.random.seed(ndim)
		for npts in 100, 400, 1000, 4000:
			points = np.random.uniform(size=(npts,ndim))
			transformLayer = ScalingLayer()
			region = MLFriends(points, transformLayer)
			
			niter = 0
			total_duration = 0
			while total_duration < 1:
				start = time.time()
				maxr = region.compute_maxradiussq(nbootstraps=20)
				total_duration += time.time() - start
				niter += 1
			print('%5d | %5d | %.2fms  val=%f' % (ndim, npts, total_duration * 1000 / niter, maxr))
			plotpoints.append((npts, total_duration * 1000 / niter / npts**2))
		plt.plot(*zip(*plotpoints), label='ndim=%d' % ndim)
	
	plt.xlabel('Number of live points')
	plt.ylabel('Duration [ms] / nlive$^2$')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend(loc='best', prop=dict(size=10))
	plt.savefig('testmaxradius.pdf', bbox_inches='tight')
	plt.close()
	
	
if __name__ == '__main__':
	test_maxradius()
