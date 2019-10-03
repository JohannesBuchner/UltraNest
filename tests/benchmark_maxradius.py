from ultranest.mlfriends import MLFriends, ScalingLayer, AffineLayer
import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_maxradius():
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
	

def benchmark_transform():
	npts = 400
	for layer in 'scale', 'affine':
		print(" ndim | duration  [%s]" % layer)
		tplotpoints = []
		rplotpoints = []
		nplotpoints = []
		for ndim in 2, 4, 8, 16, 32, 64, 128, 256,:
			np.random.seed(ndim)
			points = np.random.uniform(0.4, 0.6, size=(npts,ndim))
			transformLayer = ScalingLayer() if layer == 'scale' else AffineLayer() 
			region = MLFriends(points, transformLayer)
			region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
			region.create_ellipsoid()
			
			niter = 0
			total_duration = 0
			while total_duration < .1:
				start = time.time()
				u = region.transformLayer.untransform(np.random.normal(size=(ndim)))
				region.transformLayer.transform(u)
				total_duration += time.time() - start
				niter += 1
			print('%5d | %.2fms ' % (ndim, total_duration * 1000 / niter))
			tplotpoints.append((ndim, total_duration * 1000 / niter))

			niter = 0
			total_duration = 0
			while total_duration < .1:
				u = np.random.normal(0.5, 0.1, size=(10, ndim))
				start = time.time()
				region.inside(u)
				total_duration += time.time() - start
				niter += 1
			print('%5d |          %.2fms ' % (ndim, total_duration * 1000 / niter))
			rplotpoints.append((ndim, total_duration * 1000 / niter))
			
			niter = 0
			total_duration = 0
			while total_duration < .1:
				u = np.random.normal(0.5, 0.1, size=(10, ndim))
				start = time.time()
				array = np.empty((10), dtype=int)
				array[:] = -1
				array = np.empty((10), dtype=int)
				array[:] = -1
				array = np.empty((10), dtype=int)
				array[:] = -1
				total_duration += time.time() - start
				niter += 1
			print('%5d |                 %.2fms ' % (ndim, total_duration * 1000 / niter))
			nplotpoints.append((ndim, total_duration * 1000 / niter))
		plt.plot(*zip(*tplotpoints), label=layer + ' transform')
		plt.plot(*zip(*rplotpoints), label=layer + ' region.inside')
		plt.plot(*zip(*nplotpoints), label=layer + ' array')
	
	plt.xlabel('Number of dimensions')
	plt.ylabel('Duration [ms]')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend(loc='best', prop=dict(size=10))
	plt.savefig('testtransform.pdf', bbox_inches='tight')
	plt.close()
		
if __name__ == '__main__':
	#benchmark_maxradius()
	benchmark_transform()
