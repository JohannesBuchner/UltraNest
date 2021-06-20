from __future__ import print_function, division
import numpy as np
from ultranest.ordertest import UniformOrderAccumulator, infinite_U_zscore

def test_invalid_order():
	sample_acc = UniformOrderAccumulator()
	sample_acc.add(2, 3)
	try:
		sample_acc.add(4, 3)
		assert False
	except ValueError:
		pass

def test_diff_expand():
	sample_acc = UniformOrderAccumulator()
	sample_acc.add(1, 3)
	sample_acc.add(4, 5)
	sample_acc.add(5, 6)

def test_order_correctness():
	np.random.seed(1)
	Nlive = 400
	N = 1000
	nruns = []
	for frac in 1, 0.9:
		print("frac:", frac)
		sample_acc = UniformOrderAccumulator()
		runlength = []
		samples = []
		for i in range(N):
			order = np.random.randint(0, Nlive * frac)
			sample_acc.add(order, Nlive)
			samples.append(order)
			zscore = sample_acc.zscore
			assert np.isclose(zscore, infinite_U_zscore(np.asarray(samples), Nlive)), (zscore, infinite_U_zscore(np.asarray(samples), Nlive), samples)
			if abs(zscore) > 3:
				runlength.append(len(sample_acc))
				print("split after %d" % (runlength[-1]))
				sample_acc.reset()
				samples = []
		print('runlength:', runlength)
		nruns.append(len(runlength))
	
	nruns1, nruns2 = nruns
	print("number of runs:", nruns1, nruns2)
	assert nruns1 == 0, (nruns1)
	assert nruns2 > 0, (nruns1, nruns2)

if __name__ == '__main__':
	test_order_correctness()
