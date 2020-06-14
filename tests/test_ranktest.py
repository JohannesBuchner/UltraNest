from __future__ import print_function, division
import numpy as np
from ultranest.ranktest import RankAccumulator

def test_ranker():
	np.random.seed(1)
	Nlive = 400
	N = 1000
	nruns = []
	for frac in 1, 0.9:
		print("frac:", frac)
		sample_acc = RankAccumulator(Nlive)
		sample_ref = RankAccumulator(Nlive)
		runlength = []
		for i in range(N):
			sample_ref += np.random.randint(0, Nlive)
			sample_acc += np.random.randint(0, Nlive * frac)
			zscore = sample_acc - sample_ref
			if abs(zscore) > 3:
				runlength.append(len(sample_ref))
				print("split after %d" % (runlength[-1]))
				sample_acc.reset()
				sample_ref.reset()
		print('runlength:', runlength)
		nruns.append(len(runlength))
	
	nruns1, nruns2 = nruns
	print("number of runs:", nruns1, nruns2)
	assert nruns1 == 0, (nruns1)
	assert nruns2 > 0, (nruns1, nruns2)

def test_expand():
	sample_acc = RankAccumulator(3)
	try:
		sample_acc += 3
		assert False
	except IndexError:
		pass
	sample_acc.expand(4)
	sample_acc += 3

if __name__ == '__main__':
	test_expand()
	test_ranker()
