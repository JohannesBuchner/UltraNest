#!/usr/bin/env python3
"""

This script tests the UltraNest stepsamplers
in a few configurations with a real model.

"""

import numpy as np
import ultranest, ultranest.stepsampler

# velocity dispersions of dwarf galaxies by van Dokkum et al., Nature, 555, 629 https://arxiv.org/abs/1803.10237v1
values = np.array([15, 4, 2, 11, 1, -2, -1, -14, -39, -3])
values_lo = np.array([7, 16, 6, 3, 6, 5, 10, 6, 11, 13])
values_hi = np.array([7, 15, 8, 3, 6, 6, 10, 7, 14, 14])

def run():
	n_data = len(values)
	samples = []

	for i in range(n_data):
		# draw normal random points
		u = np.random.normal(size=400)
		v = values[i] + np.where(u < 0, u * values_lo[i], u * values_hi[i])
		samples.append(v)
	data = np.array(samples)

	Nobj, Nsamples = data.shape
	minval = -80
	maxval = +80
	ndim = 8
	viz_callback = None

	bins = np.linspace(minval, maxval, ndim+1)
	binned_data = np.array([np.histogram(row, bins=bins)[0] for row in data])
	param_names = ['bin%d' % (i+1) for i in range(ndim)]

	def likelihood(params):
		"""Histogram model"""
		return np.log(np.dot(binned_data, params) / Nsamples + 1e-300).sum()

	def transform_dirichlet(quantiles):
		"""Histogram distribution priors"""
		# https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
		# first inverse transform sample from Gamma(alpha=1,beta=1), which is Exponential(1)
		gamma_quantiles = -np.log(quantiles)
		# dirichlet variables
		return gamma_quantiles / gamma_quantiles.sum()

	stepsamplers = [
		ultranest.stepsampler.RegionBallSliceSampler(40, region_filter=False, adaptive_nsteps='move-distance'),
		ultranest.stepsampler.RegionSliceSampler(40, region_filter=False),
		ultranest.stepsampler.CubeSliceSampler(40, region_filter=True),
		ultranest.stepsampler.RegionMHSampler(40, region_filter=False),
		ultranest.stepsampler.RegionSequentialSliceSampler(40, region_filter=True, adaptive_nsteps='move-distance'),
		ultranest.stepsampler.SpeedVariableRegionSliceSampler(
			step_matrix=[[1], [1,2,3], Ellipsis, np.ones(len(param_names), dtype=bool)], nsteps=40, region_filter=False),
	]
	for stepsampler in stepsamplers:
		print(stepsampler)
		sampler = ultranest.ReactiveNestedSampler(
			param_names, likelihood, transform_dirichlet)
		sampler.stepsampler = stepsampler
		sampler.run(frac_remain=0.5, viz_callback=viz_callback)
		sampler.print_results()

if __name__ == '__main__':
	run()
