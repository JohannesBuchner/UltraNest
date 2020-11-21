import numpy as np
from ultranest import ReactiveNestedSampler

def mytransform(cube):
    return cube * 2 - 1

def mylikelihood(params):
    centers = 0.1 * np.arange(params.shape[1]).reshape((1, -1))
    return -0.5 * (((params - centers) / 0.01)**2).sum(axis=1)

paramnames = ["a", "b", "c"]
sampler = ReactiveNestedSampler(paramnames, mylikelihood, transform=mytransform, vectorized=True)
sampler.run()
sampler.print_results()
sampler.plot()
