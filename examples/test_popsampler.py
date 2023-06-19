#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ultranest import ReactiveNestedSampler
from ultranest.mlfriends import RobustEllipsoidRegion, SimpleRegion, ScalingLayer
import ultranest.popstepsampler
import matplotlib.pyplot as plt

def main(generate_direction_method, ndim, nsteps, popsize, verbose=False):

    np.random.seed(1)

    logsigma = -5

    sigma = np.logspace(-1, logsigma, ndim)
    width = 1 - 5 * sigma
    width[width < 1e-20] = 1e-20
    centers = (np.sin(np.arange(ndim)/2.) * width + 1.) / 2.
    #sigma[:] = 0.01
    #centers[:] = 0.5

    norm = -0.5 * np.log(2 * np.pi * sigma**2).sum()
    def loglike(theta):
        return -0.5 * (((theta - centers) / sigma)**2).sum(axis=1) + norm

    def transform(x):
        return x

    paramnames = ['param%d' % (i+1) for i in range(ndim)]

    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, vectorized=True)
    
    # ellipsoidal:
    region_class = RobustEllipsoidRegion
    # ellipsoidal axis-aligned:
    #sampler.transform_layer_class = ScalingLayer
    #region_class = SimpleRegion
    
    sampler.stepsampler = ultranest.popstepsampler.PopulationRandomWalkSampler(
        popsize=popsize, nsteps=nsteps, scale=1. / len(paramnames),
        generate_direction=getattr(ultranest.popstepsampler, generate_direction_method), log=verbose,
    )
    results = sampler.run(
        frac_remain=0.01, update_interval_volume_fraction=0.01, 
        max_num_improvement_loops=0, min_num_live_points=400, 
        viz_callback=None, region_class=region_class
    )
    sampler.print_results()
    stats = results['posterior']
    plt.errorbar(x=np.arange(ndim), y=stats['mean'] - centers, yerr=stats['stdev'] / sigma, color='k')
    plt.savefig('populationstepsampler_%d.pdf' % ndim)
    plt.close()
    #sampler.plot_trace()

if __name__ == '__main__':
    import sys
    main(sys.argv[1], *map(int, sys.argv[2:5]), verbose='--verbose' in sys.argv[1:])
