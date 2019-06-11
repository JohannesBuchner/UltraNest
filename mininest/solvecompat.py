"""

from mininest.solvecompat import pymultinest_solve_compat as solve

is a drop-in replacement for

from pymultinest.solve import solve

"""


import numpy as np
import string

from .integrator import NestedSampler, ReactiveNestedSampler

def pymultinest_solve_compat(LogLikelihood, Prior, n_dims, 
        paramnames=None,
        outputfiles_basename=None, resume=False, 
        n_live_points=400, evidence_tolerance=0.5, 
        seed=-1, max_iter=0, wrapped_params=None, **kwargs
    ):

    def vectorized_loglike(thetas):
        return np.asarray([LogLikelihood(theta) for theta in thetas])

    def vectorized_transform(cubes):
        return np.asarray([Prior(cube) for cube in cubes]).reshape((-1, n_dims))

    if paramnames is None:
        paramnames = list(string.ascii_lowercase)[:n_dims]

    sampler = ReactiveNestedSampler(paramnames, 
        vectorized_loglike, 
        transform=vectorized_transform, 
        min_num_live_points=n_live_points,
        log_dir=outputfiles_basename, 
        append_run_num=not resume,
        wrapped_params=wrapped_params,
    )
    sampler.run(dlogz=evidence_tolerance, max_iters=max_iter)
    sampler.print_results()
    results = sampler.results
    sampler.plot()
    return dict(logZ=results['logz'],
        logZerr=results['logzerr'],
        samples=results['samples'])
	
