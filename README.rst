MiniNest Alpha
===============

Pre-release alpha software. Will probably be renamed later.

Nested sampling for Bayesian inference on arbitrary user-defined likelihoods.

Features
=========

* Pythonic

  * pip installable
  * Easy to program for: Sanity checks with meaningful errors
  * Can control the run programmatically and check status
  * Reasonable defaults, but customizable

* Checkpointing and resuming, even with different number of live points
* Wrapped/circular parameters, derived parameters
* Tracking solution modes
* Run-time visualisations and exploration information
* Corner plots, run and parameter exploration diagnostic plots

* Robust exploration easily handles:

  * Multiple modes 
  * Degenerate parameter spaces such as bananas or tight correlations
  * Uses the robust, parameter-free MLFriends algorithm (metric learning RadFriends, Buchner+14,+19)

* strategic nested sampling

  * can vary (increase) number of live points (similar to dynamic nested sampling)
  * can sample clusters optimally (e.g., at least 50 points per cluster)
  * can target minimizing parameter estimation uncertainties
  * can target a desired evidence uncertainties
  * can target a desired number of effective samples
  * or any combination of the above

* Very fast

  * some functions implemented in Cython
  * vectorized likelihood function calls
  * MPI-support


Coming soon
=============

* Support for high-dimensional problems (>=20d)

Usage
=============

Install::

        $ python3 setup.py install --user

Run tests::

        $ python3 setup.py test

Run an example::

        $ cd examples/
        $ python3 testsine.py

Run the sampler::

    from mininest import ReactiveNestedSampler
    
    paramnames = ['theta', 'L']
    
    # vectorized prior transform function. 
    # x is a unit cube (N, x_dims). 
    # returns transformed parameters (N, num_params)
    def transform(x):
        return 10 * x - 5.
    
    # theta are the transformed parameters (N, num_params). 
    # returns loglikelihood (N values)
    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
        return like
    
    sampler = ReactiveNestedSampler(paramnames, 
        loglikelihood, 
        transform=prior_transform, 
        min_num_live_points=400, 
        log_dir='logs/myproblem', # where to store 
        append_run_num=False, # set to true to start fresh
        wrapped_params=None, # or [False, True, False, False]
    )
    sampler.run()
    sampler.print_results()
    results = sampler.results
    print('results information:', results)
    sampler.plot()


PyMultinest compatibility layer allows a drop-in replacement::

    # instead of "from pymultinest.solve import solve", we use:
    from mininest.solvecompat import pymultinest_solve_compat as solve
    
    # the rest is exactly as in PyMultinest:
    
    ndim = 2
    sigma = 0.01
    centers = 0.5
    paramnames = ['a', 'b']

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum() - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
        return like

    def transform(x):
        return 10 * x - 5.

    result = solve(LogLikelihood=loglike, Prior=transform, 
        n_dims=ndim, outputfiles_basename=None,
        verbose=True, resume=True, importance_nested_sampling=False)
    
    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(paramnames, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))




Licence
============

Closed-source at the moment, will be released as open source later.

