=========
UltraNest
=========

Fit and compare complex models reliably and rapidly with advanced sampling techniques.

.. image:: https://img.shields.io/pypi/v/ultranest.svg
        :target: https://pypi.python.org/pypi/ultranest

.. image:: https://img.shields.io/travis/JohannesBuchner/ultranest.svg
        :target: https://travis-ci.org/JohannesBuchner/ultranest

.. image:: https://readthedocs.org/projects/ultranest/badge/?version=latest
        :target: https://ultranest.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Pre-release alpha software.

Nested sampling for Bayesian inference on arbitrary user-defined likelihoods.

Features
--------

* Pythonic

  * pip installable
  * Easy to program for: Sanity checks with meaningful errors
  * Can control the run programmatically and check status
  * Reasonable defaults, but customizable

* Checkpointing and resuming, even with different number of live points
* Wrapped/circular parameters, derived parameters
* Fast-slow parameters
* Tracking solution modes
* Run-time visualisations and exploration information
* Corner plots, run and parameter exploration diagnostic plots
* Robust ln(Z) error bars

* Robust exploration easily handles:

  * Multiple modes 
  * Degenerate parameter spaces such as bananas or tight correlations
  * Uses the robust, parameter-free MLFriends algorithm (metric learning RadFriends, Buchner+14,+19)
  * Region follows new live points

* High-dimensional problems with slice sampling (or ellipsoidal sampling, FlatNUTS, etc.),
  inside region.

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
  * Use multiple cores, fully parallelizable from laptops to clusters
  * MPI support


TODO
----

* Documentation:
  * Example line fit
  * Example line fit with heterogeneous xy errors
  * Example line fit with outliers (mixture of 2 models)
  * Example power law fit
  * Example intrinsic 1d distribution from measurement errors, 2-component mixture
  * Example extracting more posterior points (ESS)
  * Example spectral line fit, white and GP
  * Example time series analysis (2-component fourier)
  * Example low-d Bayesian GP emulator as pre-filter to model evaluation
  * Example in jupyter notebook
  * Example resuming from similar data
  * Example verifying integration with VB+IS
  * Example running with MPI parallelisation
  * Example: Use external plotter to show posterior distribution uncertainty
  * Example: Improve quality by successively forgetting points below some nsteps,
    doubling the nsteps, and checking how lnZ changes. Stop when it remains
    consistent.

Usage
-----

Install::

        $ python3 setup.py install --user

Run tests::

        $ python3 setup.py test

Run an example::

        $ cd examples/
        $ python3 testsine.py

Run the sampler::

    from ultranest import ReactiveNestedSampler
    
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
    from ultranest.solvecompat import pymultinest_solve_compat as solve
    
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
-------

GPLv3 (see LICENCE file). If you require another license, please contact me.

