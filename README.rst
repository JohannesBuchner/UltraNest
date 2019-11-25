=========
UltraNest
=========

Fit and compare complex models reliably and rapidly with advanced sampling techniques.

.. image:: https://img.shields.io/pypi/v/ultranest.svg
        :target: https://pypi.python.org/pypi/ultranest

.. image:: https://img.shields.io/travis/JohannesBuchner/ultranest.svg
        :target: https://travis-ci.org/JohannesBuchner/ultranest

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/UltraNest/
        :alt: Documentation Status

Correctness. Speed. Ease of use. 🦔

About
-----

When scientific models are compared to data, two tasks are important:
1) contraining the model parameters and 2) comparing the model to other models.
Different techniques have been developed to explore model parameter spaces.
This package implements a Monte Carlo technique called nested sampling.

**Nested sampling** allows Bayesian inference on arbitrary user-defined likelihoods.
In particular, posterior probability distributions on model parameters
are constructed, and the marginal likelihood ("evidence") Z is computed.
The former can be used to describe the parameter constraints of the data,
the latter can be used for model comparison (via `Bayes factors`) 
as a measure of the prediction parsimony of a model.

In the last decade, multiple variants of nested sampling have been 
developed. These differ in how nested sampling finds better and
better fits while respecting the priors 
(constrained likelihood prior sampling techniques), and whether it is 
allowed to go back to worse fits and explore the parameter space more.

This package develops novel, advanced techniques for both (See How it works).
They are especially remarkable for being free of tuning parameters 
and theoretically justified. Beyond that, UltraNest has support for 
Big Data sets and high-performance computing applications.

UltraNest is intended for fitting complex physical models with slow
likelihood evaluations, with one to hundreds of parameters.
UltraNest intends to replace heuristic methods like multi-ellipsoid
nested sampling and dynamic nested sampling with more rigorous methods.
UltraNest also attempts to provide feature parity compared to other packages
(such as MultiNest).

However, UltraNest is still in beta. You can help by
testing it and reporting issues. Code contributions for fixes and 
new features are also welcome.
See the `Contributing page <https://johannesbuchner.github.io/UltraNest/contributing.html>`_.

Features
---------

* Pythonic

  * pip installable
  * Easy to program for: Sanity checks with meaningful errors
  * Can control the run programmatically and check status
  * Reasonable defaults, but customizable
  * Thoroughly tested with many unit and integration tests

* Robust exploration easily handles:

  * Degenerate parameter spaces such as bananas or tight correlations
  * Multiple modes/solutions in the parameter space
  * Robust, parameter-free MLFriends algorithm 
    (metric learning RadFriends, Buchner+14,+19), with new improvements
    (region follows new live points, clustering improves metric iteratively).
  * High-dimensional problems with slice sampling (or ellipsoidal sampling, FlatNUTS, etc.),
    inside region.
  * Wrapped/circular parameters, derived parameters
  * Fast-slow parameters

* strategic nested sampling

  * can vary (increase) number of live points (akin to dynamic nested sampling, but with different targets)
  * can sample clusters optimally (e.g., at least 50 points per cluster/mode/solution)
  * can target minimizing parameter estimation uncertainties
  * can target a desired evidence uncertainty threshold
  * can target a desired number of effective samples
  * or any combination of the above
  * Robust ln(Z) uncertainties by bootstrapping live points.

* Lightweight and fast

  * some functions implemented in Cython
  * vectorized likelihood function calls
  * Use multiple cores, fully parallelizable from laptops to clusters
  * MPI support

* Advanced visualisation and crash recovery:

  * Checkpointing and resuming, even with different number of live points
  * Run-time visualisations and exploration information
  * Corner plots, run and parameter exploration diagnostic plots


TODO
^^^^

* Documentation:

  * Example power law fit
  * Example spectral line fit, white and GP
  * Example low-d Bayesian GP emulator as pre-filter to model evaluation
  * Example verifying integration with VB+IS

Usage
^^^^^

Read the full documentation at:

https://johannesbuchner.github.io/UltraNest/


Licence
^^^^^^^

GPLv3 (see LICENCE file). If you require another license, please contact me.

Icon made by `Freepik <https://www.flaticon.com/authors/freepik>`_.
