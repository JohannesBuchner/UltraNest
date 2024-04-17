=========
UltraNest
=========

Fit and compare complex models reliably and rapidly with advanced sampling techniques.

.. image:: https://img.shields.io/pypi/v/ultranest.svg
        :target: https://pypi.python.org/pypi/ultranest

.. image:: https://circleci.com/gh/JohannesBuchner/UltraNest/tree/master.svg?style=shield
        :target: https://circleci.com/gh/JohannesBuchner/UltraNest

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/UltraNest/
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/GitHub-JohannesBuchner%2FUltraNest-blue.svg?style=flat
        :target: https://github.com/JohannesBuchner/UltraNest/
        :alt: Github repository

.. image:: https://joss.theoj.org/papers/10.21105/joss.03001/status.svg
        :target: https://doi.org/10.21105/joss.03001
        :alt: Software paper

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

This package develops novel, advanced techniques for both (See 
`How it works <https://johannesbuchner.github.io/UltraNest/method.html>`_).
They are especially remarkable for being free of tuning parameters 
and theoretically justified. Beyond that, UltraNest has support for 
Big Data sets and high-performance computing applications.

UltraNest is intended for fitting complex physical models with slow
likelihood evaluations, with one to hundreds of parameters.
UltraNest intends to replace heuristic methods like multi-ellipsoid
nested sampling and dynamic nested sampling with more rigorous methods.
UltraNest also attempts to provide feature parity compared to other packages
(such as MultiNest).

You can help by testing UltraNest and reporting issues. Code contributions are welcome.
See the `Contributing page <https://johannesbuchner.github.io/UltraNest/contributing.html>`_.

Features
---------

* Pythonic

  * pip and conda installable
  * Easy to program for: Sanity checks with meaningful errors
  * Can control the run programmatically and check status
  * Reasonable defaults, but customizable
  * Thoroughly tested with many unit and integration tests
  * NEW: supports likelihood functions written in `Python <https://github.com/JohannesBuchner/UltraNest/tree/master/languages/python>`_, `C <https://github.com/JohannesBuchner/UltraNest/tree/master/languages/c>`_, `C++ <https://github.com/JohannesBuchner/UltraNest/tree/master/languages/c%2B%2B>`_, `Fortran <https://github.com/JohannesBuchner/UltraNest/tree/master/languages/fortran>`_, `Julia <https://github.com/JohannesBuchner/UltraNest/tree/master/languages/julia>`_ and `R <https://github.com/JohannesBuchner/UltraNest/tree/master/languages/r>`_

* Robust exploration easily handles:

  * Degenerate parameter spaces such as bananas or tight correlations
  * Multiple modes/solutions in the parameter space
  * Robust, parameter-free MLFriends algorithm 
    (metric learning RadFriends, Buchner+14,+19), with new improvements
    (region follows new live points, clustering improves metric iteratively, 
    NEW in v4.0: refined local metric).
  * High-dimensional problems with hit-and-run sampling
  * Wrapped/circular parameters, derived parameters
  * Fast-slow parameters

* Lightweight and fast

  * some functions implemented in Cython
  * `vectorized likelihood function calls <https://johannesbuchner.github.io/UltraNest/performance.html>`__, 
    optimally supporting models with deep learning emulators
  * Use multiple cores, fully parallelizable from laptops to computing clusters
  * `MPI support <https://johannesbuchner.github.io/UltraNest/performance.html>`__

* Advanced visualisation and crash recovery:

  * Live view of the exploration for Jupyter notebooks and terminals
  * Publication-ready visualisations
  * Corner plots, run and parameter exploration diagnostic plots
  * Checkpointing and resuming, even with different number of live points
  * `Warm-start: resume from modified data / model <https://johannesbuchner.github.io/UltraNest/example-warmstart.html>`__

* strategic nested sampling

  * can vary (increase) number of live points (akin to dynamic nested sampling, but with different targets)
  * can sample clusters optimally (e.g., at least 50 points per cluster/mode/solution)
  * can target minimizing parameter estimation uncertainties
  * can target a desired evidence uncertainty threshold
  * can target a desired number of effective samples
  * or any combination of the above
  * Robust ln(Z) uncertainties by bootstrapping live points.

Usage
^^^^^

`Get started! <https://johannesbuchner.github.io/UltraNest/using-ultranest.html>`_

Read the full documentation with tutorials at:

https://johannesbuchner.github.io/UltraNest/

`API Reference: <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`_.

`Code repository: https://github.com/JohannesBuchner/UltraNest/ <https://github.com/JohannesBuchner/UltraNest/>`_

Licence
^^^^^^^

How to `cite UltraNest <https://johannesbuchner.github.io/UltraNest/issues.html#how-should-i-cite-ultranest>`_.

GPLv3 (see LICENCE file). If you require another license, please contact me.

The cute hedgehog icon was made by `Freepik <https://www.flaticon.com/authors/freepik>`_.
It symbolises UltraNest's approach of carefully walking up a likelihood,
ready to defend against any encountered danger.
