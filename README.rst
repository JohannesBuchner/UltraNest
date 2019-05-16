MiniNest Alpha
===============

Pre-release alpha software. Will probably be renamed later.

Features
=========

* Nested sampling for Bayesian inference on arbitrary user-defined likelihoods
* Robust, parameter-free MLFriends algorithm (metric learning RadFriends, Buchner (2014), Buchner (2019)
* pip installable
* MPI-support
* Wrapped/circular parameters
* Derived parameters
* Tracking solution modes
* Neat visualisations and exploration information during the run
* Very fast implementation

  * some functions implemented in Cython
  * vectorized likelihood function calls



Coming soon
=============

* Support for stopping a run programmatically, and receiving feedback
* Support for high-dimensional problems (>=20d)
* Support resuming an interrupted run
* Support for variable number of live points, posterior bulking

Usage
=============

Install::

        $ python3 setup.py install --user

Run tests::

        $ python3 setup.py test

Run an example::

        $ cd examples/
        $ python3 testsine.py





Licence
============

Closed-source at the moment, will be released as open source later.
