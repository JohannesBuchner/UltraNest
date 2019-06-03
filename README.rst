MiniNest Alpha
===============

Pre-release alpha software. Will probably be renamed later.

Features
=========

* Nested sampling for Bayesian inference on arbitrary user-defined likelihoods
* Robust, parameter-free MLFriends algorithm (metric learning RadFriends, Buchner (2014), Buchner (2019)
* pip installable
* MPI-support
* Checkpointing and resuming
* Resuming from a run with different number of live points
* Wrapped/circular parameters
* Storing derived values
* Tracking solution modes
* Run-time visualisations and exploration information
* Very fast

  * some functions implemented in Cython
  * vectorized likelihood function calls


Coming soon
=============

* Support for stopping a run programmatically, and receiving feedback
* Support for high-dimensional problems (>=20d)
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

