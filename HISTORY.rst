==============
Release Notes
==============


3.6.0 (2023-06-22)
------------------

* add PopulationRandomWalkSampler: vectorized Gaussian random walks for GPU/JAX-powered likelihoods
* limit initial widening to escape plateau (issue #81)


3.5.0 (2022-09-05)
------------------

* add hot-resume: resume from a similar fit (with different data)
* fix post_summary.csv column order
* fix build handling for non-pip systems (pyproject.toml)
* more efficient handling of categorical variables


3.4.0 (2022-04-05)
------------------

* add differential evolution proposal for slice sampling, recommend it
* fix revert of step sampler when run out of constraint, in MPI
* add SimpleRegion: axis-aligned ellipsoidal for very high-d.


3.3.3 (2021-09-17)
------------------

* pretty marginal posterior plot to stdout
* avoid non-terminations when logzerr cannot be reached
* add RobustEllipsoidRegion: ellipsoidal without MLFriends for high-d.
* add WrappingEllipsoid: for additional rejection.
* bug fixes on rank order test
* add resume-similar
* modular step samplers


3.0.0 (2020-10-03)
------------------

* Accelerated Hit-and-Run Sampler added
* Support for other languages (C, C++, Julia, Fortran) added
* Insertion order test added
* Warm-start added
* Rejection sampling with transformed ellipsoid added

2.2.0 (2020-02-07)
------------------

* allow reading UltraNest outputs without ReactiveNestedSampler instance

2.1.0 (2020-02-07)
------------------

* adaptive number of steps for slice and hit-and-run samplers.

2.0.0 (2019-10-03)
------------------

* First release.

1.0.0 (2014)
------------------

* A simpler version referenced in Buchner et al. (2014),
  combining RadFriends with an optional Metropolis-Hastings proposal.
