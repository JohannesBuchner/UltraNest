==============
Release Notes
==============

4.4.0 (2024-12-13)
------------------
* Compatible with numpy version 2 and above. Any remaining errors like "ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject" are due to ultranest and numpy being installed with different versions. Reinstall numpy and ultranest in that case.

4.3.0 (2024-04-12)
------------------
* added :py:class:`ultranest.popstepsampler.PopulationSimpleSliceSampler`: Vectorized, fixed-batch size slice sampler. (`PR <https://github.com/JohannesBuchner/UltraNest/pull/118>`_ by Benjamin Beauchesne) 
* validation of passed parameter names (`PR <https://github.com/JohannesBuchner/UltraNest/pull/133>`_ by svaverbe)
* documentation improvements, including documenting results dictionary (`PR <https://github.com/JohannesBuchner/UltraNest/pull/138>`_ by Jacopo Tissino)
* make scipy actually optional (`PR <https://github.com/JohannesBuchner/UltraNest/pull/141>`_ by Matthew Kirk)
* linting

4.2.0 (2024-02-15)
------------------

* new :py:class:`ultranest.mlfriends.LocalAffineLayer` for metric learning, set as default (see `issue 124 <https://github.com/JohannesBuchner/UltraNest/issues/124>`_)
* add Highest Density Interval function (ultranest.plot.highest_density_interval_from_samples)
* corner plot style with higher signal-to-ink ratio.
* bug fixes in popstepsampler

4.1.0 (2024-02-15)
------------------

* add number of steps calibrator :py:class:`ultranest.calibrator.ReactiveNestedCalibrator`
* add relative jump distance diagnostic for step samplers
* make population step samplers more consistent with other step samplers

4.0.0 (2024-02-15)
------------------

* new :py:class:`ultranest.mlfriends.MaxPrincipleGapAffineLayer` for metric learning, set as default

3.6.5 (2023-07-18)
------------------

* documentation improvements
* logging with MPI fixes `by adipol-ph <https://github.com/JohannesBuchner/UltraNest/issues/109>`_ and `by gregorydavidmartinez <https://github.com/JohannesBuchner/UltraNest/issues/110>`_
* more flexible plotting `by facero <https://github.com/JohannesBuchner/UltraNest/issues/108>`_

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
