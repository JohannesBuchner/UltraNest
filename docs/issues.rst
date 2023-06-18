.. _faq:

===========================
Frequently Asked Questions
===========================

.. contents:: :local:

How can I ask a question not covered here?
------------------------------------------

See `Contributing <contributing.rst>`_ for how to report bugs and ask questions.

Opening a github issue is preferred, because then other people can find the question and answer.

How do I suppress the output?
-----------------------------

To suppress the live point visualisations, set ``viz_callback=False`` in ``sampler.run()``.

To suppress the status line, set ``show_status=False`` in `sampler.run()``.

See the documentation of :py:meth:`ultranest.ReactiveNestedSampler.run()`.

To suppress the logging to stderr, set up a logging handler::

    import logging
    logger = logging.getLogger("ultranest")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('[ultranest] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

You may want to alter the above to log to a file instead. See the `logging python module <https://docs.python.org/3/library/logging.html>`_ docs.

To completely turn off logging, you can use::

    import logging
    logger = logging.getLogger("ultranest")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)


How should I choose the number of live points?
-----------------------------------------------

Try with 400 first.

In principle, choosing a very low number allows nested sampling to
make very few iterations and go to the peak quickly. However,
the space will be poorly sampled, giving a large region and thus
low efficiency, and potentially not seeing interesting modes.
Therefore, a value above 100 is typically useful.

How can I use fast-slow parameters?
-----------------------------------

If you want some parameters to be updated less frequently than others
in the search for a new point,
use the :class:`ultranest.stepsampler.SpeedVariableRegionSliceSampler`.
The `high-dimensional tutorials <example-sine-highd.html>`_ shows how to use a step sampler.

How can I speed up my problem?
------------------------------

Computationally:

 * Speed up the likelihood function, for example by using numpy more efficiently
 * Vectorize the likelihood function (see `Tour of the features <performance.rst>`_).

Algorithmically:

 * Modify the transform to better map the live points to data constraints.
 * Use a step sampler (see `the high-d tutorial example <example-sine-highd.html>`)
 * Try to alter your model

How can I verify that the step sampler is correct?
--------------------------------------------------

 * Increase (double) the number of steps. If the logz remains the same,
   chances are good that the result is reliable.

What does the status line mean?
--------------------------------

You may see lines like this::

    Z=0.3(43.44%) | Like=89.39..96.29 [89.3916..89.3936]*| it/evals=39671/11660719 eff=0.2939% N=400

Here is what the parts mean::

                                                                             total number       number
                                                                   current   of likelihood      of live
                                                                  iteration  evaluations        points
                                                                      |       |                   |
    Z=0.3(43.44%) | Like=89.39..96.29 [89.3916..89.3936]*| it/evals=39671/11660719 eff=0.2939% N=400
       |    |           ------+-----   -------+--------                                 |
       |    |                 |               |                                    current effiency
       |    |                 |               +- likelihood range targeted by strategy
       |    |                 +- lowest and highest likelihood
       |    |                    of current live points
       |    |
       |    +- Progress. Completed fraction of the integral.
       |                 (related to frac_remain)
       |
       +- current logz estimate

What does the live point display mean?
------------------------------------------

You may see displays like this::

    Mono-modal Volume: ~exp(-5.89) * Expected Volume: exp(-2.02)

    param1:      +0.0|         *********************************         |     +1.0
    param2:      +0.0|         *********************************         |     +1.0
    ...


They are very useful if you understand them. Here is what the parts mean::

       how many         how large the                  ow large the
       clusters        volume should be               MLFriends region
         |            based on iteration              is (not subtracting overlaps)
         |                   |                             |
    Mono-modal Volume: ~exp(-5.89) * Expected Volume: exp(-2.02)

    For each parameter you will find a simple linear plot of the live points:

    param1:      +0.0|         *********************************         |     +1.0
      |            |                   where live points are                     |
      |          lower value                                               upper value
    parameter name

    Live points are shown as *, or numbers, which indicate which cluster they
    belong to. Sometimes too many clusters are being found, but that does
    not make the result incorrect. Increasing the number of live points
    can avoid this (use >100).

How do I use UltraNest with C / C++ / Fortran / Julia / R?
----------------------------------------------------------

Examples are available for C, C++, Fortran, Julia and R
at https://github.com/JohannesBuchner/UltraNest/tree/master/languages

These implement the same prior and likelihood functions. The
functions are vectorized to reduce the number of function calls.

The C, C++ and Fortran functions are compiled to a dynamic library,
which is used from Python.

The Julia and R example differ in that the code is run from the Julia/R
environment, calling Python which calls back Julia/R.


What is the difference between UltraNest and MultiNest/dynesty/...
------------------------------------------------------------------

MultiNest, PyMultiNest, nestle, dynesty, NestedSamplers.jl

Correctness:

* MultiNest, PyMultiNest, nestle, dynesty, NestedSamplers.jl implement and default to
  multi-ellipsoidal rejection. This is a heuristic technique
  known to give biased results for some problems, including the
  `hyper-rectangle <https://arxiv.org/abs/1407.5459>`_ and the `LogGamma <https://arxiv.org/abs/1304.7808>`_ problems,
  as well as real-world applications.
  This problem does not really go away with decreasing efficiency (increasing the ellipsoid scale factor).
  All implementations of MultiNest will suffer this issue.

* MultiNest also implements importance nested sampling, which has been claimed to
  `reduce this problem <https://ui.adsabs.harvard.edu/abs/2019OJAp....2E..10F/abstract>`_.
  However, importance nested sampling severely `under-estimates the uncertainties <https://ui.adsabs.harvard.edu/abs/2020AJ....159...73N/abstract>`_.

* UltraNest implements MLFriends, a parameter-free algorithm that derives
  a safe region by learning from the points themselves using cross-validation.

* UltraNest implements safer uncertainty estimation.
  It incorporates the scatter in both volume estimates and likelihood estimates.
  The other libraries only support a static volume uncertainty estimate.
  UltraNest additionally uses bootstrapping to emulate multiple runs.

Implementation differences:

* UltraNest supports writing to disk (check-pointing) and resuming.
  nestle and dynesty do not support this.

* UltraNest supports parallelisation out-of-the-box.
  MPI support is built-in, like in MultiNest, allowing scaling to clusters.

* MultiNest, dynesty, nestle, and other algorithms need to pause parallelisation
  while the main process clusters the live points into a region.
  UltraNest also distributes this step.
  UltraNest does not discard additional newly discovered live points in other processes,
  and allows step samplers to resume when the likelihood threshold is raised.
  This saves model evaluations.

* UltraNest gives more helpful error messages when the likelihood has a bug.

* UltraNest has a visualisation to observe the current live points.
  This allows interrupting the run when the model seems wrong,
  without needing to wait for the full results.

Algorithmic speed (number of likelihood evaluations):

* For problems with an ellipsoid likelihood shapes, the algorithms of MultiNest and UltraNest are equally fast.
  For more complicated problems with up to 3 parameters, UltraNest is typically faster,
  while for higher dimensional problems, it depends.
  Sometimes the MultiNest algorithm requires fewer model evaluations by factors of several.
  However, the above safety caveats apply -- UltraNest favors correctness over speed.

Computational speed:

* nestle is a very small library implemented in pure python, which is fast
  for single-processors. dynesty is also pure python, but
  substantially slower due to design choices and depth of the call stack.
  UltraNest tries to keep the call stack shallow, and uses Cython to
  accelerate some portions of the code, making it on average faster than dynesty.

* UltraNest allows the user to define vectorized likelihood functions.
  This reduces the number of python function calls, making it much faster.
  UltraNest also supports this for the C/C++/Fortran/Julia languages.

* Paired with MPI parallelisation.

Distribution differences:

* MultiNest uses a custom licence which is not open source by OSI standards.
  UltraNest is free and open source software.

* MultiNest, being written in Fortran, requires manual compilation.
  UltraNest, nestle and dynesty can be installed using pip and conda.
  This allows easy integration as dependency into other packages.

What is the difference between UltraNest and PolyChord?
-------------------------------------------------------

* When UltraNest is run with a slice stepsampler, it is very similar to PolyChord.

* UltraNest supports combining of region rejection and random walking,
  which avoids unnecessary model evaluations.

* UltraNest also allows running with MLFriends first, and resuming with random walking.

* UltraNest allows auto-tuning of the number of steps during the run.
  In PolyChord this parameter has to be hand-tuned.

* PolyChord is not open-source, but uses a custom licence.
  UltraNest is free and open source software.

What is the difference between UltraNest and dyPolyChord/dynesty?
-----------------------------------------------------------------

First, see the differences to PolyChord and dynesty above:
`What is the difference between UltraNest and MultiNest/dynesty/...`,
`What is the difference between UltraNest and PolyChord?`.

Here, the different methods to vary the live points is discussed.

The dynamic nested sampling implemented in DyPolyChord/dynesty is
a heuristic optimization with a fudge-factor to balance improving
the posterior samples or the integral.

UltraNest implements a more general and rigorous approach to varying the number
of live points (tree search view, Buchner et al., in prep).
The number of live points can be tuned to increase towards multiple,
independent goals, including the integral accuracy,
posterior weight accuracy, reliability to subsampling (KL).

UltraNest is implementing in this way both 
`nested sampling and sequential Monte Carlo <https://arxiv.org/abs/1805.03924>`_.

UltraNest also allows the number of live points to increase when
clusters are detected. This is not supported in dynesty.
PolyChord (and MultiNest) splits the nested sampling runs into completely independent runs,
however this step is not well-understood in the literature.

What is the difference between UltraNest and emcee
--------------------------------------------------

emcee can work okay if the posterior is a mono-modal, multi-variate gaussian.
UltraNest handles a wider range of problems. This includes
multiple solutions/modes, non-linear correlation among parameters and posteriors with heavy or light tails.

emcee requires MCMC convergence checks which are tricky to get correct.

How should I cite UltraNest?
------------------------------

The main algorithm (MLFriends) is described in:

* Buchner, J. (2014): `A statistical test for Nested Sampling algorithms <https://arxiv.org/abs/1407.5459>`_ (`bibtex <https://ui.adsabs.harvard.edu/abs/2016S%26C....26..383B/exportcitation>`__)
* Buchner, J. (2019): `Collaborative Nested Sampling: Big Data versus Complex Physical Models <https://arxiv.org/abs/1707.04476>`_ (`bibtex <https://ui.adsabs.harvard.edu/abs/2019PASP..131j8005B/exportcitation>`__)

The UltraNest software package is presented in:

* Buchner, J. (2021): `UltraNest -- a robust, general purpose Bayesian inference engine <https://arxiv.org/abs/2101.09604>`_ (`bibtex <https://ui.adsabs.harvard.edu/abs/2021arXiv210109604B/exportcitation>`__)

So it is appropriate to write something like::

    We derive posterior probability distributions and the Bayesian
    evidence with the nested sampling Monte Carlo algorithm
    MLFriends (Buchner, 2014; 2019) using the
    UltraNest\footnote{\url{https://johannesbuchner.github.io/UltraNest/}} package (Buchner 2021).

If you use the corner plot, also cite ``corner``.
If you use the trace or run plot, also cite ``dynesty``.

Below are references for nested sampling in general. 

* Skilling, J. (2004): `Nested sampling <https://aip.scitation.org/doi/abs/10.1063/1.1835238>`_ (`bibtex <https://scholar.googleusercontent.com/scholar.bib?q=info:GmwWqzssMkkJ:scholar.google.com/&output=citation&scisdr=CgXYiBeiEMaTm_3KeVQ:AAGBfm0AAAAAYH7MYVQr0IWk3cGY_rySOzhvz51rrDuz&scisig=AAGBfm0AAAAAYH7MYdBK9Zj-2qYmMSqs5Fz3rlc0G5Px&scisf=4&ct=citation&cd=-1&hl=de>`__)
* Buchner, J. (2021): `Nested sampling methods <https://arxiv.org/abs/2101.09675>`_ (`bibtex <https://ui.adsabs.harvard.edu/abs/2021arXiv210109675B/exportcitation>`__)

These are useful when referring to the algorithm framework, 
if you want to discuss its benefits in general (for example, the global parameter space exploration,
dealing well with multiple modes). The second also contrasts
different implementations.
