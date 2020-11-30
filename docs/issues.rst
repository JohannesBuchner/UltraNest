.. _faq:

===========================
Frequently Asked Questions
===========================

.. contents:: :local:

How can I add a question here?
------------------------------

See `Contributing <contributing.rst>`_ for how to report bugs and ask questions. (hint: open a github issue)

How do I suppress the output?
-----------------------------

To suppress the logging to stdout, you can configure your own logger::

    import logging
    logger = logging.getLogger("ultranest")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('[{}] [%(levelname)s] %(message)s'.format(module_name))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

You may want to alter the above to log to a file only. See the logging python module docs.

To suppress the live point visualisations, set ``viz_callback=False`` in ``sampler.run()``.

To suppress the status line, set ``show_status=False`` in ``sampler.run()``.

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

How do I use UltraNest with C / C++ / Fortran / Julia?
-------------------------------------------------------

Examples are available for C, C++, Fortran and Julia: https://github.com/JohannesBuchner/UltraNest/lang

These implement the same prior and likelihood functions. The
functions are vectorized to reduce the number of function calls.

The C, C++ and Fortran functions are compiled to a dynamic library,
which is used from Python.

The Julia example differs in that the code is run from the Julia
environment, calling Python which calls back Julia.


How should I cite UltraNest?
------------------------------

The main method (MLFriends) is described in:

* Buchner, J. (2014): `A statistical test for Nested Sampling algorithms <https://arxiv.org/abs/1407.5459>`_ (`bibtex <https://scholar.googleusercontent.com/scholar.bib?q=info:yTXfZFtI_vYJ:scholar.google.com/&output=citation&scisdr=CgXWBaaNEK7YkjRrxRc:AAGBfm0AAAAAX8Nu3Rcw9Cl4oM-O20oJrdlLRUQB2IhI&scisig=AAGBfm0AAAAAX8Nu3ftd-SBQKMLTBDogxM9U1n9uLKdi&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1>`__)
* Buchner, J. (2019): `Collaborative Nested Sampling: Big Data versus Complex Physical Models <https://arxiv.org/abs/1707.04476>`_ (`bibtex <https://ui.adsabs.harvard.edu/abs/2019PASP..131j8005B/exportcitation>`__)

So it is appropriate to write something like

.. code-block:: none

    We derive posterior probability distributions and the Bayesian
    evidence with the nested sampling Monte Carlo algorithm
    MLFriends (Buchner, 2014; 2019) using
    UltraNest\footnote{\url{https://johannesbuchner.github.io/UltraNest/}}.

If you use the corner plot, also cite ``corner``.
If you use the trace or run plot, also cite ``dynesty``.
