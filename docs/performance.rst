.. _performance:

====================================
Features
====================================


This tutorial demonstrates:

* How to make a program that uses nested sampling
* How to store and resume runs
* The meaning of the output files
* How to use UltraNest in 100 dimensions
* How to speed up likelihood functions with vectorization
* How to write a program with UltraNest
* How to execute on multiple cores

Usage in a program
===================

Lets write a script simple.py. It defines a problem through prior and likelihood functions,
and analyses it.

To understand it, have a look first the `Basic usage <using-ultranest.html>`_ page.

.. literalinclude:: simple.py
   :language: python3

Running this, you should see outputs like:

.. code-block:: none

        Mono-modal Volume: ~exp(-9.21) * Expected Volume: exp(-4.83)

        param1:      +0.0|            +0.3  ***************  +0.5                         |     +1.0
        param2:      +0.0|                   +0.4  ***************  +0.6                  |     +1.0
        param3:      +0.0|                         +0.5  ***************  +0.7            |     +1.0

        Z=-0.2(75.84%) | Like=3.59..4.14 [-37.4805..3.8990] | it/evals=2080/3730 eff=62.4625% N=400 

This reports

1. the current sampling region size and how many clusters it found.
2. for each parameter, a linear plot where the live points (shown as \*) are currently exploring.
3. Current estimate of logz, percentage of completion, live point log-likelihood values, 
   log-likelihood value range being targeted at the moment
4. The current iteration, how many likelihood function evaluations have been performed,
   the current efficiency and the number of current live points.
   
At the end, you should see:

.. code-block:: none

        [ultranest] [INFO] Explored until L=4  ..4.0913]*| it/evals=3440/6029 eff=61.1121% N=400 
        [ultranest] [INFO] Likelihood function evaluations: 6077
        [ultranest] [INFO] Writing samples and results to disk ...
        [ultranest] [INFO] Writing samples and results to disk ... done
        [ultranest] [INFO]   logZ = 0.0331 +- 0.0725
        [ultranest] [INFO] Effective samples strategy satisfied (ESS = 1875.4, need >400)
        [ultranest] [INFO] Posterior uncertainty strategy is satisfied (KL: 0.47+-0.06 nat, need <0.50 nat)
        [ultranest] [INFO] Evidency uncertainty strategy is satisfied (dlogz=0.16, need <0.5)
        [ultranest] [INFO]   logZ error budget: single: 0.08 bs:0.07 tail:0.01 total:0.07 required:<0.50
        [ultranest] [INFO] done iterating.

This indicates that all three strategies are satisfied and no further 
improvements are needed.

sampler.print_results() gives a brief summary of logz and its uncertainties,
and the parameter constraints:

.. code-block:: none

        logZ = 0.042 +- 0.101
          single instance: logZ = 0.042 +- 0.081
          bootstrapped   : logZ = 0.033 +- 0.101
          tail           : logZ = +- 0.010

            param1              0.40 +- 0.10
            param2              0.500 +- 0.099
            param3              0.602 +- 0.098

Some features worth noting here:

* UltraNest shows what it is currently exploring. This is especially useful for debugging models.
* Key diagnostic plots are included in the output folder (see below).
* The program can resume from crashes -- even if run with a different number of live points.

Output files
============

If a `log_dir` directory was specified, you will find these files:

* debug.log: A debug log of the run

  * Please attach it or the stdout output when you open a `Github issue <https://github.com/JohannesBuchner/UltraNest/issues>`_.
  * This contains the efficiency and progress of the sampling.

* info folder: machine-readable summaries of the posterior

  * **post_summary.csv**: for each parameter: mean, std, median, upper and lower 1 sigma error. Can be read with `pandas.read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`_.
  * **results.json**: Contains detailed output of the nested sampling run. Can be read with `json.load <https://docs.python.org/3/library/json.html>`_.

    * paramnames: parameter names
    * ncall, niter: Number of likelihood calls, nested sampling iterations
    * maximum_likelihood: highest loglikelihood point found so far
    * H, Herr: (global) information gain
    * ess: effective sample size
    * logz, logzerr: ln(Z) and its uncertainty. logzerr_tail is the remainder integral contribution, logzerr_bs is from bootstrapping
    * posterior: for each parameter: mean, std, median, upper and lower 1 sigma error, and `information gain <https://arxiv.org/abs/2205.00009>`_.
    * insertion_order_MWW_test: MWW test results (see Buchner+21 in prep)

* chains: machine-readable chains

  * **equal_weighted_post.txt**: equally weighted posterior samples (similar to a Markov chain). Each column corresponds to one parameter.

    * You can make a corner plot from this.

  * weighted_post.txt: posterior samples with a weight attached. 

    * This is made by nested sampling directly, and the above is produced from this. However, carrying the weights around is cumbersome.
    * getdist compatible. columns are Weight, -loglikelihood, parameter value (d times). 

  * weighted_post_untransformed.txt: same as above, but in coordinates before the prior transformation.
  * run.txt: for each iteration, ln(z) and error, ln(volume), number of live points, log-likelihood threshold, posterior point weight (likelihood x volume) and insertion rank of newly sampled point.

* plots: Visualisations (by plot functions)

  * corner.pdf: corner/pairs plot of the marginal and conditional parameter posteriors.

    * Useful for investigating degeneracies and which parameters were learned.

  * trace.pdf: diagnostic plot showing problem structure

    * Visualises how each parameter's range was reduced as the nested sampling proceeds.
    * Color indicates where the bulk of the posterior lies.
    * Useful to understand the structure of the inference problem, and which parameters are learned first.

  * run.pdf: diagnostic plot showing integration progress

    * Visualises how the number of live points, likelihood and posterior weight evolved through the nested sampling run.
    * Visualises the evidence integration and its uncertainty.

All of the above can be written, but are never read, by ultranest.ReactiveNestedSampler. The only file used to
read the state of a previous run is:

* results/points.hdf5: file storing all sampled points. Used for resuming.

  * this is an internal file.
  * ncalls: number of likelihood calls
  * points: the columns are: likelihood threshold under which the point was sampled, likelihood of the point, a quality indicator (0 for MLFriends, otherwise the number of steps in the step sampler), u-space (unit cube) coordinates, p-space (transformed parameters) coordinates.

You can safely store additional files and plots in the sub-folders.

Speed ups
===========

Lets go to some more advanced usage examples: Integrating a 100-dimensional gaussian.
For that, we have to make a few modifications to enhance the
**computational speed**. Enhancing the **algorithmic speed** (number of likelihood evaluations
needed per iterations) is discussed in the next section.

Implementing a gaussian likelihood can be done in a few ways.

Very slow:

.. code-block:: python3

    def loglike(theta):
        return scipy.stats.norm(centers, sigma).logpdf(theta).sum()

Creating scipy.stats random variable object is slow. 
Calling the functions directly is better:

.. code-block:: python3

    def loglike(theta):
        return scipy.stats.norm.logpdf(theta, centers, sigma).sum()

We can improve further by implementing the function ourself:

.. code-block:: python3

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum() - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
        return like

Finally, we can make a vectorized function, which can process *many* points at the same time. 
This reduces function calls.

.. code-block:: python3

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
        return like

To use this function, pass ``vectorized=True`` to ReactiveNestedSampler.
Lets see how this looks like in a full program.

Vectorized full program
------------------------

Below is a Python program that implements a gaussian likelihood,
and allows the user to specify the problem dimension and a few sampler parameters.

.. literalinclude:: gauss.py
   :language: python3

Note that our likelihood is vectorized, and we pass ``vectorized=True``.

A similar program is included in the git repository as *examples/testasymgauss.py*.


High-dimensional models
========================

In high-dimensional spaces, MLFriends by itself is inefficient, so
we have to use a step sampling technique. There are several implemented
in :mod:`ultranest.stepsampler`. Here we are using slice sampling
that learns the direction from the existing live points.
This is similar to PolyChord, except the region is also used to reject
distant proposals, and the clustering is better justified (based on MLFriends).

Lets run our program on a 100-dimensional gauss:

.. code-block:: bash

        python3 gauss.py --x_dim=100 --num_live_points=400 --slice  --slice_steps=100


After a while (a few hours on my laptop), this will have traversed the parameter space::

        Z=0.3(43.44%) | Like=89.39..96.29 [89.3916..89.3936]*| it/evals=39671/11660719 eff=0.2939% N=400 

        param1  :      +0.0|                        +0.2  0 0000  0100000000000000000000100010001000010100010000000010000 0 0  00 0       0  +0.8                 |     +1.0
        param2  :      +0.0|                                      +0.3  0   0  0  001 0 00000100000100010110001010000010000000000000000 00 0   00 00  +0.9        |     +1.0
        param3  :      +0.0|                                                   +0.4  00 0  0  000001010000100000010000010001010000100000000000000000000000 0 0 0  |     +1.0
        param4  :      +0.0|                                                     +0.4  0  0   0  000000 001 0000000000100000010001100000000010010000000000 000  00|     +1.0
        param5  :      +0.0|                                                      +0.5  0 0 000000000000 00000000000100000100011001000101100000000000000000 0000  |     +1.0
        param6  :      +0.0|                                               +0.4  0 0  0000000 000000010110000000110001100100000000000100000000    0 0   0         |     +1.0
        param7  :      +0.0|                               +0.3  00 0 00  00000000000000010000000000110010010100000000010000000 0 0  0    0  +0.8                 |     +1.0
        param8  :      +0.0|          +0.1  0     0  100 00101000000000000000010010000010110000000000010000000  0 0   0   0  +0.7                                 |     +1.0
        param9  :     +0.00|     0   000  00000000000000100000100011100010010010000000100000000000000 00   0  +0.59                                               |    +1.00
        param10 :     +0.00|   00000000000001010100000100100101010000000000000100000000  00 000  +0.50                                                            |    +1.00
        param11 :     +0.00| 0 0 00000000 0000010000000101001000000101000001010000000000 0 000 0  +0.51                                                           |    +1.00
        param12 :     +0.00|    0 0   00 0000 00000100000000000010000010110010000001001000000000 000 00  0 0 00  +0.61                                            |    +1.00
        param13 :      +0.0|               +0.2  0 00  0 0000000000000010010000000110000110000000000010001100 000 00  0       0  +0.7                             |     +1.0
        param14 :      +0.0|                             +0.3  0  0    0   00000000000000001100000001010001001000010001000000000  001000      0   0  +0.9         |     +1.0
        param15 :      +0.0|                                                   +0.4  00000000000000010011000000000010010000000010000100010000 00 1 0000  +0.9     |     +1.0
        param16 :      +0.0|                                                  +0.4  01        000 010 00000000010001100010000000000000000000000100010000000 00    |     +1.0
        param17 :      +0.0|                                                    +0.4  0     0000 0 00000000000000010000000011000110000111000000000000000000000 0 0|     +1.0
        param18 :      +0.0|                                                    +0.4  01 0  000 0000000010100000000100100010001000100000100000000000 000    0 0   |     +1.0
        param19 :      +0.0|                 +0.2  0            0    0  0 000 0000000010101100000001101100000000000000000000000 00000000 000   0  +0.9            |     +1.0
        param20 :      +0.0|            +0.1  0          0   000000  0001000000000100000010000000010100000011001000000 00000010  +0.7                             |     +1.0
        param21 :     +0.00|       0             010 00000000000000000110010000100000110000010000  0000000000  1  +0.63                                           |    +1.00
        param22 :     +0.00|  00 0 0000001 00000100000010001110110100000000000000000000000000000 00      0  +0.58                                                 |    +1.00
        param23 :    +0.000| 0 0000000000000000000001000110000010011000110000000000000 0 00 0     0        0  +0.597                                              |   +1.000
        param24 :    +0.000|0  00000 00001000000010000000010000000001000001001011000010000     000       0  +0.577                                                |   +1.000
        param25 :     +0.00|      0        0 1    0000000010000000000010100000000010001011010000000000000  0   0 0  +0.64                                         |    +1.00
        param26 :      +0.0|                     +0.2  00   00 00000010010000000000000010100010100001010000100000000000000 0000  +0.7                             |     +1.0
        param27 :      +0.0|                                         +0.4  0 1   0 0100000001000000000000011001000100000010000000000001 00  00  +0.9              |     +1.0
        param28 :      +0.0|                                                 +0.4  0 00     00 01 00000000100000000000001100010000000000011001000000   0  01      |     +1.0
        param29 :      +0.0|                                                          +0.5  1  0000000 0  000000000000001110001000000100000100010001000000000 00 0|     +1.0
        param30 :      +0.0|                                                    +0.4  0 0   0  00 000000100000000010000001000001001100010000000001000100 0     0  |     +1.0
        param31 :      +0.0|                                              +0.4  0 00  00  00000000000000100001000001000111010000000010000000000000               0|     +1.0
        param32 :      +0.0|                              +0.3  0 000 0000100000000000000000101101000010000000001010000000 0000000   0  +0.8                      |     +1.0
        param33 :      +0.0|                 +0.2  0 00  000000010010100000010000000001000000000000000101001000000000       0  +0.7                               |     +1.0
        param34 :     +0.00|      0 0 0 00 00000000000010000100000100000000010001000011000000 0 0 0 1 000   0   0     0  +0.67                                    |    +1.00
        param35 :   +0.0000|0  0 0000000000001010000000000100000100000001101000000000000 01000 0  +0.5073                                                         |  +1.0000
        param36 :    +0.000| 0 000110000000000001000110000100100000100010000000000000000000  00 0  +0.508                                                         |   +1.000
        param37 :    +0.000|0   0   00 00000101 00000010001000000001010010000000010000000000000100    0000  +0.579                                                |   +1.000
        param38 :      +0.0|           +0.1  0     0 0000 0 0000000000010000000100100000000000100100000010000001 00    0  +0.7                                    |     +1.0
        param39 :      +0.0|                             +0.3  0  0000 000010 0000000100000100000001011000010000000010001000 000000 0      0  +0.8                |     +1.0
        param40 :      +0.0|                                              +0.4  0    0 0 00000 00010000000100100010000001110010000000000000000000 000 0 0         |     +1.0
        param41 :      +0.0|                                                     +0.4  0  0 00000000 100000100101010000000000100000000110000000100000000000 0   00|     +1.0
        param42 :      +0.0|                                                       +0.5  0 00 00 0  00 0000000100001000100001000000010101101000000000000 000   000|     +1.0
        param43 :      +0.0|                                                     +0.4  0    000000000010000110000000000100001100001000100000000000000  0000   0 0 |     +1.0
        param44 :      +0.0|                                   +0.3  0      0   0 000000110000001000010100000000110001000010000000000000000000          00        |     +1.0
        param45 :      +0.0|                +0.2  0        000   000 0 0000001010000000001000101000000010001100000000000000        0  +0.8                        |     +1.0
        param46 :    +0.000|0          0   0 000 00 0001 000000111000000010011000000000000000010000000 00 0   00  +0.625                                          |   +1.000
        param47 :     +0.00|        0 00 0010000000000000000101010000100000000000100100000100000000  +0.53                                                        |    +1.00
        param48 :     +0.00|    00 00 0001000000000000110000101101000000001000000 00000  0  0          0  +0.56                                                   |    +1.00
        param49 :    +0.000| 0     100010000000000000000010100110000010000000000001000000 0010   00  +0.527                                                       |   +1.000
        param50 :     +0.00|      0 0 0       000000000000010000101100001100000100000000000000010000000 10 00   0  +0.63                                          |    +1.00
        param51 :      +0.0|                  +0.2  0 0 0000 0 0000000100000000010000011000110001000001000000000 000000    0  +0.7                                |     +1.0
        param52 :      +0.0|                                      +0.3  0 00000000100000010010010000000000001000101010000000000000  0 00000  +0.8                 |     +1.0
        param53 :      +0.0|                                                   +0.4  0    010 000000000100000000101011000000000010100000000000000000000 0  0   0  |     +1.0
        param54 :      +0.0|                                                       +0.5  0    00  0  000000011000000000001011000110000000000000000000100000 000000|     +1.0
        param55 :      +0.0|                                              +0.4  0         0   0000000000 01000000000000010110100100100000000001000000 0 000  0    |     +1.0
        param56 :      +0.0|                                        +0.3  0     0   00  00 010000010010100000000001100110100000000000000000000000000  00 00   0   |     +1.0
        param57 :      +0.0|                                     +0.3  0000100000000000000111000000010010000000110000000000100000000 0 00 0     0  +0.9           |     +1.0
        param58 :      +0.0|               +0.2  0        0 0 0000000000000010000011010010011000000000010000000 000 00 0   00  0  +0.7                            |     +1.0
        param59 :     +0.00|         00   00 00000000000110000001000000001100000001001000100000000000 00001 0     0  +0.64                                        |    +1.00
        param60 :     +0.00|  0  00 0000 0 000000100110111000000000010000001000010000000 0000  0  +0.50                                                           |    +1.00
        param61 :    +0.000| 000000000000000000001010000000000010010010010010000000000000 00 0 0  +0.507                                                          |   +1.000
        param62 :    +0.000|0       00  100000010010000001000010000000001000000000000101000000 1 0 0                     0  +0.700                                |   +1.000
        param63 :      +0.0|         +0.1  0   0000     000000000000000000000000100101011011000000000000100 00 00  0  0   0  +0.7                                 |     +1.0
        param64 :      +0.0|                                +0.3  0  00 000000000000000000110000010110000100001000000001 01 000000       0  +0.8                  |     +1.0
        param65 :      +0.0|                                       +0.3  0       0101 0000000000001000001000000100001000000100001000000000000 0  00 0  00         |     +1.0
        param66 :      +0.0|                                                       +0.5  0 000000 0 00000000100000001000000100100000000000000100000 0100000   00  |     +1.0
        param67 :      +0.0|                                                            +0.5  00  0000  0000001001000000000011000000100000010000000001100000 01   |     +1.0
        param68 :      +0.0|                                                       +0.5  0  000 0000000000000000100010100000100001101001000010000000 0 00  00 0   |     +1.0
        param69 :      +0.0|                                       +0.3  0    00 0   00 00100000000000000001000000001101000100001000010000000000 0         0      |     +1.0
        param70 :      +0.0|                              +0.3  0   0 000000010 10000010100000000011000010110000000000000 000  000  +0.8                          |     +1.0
        param71 :      +0.0|       +0.1  0 0000 0 0  0 1010 000000000000000110000100100010100000000001 00000 00 0  0 0  +0.7                                      |     +1.0
        param72 :     +0.00|    0  000 00000100000000001000110001000000010100000000001000000 0000  0  0  +0.55                                                    |    +1.00
        param73 :    +0.000|0 00000000000000000000010010010000011010000010000000000000 000 000      0  +0.540                                                     |   +1.000
        param74 :     +0.00|  0 000  000 0000100010000001000000101010000000001010001000000000 0 0 00     0   0  +0.61                                             |    +1.00
        param75 :     +0.00|         0  0000010  00001000100010001010001010000000000000000000010000 000   0     0    0  +0.67                                     |    +1.00
        param76 :      +0.0|          +0.1  0     0   0 0  00000000100100000000010110100000010010000100000000000 000000  +0.7                                     |     +1.0
        param77 :      +0.0|                                       +0.3  000      001000100000010000000100000010000010001000000000 0 00  0  0  +0.8               |     +1.0
        param78 :      +0.0|                                      +0.3  0  0 0    0   0  0 00000000010000000000011000001101010000010000000000000000000 0  +0.9    |     +1.0
        param79 :      +0.0|                                                     +0.4  0  00  00 0100 01100000000000000001000000000100010000000010001000 0 00 0 1 |     +1.0
        param80 :      +0.0|                                                          +0.5  00   000000000010001011000000000000001010100100000000000000010000   0 |     +1.0
        param81 :      +0.0|                                               +0.4  0 0   0000000  0000000000100001010001100000100000000000000000 11000 0000         |     +1.0
        param82 :      +0.0|                                      +0.3  0 00  0000000000001010010000001000100000001000000000000010000 0   1 00  +0.9              |     +1.0
        param83 :      +0.0|                +0.2  0   0 000 000000000000000001010100010100100000000000000000 01 001000  000  0     0  +0.8                        |     +1.0
        param84 :     +0.00|  0            0 00  00101000000000001000000001000011000100000000001100  0000 000  0    0  +0.66                                      |    +1.00
        param85 :    +0.000|0  00000000000000100010001001001010000000000010010000000000 0 0  00    0 0         0  +0.622                                          |   +1.000
        param86 :    +0.000|000 10 0000000100100000000000100001010100000100000000100000000 0 00       0  +0.556                                                   |   +1.000
        param87 :     +0.00|  0    0 000 0001000000000000001111101100000010000000000000000000  00000 0   0  +0.58                                                 |    +1.00
        param88 :      +0.0|          +0.1  0  0 00000000000000000110000000011101010001000000000000 000000   00 0  0  +0.7                                        |     +1.0
        param89 :      +0.0|                           +0.2  0000 0000000000000000001000100001111010000000000001000000000 00  0 0  0 00        0  +0.9            |     +1.0
        param90 :      +0.0|                                         +0.4  00   0  00 0000 0000000010100000000101000110000000000100000110000000   0 0  +0.9       |     +1.0
        param91 :      +0.0|                                         +0.4  0            0   0 0000000000110000101000000110000001000000000000000000000000 0   0 0  |     +1.0
        param92 :      +0.0|                                                         +0.5  000 00 000000 0000000000001001010000000000011100000010001000 00 0   10 |     +1.0
        param93 :      +0.0|                                                              +0.5  00 000 00000001000001000010000101100000000001001010000000000      |     +1.0
        param94 :      +0.0|                                              +0.4  0 00 0010000000010000000010100000100101011000000000000000000  0 000      0        |     +1.0
        param95 :      +0.0|                          +0.2  0 0   00   100010000000000000001000011000000101010000000100 000 00 000 0   0           0  +0.9        |     +1.0
        param96 :      +0.0|         +0.1  0    00  00000000100000000000000001000011000101010010000000000000000 00       0  +0.7                                  |     +1.0
        param97 :    +0.000|0     0 0  0 000010000100100010001001000000000000000000100000000000001000010 0  +0.580                                                |   +1.000
        param98 :     +0.00| 0   00  0010000000001000000011000010000101000000100000 000000 0  00 0  +0.52                                                         |    +1.00
        param99 :     +0.00|  0   00000000000000000100001001011001100100000000000000 000000 0  00 0 00000  +0.57                                                  |    +1.00
        param100:     +0.00|  0      0 0 01  0000000000100000010000010100010100000001000100000000000 00 0     0       0  +0.67                                    |    +1.00


The integral is given as::

        logZ = 1.043 +- 0.846
          single instance: logZ = 1.043 +- 0.458
          bootstrapped   : logZ = 1.084 +- 0.743
          tail           : logZ = +- 0.405

This result is close to the analytic value (0) on infinite bounds 
(the prior boundaries slightly increase the result).

We can test whether the slice sampler is good enough by doubling 
the number of steps, until the ln(Z) estimate is stable.

Parallelisation
====================

Your likelihood function may already be using multiple cores,
whether your intended to or not, due to underlying libraries (e.g., numpy).
You can control this with the OMP_NUM_THREADS environment variable:

.. code-block:: bash

        # avoid automatic parallelisation
        export OMP_NUM_THREADS=1

If the likelihood is not parallelised, ultranest can parallelize
its execution to multiple cores.

Using multiple cores
--------------------

To use multiple processors and cores, scaling UltraNest all the way to 
large computing clusters, you can parallelise the program with MPI:

* No code changes are required. 
* You need to install MPI (for example, OpenMPI) and mpi4py (pip install mpi4py).
* Then run your script with mpiexec:

.. code-block:: bash
        
        mpiexec -np 4 python3 gauss.py --x_dim=100 --num_live_points=400 --slice  --slice_steps=100


This launches four scripts which are started in parallel, and ultranest 
coordinates them.

Use as many scripts as processors. If memory is a concern, look into shared memory solutions.

GPU-acceleration
====================

Some models today use probabilistic programming languages, such as JAX,
which allows fast model evaluations on GPUs and CPUs.

UltraNest supports such models with vectorization (see above).

For high-dimensional, cheap, vectorized models, the 
:py:mod:`popstepsampler` implements vectorized versions.

More features
===================

To find more features and details such as ...

* Circular/wrapped parameter spaces
* Model comparison of empirical and physical models
* Quantifying posterior uncertainty
* Visualisation and interoperation with getdist, pandas, matplotlib, ...
* Using in a Jupyter notebook
* all the step samplers and slice samplers available

... see the tutorials!
