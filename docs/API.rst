API
===

`Full API documentation on one page <ultranest.html>`_

The main interface is :py:class:`ultranest.integrator.ReactiveNestedSampler`, 
also available as `ultranest.ReactiveNestedSampler`.


Modules commonly used directly:
--------------------------------------------------------------------------------

 * :py:mod:`ultranest.integrator`: Nested sampling integrators
 * :py:mod:`ultranest.plot`: Plotting utilities
 * :py:mod:`ultranest.stepsampler`: MCMC-like step sampling
 * :py:mod:`ultranest.popstepsampler`: Vectorized step samplers
 * :py:mod:`ultranest.calibrator`: Calibration of step sampler
 * :py:mod:`ultranest.solvecompat`: Drop-in replacement for pymultinest.solve.
 * :py:mod:`ultranest.hotstart`: Warm start

Internally used modules:
--------------------------------------------------------------------------------

 * :py:mod:`ultranest.mlfriends`: Region construction methods
 * :py:mod:`ultranest.netiter`: Graph-based nested sampling
 * :py:mod:`ultranest.ordertest`: U test for a uniform distribution of integers
 * :py:mod:`ultranest.stepfuncs`: Efficient helper functions for vectorized step-samplers
 * :py:mod:`ultranest.store`: Storage for nested sampling points
 * :py:mod:`ultranest.viz`: Live point visualisations

Experimental modules, no guarantees:
--------------------------------------------------------------------------------

 * :py:mod:`ultranest.dychmc`: Constrained Hamiltanean Monte Carlo step sampling.
 * :py:mod:`ultranest.dyhmc`: Experimental constrained Hamiltanean Monte Carlo step sampling
 * :py:mod:`ultranest.flatnuts`: FLATNUTS is a implementation of No-U-turn sampler 
 * :py:mod:`ultranest.pathsampler`: MCMC-like step sampling on a trajectory
 * :py:mod:`ultranest.samplingpath`: Sparsely sampled, virtual sampling path.


Alphabetical list of submodules
-------------------------------

.. toctree::
   :maxdepth: 2

   ultranest


