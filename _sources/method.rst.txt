=============
Method
=============

UltraNest combines several methods to improve nested sampling, 
and will be described in detail in a future paper. The key elements are:

1. constrained-likelihood sampling algorithm

   For sampling a new, independent point, primarily implements MLFriends.
   MLFriends places ellipsoids around each live point and draws from them.
   The ellipsoid shape is determined by the live point distribution (Mahalanobis distance), 
   the ellipsoid size is determined by cross-validation.
   For high-dimensional inference, parameter-free algorithms such as HARM and DyCHMC are available.

2. nested sampler

   This component manages the live point population, and can add or remove points.
   UltraNest supports several advanced strategies for injecting more points 
   when needed to populated clusters, to reach the effective sample size,
   integration uncertainty or posterior accuracy.

3. nested integrator

   This component assigns weights to the sampled points. UltraNest implements
   a bootstrapping scheme that simulates other runs with fewer live points.
   This gives robust and realistic uncertainties.

4. termination criterion

   In principle, termination can occur when the dead, removed points have more
   weight than the live points. However, in practice, very small peaks can be discovered
   on relatively flat likelihood surfaces, 
   for example when fitting multiple components to a data series.
   Therefore, UltraNest integrates until the live point weights are insignificant (*frac_remain=0.01*).
   For noisy likelihoods, termination when the live points are within a tolerance of *Lepsilon* can be requested.


Visualisation
--------------

The animation below gives an idea how the algorithm proceeds.
Choose different 2d target distributions to explore, and compare to a 
simple MCMC algorithm.

.. raw:: html

	<iframe src="_static/mcmc-demo/app.html" style="width:100%; height:400px;"></iframe>


1. Initially, points are drawn from the entire parameter space (green) according to the prior.
2. The lowest of these live point (worst fit) is removed, and a better one sought.
   At each removal, the volume sampled by the live point shrinks by a constant
   factor.
3. To still sample from the prior, MLFriends creates ellipsoids around all 
   live points and samples from them. The ellipsoid size is determined 
   by bootstrapping: Some points are randomly left out, and the ellipsoids
   have to be large enough so that they could have been sampled. This is
   repeated several times. In UltraNest, the ellipsoid shape is learnt
   as well.
4. Nested sampling proceeds to the peak, keeping track of the likelihood.
   The volume becomes smaller and smaller. At some point, the remainder
   does not contribute any probability mass, and the exploration is finished.
5. The removed points are weighted by their likelihood and the volume they 
   represent. These are the posterior samples (histograms).

The sampling can become inefficient, for example for high-dimensional
problems. UltraNest provides MCMC-based methods to find a new point.

The animation is based on work by Chi Feng https://chi-feng.github.io/mcmc-demo/
and is MIT licenced. The RadFriends implementation was contributed by Johannes Buchner.


Literature
------------

On the theory behind nested sampling:

* Skilling, J. (2004): Nested sampling
* Chopin, N. & Robert, C. (2008): Properties of Nested Sampling
* Evans, M. (2007): Discussion of nested sampling for Bayesian computations by John Skilling 
* Skilling, J. (2009): Nested sampling's convergence
* Walter, C. (2014): Point Process-based Monte Carlo estimation

For an introduction of constrained-likelihood prior sampling methods and verification:

* Buchner, J. (2014): A statistical test for Nested Sampling algorithms
* Higson, E.; Handley, W.; Hobson, M. & Lasenby, A. (2019) NESTCHECK: diagnostic tests for nested sampling calculations
* Buchner, J. (2021): Nested Sampling Methods

On analysing many data sets:

* Buchner, J. (2019): Collaborative Nested Sampling: Big Data versus Complex Physical Models


Code Concepts
==============

This content gives some hints for those who want to dive into the codebase.

Spaces
---------------

UltraNest maps four spaces:

* u: The unit cube space
* t: A affine projection of the unit cube space (called transformLayer)
* p: The physical parameter space
* L/logl: Likelihood values

The transformations are made as follows:

* unit cube <-> transformLayer space: region.transformLayer.transform() and untransform()
* unit cube -> physical parameter space: user-provided prior transform function
* physical parameter space -> likelihood values: user-provided likelihood function

Nested sampler & integrator
---------------------------

A tree-based concept envisions the full prior volume as the root of a tree.
Branches indicate divisions. The first children of the tree root are 
sampled directly from the prior, and are called *roots* in the code (although
they are in reality the first branches).

The nested sampling algorithm is a breadth-first search of this tree,
with nodes expanded as needed. Replacements of a node by its children
is a nested sampling shrinkage. When children are added to nodes is
driven by software agents. These agents can optimize for various strategies.

Four strategies are implemented. They can be independently activated by the user:

* Obtain a logz error below a threshold (*dlogz*)
* Obtain a certain number of effective samples (*min_ess*)
* Keep a certain number of live points (width of tree) per cluster (*cluster_num_live_points*).
* Obtain a posterior uncertainty below a threshold (*dKL*).

The breadth-first search starts with the *roots*. Additionally,
alternative breadth-first searches with some *roots* left out 
(bootstrapping) is simulated simultaneously. The breadth-first search
computes logz and adds uncertainty contributions from:

* The limited shrinkage estimate (information H)
* The bootstraps (simulating alternative, repeated nested sampling runs)
* The noise in shrinkage estimates (by sampling it a binomial distribution in each bootstrap realisation)

The combination makes UltraNest's logz uncertainties very reliable.

Constructing MLFriends regions
-------------------------------

Robust likelihood-constrained prior sampling (LCPS) is achieved with the
parameter-free **MLFriends** algorithm.

MLFriends (Buchner 2019) is a substantial improvement upon the original RadFriends algorithm (2014 paper).
RadFriends obtains regions by leaving some points out, and testing how
large spheres around the live points have to be to be able to recover them.
This bootstrapping is repeated many times (bootstrapping), to be robust.
This way, RadFriends learns automatically about multiple modes.
RadFriends is slow, because it does not learn the relative sizes of different parameters,
nor the correlation between them. MLFriends (2019 paper) improves by obtaining the covariance
of the live points, and thus uses a Mahalanobis distance. 
This learned metric helps reduce the region, and accelerates sampling.

At the same time, MLFriends provides natural agglomerate clustering -- points within each others
ellipsoids belong to the same cluster.
UltraNest improves MLFriends further by iteratively 

1. compute MLFriends ellipsoid radius with a metric (euclidean initially)
2. Identify clusters with that radius
3. Shift clusters on top of each other and learn a new metric from the covariance.

This scheme iteratively improves the metric, and is robust when clusters appear (
otherwise the metric may be primarily measuring the distance between clusters).

Finally, UltraNest only accepts new regions when they reduce the size of the sampling space.
This is necessary because when a cluster disappears, it has few points and would
be merged back into the bulk, increasing the radius drastically. Requiring shrinkage
avoids this problem. By starting the bootstrapping always with the same random numbers,
over-shrinkage by repeated invocations is avoided.

UltraNest also adds a enveloping ellipsoid. This can help in 
mono-modal gaussian problems. Contrary to Mukherjee, Parkinson & Liddle (2006), 
the ellipsoid enlargement is learned by bootstrapping, and is thus robust and
well-justified. Both regions are used as filters.

Another improvement is that in MPI-parallelised runs, UltraNest 
distributes the RadFriends bootstraps. This makes clustering very fast
(MultiNest has to stop for clustering on the master node).

Sampling from regions
-------------------------------

UltraNest samples in four region sampling schemes:

* Sample from unit cube, filter with MLFriends & ellipsoid
* Sample from MLFriends points, filter with unit cube & ellipsoid
* Sample from MLFriends region bounding box, filter with unit cube & ellipsoid
* Sample from ellipsoid, filter with unit cube & MLFriends region

UltraNest switches between these methods when the current on becomes inefficient.
Efficiency here means being able to draw points, not whether they are
above the likelihood contour.

Transformed ellipsoid
-------------------------------

Additionally, a ellipsoid is also built in transformed space and used
for rejecting points. This is only done when the transform is non-linear.
The parameter transformation, perhaps being closer to the data constraints,
allows the ellipsoid to follow.

Users can thus tune the transform to improve the sampling efficiency.

For example:

* L-shapes: when two parameters are added with a log-uniform prior, it could be wise to transform them into linear space.
* funnel-shapes: in hierarchical models, the variance and means should be decorrelated -- providing the means as a fraction of the variance allows the ellipsoid to follow funnel-shapes.
