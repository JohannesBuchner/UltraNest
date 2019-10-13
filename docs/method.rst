=============
How it works
=============

UltraNest combines several methods, and will be described in detail 
in a future paper.

The animation below gives an idea how the algorithm proceeds through visualisation.
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

On analysing many data sets:

* Buchner, J. (2019): Collaborative Nested Sampling: Big Data versus Complex Physical Models


Code Concepts
---------------

This content gives some hints for those who want to dive into the codebase.

UltraNest maps four spaces:

* u: The unit cube space
* t: A affine projection of the unit cube space (called transformLayer)
* p: The physical parameter space
* L/logl: Likelihood values

The transformations are made as follows:

* unit cube <-> transformLayer space: region.transformLayer.transform() and untransform()
* unit cube -> physical parameter space: user-provided prior transform function
* physical parameter space -> likelihood values: user-provided likelihood function

The tree-based concept envisions the full prior volume as the root of a tree.
Branches indicate divisions. The first children of the tree root are 
sampled directly from the prior, and are called *roots* in the code (although
they are the first branches).

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

MLFriends (2019 paper) is a substantial improvement upon the original RadFriends algorithm (2014 paper).
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
1) using metric for getting MLFriends ellipsoid radius 
2) using radius to identify clusters.
3) shift clusters on top of each other and learn a new metric.
Another improvement is that in MPI-parallelised runs, UltraNest 
distributes the RadFriends bootstraps. This makes clustering very fast
(MultiNest has to stop for clustering on the master).

UltraNest also adds a enveloping ellipsoid. This can help in 
mono-modal gaussian problems. Contrary to Mukherjee, Parkinson & Liddle (2006), 
the ellipsoid enlargement is learned by bootstrapping, and is thus robust and
well-justified. Both regions are used as filters.
UltraNest samples in four region sampling schemes:

* Sample from unit cube, filter with MLFriends & ellipsoid
* Sample from MLFriends points, filter with unit cube & ellipsoid
* Sample from MLFriends region bounding box, filter with unit cube & ellipsoid
* Sample from ellipsoid, filter with unit cube & MLFriends region

UltraNest switches between these methods when the current on becomes inefficient.
Efficiency here means being able to draw points, not whether they are
above the likelihood contour.

