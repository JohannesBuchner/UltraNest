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
