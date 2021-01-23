---
title: 'UltraNest - a robust, general purpose Bayesian inference engine'
tags:
  - Python
  - Bayesian inference
  - Nested Sampling
  - Monte Carlo
  - Bayes factors
authors:
  - name: Johannes Buchner
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2, 3, 4"
affiliations:
 - name: Max Planck Institute for Extraterrestrial Physics, Giessenbachstrasse, 85741 Garching, Germany. 
   index: 1
 - name: Millenium Institute of Astrophysics, Vicuña MacKenna 4860, 7820436 Macul, Santiago, Chile . . . 
   index: 2
 - name: Pontificia Universidad Católica de Chile, Instituto de Astrofísica, Casilla 306, Santiago 22, Chile. 
   index: 3
 - name: Excellence Cluster Universe, Boltzmannstr. 2, D-85748, Garching, Germany  
   index: 4

date: 21 January 2021
bibliography: paper.bib

---

# Summary

UltraNest is a general-purpose Bayesian inference package for parameter
estimation and model comparison. It allows fitting arbitrary models specified as
likelihood functions written in Python, C, C++, Fortran, Julia or R.
With a focus on correctness and speed (in that order),
UltraNest is especially useful for multi-modal or non-Gaussian parameter spaces,
computational expensive models, in robust pipelines. 
Parallelisation to computing clusters and resuming incomplete runs is available.

# Statement of need

When scientific models are compared to data, two tasks are important: 
1) contraining the model parameters and 2) comparing the model to other models. 
While several open source, Bayesian model fitting packages are available that 
can be easily tied to existing models,
they are difficult to run such that the result is reliable and user interaction is minimized.
A chicken-and-egg problem is that one does not know a priori the posterior distribution
of a given likelihood, prior and data set, and cannot chose a sampler that performs well.
For example, Markov Chain Monte Carlo convergence checks may suggest good results,
while in fact another distant but important posterior peak has remained unseen.
Current and upcoming large astronomical surveys require characterising 
a large number of highly diverse objects, which requires reliable analysis pipelines.
This is what UltraNest was developed for.

Nested sampling [NS, @Skilling2004] allows Bayesian inference on arbitrary user-defined likelihoods.
Additional to computing parameter posterior samples, 
it also estimates the marginal likelihood (“evidence”, $Z$).
Bayes factors between two competing models $B=Z_1/Z_2$ are 
a measure of the relative prediction parsimony of the models, 
and form the basis of Bayesian model comparison.
By performing a global scan of the parameter space from the 
worst to best fits to the data, NS performs well also in multi-modal settings.

In the last decade, several variants of NS have been developed. 
These include (1) how better and better fits are found while 
respecting the priors,
(2) whether it is allowed to go back to worse fits and explore the parameter space more,
and (3) diagnostics through tests and visualisations. 
UltraNest develops novel, state-of-the-art techniques for all of the above. 
They are especially remarkable for being free of tuning parameters and 
theoretically justified.

Currently available efficient NS implementations such as `MultiNest` [@Feroz2009] and its
open-source implementations rely on a heuristic algorithm which has shown biases
when the likelihood contours are not ellipsoidal [@Buchner2014stats;@Nelson2020].
UltraNest instead implements better motivated self-diagnosing algorithms,
and improved, conservative uncertainty propagation.
In other words, UltraNest prioritizes robustness and correctness, and maximizes 
speed second. For potentially complex posteriors where the user 
is willing to invest computation for obtaining a 
gold-standard exploration of the entire posterior distribution in one run,
UltraNest was developed.

![Logo of UltraNest; made by https://www.flaticon.com/authors/freepik](docs/static/logo.png)

# Method

NS methods are systematically reviewed in Buchner et al., submitted,
highlighting also the approaches used in UltraNest.

The basic outline of vanilla NS [see @Skilling2004 for details] is as follows:

A set of $N$ live points is drawn from the unit hypercube (u-space).
A inverse cumulative prior transform converts to physical parameter units (v-space),
and the likelihood $L$ evaluated.
NS then repeatedly replaces the current worst likelihood
through likelihood-constrained prior sampling (LRPS).
At each iteration (represented by the removed, dead point), 
the prior space investigated shrinks by approximately $V_{i+1}/V_i = (N - 1)/N$,
starting from the entire prior volume $V_i=1$.
The dead point becomes a posterior sample 
with weight $w_i=L_i\times V_i$, yielding the posterior distribution 
and the evidence estimate $Z_i=\sum_{j=1}^i w_i$.
The iteration procedure can terminate when the live points become unimportant,
i.e. when $w_{live}=V_{i+1}\max_{i=1}^N L_{live,i}\ll Z_i$. 

## Reactive NS

Instead of iterating with a fixed array of live points, UltraNest 
uses a tree. The root of the tree represents the entire prior volume,
and its child nodes are samples from the entire prior.
A breadth-first search is run, which keeps a stack of the opened nodes
sorted by likelihood. 
When encountering a node, attaching a child to it is decided by several criteria.

Reactive NS is a flexible generalisation of the 
Dynamic NS [@Higson2017], which used a simple heuristic
for identifying where to add more live points. The tree formulation
of Reactive NS makes implementing error propagation and 
variable number of live points straight-forward.

## Integration procedure

UltraNest computes conservative uncertainties on $Z$ and the posterior weights.
Several Reactive NS explorers are run which see only parts of the tree,
specifically a bootstrapped subsample of the root edges.
For each sample, each explorer estimate a weight (0 if it is blind to it),
and an estimate of the evidence. The ensemble gives an uncertainty distribution.

The bootstrapped integrators is an evolution over 
single-bulk evidence uncertainty measures and includes the scatter 
in likelihoods (by bootstrapping) and volume estimates [by beta sampling; @Skilling2004].

## LRPS procedures in UltraNest

The live points all fulfill the current likelihood threshold, therefore
they can be used to trace out the neighbourhood where a new, independent prior sample
can be generated that also fulfills the threshold. 
Region-based sampling uses rejection sampling using constructed geometries.

UltraNest combines three region constructions, 
and uses their intersection: 
MLFriends [@Buchner2019c, based on RadFriends by @Buchner2014stats], 
a bootstrapped single ellipsoid in u-space and another in v-space.
The last one drastically helps when one parameter constraint scales with another,
(e.g., funnels).
UltraNest dynamically chooses whether to draw samples 
from the entire prior, 
the single u-space ellipsoid or MLFriends ellipsoids (accounting for overlaps),
and filtered by the other constraints (including the transformed v-space ellipsoid).

Useful for high dimensional problems ($d>20$), UltraNest supports several types of 
Monte Carlo random walks, including:

* Slice sampling [as in `Polychord`, @Handley2015a]
* Hit-and-run sampling
* Constrained Hamiltonian Monte Carlo with No-U turn sampling [similar to `NoGUTS`, @Griffiths2019]

# Features

* Run-time visualisation
* Posterior visualisations
* Diagnostic test of run quality
* MPI parallelisation
* Resuming
* Models written in Python, C, C++, Fortran, Julia [@ultranestjl], R, and Javascript [@ultranestjs].

[Extensive documentation](https://johannesbuchner.github.io/UltraNest/) is available.

# Acknowledgements

I am very thankful to Fred Beaujean, Josh Speagle and J. Michael Burgess for insightful conversations.

# References
