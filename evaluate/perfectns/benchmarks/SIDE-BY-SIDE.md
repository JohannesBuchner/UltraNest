# Side-by-side sampling speed benchmark

SnowballingNestedSampler, custom integrator, K=10000 live points base,
P=100, first snowball rotation only, capped at 10000 batch advances /
30000 released dead points (whichever comes first) so every
configuration performs the same integration work. Median wall time of
seeds 1-3 on an otherwise idle 8-core laptop. `legacy` is the
per-walker-loop quadratic slice direction (before its vectorization);
mlfriends/covariance/cube are the region classes (RobustEllipsoidRegion
with MLFriends machinery, CovarianceRegion, CubeRegion).

## gauss-50d

| proposal        | region     | wall/s | dead points | evals   | evals/s |
|-----------------|------------|-------:|------------:|--------:|--------:|
| quadratic-slice | mlfriends (legacy) | 44.25 | 30100 | 348400 | 7895 |
| quadratic-slice | mlfriends  | 5.15 | 30000 | 347400 | 67448 |
| quadratic-slice | covariance | 4.93 | 30000 | 348500 | 70199 |
| quadratic-slice | cube       | 4.95 | 30000 | 348500 | 70337 |
| mix             | mlfriends  | 3.36 | 30050 | 337300 | 99016 |
| mix             | covariance | 2.98 | 30097 | 399900 | 134399 |
| mix             | cube       | 2.74 | 30000 | 403400 | 147434 |

## banana-20d

| proposal        | region     | wall/s | dead points | evals   | evals/s |
|-----------------|------------|-------:|------------:|--------:|--------:|
| quadratic-slice | mlfriends (legacy) | 29.73 | 30000 | 330800 | 11136 |
| quadratic-slice | mlfriends  | 3.77 | 30000 | 328800 | 87333 |
| quadratic-slice | covariance | 3.64 | 30000 | 329700 | 90560 |
| quadratic-slice | cube       | 3.96 | 30000 | 329700 | 83448 |
| mix             | mlfriends  | 2.59 | 30000 | 308300 | 121068 |
| mix             | covariance | 2.71 | 30000 | 351100 | 130194 |
| mix             | cube       | 2.11 | 30000 | 318100 | 149930 |

## Observations

- The region class barely matters for the quadratic slice: it only uses
  the region's live points, not the axes, and the covariance/cube runs
  have identical trajectories (identical evaluation counts). The
  MLFriends machinery (maxradius bootstrapping, enlargement) is wasted
  compute for it, and its bootstrap draws even shift the numpy RNG
  stream.
- The mixture proposal is the fastest configuration at K=10000: its
  direction generation is about twice as cheap as the quadratic fit
  (no per-walker nearest-neighbour search or 3x3 solves), which more
  than compensates its slightly longer walks (~13 vs ~11.6 likelihood
  evaluations per dead point on gauss-50d). mix + cube / mix +
  covariance are 1.7-1.9x faster than the quadratic slice + mlfriends.
- For the mixture proposal the cube region is the fastest (the
  region-oriented half becomes a plain isotropic direction); the
  covariance axes do not reduce the evaluation count.
- Full-run correctness (banana-2d, K=400, 8 rotations, analytic
  lnZ = -2.7693): quadratic + CovarianceRegion 83.3s,
  lnZ = -2.7676 +- 0.0059; mix + CovarianceRegion 40.5s,
  lnZ = -2.7652 +- 0.0059.

## P sweep with the mixture proposal (covariance region, K=10000)

| P    | gauss-50d wall/s | banana-20d wall/s |
|-----:|-----------------:|------------------:|
| 100  | 1.95 | 1.94 |
| 250  | 1.78 | 1.70 |
| 500  | 1.77 | 1.24 |
| 1000 | 1.82 | 1.21 |
| 2000 | 2.02 | 1.31 |

The optimum is P~500-1000 for the mixture proposal (37% faster than
P=100 on banana-20d, 9% on gauss-50d); with the quadratic slice the
optimum is P~250-500 (~15-20% faster than P=100). The default stays at
the cap of 100: the gain for the default configuration is below the
30% adoption threshold, and P=100 is optimal at K=400. Pass
`P=500` (or higher) explicitly for large-K mixture-proposal runs.

The default region class is CovarianceRegion; CubeRegion is allowed for
step samplers that do not use the region axes. The default direction
proposal remains the quadratic slice (it follows the ridge by design);
the mixture proposal is available via
`stepsampler=generate_mixture_random_direction`.
