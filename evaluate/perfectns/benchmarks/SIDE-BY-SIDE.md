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

## Nearest-neighbour search skipped; region pool padded to >=400 points

The quadratic slice direction no longer excludes the region point
nearest to each walker's start (the search was the dominant direction
cost, ~40-50us of ~50-70us per direction at nlive=4000, guarding an
event with probability 3/nlive); the fit picks are three distinct
random live points. The region proposal pool is the live points when
K >= 400, padded with the most recent dead points from the store to at
least 400 points otherwise (proposal diversity at small K).

Same-batch comparison (K=10000, P=100, 30000 dead points, median of 3
seeds; before = with the search and live-points-only pool):

| proposal        | problem   | before | after | speedup |
|-----------------|-----------|-------:|------:|--------:|
| quadratic-slice | gauss-50d | 3.62s  | 2.50s | 1.45x   |
| quadratic-slice | banana-20d| 3.27s  | 2.01s | 1.63x   |
| mix             | gauss-50d | 2.25s  | 2.08s | 1.08x   |
| mix             | banana-20d| 1.96s  | 1.81s | 1.08x   |

Full-run correctness with the padded pool (banana-2d, 8 rotations,
analytic lnZ = -2.7693): K=50 4.3s, lnZ = -2.7783 +- 0.0167; K=100
6.1s, lnZ = -2.7485 +- 0.0117; K=400 27.1s, lnZ = -2.7744 +- 0.0059.
Small-K runs now complete in seconds while keeping >=400 proposal
points.

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
