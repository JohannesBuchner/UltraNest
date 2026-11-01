# perfectns testing pipeline

Tests nested sampling integrators on the validation problems from
`ultranest/perfectns.py` (gauss, box, funnel, banana, hyperpyramid).

## Layout

The pipeline runs ONE (problem, dimension) x (integrator, proposal)
combination and writes its plots and results into

    output/<problemname>-<dim>/<integrator>-<proposal>/

for example `output/funnel-10d/ultranest-mlfriends/` or
`output/banana-2d/snowballingns-quadratic-slice/`.

## Registered integrator-proposal combinations

- `ultranest-mlfriends` — ReactiveNestedSampler with the default
  MLFriends region machinery
- `ultranest-de-mix` — ReactiveNestedSampler with a slice sampler using
  the mixture differential-evolution direction
- `snowballingns-quadratic-slice` — the SnowballingNestedSampler with
  the quadratic slice direction proposal (the default)
- `snowballingns-mix` — the SnowballingNestedSampler with the mixture
  differential-evolution direction proposal

## Usage

```sh
python evaluate/perfectns/pipeline.py --problem banana --dim 2 \
    --combo snowballingns-quadratic-slice
```

or through the Makefile, which drives combinations as make targets (the
final output file of each combination is `results.json`):

```sh
make                                        # the default matrix
make output/banana-10d/snowballingns-quadratic-slice/results.json
make PROBLEMS=funnel DIMS=10 COMBOS="snowballingns-quadratic-slice"
make K=100 MAX_ROTATIONS=3 output/banana-2d/snowballingns-mix/results.json
make clean
```

Each combination folder contains `results.json` / `results.txt` (lnZ,
error, iteration and evaluation counts, wall time, analytic reference
where available) and the plots: `snowballingns_panels.png` +
`rjd_distributions.png` for snowballingns runs (also
`sampler.plot_rotations()`, `plot_corner()`, `plot_trace()`);
`ultranest_cornerplot.png` for ultranest runs.

## Speed benchmark

`benchmark_speed.py` times the sampling hot path (batch advances:
propose, transform, loglike, receive, release + integration) on
gauss-50d and banana-20d at K=10000, first snowball rotation only,
capped at `--max-batches` batch advances and `--max-dead` released dead
points (whichever comes first) so every configuration performs the same
integration work. `--legacy-direction` runs the per-walker-loop
quadratic slice direction (the pre-vectorization implementation) for
the baseline. `--proposal quadratic-slice|mix` and
`--region mlfriends|covariance|cube` select the direction proposal and
region class. Results go into `benchmarks/benchmark_speed-*.json`.

```sh
make benchmark    # or:
python benchmark_speed.py --problem gauss --dim 50 --P 100
```
