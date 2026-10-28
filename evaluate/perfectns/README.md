# perfectns testing pipeline

Tests nested sampling integrators on the validation problems from
`ultranest/perfectns.py` (gauss, box, funnel, banana, hyperpyramid).

## Layout

Each (problem, dimension) x (integrator, proposal) combination writes
its plots and results into

    output/<problemname>-<dim>/<integrator>-<proposal>/

for example `output/funnel-10d/ultranest-mlfriends/` or
`output/banana-10d/snowballingns-mix/`.

## Registered integrator-proposal combinations

- `ultranest-mlfriends` — ReactiveNestedSampler with the default
  MLFriends region machinery
- `ultranest-de-mix` — ReactiveNestedSampler with a slice sampler using
  the mixture differential-evolution direction
- `snowballingns-mix` — the snowballing nested sampler
  (`ultranest/snowballingns`) with its built-in mixture proposal

Step-sampler based combinations (e.g. `snowballingns-quadratic-slice`)
plug into the `COMBOS` registry once
`ultranest/snowballingns/stepsamplers.pyx` exists.

## Usage

```sh
python evaluate/perfectns/pipeline.py --problem banana --dim 2 --combos all
python evaluate/perfectns/pipeline.py --problems gauss,banana --dims 2,10 \
    --combos ultranest-mlfriends,snowballingns-mix
python evaluate/perfectns/pipeline.py --problems funnel --dims 10 \
    --combos snowballingns-mix --max-rotations 5
```

Each combination folder contains `results.json` / `results.txt` (lnZ,
error, iteration and evaluation counts, wall time, analytic reference
where available) and the plots: `snowballingns_panels.png` +
`rjd_distributions.png` for snowballingns runs; `ultranest_cornerplot.png`
for ultranest runs.
