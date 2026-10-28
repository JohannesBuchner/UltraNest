# SnowballingNestedSampler integration plan

How to integrate the snowballing nested sampler
(`ultranest/snowballingns/`) into the existing UltraNest
infrastructure.

## What exists now

- `ultranest/snowballingns/state.pyx` — the `SnowballState` state
  machine: threads, walkers, the burial buffer, the growing HDF5 store,
  and the burial-gate incremental integrator (append-only ingestion in
  threshold order with final weights).
- `ultranest/snowballingns/integrator.py` — the run driver:
  `run_snowballing()` (the snowball rotation loop: batch advances,
  termination, widening, record assembly, panel plots).
- `ultranest/snowballingns/utils.py` — the per-rotation record store,
  RJD diagnostics, the Gompertz looping criterion, panel plots.
- `evaluate/perfectns/pipeline.py` — the test pipeline over the
  perfectns validation problems, writing to
  `output/<problem>-<dim>/<integrator>-<proposal>/`.

## Target API

A class similar to `ReactiveNestedSampler`, with a configurable step
sampling function:

```python
from ultranest.snowballingns import SnowballingNestedSampler

sampler = SnowballingNestedSampler(
    paramnames, loglike, transform=transform, vectorized=True,
    K=400, P=20)
sampler.stepsampler = QuadraticSliceSampler(nsteps=4)  # optional;
                                                       # default: built-in mixture
result = sampler.run(frac_remain=0.5, max_rotations=8)
# result: ultranest-compatible dict (logz, logzerr, samples, posterior,
# ...) plus the HDF5 point and rotation-record stores
```

Key differences from ReactiveNestedSampler:

- a step sampler must be chosen (the proposal is integral to the
  snowball's validity: the origin-eligible start selection),
- the run is a sequence of snowball rotations with widening, not a
  single sequential ladder,
- the integral is re-materialized each rotation; the termination is
  frac_remain on the incrementally maintained values plus the Gompertz
  looping criterion,
- completed points live in a growing HDF5 store, not in memory.

## Architecture options

**(a) Reuse ReactiveNestedSampler, feeding it points as they are
learned, running a different sampler internally.**
ReactiveNestedSampler's loop is built around the sequential front: one
point replaced at a time, a region sequence, a point stack for resume,
per-iteration callbacks. The snowball's batch advances of P threads with
per-rotation re-integration do not fit that loop; feeding it points as
learned would fight the point-stack and resume logic. **Not
recommended.**

**(b) Refactor ReactiveNestedSampler so it can be subclassed.**
Extract the integration, termination, output-writing and progress
machinery of `ultranest/integrator.py` (1000+ lines) into a base class
that both ReactiveNestedSampler and SnowballingNestedSampler extend.
Cost: a large refactor with risk to the production sampler. Benefit:
shared progress reporting, output files, resume. The shared surface is
modest, and the snowball's integration and termination differ
fundamentally (per-rotation re-materialization, HDF5 store, Gompertz
looping). **Possible later; not the first step.**

**(c) Independent class reusing leaf utilities.** — **Recommended.**
`SnowballingNestedSampler` is its own class reusing ultranest's leaf
utilities where they fit: `ultranest.ordertest`, the plot conventions,
the results-dict format (logz, logzerr, samples, posterior), and the
stepsampler conventions (a `.stepsampler` attribute, as in
ReactiveNestedSampler). The control flow (rotations, batch advances,
re-materialization) stays in the snowballingns package. This is the
smallest-risk path and matches the existing package layout.

## Step sampler

The current state machine has the walker phase machinery built in
(Cython): mixture proposal direction over origin-eligible thread ends,
stepping-out, bisection, nsteps accepted steps. To make the proposal
configurable:

- extract the walker phase machinery from `state.pyx` behind a small
  interface: `setup_walkers(ends, thresholds, mask)`, `propose()`,
  `receive(Lnew)` with per-walker thresholds,
- implement alternative proposals in
  `ultranest/snowballingns/stepsamplers.pyx`, e.g. a quadratic slice
  sampler (parabolic fit of the likelihood along the chord), registering
  them in the pipeline's `COMBOS` (e.g. `snowballingns-quadratic-slice`).

**Can `ultranest/popstepsampler.py` be reused instead?** Largely yes at
the pattern level: `PopulationSliceSampler` is the same lockstep
machinery (vectorized stepping-out brackets over a population,
`setup_brackets(mask_starting, region)`, `advance(transform, loglike,
Lmin, region)`, `shift()`, generation tracking, revert-until-all-above-
Lmin). The differences: (1) its `advance` takes a scalar Lmin, the
snowball's batch advance has per-walker thresholds (small
generalization); (2) starts come from region-based proposals, the
snowball needs the origin-eligible start selection (the
perfect-sampling conditioning); (3) the state machine owns the store and
integration, so the sampler only produces replacements. It is pure
Python — wrapping it keeps a second implementation path alive at some
runtime cost. **Recommendation**: port the bracket-stepping pattern into
`stepsamplers.pyx` (Cython, per-walker thresholds, origin-eligible
starts), keeping `popstepsampler.py`'s interface shape so a Python-level
fallback stays possible.

## MPI

Deferred — we choose not to support MPI yet. Design constraints kept:

- the vectorization mode is fixed throughout the run (helps jax-based
  likelihoods),
- the likelihood function is called through a single vectorized
  evaluation per likelihood-function-evaluation step.

Later, an MPI-distributed evaluation slots in without touching the
state machine: the P-proposal batch is split across ranks, likelihoods
gathered on rank 0, where the state machine and the burial logic run
sequentially. Caveat: the batches are P-sized (small), so MPI overhead
may dominate unless K and P grow substantially — another reason to
defer.

## Milestones

1. **done** — the `ultranest/snowballingns` package (state, integrator,
   utils) and the `evaluate/perfectns` test pipeline.
2. Extract the walker phase machinery behind a stepsampler interface;
   `stepsamplers.pyx` with the built-in mixture and a quadratic-slice
   proposal; register `snowballingns-quadratic-slice` in the pipeline.
3. `SnowballingNestedSampler` class with the ultranest-compatible
   results dict (posterior samples = the buried dead points weighted by
   their strips + the final thread ends), progress reporting, plots.
4. Resume/warmstart from the HDF5 store (the evaluated points are kept,
   so a warmstart is the natural restart).
5. MPI support (optional, deferred).
