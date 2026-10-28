"""
Benchmark for the perfectns rejection problems at high dimension.

For each problem, a short instrumented run locates the beginning, middle
and end phases of a nested sampling run (by likelihood threshold Lmin).
At each phase, live points are drawn from the exact constrained prior,
a region is built, and a few iterations of the region-based rejection
path are timed: candidate generation (region.sample), likelihood
evaluation and region rebuilding. Reports the proposal acceptance rate,
the per-candidate cost, and the effective accepted samples per second.

Also compares the vectorized numpy candidate generation against a plain
python loop for the bbox proposal, to quantify the vectorization gain.

Usage::

    python tests/benchmark_perfectns.py [--dim 20] [--nlive 200]
"""
import argparse
import time

import numpy as np

from ultranest import ReactiveNestedSampler
from ultranest.mlfriends import LocalAffineLayer
from ultranest.perfectns import (
    PerfectRegion, NonIsotropicGaussianLRPS, UncorrelatedFunnelLRPS,
    BoxGaussianLRPS, BananaGaussianLRPS, HyperPyramidLRPS)


def get_problem(name, dim):
    if name == 'gauss':
        return NonIsotropicGaussianLRPS.standard_problem(dim)
    if name == 'funnel':
        return UncorrelatedFunnelLRPS(dim, trunc_sigmas=6.0)
    if name == 'banana':
        return BananaGaussianLRPS.standard_problem(dim)
    if name == 'hyperpyramid':
        return HyperPyramidLRPS(dim)
    raise ValueError(name)


def find_phase_lmins(problem, nlive, max_ncalls, seed):
    """Short instrumented run; return (beginning, middle, end) Lmin."""
    records = []
    orig = problem.proposal

    def wrapped(Lmin, rng, nsamples, *args, **kwargs):
        cands, ok = orig(Lmin, rng, nsamples, *args, **kwargs)
        records.append(Lmin)
        return cands, ok

    problem.proposal = wrapped
    np.random.seed(seed)
    sampler = ReactiveNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True)
    try:
        sampler.run(min_num_live_points=nlive, frac_remain=0.001,
                    region_class=PerfectRegion.with_problem(problem),
                    max_ncalls=max_ncalls)
    except Exception:
        pass
    problem.proposal = orig
    if len(records) == 0:
        raise RuntimeError('no proposals recorded')
    return records[0], records[len(records) // 2], records[-1]


def build_region(problem, Lmin, nlive, rng):
    """Live points from the exact constrained prior at Lmin, plus region."""
    us = problem.sample(rng, nlive, Lmin)
    layer = LocalAffineLayer()
    layer.optimize(us, us)
    region = PerfectRegion(us, layer, problem=problem)
    r, f = region.compute_enlargement(nbootstraps=30)
    region.maxradiussq = r
    region.enlarge = f
    region.create_ellipsoid(minvol=1e-30)
    return region


def benchmark_phase(problem, Lmin, nlive, ndraw, niters, seed):
    """Time a few rejection iterations at the given threshold."""
    rng = np.random.RandomState(seed)
    region = build_region(problem, Lmin, nlive, rng)

    # region rebuild cost (paid every volume-fraction interval)
    t0 = time.time()
    for _ in range(3):
        r, f = region.compute_enlargement(nbootstraps=30)
        region.maxradiussq = r
        region.enlarge = f
        region.create_ellipsoid(minvol=1e-30)
    t_rebuild = (time.time() - t0) / 3

    # a few iterations: region.sample (proposal) + likelihood evaluation
    t_prop = 0.0
    t_like = 0.0
    nreturned = 0
    for _ in range(niters):
        t0 = time.time()
        u = region.sample(nsamples=ndraw)
        t1 = time.time()
        problem.loglike(problem.transform(u))
        t2 = time.time()
        t_prop += t1 - t0
        t_like += t2 - t1
        nreturned += len(u)

    ntotal = niters * ndraw
    acc = nreturned / ntotal
    return dict(
        Lmin=Lmin, acceptance=acc,
        us_per_candidate=1e6 * (t_prop) / ntotal,
        us_per_likelihood=1e6 * (t_like) / max(1, nreturned),
        accepted_per_second=nreturned / (t_prop + t_like) if t_prop + t_like > 0 else np.nan,
        t_rebuild_ms=1e3 * t_rebuild,
    )


def benchmark_loop_vs_vectorized(problem, Lmin, naccepted, seed):
    """Time vectorized candidate generation against a python loop."""
    rng = np.random.RandomState(seed)

    t0 = time.time()
    n_vec = 0
    while n_vec < naccepted:
        candidates, ok = problem.proposal(Lmin, rng, 100000)
        n_vec += int(ok.sum())
    t_vec = time.time() - t0

    # plain python loop reference (bbox-style: one candidate at a time)
    t0 = time.time()
    n_loop = 0
    while n_loop < naccepted:
        x = rng.uniform(0.0, 1.0)
        if problem.loglike(np.asarray(x).reshape((1, -1)))[0] >= Lmin:
            n_loop += 1
    t_loop = time.time() - t0
    return t_vec / n_vec * 1e6, t_loop / n_loop * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--nlive', type=int, default=200)
    parser.add_argument('--ndraw', type=int, default=2000)
    parser.add_argument('--niters', type=int, default=10)
    parser.add_argument('--max-ncalls', type=int, default=6000,
                        help='cap for the phase-finding run')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    print('perfectns benchmark: d=%d, nlive=%d, ndraw=%d, niters=%d' % (
        args.dim, args.nlive, args.ndraw, args.niters))
    for name in ['gauss', 'funnel', 'box', 'banana', 'hyperpyramid']:
        problem = get_problem(name, args.dim)
        print('=' * 78)
        print('%s (d=%d)' % (name, args.dim), flush=True)
        try:
            beginning, middle, end = find_phase_lmins(
                problem, args.nlive, args.max_ncalls, args.seed)
        except RuntimeError as e:
            print('  phase finding failed: %s' % e)
            continue
        print('  phases: beginning Lmin=%.4g  middle Lmin=%.4g  end Lmin=%.4g' % (
            beginning, middle, end), flush=True)
        for phase, Lmin in [('beginning', beginning), ('middle', middle),
                            ('end', end)]:
            stats = benchmark_phase(problem, Lmin, args.nlive, args.ndraw,
                                    args.niters, args.seed)
            print('  %-10s Lmin=%12.4g: acceptance=%9.2e  '
                  'proposal %6.2f us/cand  likelihood %6.2f us/eval  '
                  '%8.1f accepted/s  rebuild %6.2f ms' % (
                      phase, Lmin, stats['acceptance'],
                      stats['us_per_candidate'], stats['us_per_likelihood'],
                      stats['accepted_per_second'], stats['t_rebuild_ms']),
                  flush=True)
        # vectorization gain (bbox-style cube proposal, likelihood only)
        try:
            us_vec, us_loop = benchmark_loop_vs_vectorized(
                problem, min(beginning, 0.0), naccepted=20, seed=args.seed)
            print('  vectorized vs python loop (cube proposal): '
                  '%.2f us vs %.2f us per accepted sample (%.0fx)' % (
                      us_vec, us_loop, us_loop / max(us_vec, 1e-12)), flush=True)
        except Exception as e:
            print('  vectorization comparison skipped: %s' % e)


if __name__ == '__main__':
    main()
