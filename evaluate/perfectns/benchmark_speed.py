"""Speed benchmark for the snowballingns custom integrator
=====================================================

Times the sampling hot path (batch advances: propose, transform,
loglike, receive, release + integration) on the perfectns validation
problems, one snowball rotation at a time.

Protocol: K live points base, first rotation only, capped at
``--max-batches`` batch advances and ``--max-dead`` released dead
points, whichever comes first -- the same integration work for every
configuration, cut short before the termination criterion fires (a
full first rotation at K=10000 takes orders of magnitude longer).

Each configuration runs in its own temporary directory (the HDF5 point
store is part of the measured pipeline) which is removed afterwards.

Usage::

    python benchmark_speed.py --problem gauss --dim 50 \
        --P 100 250 500 1000 2000
    python benchmark_speed.py --problem banana --dim 20 --P 100 \
        --proposal mix --region covariance \
        --profile --profile-out /tmp/prof.out
"""
import argparse
import cProfile
import json
import os
import pstats
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # the repository root

from ultranest.perfectns import (BananaGaussianLRPS,
                                 NonIsotropicGaussianLRPS)


def make_problem(name, dim):
    """Test problem from perfectns.py by name."""
    if name == 'banana':
        return BananaGaussianLRPS.standard_problem(dim)
    if name == 'gauss':
        return NonIsotropicGaussianLRPS.standard_problem(dim)
    raise ValueError("unknown problem=%s" % name)


def run_once(problem, K, P, seed, max_batches, max_dead, profile=None,
             tmpdir=None, legacy_direction=False, proposal='quadratic-slice',
             region='mlfriends'):
    """One capped first-rotation run; returns the timing summary."""
    from ultranest.snowballingns.integrator import SnowballingNestedSampler
    from ultranest.snowballingns.utils import CubeRegion, CovarianceRegion
    from ultranest.mlfriends import RobustEllipsoidRegion
    region_class = {'mlfriends': RobustEllipsoidRegion,
                    'covariance': CovarianceRegion,
                    'cube': CubeRegion}[region]
    if legacy_direction:
        # the per-walker-loop direction generator as it was before the
        # vectorization (for the baseline)
        import ultranest.stepfuncs as stepfuncs
        from ultranest.stepfuncs import generate_mixture_random_direction

        def legacy_quadratic(ui, region, scale=1):
            ui = np.asarray(ui)
            if ui.ndim == 1:
                ui = ui.reshape((1, -1))
            nsamples, ndim = ui.shape
            us = region.u
            nlive = us.shape[0]
            out = np.zeros((nsamples, ndim))
            if nlive < 5:
                return generate_mixture_random_direction(ui, region,
                                                         scale=scale)
            for j in range(nsamples):
                i0 = int(np.argmin(((us - ui[j].reshape((1, -1))) ** 2)
                                   .sum(axis=1)))
                perm = np.random.permutation(nlive)
                perm = perm[perm != i0][:3]
                if len(perm) < 3:
                    out[j] = generate_mixture_random_direction(
                        ui[j], region, scale=scale)
                    continue
                p = us[perm]
                k = int(np.random.randint(ndim))
                others = [d2 for d2 in range(ndim) if d2 != k]
                mu = us.mean(axis=0)
                svals = p[:, k] - mu[k]
                V = np.array([np.ones(3), svals, svals ** 2]).T
                try:
                    coefs = np.linalg.solve(V, p[:, others] - mu[others])
                except np.linalg.LinAlgError:
                    out[j] = generate_mixture_random_direction(
                        ui[j], region, scale=scale)
                    continue
                if np.abs(svals - svals.mean()).max() == 0:
                    out[j] = generate_mixture_random_direction(
                        ui[j], region, scale=scale)
                    continue
                if np.random.uniform() < 0.5:
                    ds = svals.max() - svals.min()
                    out[j, k] = ds
                    out[j, others] = (coefs[1, :] + coefs[2, :]
                                      * (svals.max() + svals.min())) * ds
                else:
                    out[j] = np.random.normal(size=ndim)
            return out * scale
        stepfuncs.generate_quadratic_slice_direction = legacy_quadratic
    if proposal == 'mix':
        from ultranest.stepfuncs import generate_mixture_random_direction
        stepsampler = generate_mixture_random_direction
    else:
        stepsampler = None  # default: quadratic slice direction
    sampler = SnowballingNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True, num_live_points_base=K, P=P, num_steps=1,
        stepsampler=stepsampler, region_class=region_class,
        integrator='custom', log_dir=tmpdir)
    t0 = time.time()
    if profile is None:
        sampler.run(frac_remain=0.5, max_rotations=1, seed=seed,
                    show_status=False, max_batches=max_batches,
                    max_dead=max_dead, plot_trace=False, plot_corner=False)
    else:
        profiler = cProfile.Profile()
        profiler.enable()
        sampler.run(frac_remain=0.5, max_rotations=1, seed=seed,
                    show_status=False, max_batches=max_batches,
                    max_dead=max_dead, plot_trace=False, plot_corner=False)
        profiler.disable()
        profiler.dump_stats(profile)
    wall_time = time.time() - t0
    nburied = (sampler.rotation_summaries[-1]['nburied']
               if sampler.rotation_summaries else 0)
    return dict(P=P, K=K, seed=seed, wall_time=wall_time,
                nburied=nburied, ncall=sampler.ncall,
                proposal=proposal, region=region,
                logz=sampler.results['logz'],
                logzerr=sampler.results['logzerr'],
                evals_per_s=sampler.ncall / wall_time
                if wall_time > 0 else float('nan'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', default='gauss',
                        choices=('gauss', 'banana'))
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--K', type=int, default=10000)
    parser.add_argument('--P', type=int, nargs='+', default=[100])
    parser.add_argument('--max-batches', type=int, default=10000,
                        help='cap on batch advances per run '
                             '(default: 10000)')
    parser.add_argument('--max-dead', type=int, default=30000,
                        help='cap on released dead points per run '
                             '(default: 30000; binds first for every P, '
                             'so all configurations do the same work)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--profile', type=str, default=None,
                        help='cProfile stats output file (one run, '
                             'first P, first seed)')
    parser.add_argument('--legacy-direction', action='store_true',
                        help='use the per-walker-loop quadratic slice '
                             'direction (the pre-vectorization baseline)')
    parser.add_argument('--proposal', default='quadratic-slice',
                        choices=('quadratic-slice', 'mix'),
                        help='direction proposal (default: quadratic-slice)')
    parser.add_argument('--region', default='mlfriends',
                        choices=('mlfriends', 'covariance', 'cube'),
                        help='region class (default: mlfriends, the '
                             'pre-simple-region reference)')
    parser.add_argument('--outdir', type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'benchmarks'))
    args = parser.parse_args()

    problem = make_problem(args.problem, args.dim)
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for P in args.P:
        for seed in args.seeds:
            tmpdir = tempfile.mkdtemp(prefix='bench-snowball-')
            try:
                row = run_once(problem, args.K, P, seed, args.max_batches,
                               args.max_dead,
                               profile=(args.profile if P == args.P[0]
                                        and seed == args.seeds[0]
                                        else None),
                               tmpdir=tmpdir,
                               legacy_direction=args.legacy_direction,
                               proposal=args.proposal, region=args.region)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
            rows.append(row)
            print('    P=%-5d seed=%d: %.2fs, %d dead points, '
                  '%d evals, %.0f evals/s, lnZ=%.3f +- %.3f'
                  % (P, seed, row['wall_time'], row['nburied'],
                     row['ncall'], row['evals_per_s'], row['logz'],
                     row['logzerr']), flush=True)
    # summarize: median wall time per P
    print()
    print('%s-%dd, K=%d, %s proposal on the %s region, '
          'caps: %d batches / %d dead points'
          % (args.problem, args.dim, args.K, args.proposal, args.region,
             args.max_batches, args.max_dead))
    print('%-8s %-10s %-10s %-12s' % ('P', 'wall/s', 'dead', 'evals/s'))
    summary = []
    for P in args.P:
        ts = [r['wall_time'] for r in rows if r['P'] == P]
        dead = int(np.median([r['nburied'] for r in rows if r['P'] == P]))
        ev = np.median([r['evals_per_s'] for r in rows if r['P'] == P])
        med = float(np.median(ts))
        summary.append(dict(P=P, wall_time_median=med, nburied=dead,
                            evals_per_s=float(ev)))
        print('%-8d %-10.2f %-10d %-12.0f' % (P, med, dead, ev))
    base = summary[0]['wall_time_median']
    for s in summary:
        s['speedup_vs_first'] = base / s['wall_time_median']
    out = dict(problem=args.problem, dim=args.dim, K=args.K,
               proposal=args.proposal, region=args.region,
               max_batches=args.max_batches, max_dead=args.max_dead,
               runs=rows, summary=summary)
    fn = os.path.join(args.outdir, 'benchmark_speed-%s-%dd-K%d-%s-%s%s.json'
                      % (args.problem, args.dim, args.K, args.proposal,
                         args.region,
                         '-legacy' if args.legacy_direction else ''))
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', fn)


if __name__ == '__main__':
    main()
