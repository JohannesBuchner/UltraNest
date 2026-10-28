"""
Pipeline for testing nested sampling integrators on the perfectns
validation problems
====================================================================

Runs a (problem, dimension) x (integrator, proposal) matrix and writes
the output plots and results into folders laid out as::

    <outdir>/<problemname>-<dim>/<integrator>-<proposal>/

for example funnel-10d/ultranest-mlfriends/ or gauss-2d/ultranest-de-mix/
or banana-10d/snowballingns-quadratic-slice/.

The problems come from ultranest.perfectns (gauss, box, funnel, banana,
hyperpyramid). The integrator-proposal combinations are registered in
COMBOS; currently available:

* ``ultranest-mlfriends``: ReactiveNestedSampler with the default
  MLFriends region machinery,
* ``ultranest-de-mix``: ReactiveNestedSampler with a slice sampler using
  the mixture differential-evolution direction,
* ``snowballingns-mix``: the snowballing nested sampler with its
  built-in mixture proposal. Step-sampler based combinations (e.g.
  ``snowballingns-quadratic-slice``) plug in here once
  ultranest/snowballingns/stepsamplers.pyx exists.

Usage::

    python evaluate/perfectns/pipeline.py --problem banana --dim 2 \
        --combos all
    python evaluate/perfectns/pipeline.py --problems gauss,banana \
        --dims 2,10 --combos ultranest-mlfriends,snowballingns-mix
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultranest.perfectns import (
    BananaGaussianLRPS, BoxGaussianLRPS, HyperPyramidLRPS,
    NonIsotropicGaussianLRPS, UncorrelatedFunnelLRPS, mc_prior_volume_table)


def make_problem(name, dim):
    """Test problem from perfectns.py by name."""
    if name == 'banana':
        return BananaGaussianLRPS.standard_problem(dim)
    if name == 'hyperpyramid':
        return HyperPyramidLRPS(dim)
    if name == 'funnel':
        return UncorrelatedFunnelLRPS(dim, trunc_sigmas=6.0)
    if name == 'gauss':
        return NonIsotropicGaussianLRPS(dim)
    if name == 'box':
        return BoxGaussianLRPS(dim)
    raise ValueError("unknown problem=%s" % name)


def run_ultranest(problem, outdir, nlive=400, frac_remain=0.5, seed=1,
                  nsteps=4, proposal='mlfriends', **kw):
    """Run ReactiveNestedSampler with the given region/proposal; writes
    the ultranest plots and a results summary into outdir."""
    from ultranest import ReactiveNestedSampler
    sampler = ReactiveNestedSampler(problem.paramnames, problem.loglike,
                                    transform=problem.transform,
                                    vectorized=True)
    if proposal == 'de-mix':
        from ultranest.stepsampler import (SliceSampler,
                                           generate_mixture_random_direction)
        sampler.stepsampler = SliceSampler(
            nsteps=nsteps,
            generate_direction=generate_mixture_random_direction)
    t0 = time.time()
    result = sampler.run(min_num_live_points=nlive,
                         frac_remain=frac_remain)
    dt = time.time() - t0
    summary = dict(integrator='ultranest', proposal=proposal,
                   logz=float(result['logz']),
                   logzerr=float(result['logzerr']),
                   niter=int(result['niter']),
                   ncall=int(result['ncall']),
                   nlive=nlive, wall_time=dt)
    try:
        # runplot/traceplot need the stored-results format (with logvol);
        # the corner plot works with the run() result directly
        import ultranest.plot
        fig = ultranest.plot.cornerplot(result)
        if fig is not None:
            fig.savefig(os.path.join(outdir, 'ultranest_cornerplot.png'),
                        dpi=110)
    except Exception as e:
        print('  ultranest cornerplot failed:', type(e).__name__, e)
    return summary


def run_snowballingns(problem, outdir, K=400, P=20, nsteps=1,
                      frac_remain=0.5, max_rotations=8, seed=1,
                      vol_ndraw=2000000, vol_nkeep=50001, **kw):
    """Run the snowballing nested sampler with its built-in mixture
    proposal; writes the panel plots and a results summary into
    outdir."""
    from ultranest.snowballingns.integrator import run_snowballing
    transform, loglike = problem.transform, problem.loglike
    d = len(problem.paramnames)
    if isinstance(problem, (BananaGaussianLRPS, NonIsotropicGaussianLRPS,
                            BoxGaussianLRPS)):
        vol_func = mc_prior_volume_table(problem, ndraw=vol_ndraw,
                                         nkeep=vol_nkeep)
    else:
        try:
            vol_func = problem.make_volume_function()
        except Exception:
            vol_func = None
    result = run_snowballing(transform, loglike, d, K=K, P=P, nsteps=nsteps,
                             frac_remain=frac_remain,
                             max_rotations=max_rotations, seed=seed,
                             outdir=outdir, vol_func=vol_func)
    return dict(integrator='snowballingns', proposal='mix',
                logz=float(result['logz']),
                logzerr=float(result['logzerr']),
                nrot=len(result['results']),
                nrounds=int(result['nrounds']),
                nlive=result['nlive'], wall_time=result['wall_time'])


COMBOS = {
    'ultranest-mlfriends': dict(
        runner=run_ultranest, kwargs=dict(proposal='mlfriends')),
    'ultranest-de-mix': dict(
        runner=run_ultranest, kwargs=dict(proposal='de-mix')),
    'snowballingns-mix': dict(
        runner=run_snowballingns, kwargs=dict()),
}


def run_combo(name, problem, problemname, dim, outdir, args):
    """Run one integrator-proposal combination and write its summary."""
    combo_dir = os.path.join(outdir, '%s-%dd' % (problemname, dim), name)
    os.makedirs(combo_dir, exist_ok=True)
    spec = COMBOS[name]
    kwargs = dict(spec['kwargs'])
    kwargs.update(nlive=args.nlive, K=args.K, P=args.P, nsteps=args.nsteps,
                  frac_remain=args.frac_remain, seed=args.seed,
                  max_rotations=args.max_rotations)
    print('=== %s-%d / %s ===' % (problemname, dim, name), flush=True)
    t0 = time.time()
    summary = spec['runner'](problem, combo_dir, **kwargs)
    summary['wall_time'] = time.time() - t0
    try:
        summary['analytic_logz'] = float(problem.analytic_lnZ())
    except Exception:
        summary['analytic_logz'] = None
    summary['args'] = dict(nlive=args.nlive, K=args.K, P=args.P,
                           nsteps=args.nsteps,
                           frac_remain=args.frac_remain, seed=args.seed,
                           max_rotations=args.max_rotations)
    with open(os.path.join(combo_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(combo_dir, 'results.txt'), 'w') as f:
        f.write('%s-%dd / %s\n' % (problemname, dim, name))
        for key, value in summary.items():
            if key != 'args':
                f.write('%-12s %s\n' % (key, value))
    print('    lnZ = %s +- %s (%s), %.1fs -> %s'
          % (summary['logz'], summary['logzerr'],
             ('analytic %s' % summary['analytic_logz']
              if summary['analytic_logz'] is not None else 'no analytic'),
             summary['wall_time'], combo_dir), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems', default='banana,gauss',
                        help='comma-separated problem names')
    parser.add_argument('--dims', default='2,10',
                        help='comma-separated dimensionalities')
    parser.add_argument('--combos', default='all',
                        help="comma-separated integrator-proposal "
                             "combinations, or 'all'")
    parser.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--nlive', type=int, default=400)
    parser.add_argument('--K', type=int, default=400)
    parser.add_argument('--P', type=int, default=20)
    parser.add_argument('--nsteps', type=int, default=1)
    parser.add_argument('--frac-remain', type=float, default=0.5)
    parser.add_argument('--max-rotations', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--vol-ndraw', type=int, default=2000000)
    parser.add_argument('--vol-nkeep', type=int, default=50001)
    args = parser.parse_args()

    problems = args.problems.split(',')
    dims = [int(d) for d in args.dims.split(',')]
    if args.combos.strip() == 'all':
        combos = sorted(COMBOS.keys())
    else:
        combos = [c.strip() for c in args.combos.split(',')]
        for c in combos:
            if c not in COMBOS:
                raise ValueError("unknown combo=%s, available: %s"
                                 % (c, sorted(COMBOS.keys())))

    for problemname in problems:
        for dim in dims:
            problem = make_problem(problemname, dim)
            for name in combos:
                run_combo(name, problem, problemname, dim, args.outdir,
                          args)


if __name__ == '__main__':
    main()
