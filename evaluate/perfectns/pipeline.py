"""
Pipeline for testing nested sampling integrators on the perfectns
validation problems
====================================================================

Runs ONE (problem, dimension) x (integrator, proposal) combination and
writes the output plots and results into::

    <outdir>/<problemname>-<dim>/<integrator>-<proposal>-K<K>/

for example funnel-8d/ultranest-mlfriends-K400/ or
gauss-2d/ultranest-de-mix-K400/ or
banana-8d/snowballingns-quadratic-slice-seed1-K400/.

The problems come from ultranest.perfectns (gauss, box, funnel, banana,
hyperpyramid). The integrator-proposal combinations are registered in
COMBOS; currently available:

* ``ultranest-mlfriends``: ReactiveNestedSampler with the default
  MLFriends region machinery,
* ``ultranest-de-mix``: ReactiveNestedSampler with a slice sampler using
  the mixture differential-evolution direction,
* ``snowballingns-quadratic-slice``: the SnowballingNestedSampler with
  the quadratic slice direction proposal (the default),
* ``snowballingns-mix``: the SnowballingNestedSampler with the mixture
  differential-evolution direction proposal.

The Makefile in this folder drives combinations through this script;
each combination's final output file is ``results.json``.

Usage::

    python evaluate/perfectns/pipeline.py --problem banana --dim 2 \
        --combo snowballingns-quadratic-slice
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # the repository root

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


def run_ultranest(problem, outdir, K=400, frac_remain=0.5, seed=1,
                  nsteps=4, proposal='mlfriends', **kw):
    """Run ReactiveNestedSampler with the given region/proposal; writes
    the corner plot and a results summary into outdir."""
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
    result = sampler.run(min_num_live_points=K,
                         frac_remain=frac_remain)
    dt = time.time() - t0
    summary = dict(integrator='ultranest', proposal=proposal,
                   logz=float(result['logz']),
                   logzerr=float(result['logzerr']),
                   niter=int(result['niter']),
                   ncall=int(result['ncall']),
                   nlive=K, wall_time=dt)
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
                      stepsampler='quadratic-slice',
                      integrator='multicounter',
                      vol_ndraw=2000000, vol_nkeep=50001, **kw):
    """Run the SnowballingNestedSampler with the given direction
    proposal; writes the panel plots and a results summary into
    outdir."""
    from ultranest.snowballingns.integrator import SnowballingNestedSampler
    from ultranest.stepfuncs import (generate_mixture_random_direction,
                                     generate_quadratic_slice_direction)
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
    direction = {
        'quadratic-slice': generate_quadratic_slice_direction,
        'mix': generate_mixture_random_direction,
    }[stepsampler]
    sampler = SnowballingNestedSampler(
        problem.paramnames, loglike, transform=transform, vectorized=True,
        num_live_points_base=K, num_steps=nsteps, stepsampler=direction,
        integrator=integrator, log_dir=outdir)
    result = sampler.run(frac_remain=frac_remain, max_rotations=max_rotations,
                         seed=seed)
    sampler.plot_rotations()
    return dict(integrator='snowballingns-' + integrator,
                proposal=stepsampler, seed=seed,
                logz=float(result['logz']),
                logzerr=float(result['logzerr']),
                nrot=sampler.nrotations,
                nrounds=int(result['nrounds']) if 'nrounds' in result
                else int(result['ncall']),
                nlive=result['nlive'], wall_time=result['wall_time'])


COMBOS = {
    'ultranest-mlfriends': dict(
        runner=run_ultranest, kwargs=dict(proposal='mlfriends')),
    'ultranest-de-mix': dict(
        runner=run_ultranest, kwargs=dict(proposal='de-mix')),
}
# the snowballingns combinations, one per seed, for both integrators
# (the graph-based MultiCounter and the custom 1/K-strip integrator) and
# both direction proposals
for _seed in (1, 2, 3, 4):
    for _proposal in ('quadratic-slice', 'mix'):
        for _integrator, _iname in (('multicounter', ''),
                                    ('custom', '-custom')):
            COMBOS['snowballingns%s-%s-seed%d'
                   % (_iname, _proposal, _seed)] = dict(
                runner=run_snowballingns,
                kwargs=dict(stepsampler=_proposal, integrator=_integrator,
                            seed=_seed))


def run_combo(name, problem, problemname, dim, outdir, args):
    """Run one integrator-proposal combination and write its summary."""
    combo_dir = os.path.join(outdir, '%s-%dd' % (problemname, dim),
                             '%s-K%d' % (name, args.K))
    os.makedirs(combo_dir, exist_ok=True)
    spec = COMBOS[name]
    kwargs = dict(spec['kwargs'])
    # the seed comes from the combo name (e.g. snowballingns-mix-seed3)
    seed = int(name.split('-seed')[1]) if '-seed' in name else args.seed
    kwargs.update(K=args.K, P=args.P, nsteps=args.nsteps,
                  frac_remain=args.frac_remain, seed=seed,
                  max_rotations=args.max_rotations)
    print('=== %s-%d / %s ===' % (problemname, dim, name), flush=True)
    t0 = time.time()
    summary = spec['runner'](problem, combo_dir, **kwargs)
    summary['wall_time'] = time.time() - t0
    try:
        summary['analytic_logz'] = float(problem.analytic_lnZ())
    except Exception:
        summary['analytic_logz'] = None
    summary['args'] = dict(K=args.K, P=args.P,
                           nsteps=args.nsteps, frac_remain=args.frac_remain,
                           seed=seed, max_rotations=args.max_rotations)
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
    parser.add_argument('--problem', default='banana',
                        help='problem name')
    parser.add_argument('--dim', type=int, default=2,
                        help='dimensionality')
    parser.add_argument('--combo', default='snowballingns-quadratic-slice',
                        help='integrator-proposal combination (one of: %s)'
                             % ', '.join(sorted(COMBOS.keys())))
    parser.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--P', type=int, default=None,
                        help='threads advanced per batch advance '
                             '(default: half the live points, max 100)')
    parser.add_argument('--nsteps', type=int, default=1)
    parser.add_argument('--frac-remain', type=float, default=0.5)
    parser.add_argument('--max-rotations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--vol-ndraw', type=int, default=2000000)
    parser.add_argument('--vol-nkeep', type=int, default=50001)
    args = parser.parse_args()
    if args.combo not in COMBOS:
        raise ValueError("unknown combo=%s, available: %s"
                         % (args.combo, sorted(COMBOS.keys())))

    problem = make_problem(args.problem, args.dim)
    run_combo(args.combo, problem, args.problem, args.dim, args.outdir, args)


if __name__ == '__main__':
    main()
