"""
Variable-speed nested sampling IV: simultaneous windows, vectorized
===================================================================

The W windows (equal-iteration segments of a pilot run) are processed
simultaneously: the sampler keeps W ensembles of live points (one per
window, initialised from the pilot's live-point population at the window
boundary) and advances them in lockstep with a vectorized slice sampler.
The vectorized sampler keeps each walker's state -- the proposal
direction, the stepping-out interval (left, right), the bisection
offset, the chain history and the proposal scale -- in arrays, and
evaluates all active walkers' proposals with one vectorized transform
call and one vectorized likelihood call per round. No threads or
processes are used; the simultaneity comes from the vectorization of
the likelihood and transform functions.

The improvement-loop machinery carries over per window: in each loop
every window's ensemble re-runs its window with chains of the window's
current arc length; from the third loop on, windows whose geometric-mean
RJD is still improving (Gompertz criterion) double their arc length,
the others are frozen. RJD, insertion orders and lnX drift are tracked
per arc, and the same panels as in variablespeedns3 are produced.

Usage::

    python evaluate/variablespeedns4.py [--problem banana] [--dim 2]
        [--nlive 400] [--max-loops 11] [--outdir ...]
"""
import argparse
import os
import sys

import numpy as np
from numpy import exp, log, log1p, logaddexp
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variablespeedns3 import (ArcSampler, sample_point, make_problem,
                              make_region, mean_pair_distance,
                              safe_mixture_direction, segment_geomeans,
                              gompertz_decision, plot_panels,
                              plot_rjd_distributions)
from ultranest.perfectns import mc_prior_volume_table


class VectorizedSliceSampler:
    """Slice sampler advancing one walker per ensemble simultaneously.

    Each walker keeps its own slice state -- the proposal direction, the
    stepping-out interval (left, right), the bisection offset, the chain
    history and the proposal scale -- in arrays. All active walkers'
    proposals are evaluated with one vectorized transform call and one
    vectorized likelihood call per round.

    The per-walker state machine replicates SliceSampler: from a fresh
    interval the left stepping-out proposal is made (acceptance doubles
    left, rejection moves to the right proposal); acceptance doubles
    right, rejection starts the bisection; a bisecting draw above the
    likelihood threshold is appended to the chain history and starts a
    fresh interval, a rejection shrinks the interval toward the current
    offset. Proposals outside the unit cube count as rejections.
    """

    PHASE_LEFT, PHASE_RIGHT, PHASE_BISECT = 0, 1, 2

    def __init__(self, nwalkers, ndim, generate_direction, scale=1.0):
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.generate_direction = generate_direction
        self.active = np.zeros(nwalkers, dtype=bool)
        self.fresh = np.zeros(nwalkers, dtype=bool)
        self.phase = np.zeros(nwalkers, dtype=int)
        self.v = np.zeros((nwalkers, ndim))
        self.left = np.ones(nwalkers)
        self.right = np.ones(nwalkers)
        self.uoff = np.zeros(nwalkers)
        self.scale = np.full(nwalkers, float(scale))
        self.next_scale = np.full(nwalkers, float(scale))
        self.history = [[] for _ in range(nwalkers)]
        self.chain_Lmin = np.zeros(nwalkers)
        self.nsteps = np.ones(nwalkers, dtype=int)
        self.rounds = np.zeros(nwalkers, dtype=int)
        self.last_jump = [None] * nwalkers

    def start_chain(self, k, u_start, L_start, Lmin, nsteps):
        """Start walker k's replacement chain from the given live point."""
        self.active[k] = True
        self.fresh[k] = True
        u_start = np.asarray(u_start, dtype=float)
        self.history[k] = [(u_start.copy(), float(L_start))]
        self.chain_Lmin[k] = float(Lmin)
        self.nsteps[k] = int(nsteps)
        self.rounds[k] = 0
        self.last_jump[k] = None

    def advance(self, transform, loglike, regions):
        """One round for all active walkers.

        regions: region object per walker (used for the direction
        generation). Returns a list of (k, u_new, L_new) for the walkers
        whose chain completed in this round.
        """
        act = np.where(self.active)[0]
        if len(act) == 0:
            return []
        self.rounds[act] += 1
        # fresh intervals: generate directions, reset the interval
        for k in act[self.fresh[act]]:
            ui = self.history[k][-1][0]
            self.v[k] = self.generate_direction(ui, regions[k])
            self.left[k] = -self.scale[k]
            self.right[k] = self.scale[k]
            self.uoff[k] = 0.0
            self.phase[k] = self.PHASE_LEFT
            self.fresh[k] = False
        # one proposal per active walker
        prop = np.empty((len(act), self.ndim))
        u_draw = np.zeros(len(act))
        for i, k in enumerate(act):
            ui = self.history[k][-1][0]
            if self.phase[k] == self.PHASE_LEFT:
                prop[i] = ui + self.v[k] * self.left[k]
            elif self.phase[k] == self.PHASE_RIGHT:
                prop[i] = ui + self.v[k] * self.right[k]
            else:
                u = np.random.uniform(self.left[k], self.right[k])
                u_draw[i] = u
                prop[i] = ui + self.v[k] * u
        # all walkers' proposals in one vectorized transform + likelihood call
        pnew = transform(prop)
        Lnew = np.asarray(loglike(pnew), dtype=float).reshape(-1)
        inside = np.all((prop > 0) & (prop < 1), axis=1)
        accepted = (Lnew > self.chain_Lmin[act]) & inside
        # per-phase updates (masks computed before any mutation)
        phases = self.phase[act].copy()
        m0 = phases == self.PHASE_LEFT
        m1 = phases == self.PHASE_RIGHT
        m2 = phases == self.PHASE_BISECT
        # stepping out left: accepted -> double left; rejected -> right
        self.left[act[m0 & accepted]] *= 2
        self.phase[act[m0 & ~accepted]] = self.PHASE_RIGHT
        # stepping out right: accepted -> double right; rejected -> bisect
        self.right[act[m1 & accepted]] *= 2
        go_bisect = act[m1 & ~accepted]
        self.phase[go_bisect] = self.PHASE_BISECT
        if len(go_bisect):
            grow = ((-self.left[go_bisect] > self.next_scale[go_bisect])
                    | (self.right[go_bisect] > self.next_scale[go_bisect]))
            self.next_scale[go_bisect[grow]] *= 1.1
            self.next_scale[go_bisect[~grow]] /= 1.1
        # bisecting: accepted -> append to history, fresh interval;
        # rejected -> shrink the interval toward the current offset
        for i in np.where(m2 & accepted)[0]:
            k = act[i]
            self.history[k].append((prop[i].copy(), float(Lnew[i])))
            self.fresh[k] = True
        for i in np.where(m2 & ~accepted)[0]:
            k = act[i]
            if u_draw[i] == 0:
                pass
            elif u_draw[i] < 0:
                self.left[k] = u_draw[i]
            else:
                self.right[k] = u_draw[i]
        # completed chains
        completed = []
        for k in act:
            if len(self.history[k]) > self.nsteps[k]:
                u_new, L_new = self.history[k][-1]
                self.last_jump[k] = (self.history[k][0][0].copy(),
                                     u_new.copy())
                # finalise the proposal scale as SliceSampler.finalize_chain
                nudge = 1.1 ** (1.0 / self.nsteps[k])
                ns = min(max(self.next_scale[k], self.scale[k] / nudge**10),
                         self.scale[k] * nudge**10)
                self.next_scale[k] = ns
                self.scale[k] = ns
                self.active[k] = False
                completed.append((k, u_new.copy(), float(L_new)))
        return completed


def pilot_run(problem, args):
    """Pilot nested sampling run with nsteps=1: records the arcs and the
    live-point population at every iteration (the window ensembles are
    initialised from the populations at the window boundaries)."""
    ndim = len(problem.paramnames)
    transform, loglike = problem.transform, problem.loglike
    nlive = args.nlive
    np.random.seed(args.seed)
    u = np.random.uniform(size=(nlive, ndim))
    L = np.asarray(loglike(transform(u)), dtype=float).flatten()
    region = make_region(u, ndim)
    sampler = ArcSampler(nsteps=1,
                         generate_direction=safe_mixture_direction)

    logVrem = 0.0
    logZ = -1e300
    h = 0.0
    Lmax = float(L.max())
    arcs = []
    populations = [(u.copy(), L.copy())]
    total = 0
    while total < args.max_iters:
        worst = int(np.argmin(L))
        Lmin = float(L[worst])
        unew, Lnew, u_start, u_end = None, None, None, None
        for _attempt in range(100):
            res = sample_point(sampler, region, Lmin, u, L, transform,
                               loglike)
            if res is not None:
                unew, Lnew, u_start, u_end = res
                break
            sampler = ArcSampler(nsteps=1,
                                 generate_direction=safe_mixture_direction)
        if unew is None:
            raise RuntimeError('pilot sampler could not produce a point at '
                               'Lmin=%g' % Lmin)
        u_start = np.asarray(u_start, dtype=float)
        u_end = np.asarray(u_end, dtype=float)
        rjd = float(np.linalg.norm(u_end - u_start) / mean_pair_distance(u))
        order = int((L < Lnew).sum())
        u[worst] = unew
        L[worst] = Lnew
        logwidth = logVrem + log1p(-exp(-1.0 / nlive))
        logwt = logwidth + Lmin
        logZ_new = logaddexp(logZ, logwt)
        h = (exp(logwt - logZ_new) * Lmin
             + exp(logZ - logZ_new) * (h + logZ) - logZ_new)
        logZ = logZ_new
        logVrem -= 1.0 / nlive
        Lmax = max(Lmax, Lnew)
        arcs.append(dict(Lmin=Lmin, u=unew, L=Lnew, u_start=u_start,
                         rjd=rjd, rjd1=rjd, order=order, x=-logVrem,
                         cur_len=1, nsteps=1))
        total += 1
        populations.append((u.copy(), L.copy()))
        region.u[worst] = u[worst]
        region.unormed[worst] = region.transformLayer.transform(u[worst])
        region.ellipsoid_center = np.mean(region.u, axis=0)
        if total % nlive == 0:
            region = make_region(u, ndim)
        logZremain = Lmax + logVrem
        if logZremain - logZ < log(args.frac_remain) \
                or (Lmax - float(L.min())) < args.Lepsilon:
            break

    logzerr = float(np.sqrt(max(h, 0.0) / nlive))
    logz = float(logaddexp(logZ, Lmax + logVrem))
    return arcs, populations, dict(logz=logz, logzerr=logzerr, total=total)


def multi_ensemble_loop(snapshots, boundary_x0, targets, nsteps, nlive,
                        problem, args, vol_func):
    """One improvement loop: all W window ensembles run their window
    simultaneously with the vectorized slice sampler.

    Each ensemble is initialised from its boundary snapshot and produces
    targets[k] replacements with chains of nsteps[k] accepted steps. All
    active walkers advance in lockstep; each round evaluates their
    proposals with one vectorized transform and likelihood call.

    Returns (records, final Lmax over all ensembles).
    """
    W = len(snapshots)
    ndim = len(problem.paramnames)
    transform, loglike = problem.transform, problem.loglike
    Lgrid = getattr(vol_func, 'Lgrid', None)
    L_covered = float(Lgrid[-1]) if Lgrid is not None else np.inf
    sampler = VectorizedSliceSampler(W, ndim, safe_mixture_direction)
    ensembles = []
    for k in range(W):
        u, L = snapshots[k][0].copy(), snapshots[k][1].copy()
        ensembles.append(dict(u=u, L=L, region=make_region(u, ndim),
                              count=0, target=int(targets[k]),
                              worst=int(np.argmin(L)),
                              logVrem=-float(boundary_x0[k]),
                              restarts=0))
    records = []
    while True:
        # start chains for idle walkers of unfinished ensembles
        for k, e in enumerate(ensembles):
            if e['count'] < e['target'] and not sampler.active[k]:
                Lmin = float(e['L'][e['worst']])
                i = np.random.randint(len(e['L']))
                sampler.start_chain(k, e['u'][i], e['L'][i], Lmin, nsteps[k])
        if not sampler.active.any():
            break
        completed = sampler.advance(transform, loglike,
                                    [e['region'] for e in ensembles])
        # stall guard: restart chains that take too long
        for k in np.where(sampler.active)[0]:
            if sampler.rounds[k] > 100 * sampler.nsteps[k] + 1000:
                e = ensembles[k]
                e['restarts'] += 1
                if e['restarts'] > 100:
                    raise RuntimeError('window %d chain keeps stalling at '
                                       'Lmin=%g' % (k, e['L'][e['worst']]))
                print('  window %d chain stalled after %d rounds; '
                      'restarting' % (k, sampler.rounds[k]), flush=True)
                i = np.random.randint(len(e['L']))
                sampler.start_chain(k, e['u'][i], e['L'][i],
                                    float(e['L'][e['worst']]), nsteps[k])
        for k, u_new, L_new in completed:
            e = ensembles[k]
            worst = e['worst']
            Lmin = float(e['L'][worst])
            u_start, u_end = sampler.last_jump[k]
            mpd = mean_pair_distance(e['u'])
            rjd = float(np.linalg.norm(u_end - u_start) / mpd)
            order = int((e['L'] < L_new).sum())
            e['u'][worst] = u_new
            e['L'][worst] = L_new
            e['region'].u[worst] = u_new
            e['region'].unormed[worst] = \
                e['region'].transformLayer.transform(u_new)
            e['region'].ellipsoid_center = np.mean(e['u'], axis=0)
            if (e['count'] + 1) % nlive == 0:
                e['region'] = make_region(e['u'], ndim)
            e['count'] += 1
            logwidth = e['logVrem'] + log1p(-exp(-1.0 / nlive))
            e['logVrem'] -= 1.0 / nlive
            x = -e['logVrem']
            drift = np.nan
            if Lmin <= L_covered:
                X_actual = max(float(vol_func(Lmin)), 1e-300)
                drift = log(X_actual) + x
            records.append(dict(Lmin=Lmin, L=L_new, rjd=rjd, order=order,
                                x=x, segment=k, nsteps=nsteps[k],
                                drift=drift, gap=False))
            e['worst'] = int(np.argmin(e['L']))
    Lmax_final = float(max(e['L'].max() for e in ensembles))
    records.sort(key=lambda r: r['x'])
    return records, Lmax_final


def combined_logz(records, nlive, Lmax_final):
    """Bookkeeping lnZ over the combined window records (x-sorted) plus
    the tail from the final remaining volume."""
    recs = sorted(records, key=lambda r: r['x'])
    logZ = -1e300
    h = 0.0
    x_prev = 0.0
    for r in recs:
        logwidth = -x_prev + log1p(-exp(-1.0 / nlive))
        logwt = logwidth + r['Lmin']
        logZ_new = logaddexp(logZ, logwt)
        h = (exp(logwt - logZ_new) * r['Lmin']
             + exp(logZ - logZ_new) * (h + logZ) - logZ_new)
        logZ = logZ_new
        x_prev = r['x']
    logZ = float(logaddexp(logZ, Lmax_final - x_prev))
    logzerr = float(np.sqrt(max(h, 0.0) / nlive))
    return logZ, logzerr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['banana', 'hyperpyramid',
                                              'funnel', 'gauss'],
                        default='banana')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--nlive', type=int, default=400)
    parser.add_argument('--nsegments', type=int, default=20)
    parser.add_argument('--max-loops', type=int, default=11,
                        help='loops (pilot + improvements); the arc length '
                             'doubles per improving window per loop')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--frac-remain', type=float, default=0.01)
    parser.add_argument('--Lepsilon', type=float, default=0.001)
    parser.add_argument('--max-iters', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--vol-ndraw', type=int, default=20000000)
    parser.add_argument('--vol-nkeep', type=int, default=200001)
    parser.add_argument('--outdir', default='evaluate/variablespeed_plots4')
    args = parser.parse_args()

    problem = make_problem(args)
    if args.problem in ('gauss', 'banana'):
        vol_func = mc_prior_volume_table(problem, ndraw=args.vol_ndraw,
                                         nkeep=args.vol_nkeep)
    else:
        vol_func = problem.make_volume_function()
    Lgrid = getattr(vol_func, 'Lgrid', None)
    L_covered = float(Lgrid[-1]) if Lgrid is not None else np.inf

    arcs, populations, res1 = pilot_run(problem, args)
    try:
        analytic = float(problem.analytic_lnZ())
    except Exception:
        analytic = np.nan
    W = args.nsegments
    total = len(arcs)
    bounds = np.linspace(0, total, W + 1).astype(int)
    for i, arc in enumerate(arcs):
        arc['segment'] = int(min(np.searchsorted(bounds, i,
                                                 side='right') - 1, W - 1))
        drift = np.nan
        if arc['Lmin'] <= L_covered:
            drift = log(max(float(vol_func(arc['Lmin'])), 1e-300)) + arc['x']
        arc['drift'] = drift
    print('loop 1 (pilot, nsteps=1): %d arcs, lnZ = %.4f +- %.4f' % (
        total, res1['logz'], res1['logzerr']), flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    title = '%s, dim=%d, nlive=%d, %d windows (vectorized)' % (
        args.problem, args.dim, args.nlive, W)

    pilot_records = [dict(Lmin=a['Lmin'], x=a['x'], rjd=a['rjd'],
                          order=a['order'], segment=a['segment'],
                          nsteps=1, drift=a['drift']) for a in arcs]
    snapshots = [populations[b] for b in bounds[:-1]]
    boundary_x0 = bounds[:-1] / float(args.nlive)
    targets = (bounds[1:] - bounds[:-1]).astype(int)
    del populations

    loop_records = [pilot_records]
    results = [res1]
    gm_history = [segment_geomeans(pilot_records, W)]
    nsteps = np.full(W, 2, dtype=int)  # the first improvement loop doubles
    fn = plot_panels(loop_records, results, W, args.outdir, title, analytic,
                     gm_history)
    print('wrote', fn, flush=True)
    fn = plot_rjd_distributions(loop_records, W, args.outdir, title)
    print('wrote', fn, flush=True)
    for loop in range(2, args.max_loops + 1):
        records, Lmax_final = multi_ensemble_loop(
            snapshots, boundary_x0, targets, nsteps, args.nlive, problem,
            args, vol_func)
        logz, logzerr = combined_logz(records, args.nlive, Lmax_final)
        results.append(dict(logz=logz, logzerr=logzerr, total=len(records)))
        loop_records.append(records)
        gm_history.append(segment_geomeans(records, W))
        print('loop %d: %d replacements, lnZ = %.4f +- %.4f' % (
            loop, len(records), logz, logzerr), flush=True)
        stop = False
        if len(gm_history) >= 3:
            new_nsteps = nsteps.copy()
            improving = set()
            for k in range(W):
                gms = np.asarray([h[k] for h in gm_history], dtype=float)
                if gompertz_decision(gms):
                    improving.add(k)
                    new_nsteps[k] = nsteps[k] * 2
            print('Gompertz criterion: improving %d/%d windows: %s' % (
                len(improving), W,
                ['%d->%d' % (k, new_nsteps[k])
                 for k in sorted(improving)]), flush=True)
            nsteps = new_nsteps
            if not improving:
                print('no windows predicted to improve; done.', flush=True)
                stop = True
        else:
            # before the criterion can fire, double all windows' arc length
            nsteps = nsteps * 2
        fn = plot_panels(loop_records, results, W, args.outdir, title,
                         analytic, gm_history)
        print('wrote', fn, flush=True)
        fn = plot_rjd_distributions(loop_records, W, args.outdir, title)
        print('wrote', fn, flush=True)
        if stop:
            break

    print()
    print('final lnZ = %.4f +- %.4f' % (
        results[-1]['logz'], results[-1]['logzerr']))
    if np.isfinite(analytic):
        print('analytic lnZ = %.4f' % analytic)


if __name__ == '__main__':
    main()
