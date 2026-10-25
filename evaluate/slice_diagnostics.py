"""
Nested sampling diagnostics as a function of ln(L) level
========================================================

Compares a perfect rejection-sampling reference (perfectns: i.i.d. draws
from the analytically known constrained region) against the usual
slice-sampling step sampler (the mixture proposal) with too-short chains
(nsteps in 8, 4, 3, 2, 1), as a function of ln(L) level.

The ln(L) axis is divided into nlevels equal-volume L-range windows,
defined through the analytically known prior volume V(Lmin) of the
constrained region, so the REAL volume shrinkage per iteration can be
compared with the nested sampling volume estimate. For each sampler
setting, nrep independent replicates (same nlive, fresh seeds) run
simultaneously in lockstep on one core: the live point populations are
batched as a (nrep, nlive, ndim) array so that the initial draws and
likelihood evaluations are vectorized over replicates; the step sampler
is fast, so it is simply looped over replicates. Every iteration of
every replicate is recorded into the L-range window its threshold falls
into, giving per window:

* mean true volume shrinkage / expected mean shrinkage, with the
  standard error from the scatter of the per-replicate means (the
  independent units -- iterations within a replicate are correlated),
* the insertion-order U statistic (uniformity of the insertion ranks),
* the chain relative jump distances RJD = |jump| / <mean pair distance>.

One figure per representative window shows the three diagnostics against
the sampler setting. A perfect sampler has shrinkage ratio 1, U/N = 0.5,
and RJD >= 1.

Usage::

    python evaluate/slice_diagnostics.py [--problem funnel] [--dim 10]
        [--nlive 400] [--nrep 8] [--nlevels 100] [--outdir /tmp]
"""
import argparse
import os
import time

import joblib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import special

from ultranest.mlfriends import SimpleRegion, LocalAffineLayer
from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
from ultranest.ordertest import infinite_U_zscore
from ultranest.perfectns import (GaussContour, BananaContour,
    HyperPyramidContour, ShellContour, SplitEllipsoidContour,
    UncorrelatedFunnelLRPS, mc_information_gain)


# ---------------------------------------------------------------------------
# L-range windows
# ---------------------------------------------------------------------------

def volume_window_edges(vol_func, nlevels=100, vmin_frac=1e-3):
    """Lmin bin edges for nlevels equal-volume windows: window k covers
    the volume fraction [exp(-(k+1)*d), exp(-k*d)] with
    d = ln(1/vmin_frac)/nlevels. All runs and settings share these
    windows."""
    Vfull = vol_func.Vfull
    d = np.log(1.0 / vmin_frac) / nlevels
    fracs = np.exp(-np.arange(nlevels + 1) * d)
    edges = np.array([vol_func.Lmin_of_volume(max(Vfull * fr, 1e-300))
                      for fr in fracs])
    return edges, d


# ---------------------------------------------------------------------------
# collector: nrep replicates in lockstep, batched populations
# ---------------------------------------------------------------------------

def initial_live_points_batched(problem, nlive, nrep, rng):
    """Draw live points for all replicates at once from the prior
    (uniform in u-space): (nrep, nlive, ndim)."""
    ndim = len(problem.paramnames)
    if hasattr(problem, 's_lower') and np.isfinite(problem.s_lower):
        lo = special.ndtr(problem.s_lower / problem.tau)
        hi = special.ndtr(problem.s_upper / problem.tau)
        us = np.empty((nrep, nlive, ndim))
        us[:, :, 0] = rng.uniform(lo, hi, (nrep, nlive))
        us[:, :, 1:] = rng.uniform(size=(nrep, nlive, ndim - 1))
    else:
        us = rng.uniform(size=(nrep, nlive, ndim))
    return us


def mean_pair_distance(us):
    diffs = us[:, None, :] - us[None, :, :]
    d2 = (diffs ** 2).sum(axis=2)
    iu = np.triu_indices(len(us), k=1)
    return np.sqrt(d2[iu]).mean()


def make_region(us):
    """SimpleRegion around the live points, enlarged."""
    layer = LocalAffineLayer()
    layer.optimize(us, us)
    region = SimpleRegion(us, layer)
    r, f = region.compute_enlargement(nbootstraps=10)
    region.maxradiussq = r
    region.enlarge = f
    region.create_ellipsoid(minvol=1e-30)
    return region


class RecordingSliceSampler(SliceSampler):
    """SliceSampler that remembers the start and end of the chain that
    produced each returned sample (finalize_chain resets the history,
    so the chain start must be stashed before the reset)."""

    def finalize_chain(self, region=None, Lmin=None, Ls=None):
        if len(self.history) > 0:
            self.last_jump = (self.history[0][0].copy(),
                              self.history[-1][0].copy())
        SliceSampler.finalize_chain(self, region=region, Lmin=Lmin, Ls=Ls)


def run_replicates(problem, nlive, nsteps, rng, vol_func, edges,
                   nrep=8, max_iters=None):
    """Run nrep independent nested sampling replicates simultaneously in
    lockstep (one core), each maintaining nlive live points with the
    given sampler setting.

    nsteps=None: perfectns rejection reference (i.i.d. draws from the
    analytically known constrained region). Otherwise: the usual slice
    sampler (mixture proposal) with nsteps accepted steps per sample,
    looped over replicates (it is fast); each replicate has its own
    sampler state and region.

    Every iteration of every replicate is recorded into the L-range
    window (see volume_window_edges) its threshold falls into, as
    (window, Lmin, insertion_order, true_shrinkage, rjd).
    """
    ndim = len(problem.paramnames)
    us = initial_live_points_batched(problem, nlive, nrep, rng)
    Ls = problem.loglike(problem.transform(us.reshape(-1, ndim))
                         ).reshape(nrep, nlive)

    top = float(edges[-1])
    done = Ls.min(axis=1) >= top

    mpd = np.array([mean_pair_distance(us[r]) for r in range(nrep)])
    regions = [None] * nrep
    samplers = [None] * nrep
    if nsteps is not None:
        for r in range(nrep):
            regions[r] = make_region(us[r])
            samplers[r] = RecordingSliceSampler(
                nsteps=nsteps,
                generate_direction=generate_mixture_random_direction)

    # independent proposal streams for the rejection reference
    seed0 = int(rng.randint(1 << 30))
    rngs = [np.random.RandomState(seed0 + 7919 * r) for r in range(nrep)]

    if max_iters is None:
        max_iters = 4 * int(np.log(1e3) * nlive)

    recs = [[] for _ in range(nrep)]
    it = 0
    while not done.all() and it < max_iters:
        active = np.where(~done)[0]
        Lmin_vec = Ls[active].min(axis=1)
        win_vec = np.searchsorted(edges, Lmin_vec, side='right') - 1
        i_worst = np.argmin(Ls[active], axis=1)
        V_before = vol_func(Lmin_vec)
        u_dead = us[active, i_worst, :].copy()

        u_new = np.empty((len(active), ndim))
        logl_new = np.empty(len(active))
        for k, r in enumerate(active):
            Lmin_k = float(Lmin_vec[k])
            if nsteps is None:
                # perfectns rejection reference: i.i.d. draw from the
                # analytically known constrained region (candidates are
                # in (s, w) natural coordinates: convert to u-space
                # before the batched likelihood evaluation below)
                while True:
                    candidates, ok = problem.proposal(Lmin_k, rngs[r], 256)
                    if ok.any():
                        break
                u_new[k] = problem.to_u_batch(candidates[ok, :])[0]
            else:
                while True:
                    u, p, ll, nc = samplers[r].__next__(
                        regions[r], Lmin=Lmin_k, us=us[r], Ls=Ls[r],
                        transform=problem.transform,
                        loglike=problem.loglike)
                    if u is not None:
                        break
                u_new[k] = np.asarray(u, dtype=float)
                logl_new[k] = float(ll)

        if nsteps is None:
            # one vectorized likelihood call for all replicates
            logl_new = problem.loglike(problem.transform(u_new))

        # vectorized bookkeeping
        orders = (Ls[active] < logl_new[:, None]).sum(axis=1)
        us[active, i_worst, :] = u_new
        Ls[active, i_worst] = logl_new
        Lmin_after = Ls[active].min(axis=1)
        V_after = vol_func(Lmin_after)
        shrink = V_after / V_before
        # RJD reference: the chain start. The slice chain begins at a
        # live point chosen by the sampler and can persist over several
        # iterations (history[0]); the rejection reference's chain is
        # the single draw, measured from the replaced worst point.
        jump0 = np.empty((len(active), ndim))
        for k, r in enumerate(active):
            if nsteps is None:
                jump0[k] = u_dead[k]
            else:
                jump0[k] = samplers[r].last_jump[0]
        rjd = np.linalg.norm(u_new - jump0, axis=1) / mpd[active]

        for k, r in enumerate(active):
            recs[r].append((int(win_vec[k]), float(Lmin_vec[k]),
                            int(orders[k]), float(shrink[k]),
                            float(rjd[k])))

        done[active] = Lmin_after >= top
        it += 1
        if it % 50 == 0:
            for r in active:
                mpd[r] = mean_pair_distance(us[r])
        if nsteps is not None and it % nlive == 0:
            for r in active:
                regions[r] = make_region(us[r])

    return recs


# ---------------------------------------------------------------------------
# window statistics pooled over replicates
# ---------------------------------------------------------------------------

def aggregate_replicates(all_recs, nlive, d):
    """Pool per-replicate iteration records into per-window statistics.

    all_recs: list with one record list per replicate, records
        (window, Lmin, order, shrink, rjd).

    The mean observed shrinkage is pooled over all replicates; its
    standard error comes from the scatter of the per-replicate means
    (the independent units), not from the correlated iterations within
    a replicate.
    """
    by_win = {}  # window -> list of per-replicate item lists
    for recs in all_recs:
        per_rep = {}
        for win, Lmin, order, shrink, rjd in recs:
            per_rep.setdefault(win, []).append((Lmin, order, shrink, rjd))
        for win, items in per_rep.items():
            by_win.setdefault(win, []).append(items)
    expected = np.exp(-1.0 / nlive)
    stats = []
    for win in sorted(by_win):
        reps = by_win[win]
        pooled = [item for items in reps for item in items]
        arr = np.array(pooled, dtype=float)
        orders = arr[:, 1].astype(int)

        def sem_of(values):
            if len(values) < 2:
                return np.nan
            return float(np.std(values, ddof=1) / np.sqrt(len(values)))

        rep_shrink = [np.mean([it[2] for it in items]) for items in reps]
        rep_u = [np.mean([(it[1] + 0.5) / nlive for it in items])
                 for items in reps]
        rep_rjd = [np.mean([it[3] for it in items]) for items in reps]
        stats.append(dict(
            window=win, Lmin=float(arr[:, 0].mean()),
            volfrac=float(np.exp(-(win + 0.5) * d)),
            nrep=len(reps), niter=len(pooled),
            mean_shrink=float(np.mean(arr[:, 2])),
            sem_shrink=sem_of(rep_shrink),
            U_over_N=float(np.mean((orders + 0.5) / nlive)),
            sem_U=sem_of(rep_u),
            rjd_mean=float(np.mean(arr[:, 3])),
            sem_rjd=sem_of(rep_rjd),
            zscore=float(infinite_U_zscore(orders, nlive)),
            rjds=arr[:, 3], expected=expected,
        ))
    return stats


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_panels(stats_by_setting, setting_list, outdir, nlive,
                name, dim, K, prefix='diagnostics'):
    """Three stacked panels (shrinkage ratio, insertion order U/N, RJD)
    as line plots vs lnX (0 at the left, the final negative value at the
    right), one colored line + 1sigma fill_between band per sampler
    setting."""
    colors = dict(zip(setting_list,
                      ['k'] + list(plt.cm.plasma(np.linspace(0.05, 0.8, 5)))))
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    lnX_min = 0.0
    for setting in setting_list:
        st = sorted(stats_by_setting[setting], key=lambda s: s['window'])
        lnX = np.log([s['volfrac'] for s in st])
        lnX_min = min(lnX_min, lnX.min())
        ratio = np.array([s['mean_shrink'] / s['expected'] for s in st])
        ratio_err = np.array([s['sem_shrink'] / s['expected'] for s in st])
        u = np.array([s['U_over_N'] for s in st])
        u_err = np.array([s['sem_U'] for s in st])
        rjd = np.array([s['rjd_mean'] for s in st])
        rjd_err = np.array([s['sem_rjd'] for s in st])
        c = colors[setting]
        label = '%s (median RJD %.2f)' % (
            setting, np.median([s['rjd_mean'] for s in st]))

        ax = axes[0]
        ax.plot(lnX, ratio, '-', color=c, label=label)
        ax.fill_between(lnX, ratio - ratio_err, ratio + ratio_err,
                        color=c, alpha=0.2, lw=0)

        ax = axes[1]
        ax.plot(lnX, u, '-', color=c)
        ax.fill_between(lnX, u - u_err, u + u_err, color=c, alpha=0.2, lw=0)

        ax = axes[2]
        ax.plot(lnX, rjd, '-', color=c)
        ax.fill_between(lnX, rjd - rjd_err, rjd + rjd_err,
                        color=c, alpha=0.2, lw=0)

    axes[0].axhline(1.0, color='k', lw=0.8, ls=':')
    axes[0].ticklabel_format(useOffset=False, axis='y')
    axes[0].set_ylabel('mean true shrinkage / expected')
    axes[0].set_title('%s d=%d H=%.1f: volume shrinkage ratio '
                      '(1 = correct), nlive = %d'
                      % (name, dim, K, nlive), fontsize=11)
    axes[0].legend(fontsize=8, loc='lower left', framealpha=0.9)

    axes[1].axhline(0.5, color='k', lw=0.8, ls=':')
    axes[1].set_ylabel('insertion order U / N')
    axes[1].set_title('U insertion order test (0.5 = uniform)',
                      fontsize=11)

    axes[2].axhline(1.0, color='k', lw=0.8, ls=':')
    axes[2].set_ylabel('relative jump distance RJD')
    axes[2].set_title('RJD (>1 = moved far enough)', fontsize=11)
    axes[2].set_xlabel('ln(X / X_full)')
    axes[2].set_xlim(0, lnX_min)

    fig.tight_layout()
    filename = os.path.join(outdir, '%s_%s_%dd.png'
                            % (prefix, name, dim))
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


def _run_replicates_uncached(problem, nlive, nsteps, seed, setting, nrep,
                             vol_func, edges, max_iters):
    """One run: nrep independent replicates in lockstep, batched as a
    (nrep, nlive, ndim) population so the operations vectorize over the
    replicate axis. The seed is an argument so that joblib.Memory
    memoizes runs: re-running with the same seed and settings hits the
    cache."""
    np.random.seed(seed ^ 0x5EED)
    rng = np.random.RandomState(seed)
    return run_replicates(problem, nlive, nsteps, rng, vol_func, edges,
                          nrep=nrep, max_iters=max_iters)


memory = joblib.Memory('.', verbose=0)
run_setting_reps = memory.cache(_run_replicates_uncached)


def _run_setting_job(job):
    """Worker: run one memoized 80-replicate run (module-level for
    pickling)."""
    (problem, nlive, nsteps, seed, setting, nrep, vol_func, edges,
     max_iters) = job
    t0 = time.time()
    all_recs = run_setting_reps(problem, nlive, nsteps, seed, setting,
                                nrep, vol_func, edges, max_iters)
    return setting, all_recs, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem',
                        choices=['gauss', 'funnel', 'banana',
                                 'hyperpyramid', 'shell', 'splitellipsoid'],
                        default='funnel')
    parser.add_argument('--mc-ndraw', type=int, default=100000000,
                        help='prior draws for the Monte-Carlo tabulated '
                             'prior volume (gauss, banana)')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--nlive', type=int, default=400)
    parser.add_argument('--nrep', type=int, default=80,
                        help='independent replicates per run')
    parser.add_argument('--nruns', type=int, default=5,
                        help='independent memoized runs per setting '
                             '(nruns x nrep replicates total)')
    parser.add_argument('--nlevels', type=int, default=40)
    parser.add_argument('--vmin-frac', type=float, default=1e-3,
                        help='deepest window: volume fraction V/Vfull')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='safety cap per replicate (default: 4x the '
                             'expected traversal length)')
    parser.add_argument('--outdir', default='evaluate/diagnostics_plots')
    parser.add_argument('--workers', type=int, default=None,
                        help='parallel worker processes (default: CPU count)')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.problem == 'gauss':
        problem = GaussContour(args.dim)
    elif args.problem == 'funnel':
        problem = UncorrelatedFunnelLRPS(args.dim, trunc_sigmas=6.0)
    elif args.problem == 'banana':
        problem = BananaContour(args.dim)
    elif args.problem == 'hyperpyramid':
        problem = HyperPyramidContour(args.dim)
    elif args.problem == 'shell':
        problem = ShellContour(args.dim)
    else:
        problem = SplitEllipsoidContour(args.dim)

    # information gain H for the plot title (MC fallback where the
    # analytic value is unavailable, e.g. the banana at d > 2)
    try:
        K = problem.analytic_H()
    except Exception:
        K = mc_information_gain(problem)

    # shared volume function and L-range windows: identical for all
    # runs and settings
    vol_func = problem.make_volume_function(vmin_frac=args.vmin_frac,
                                            ndraw=args.mc_ndraw)
    edges, d = volume_window_edges(vol_func, nlevels=args.nlevels,
                                   vmin_frac=args.vmin_frac)
    max_iters = args.max_iters
    if max_iters is None:
        max_iters = 4 * int(np.log(1.0 / args.vmin_frac) * args.nlive)

    # sampler settings: perfectns rejection reference, then the usual
    # slice sampler (mixture proposal) with too-short chains
    setting_list = ['rejection', 'slice8', 'slice4', 'slice3', 'slice2',
                    'slice1']
    nsteps_of = dict(rejection=None, slice8=8, slice4=4, slice3=3,
                     slice2=2, slice1=1)
    os.makedirs(args.outdir, exist_ok=True)

    jobs = []
    for si, setting in enumerate(setting_list):
        for run_i in range(args.nruns):
            seed = args.seed * 1000003 + si * 104729 + run_i * 7919
            jobs.append((problem, args.nlive, nsteps_of[setting], seed,
                         setting, args.nrep, vol_func, edges, max_iters))

    all_recs = {}
    from multiprocessing import Pool
    nproc = min(len(jobs), args.workers or os.cpu_count() or 1)
    print('%d jobs on %d workers' % (len(jobs), nproc), flush=True)
    with Pool(processes=nproc) as pool:
        for setting, recs, dt in pool.imap_unordered(_run_setting_job, jobs):
            all_recs.setdefault(setting, []).extend(recs)
            print('  %-8s run: %d replicates x %d iterations in %.1fs' % (
                setting, len(recs), len(recs[0]), dt), flush=True)

    stats_by_setting = {s: aggregate_replicates(all_recs[s], args.nlive, d)
                        for s in setting_list}

    fn = plot_panels(stats_by_setting, setting_list, args.outdir,
                     args.nlive, args.problem, args.dim, K,
                     prefix='diagnostics2')
    print('wrote', fn, flush=True)

    # full data table
    table_fn = os.path.join(args.outdir, 'diagnostics_table.txt')
    with open(table_fn, 'w') as f:
        f.write('# setting window Lmin volfrac nrep niter '
                'mean_shrink sem_shrink U_over_N sem_U zscore '
                'rjd_mean sem_rjd rjd_median\n')
        for setting in setting_list:
            for st in stats_by_setting[setting]:
                f.write(('%s %d %.6f %.4e %d %d %.6f %.6f %.4f %.4f '
                         + '%.3f %.4f %.4f %.4f\n') % (
                    setting, st['window'], st['Lmin'], st['volfrac'],
                    st['nrep'], st['niter'], st['mean_shrink'],
                    st['sem_shrink'], st['U_over_N'], st['sem_U'],
                    st['zscore'], st['rjd_mean'], st['sem_rjd'],
                    np.median(st['rjds']) if len(st['rjds']) else np.nan))
    print('wrote', table_fn)


if __name__ == '__main__':
    main()
