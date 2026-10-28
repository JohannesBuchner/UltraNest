"""
Variable-speed nested sampling III: per-segment arc extension
=============================================================

A first nested sampling run with nsteps=1 records, for every dead point,
the arc Lmin->L (threshold -> replacement point) together with the chain
start that produced it and its relative jump distance (RJD = |jump| /
<live point pair distance>). The run's iterations split into 20
equal-iteration segments (equal on the -lnX axis, since every iteration
shrinks the volume by 1/nlive).

The run is then *resumed* from the recorded arcs with the resume/point-
stack logic: each arc replaces the current worst live point once the
threshold has passed its recorded Lmin and its point exceeds it. In the
first resumed loop every arc is extended by a further nsteps accepted
steps at the current threshold, producing modified arcs Lmin->L'. From
the point where three loops have been collected, each segment's
geometric-mean RJD is fit with a Gompertz curve, and the segment is only
extended further if the next loop is predicted to improve the geometric
mean by at least half the change predicted by the linear trend slope of
the collected values; otherwise the segment's arcs are replayed as
recorded. RJD is tracked per arc throughout. Where the recorded arcs
cannot serve (the replay lags the recorded trajectory), the step sampler
provides the replacement at the segment's current arc length.

Plots, per loop, as in variablespeed_panels.png: the lnL-vs-lnX
trajectories, the per-segment insertion-order U test, the per-segment
mean RJD, and the lnX drift (the latter only where the volume function
of the test problem is available).

Usage::

    python evaluate/variablespeedns3.py [--problem banana] [--dim 2]
        [--nlive 400] [--max-loops 11] [--outdir ...]
"""
import argparse
import os

import numpy as np
from numpy import exp, log, log1p, logaddexp
from scipy import stats
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ultranest.mlfriends import MLFriends, LocalAffineLayer, ScalingLayer
from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
from ultranest.ordertest import infinite_U_zscore
from ultranest.perfectns import (
    BananaGaussianLRPS, HyperPyramidLRPS, NonIsotropicGaussianLRPS,
    UncorrelatedFunnelLRPS, mc_prior_volume_table)


class ArcSampler(SliceSampler):
    """SliceSampler that stashes the chain start and end of the arc that
    produced each returned sample (finalize_chain resets the history, so
    the chain start must be stashed before the reset)."""

    def finalize_chain(self, region=None, Lmin=None, Ls=None):
        if len(self.history) > 0:
            self.last_jump = (self.history[0][0].copy(),
                              self.history[-1][0].copy())
        SliceSampler.finalize_chain(self, region=region, Lmin=Lmin, Ls=Ls)


_direction_rejects = [0]
_stall_rejects = [0]


def safe_mixture_direction(ui, region):
    """generate_mixture_random_direction, but never returns a zero or
    non-finite direction: a zero direction makes every slice stepping-out
    proposal equal to the chain start, so the interval doubles forever
    (0 * inf = NaN, then uniform(-inf, inf) fails)."""
    for _ in range(100):
        v = generate_mixture_random_direction(ui, region)
        if np.all(np.isfinite(v)) and np.linalg.norm(v) > 1e-300:
            return v
        _direction_rejects[0] += 1
        if _direction_rejects[0] == 1:
            udup = len(np.unique(region.u, axis=0)) != len(region.u)
            axes = np.asarray(region.transformLayer.axes)
            print('  WARNING: rejected degenerate direction '
                  '(duplicate live points: %s, axes finite: %s)'
                  % (udup, bool(np.all(np.isfinite(axes)))), flush=True)
    v = np.random.normal(size=len(ui))
    return v / np.linalg.norm(v)


def make_arc_sampler():
    return ArcSampler(nsteps=1, generate_direction=safe_mixture_direction)


def sample_point(sampler, region, Lmin, u, L, transform, loglike,
                 max_calls=100000):
    """Drive the sampler until it produces a replacement point, guarding
    against stuck chains (degenerate intervals, non-finite proposals).

    Returns (unew, Lnew, u_start, u_end), or None if the sampler failed;
    the caller then restarts the chain or falls back.
    """
    calls = 0
    while True:
        try:
            unew, pnew, Lnew, nc = sampler.__next__(
                region, Lmin=Lmin, us=u, Ls=L,
                transform=transform, loglike=loglike)
        except (OverflowError, ValueError, FloatingPointError,
                np.linalg.LinAlgError) as e:
            print('  sampler error at Lmin=%.4g: %s' % (Lmin, repr(e)),
                  flush=True)
            return None
        calls += 1
        if unew is not None:
            unew = np.asarray(unew, dtype=float).flatten()
            Lnew = float(np.asarray(Lnew).flatten()[0])
            if not (np.all(np.isfinite(unew)) and np.isfinite(Lnew)):
                print('  sampler returned non-finite point at Lmin=%.4g'
                      % Lmin, flush=True)
                return None
            u_start, u_end = sampler.last_jump
            if np.linalg.norm(unew - u_start) < 1e-12:
                # the proposal scale collapsed below the float resolution:
                # the chain only creeps in ~1 ULP steps, which would drag
                # the population to the likelihood maximum; the caller
                # restarts with a fresh proposal scale
                _stall_rejects[0] += 1
                if _stall_rejects[0] <= 5:
                    print('  sampler chain only creeps (scale=%.3g) at '
                          'Lmin=%.4g; restarting with fresh scale'
                          % (sampler.scale, Lmin), flush=True)
                return None
            return unew, Lnew, u_start, u_end
        if calls > max_calls:
            print('  sampler stalled after %d calls at Lmin=%.4g'
                  % (calls, Lmin), flush=True)
            return None


def make_problem(args):
    """Test problem (the volume function is only used for the drift
    panel)."""
    if args.problem == 'banana':
        return BananaGaussianLRPS.standard_problem(args.dim)
    if args.problem == 'hyperpyramid':
        return HyperPyramidLRPS(args.dim)
    if args.problem == 'funnel':
        return UncorrelatedFunnelLRPS(args.dim, trunc_sigmas=6.0)
    if args.problem == 'gauss':
        return NonIsotropicGaussianLRPS(args.dim)
    raise ValueError("Unknown problem=%s" % args.problem)


def make_region(us, x_dim):
    """MLFriends region around the live points, enlarged."""
    if x_dim > 1:
        layer = LocalAffineLayer()
    else:
        layer = ScalingLayer()
    layer.optimize(us, us)
    region = MLFriends(us, layer)
    r, f = region.compute_enlargement(nbootstraps=10)
    region.maxradiussq = r
    region.enlarge = f
    region.create_ellipsoid(minvol=1e-30)
    return region


def mean_pair_distance(us):
    return float(pdist(us).mean())


def first_run(problem, args):
    """Loop 1: nested sampling with nsteps=1, recording the arcs.

    Each arc records the threshold Lmin, the replacement point (u, L),
    the chain start u_start, the RJD, the insertion order and the
    bookkeeping position x = -lnX. Also returns the initial population
    for the replays.
    """
    ndim = len(problem.paramnames)
    transform, loglike = problem.transform, problem.loglike
    nlive = args.nlive
    np.random.seed(args.seed)
    u = np.random.uniform(size=(nlive, ndim))
    L = np.asarray(loglike(transform(u)), dtype=float).flatten()
    u0, L0 = u.copy(), L.copy()
    region = make_region(u, ndim)
    sampler = make_arc_sampler()

    logVrem = 0.0
    logZ = -1e300
    h = 0.0
    Lmax = float(L.max())
    arcs = []
    total = 0
    while total < args.max_iters:
        worst = int(np.argmin(L))
        Lmin = float(L[worst])
        unew, Lnew, u_start, u_end = None, None, None, None
        for _attempt in range(100):
            res = sample_point(sampler, region, Lmin, u, L, transform, loglike)
            if res is not None:
                unew, Lnew, u_start, u_end = res
                break
            sampler = make_arc_sampler()
        if unew is None:
            raise RuntimeError('sampler could not produce a point at '
                               'Lmin=%g' % Lmin)
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
        arcs.append(dict(Lmin=Lmin, u=unew, L=Lnew, u_start=u_start.copy(),
                         rjd=rjd, rjd1=rjd, order=order, x=-logVrem,
                         cur_len=1, nsteps=1))
        total += 1
        region.u[worst] = u[worst]
        region.unormed[worst] = region.transformLayer.transform(u[worst])
        region.ellipsoid_center = np.mean(region.u, axis=0)
        if total % nlive == 0:
            region = make_region(u, ndim)
        logZremain = Lmax + logVrem
        if logZremain - logZ < log(args.frac_remain) \
                or (Lmax - float(L.min())) < args.Lepsilon:
            break

    # 20 equal-iteration segments (equal on the -lnX axis)
    nseg = args.nsegments
    seg_bounds = np.linspace(0, len(arcs), nseg + 1).astype(int)
    for i, arc in enumerate(arcs):
        arc['segment'] = int(min(np.searchsorted(seg_bounds, i,
                                                 side='right') - 1, nseg - 1))
    logzerr = float(np.sqrt(max(h, 0.0) / nlive))
    logz = float(logaddexp(logZ, Lmax + logVrem))
    return arcs, u0, L0, dict(logz=logz, logzerr=logzerr, total=total)


def replay(arcs, u0, L0, extend_all, marked, nlive, problem, args, vol_func):
    """One resumed loop: walk the arcs with the resume/point-stack logic.

    extend_all: extend every arc (the first resumed loop); otherwise only
    the segments in `marked` are extended (None extends all). The
    extension continues each arc's chain from its endpoint for its current
    length in accepted steps at the current threshold, doubling the arc
    length. Where no recorded arc can serve (the replay lags the recorded
    trajectory), the step sampler provides the replacement at the
    segment's current arc length.

    Returns (records, result); records hold per dead point the threshold,
    the replacement likelihood, the RJD, the insertion order, the
    bookkeeping position and the lnX drift.
    """
    ndim = len(problem.paramnames)
    transform, loglike = problem.transform, problem.loglike
    u = u0.copy()
    L = L0.copy()
    region = make_region(u, ndim)
    sampler = make_arc_sampler()
    Lgrid = getattr(vol_func, 'Lgrid', None)
    L_covered = float(Lgrid[-1]) if Lgrid is not None else np.inf

    logVrem = 0.0
    logZ = -1e300
    h = 0.0
    Lmax = float(L.max())
    stack = list(arcs)
    records = []
    total = 0
    ngap = 0
    while stack and total < args.max_iters:
        worst = int(np.argmin(L))
        Lmin = float(L[worst])
        j = next((j for j, arc in enumerate(stack)
                  if arc['Lmin'] <= Lmin and arc['L'] > Lmin), None)
        if j is not None:
            arc = stack.pop(j)
            gap = False
        else:
            arc = stack[0]
            gap = True
            ngap += 1
            if arc['Lmin'] <= Lmin:
                # the replay leads: the recorded arc's point is below the
                # current threshold (extension endpoints are less
                # threshold-biased than the 1-step recorded points), so the
                # arc can never serve again; consume it. When instead the
                # threshold has not reached arc['Lmin'] yet (the replay
                # lags), keep the arc for later.
                stack.pop(0)
        mpd = mean_pair_distance(u)
        do_extend = (extend_all or marked is None
                     or arc['segment'] in marked) and not gap
        if do_extend:
            nsteps_ext = arc['cur_len']
            res = None
            for _attempt in range(10):
                sampler.nsteps = nsteps_ext
                # continue the arc's chain from its endpoint
                sampler.new_chain()
                sampler.history = [(arc['u'].copy(), float(arc['L']))]
                res = sample_point(sampler, region, Lmin, u, L, transform,
                                   loglike,
                                   max_calls=1000 * nsteps_ext + 10000)
                if res is not None:
                    break
                # the extension failed; restart with a fresh proposal scale
                sampler = make_arc_sampler()
            if res is None:
                # the extension failed; replay the arc as recorded
                unew, Lnew, rjd = arc['u'], arc['L'], arc['rjd']
                used_nsteps = arc['cur_len']
            else:
                unew, Lnew, u_start, u_end = res
                rjd = float(np.linalg.norm(unew - arc['u_start']) / mpd)
                arc['u'], arc['L'], arc['rjd'] = unew, Lnew, rjd
                arc['cur_len'] *= 2
                used_nsteps = arc['cur_len']
        elif gap:
            sampler.nsteps = max(1, arc['cur_len'])
            unew, Lnew, u_start, u_end = None, None, None, None
            for _attempt in range(100):
                res = sample_point(sampler, region, Lmin, u, L, transform,
                                   loglike)
                if res is not None:
                    unew, Lnew, u_start, u_end = res
                    break
                sampler = make_arc_sampler()
            if unew is None:
                raise RuntimeError('sampler could not produce a gap-fill '
                                   'point at Lmin=%g' % Lmin)
            rjd = float(np.linalg.norm(u_end - u_start) / mpd)
            used_nsteps = max(1, arc['cur_len'])
        else:
            unew, Lnew, rjd = arc['u'], arc['L'], arc['rjd']
            used_nsteps = arc['cur_len']
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
        x = -logVrem
        drift = np.nan
        X_actual = max(float(vol_func(Lmin)), 1e-300)
        if Lmin <= L_covered:
            drift = log(X_actual) + x
        records.append(dict(Lmin=Lmin, L=Lnew, rjd=rjd, order=order, x=x,
                            segment=arc['segment'], nsteps=used_nsteps,
                            drift=drift, gap=gap))
        total += 1
        if total % 1000 == 0:
            print('    replay it=%d Lmin=%.4g maxL=%.4g meanL=%.4g '
                  'ndistinct=%d ngap=%d'
                  % (total, Lmin, float(L.max()), float(L.mean()),
                     len(np.unique(u, axis=0)), ngap), flush=True)
        region.u[worst] = u[worst]
        region.unormed[worst] = region.transformLayer.transform(u[worst])
        region.ellipsoid_center = np.mean(region.u, axis=0)
        if total % nlive == 0:
            region = make_region(u, ndim)

    logzerr = float(np.sqrt(max(h, 0.0) / nlive))
    logz = float(logaddexp(logZ, Lmax + logVrem))
    return records, dict(logz=logz, logzerr=logzerr, total=total,
                         ngap=ngap, nleft=len(stack))


def mark_segments(arcs, nsegments, alpha):
    """One-sided Wilcoxon on the paired per-arc RJD increase (current
    minus loop-1) per segment; returns (marked set, per-segment p)."""
    marked = set()
    pvalues = {}
    for k in range(nsegments):
        d = np.asarray([arc['rjd'] - arc['rjd1'] for arc in arcs
                        if arc['segment'] == k], dtype=float)
        p = np.nan
        if len(d) >= 10:
            try:
                stat, p = stats.wilcoxon(d, alternative='greater')
                p = float(p)
            except ValueError:
                p = np.nan
        pvalues[k] = p
        if np.isfinite(p) and p < alpha:
            marked.add(k)
    return marked, pvalues


def segment_geomeans(records, nsegments):
    """Geometric mean RJD per segment (nan where a segment is empty)."""
    out = np.full(nsegments, np.nan)
    for k in range(nsegments):
        vals = np.asarray([r['rjd'] for r in records if r['segment'] == k],
                          dtype=float)
        if len(vals):
            out[k] = np.exp(np.mean(np.log(vals)))
    return out


def _gompertz(x, A, b, c):
    return A * np.exp(-b * np.exp(-c * x))


def fit_gompertz(gms):
    """Fit a Gompertz curve g(j) = A exp(-b exp(-c j)) to the
    geometric-mean values at loop indices 0..len-1.

    Initialises the asymptote A as an extension of the last two values
    (the last trend step doubled: y_i + 2 (y_i - y_{i-1})), solves the
    double-log-linearised problem ln(-ln(y/A)) = ln b - c j by linear
    least squares, and refines all three parameters with gradient-based
    optimisation of the non-linear least squares (analytic jacobian).
    Returns (popt, ok).
    """
    j = np.arange(len(gms), dtype=float)
    try:
        ymax = float(np.nanmax(gms))
        if not np.isfinite(ymax) or ymax <= 0:
            return None, False
        # asymptote: the maximum as an extension of the last two values
        A0 = float(gms[-1]) + 2.0 * (float(gms[-1]) - float(gms[-2]))
        if not np.isfinite(A0) or A0 <= ymax:
            A0 = ymax * 1.1 + 1e-300
        u = np.asarray(gms, dtype=float) / A0
        if not np.all((u > 0) & (u < 1)):
            A0 = ymax * 1.1 + 1e-300
            u = np.asarray(gms, dtype=float) / A0
        b0, c0 = 1.0, 0.5
        if np.all((u > 0) & (u < 1)):
            # double-log linearisation with A fixed:
            # ln(-ln(y/A)) = ln b - c j
            z = np.log(-np.log(u))
            slope, intercept = np.polyfit(j, z, 1)
            if np.isfinite(slope) and np.isfinite(intercept):
                c0 = max(-float(slope), 1e-6)
                b0 = max(float(np.exp(intercept)), 1e-300)

        def residual(p):
            A, b, c = p
            return gms - _gompertz(j, A, b, c)

        def jacobian(p):
            A, b, c = p
            uu = np.exp(-c * j)
            g = _gompertz(j, A, b, c)
            return np.column_stack([-np.exp(-b * uu), g * uu,
                                    -g * b * j * uu])

        result = least_squares(residual, np.array([A0, b0, c0]),
                               jac=jacobian,
                               bounds=(np.array([0.0, 0.0, 0.0]),
                                       np.array([np.inf, np.inf, np.inf])),
                               max_nfev=200)
        return np.asarray(result.x), True
    except Exception:
        return None, False


def gompertz_decision(gms, factor=0.5):
    """Whether a segment should be improved in the next loop.

    Fits a Gompertz curve g(j) = A exp(-b exp(-c j)) to the geometric-mean
    RJD values (j = 0-based loop index) and predicts the next loop's
    improvement; improve only if it reaches `factor` times the change
    predicted by the linear trend slope fit to all collected values.
    Defaults to improving when fewer than 3 loops are collected or the fit
    fails.
    """
    if len(gms) < 3:
        return True
    slope = np.polyfit(np.arange(len(gms), dtype=float), gms, 1)[0]
    popt, ok = fit_gompertz(gms)
    if not ok:
        return True
    predicted = _gompertz(len(gms), *popt) - _gompertz(len(gms) - 1, *popt)
    # strict: a flat sequence predicts no improvement and stops the looping
    return bool(predicted > factor * slope)


def segment_stats(records, nsegments, nlive):
    """Per-segment mean RJD, U-test z-score and center position."""
    out = []
    for k in range(nsegments):
        recs = [r for r in records if r['segment'] == k]
        if not recs:
            out.append(dict(segment=k, x=np.nan, rjd=np.nan, u_z=np.nan))
            continue
        rjd = np.asarray([r['rjd'] for r in recs])
        orders = np.asarray([r['order'] for r in recs])
        xs = np.asarray([r['x'] for r in recs])
        out.append(dict(
            segment=k, x=float(xs.mean()),
            rjd=float(rjd.mean()),
            u_z=float(infinite_U_zscore(orders, int(round(nlive))))))
    return out


def plot_panels(loop_records, results, nsegments, outdir, title,
                analytic=np.nan, gm_history=None, nlive_per_loop=None,
                loop_label='loop'):
    """The variablespeed_panels.png-style figure: lnL-vs-lnX trajectories,
    the arc chain length (nsteps) vs -lnX, the per-segment U test, the
    per-segment mean RJD (one line per loop), the lnX drift, the current
    Gompertz fits per segment (dataset+curve in one color per segment),
    and lnZ per loop iteration."""
    nloop = len(loop_records)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, nloop))
    segcolors = plt.cm.tab20(np.linspace(0.0, 1.0, nsegments))
    fig, axes = plt.subplots(7, 1, figsize=(7, 21))

    xmax = 0.0
    for j, records in enumerate(loop_records):
        c = colors[j]
        xs = np.asarray([r['x'] for r in records])
        Ls = np.asarray([r['Lmin'] for r in records])
        axes[0].plot(xs, Ls, '-', color=c, lw=0.8,
                     label='%s %d' % (loop_label, j + 1))
        ns = np.asarray([r['nsteps'] for r in records], dtype=float)
        axes[1].scatter(xs, ns, s=2, color=c, alpha=0.5, linewidths=0)
        nl = 400 if nlive_per_loop is None else int(nlive_per_loop[j])
        segs = segment_stats(records, nsegments, nl)
        xc = [s['x'] for s in segs]
        axes[2].plot(xc, [s['u_z'] for s in segs], 'o-', color=c,
                     ms=3, lw=0.8)
        axes[3].plot(xc, [s['rjd'] for s in segs], 'o-', color=c,
                     ms=3, lw=0.8)
        drifts = np.asarray([r['drift'] for r in records], dtype=float)
        m = np.isfinite(drifts)
        axes[4].plot(xs[m], drifts[m], '-', color=c, lw=0.8)
        xmax = max(xmax, float(xs.max()))

    axes[0].set_ylabel('lnL')
    axes[0].set_title(title, fontsize=11)
    axes[0].legend(fontsize=8, loc='lower right', framealpha=0.9)

    axes[1].set_yscale('log', base=2)
    axes[1].set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    axes[1].set_yticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    axes[1].set_ylabel('nsteps (arc chain length)')
    axes[1].set_title('arc chain length vs -lnX (one color per %s)'
                      % loop_label,
                      fontsize=11)

    axes[2].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[2].set_ylabel('insertion order U test z-score')
    axes[2].set_title('U test per segment (0 = uniform)', fontsize=11)

    axes[3].axhline(1.0, color='k', lw=0.8, ls=':')
    axes[3].set_ylabel('mean RJD per segment')
    axes[3].set_title('chain relative jump distance (>1 = moved far '
                      'enough)', fontsize=11)

    axes[4].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[4].set_ylabel('lnX drift ln(V(Lmin) / X_bookkeeping)')
    axes[4].set_title('lnX drift (>0: actual volume lags bookkeeping, '
                      'lnZ biased)', fontsize=11)
    axes[4].set_xlabel('-ln(X) (bookkeeping)')
    for ax in axes[:5]:
        ax.set_xlim(0, xmax)

    if gm_history is not None:
        nloop_gm = len(gm_history)
        for k in range(nsegments):
            gms = np.asarray([h[k] for h in gm_history], dtype=float)
            m = np.isfinite(gms)
            if not m.any():
                continue
            it = np.arange(1, nloop_gm + 1)[m]
            axes[5].plot(it, gms[m], 'o', color=segcolors[k], ms=3)
            if m.sum() >= 3:
                popt, ok = fit_gompertz(gms[m])
                if ok:
                    jj = np.linspace(0, m.sum() - 1, 100)
                    axes[5].plot(jj + 1, _gompertz(jj, *popt), '-',
                                 color=segcolors[k], lw=1.0)
    axes[5].set_xlabel('%s iteration' % loop_label)
    axes[5].set_ylabel('geometric-mean RJD')
    axes[5].set_title('current Gompertz fits per segment (points: data, '
                      'line: fit; one color per segment)', fontsize=11)
    if nloop >= 1:
        axes[5].set_xticks(range(1, max(nloop, 1) + 1))

    iters = np.arange(1, len(results) + 1)
    axes[6].errorbar(iters, [r['logz'] for r in results],
                     yerr=[r['logzerr'] for r in results],
                     fmt='o-', color='k', ms=4, lw=1, capsize=3)
    if np.isfinite(analytic):
        axes[6].axhline(analytic, color='r', lw=1, ls='--',
                        label='analytic')
        axes[6].legend(fontsize=8)
    axes[6].set_xticks(iters)
    axes[6].set_xlabel('%s iteration' % loop_label)
    axes[6].set_ylabel('lnZ')
    axes[6].set_title('lnZ per %s iteration (error bars: '
                      'sqrt(H/nlive))' % loop_label, fontsize=11)

    fig.tight_layout()
    filename = os.path.join(outdir, 'variablespeedns3_panels.png')
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


def plot_rjd_distributions(loop_records, nsegments, outdir, title,
                           width_frac=0.1, shift_frac=0.5,
                           loop_label='loop'):
    """Separate figure: one panel per segment with the RJD distributions at
    each loop (log10-RJD histograms, colored by loop), plus the per-segment
    geometric-mean RJD saturation analysis.

    Saturation of the geometric mean is identified two ways:
    (a) the loop-to-loop shift of the geometric mean is small relative to
        the width of the distribution: |shift| < width_frac * std(log rjd);
    (b) the most recent shift is small compared to the previous shifts:
        |shift| < shift_frac * median(previous |shifts|).

    Writes <outdir>/rjd_distributions.png and a per-segment table to
    <outdir>/rjd_distributions.txt; returns the filename.
    """
    nloop = len(loop_records)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, nloop))
    ncols = 5
    nrows = int(np.ceil(nsegments / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.0 * nrows),
                             sharey=True)
    axes = np.asarray(axes).ravel()

    gms_all = []
    sat_a_all = np.full(nsegments, np.nan)
    sat_b_all = np.full(nsegments, np.nan)
    for k in range(nsegments):
        ax = axes[k]
        dists = [np.asarray([r['rjd'] for r in records
                             if r['segment'] == k], dtype=float)
                 for records in loop_records]
        lg = [np.log10(d) for d in dists if len(d) > 0]
        if lg:
            lo = min(l.min() for l in lg)
            hi = max(l.max() for l in lg)
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            for j, l in enumerate(lg):
                ax.hist(l, histtype='step', bins=20,
                        range=(lo - pad, hi + pad), color=colors[j], lw=1.1)
        lgm = np.array([np.mean(np.log(d)) if len(d) else np.nan
                        for d in dists])
        widths = np.array([np.std(np.log(d)) if len(d) > 1 else np.nan
                           for d in dists])
        gms_all.append(np.exp(lgm))
        shifts = np.diff(lgm)
        for i in range(len(shifts)):
            if not np.isfinite(sat_a_all[k]) \
                    and np.isfinite(widths[i + 1]) \
                    and abs(shifts[i]) < width_frac * widths[i + 1]:
                sat_a_all[k] = i + 2
            if not np.isfinite(sat_b_all[k]) and i >= 1:
                prev = np.abs(shifts[:i])
                prev = prev[np.isfinite(prev)]
                if len(prev) and abs(shifts[i]) \
                        < shift_frac * np.median(prev):
                    sat_b_all[k] = i + 2
        ax.set_title('seg %d (sat a: %s, b: %s)'
                     % (k, '%d' % sat_a_all[k]
                        if np.isfinite(sat_a_all[k]) else '-',
                        '%d' % sat_b_all[k]
                        if np.isfinite(sat_b_all[k]) else '-'),
                     fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel('count')
        if k >= nsegments - ncols:
            ax.set_xlabel('log10(RJD)', fontsize=8)
        ax.tick_params(labelsize=7)

    handles = [Line2D([], [], color=colors[j], lw=2,
                      label='%s %d' % (loop_label, j + 1)) for j in range(nloop)]
    fig.legend(handles=handles, loc='upper center',
               ncol=min(nloop, 11), fontsize=7, framealpha=0.9)
    fig.suptitle(title + ': RJD distributions per segment', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    filename = os.path.join(outdir, 'rjd_distributions.png')
    fig.savefig(filename, dpi=110)
    plt.close(fig)

    lines = ['%s: geometric-mean RJD per segment and loop' % title,
             'saturation (a): |geomean shift| < %g * std(log rjd), '
             '(b): |shift| < %g * median(previous |shifts|)'
             % (width_frac, shift_frac), '']
    lines.append('segment ' + ' '.join('l%-5d' % (j + 1)
                                       for j in range(nloop))
                 + '  sat_a  sat_b')
    for k in range(nsegments):
        cells = ' '.join('%5.3f' % g if np.isfinite(g) else '    -'
                         for g in gms_all[k])
        lines.append('%7d ' % k + cells
                     + '  %5s %5s' % (
                         '%d' % sat_a_all[k]
                         if np.isfinite(sat_a_all[k]) else '-',
                         '%d' % sat_b_all[k]
                         if np.isfinite(sat_b_all[k]) else '-'))
    with open(os.path.join(outdir, 'rjd_distributions.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    sa = sat_a_all[np.isfinite(sat_a_all)]
    sb = sat_b_all[np.isfinite(sat_b_all)]
    print('RJD geomean saturation (a: |shift| < %g x width; b: |shift| < '
          '%g x median previous): a in %d/%d segments (median loop %s), '
          'b in %d/%d (median loop %s)'
          % (width_frac, shift_frac, len(sa), nsegments,
             '%.0f' % np.median(sa) if len(sa) else '-',
             len(sb), nsegments,
             '%.0f' % np.median(sb) if len(sb) else '-'), flush=True)
    return filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['banana', 'hyperpyramid',
                                              'funnel', 'gauss'],
                        default='banana')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--nlive', type=int, default=400)
    parser.add_argument('--nsegments', type=int, default=20)
    parser.add_argument('--max-loops', type=int, default=11,
                        help='loops (pilot + doublings); loop k reaches arc '
                             'length 2^(k-1), so 11 loops -> 1024 steps')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='per-segment RJD increase significance')
    parser.add_argument('--frac-remain', type=float, default=0.01)
    parser.add_argument('--Lepsilon', type=float, default=0.001)
    parser.add_argument('--max-iters', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--vol-ndraw', type=int, default=20000000)
    parser.add_argument('--vol-nkeep', type=int, default=200001)
    parser.add_argument('--outdir', default='evaluate/variablespeed_plots3')
    args = parser.parse_args()

    problem = make_problem(args)
    if args.problem in ('gauss', 'banana'):
        vol_func = mc_prior_volume_table(problem, ndraw=args.vol_ndraw,
                                         nkeep=args.vol_nkeep)
    else:
        vol_func = problem.make_volume_function()

    arcs, u0, L0, res1 = first_run(problem, args)
    try:
        analytic = float(problem.analytic_lnZ())
    except Exception:
        analytic = np.nan
    nsegments = args.nsegments
    print('loop 1 (pilot, nsteps=1): %d arcs, lnZ = %.4f +- %.4f' % (
        len(arcs), res1['logz'], res1['logzerr']), flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    title = '%s, dim=%d, nlive=%d, %d segments' % (
        args.problem, args.dim, args.nlive, nsegments)

    # snapshot the loop-1 records before the replays mutate the arcs
    loop_records = [[dict(Lmin=a['Lmin'], x=a['x'], rjd=a['rjd'],
                          order=a['order'], segment=a['segment'],
                          nsteps=1, drift=np.nan) for a in arcs]]
    results = [res1]
    gm_history = [segment_geomeans(loop_records[0], nsegments)]
    improve = None  # None = extend all segments until the criterion fires
    fn = plot_panels(loop_records, results, nsegments, args.outdir, title,
                     analytic, gm_history)
    print('wrote', fn, flush=True)
    fn = plot_rjd_distributions(loop_records, nsegments, args.outdir, title)
    print('wrote', fn, flush=True)
    for loop in range(2, args.max_loops + 1):
        extend_all = (loop == 2)
        records, res = replay(arcs, u0, L0, extend_all, improve,
                              args.nlive, problem, args, vol_func)
        results.append(res)
        loop_records.append(records)
        print('loop %d: %d replacements (%d sampler gap-fills, %d arcs '
              'unused), lnZ = %.4f +- %.4f' % (
                  loop, res['total'], res['ngap'], res['nleft'],
                  res['logz'], res['logzerr']), flush=True)
        gm_history.append(segment_geomeans(records, nsegments))
        stop = False
        if loop == 2:
            marked, pvalues = mark_segments(arcs, nsegments, args.alpha)
            print('marked segments (mean RJD significantly increased, '
                  'p < %g, informational): %s' % (
                      args.alpha, sorted(marked)), flush=True)
            for k in range(nsegments):
                print('  segment %2d: RJD increase p = %s' % (
                    k, '%.3g' % pvalues[k] if np.isfinite(pvalues[k])
                    else 'nan'), flush=True)
        if len(gm_history) >= 3:
            new_improve = set(
                k for k in range(nsegments)
                if gompertz_decision(
                    np.array([h[k] for h in gm_history])))
            print('Gompertz criterion: improving %d/%d segments in the '
                  'next loop: %s' % (len(new_improve), nsegments,
                                     sorted(new_improve)), flush=True)
            if not new_improve:
                print('no segments predicted to improve; done.',
                      flush=True)
                stop = True
            improve = new_improve
        # write the plots after every loop so progress can be watched
        fn = plot_panels(loop_records, results, nsegments, args.outdir,
                         title, analytic, gm_history)
        print('wrote', fn, flush=True)
        fn = plot_rjd_distributions(loop_records, nsegments, args.outdir,
                                    title)
        print('wrote', fn, flush=True)
        if stop:
            break

    print()
    print('final lnZ = %.4f +- %.4f' % (
        results[-1]['logz'], results[-1]['logzerr']))
    if np.isfinite(analytic):
        print('analytic lnZ = %.4f' % analytic)
    print('wrote', fn, flush=True)


if __name__ == '__main__':
    main()
