"""
Variable-speed nested sampling: adapt the step count per volume window
======================================================================

Wraps :py:class:`ReactiveNestedSampler` (similar to
:py:class:`ultranest.calibrator.ReactiveNestedCalibrator`) to let the
number of step sampler steps vary between equal-volume windows of the
lnX range:

1. A pilot run with nlive live points uses a step sampler that takes only
   one step (nsteps=1) and records, per iteration, the chain relative
   jump distance RJD = |chain displacement| / <live point pair distance>.
2. An independent run (fresh seed, same nlive) with doubled steps records
   the same.
3. The lnX volume range of the pilot run is divided into nwindows
   equal-volume windows. In each window, the RJD distribution of the two
   runs is compared; where the doubled run has the higher mean RJD, that
   window of the combined run is replaced by the improved segment.
4. The number of steps is doubled again, but attempted only in the
   windows that needed the previous improvement. Each trial resumes the
   live point population at the window's lower border from the combined
   run; at the window's upper border, the population is replaced by the
   next window's population, so trials do not cascade. For the
   resumption, the Lmin threshold at which each live point was created is
   tracked at the window borders: a live point remains suitable at a
   border if it was created below the border threshold and its likelihood
   lies above it.
   The place to stop a trial is found by letting the step sampler raise
   an exception when Lmin becomes too high; the ReactiveNestedSampler
   tree is then salvaged: the live points are the childless nodes at or
   above the crash threshold (breadth-first order guarantees that the
   processed nodes below the crash threshold are gone and the unprocessed
   ones remain).

All runs are single ReactiveNestedSampler passes (no improvement loops),
so the recorded iterations are the dead points of the final tree. The
combined run stitches the per-window segments and integrates them the way
a nested sampling integrator would: the prior volume is tracked with the
mean estimator (1/nlive per dead point) through the stitched dead point
sequence. For segments from runs with too-short chains the actual prior
volume V(Lmin) of the constrained region lags this bookkeeping volume;
the lnX drift ln(V(Lmin)/X_bookkeeping) quantifies that problematic
behaviour (the pilot run's lnZ is biased by it, and it vanishes in the
windows replaced by improved segments).

Outputs, per loop iteration, as a function of -lnX: the dead point
log-likelihoods, the insertion-order U test z-score per window and the
mean RJD per window; plus a plot of ln(Z) and its error versus loop
iteration.

Usage::

    python evaluate/variablespeedns.py [--problem hyperpyramid] [--dim 4]
        [--nlive 400] [--nwindows 20] [--outdir evaluate/variablespeed_plots]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.spatial.distance import pdist

from ultranest import ReactiveNestedSampler
from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
from ultranest.ordertest import infinite_U_zscore
from ultranest.perfectns import (
    BananaGaussianLRPS, HyperPyramidLRPS, NonIsotropicGaussianLRPS,
    UncorrelatedFunnelLRPS, mc_prior_volume_table)


class WindowLimitReached(Exception):
    """Raised by the step sampler when Lmin exceeds the window's upper
    edge; carries the offending threshold for salvaging the tree."""

    def __init__(self, Lmin):
        super().__init__('Lmin exceeded the window limit: %g' % Lmin)
        self.Lmin = float(Lmin)


# ---------------------------------------------------------------------------
# recording step sampler
# ---------------------------------------------------------------------------

class RecordingSliceSampler(SliceSampler):
    """Slice sampler that records per-iteration diagnostics and can stop
    at a likelihood threshold.

    For each returned sample, remembers the start and end of the chain
    that produced it (finalize_chain resets the history, so the chain
    start must be stashed before the reset) and appends a record
    (Lmin, insertion order, RJD, nlive), where the RJD is the distance
    between the chain start and the returned point divided by the mean
    pair distance of the live points.

    If *stop_Lmin* is set, raises :py:class:`WindowLimitReached` as soon
    as the integrator asks for a point above that threshold, which aborts
    the run at the window border.
    """

    def __init__(self, *args, stop_Lmin=None, **kwargs):
        SliceSampler.__init__(self, *args, **kwargs)
        self.stop_Lmin = stop_Lmin
        self.records = []
        self.last_jump = None

    def finalize_chain(self, region=None, Lmin=None, Ls=None):
        if len(self.history) > 0:
            self.last_jump = (self.history[0][0].copy(),
                              self.history[-1][0].copy())
        SliceSampler.finalize_chain(self, region=region, Lmin=Lmin, Ls=Ls)

    def __next__(self, region, Lmin, us, Ls, transform, loglike,
                 ndraw=10, plot=False, tregion=None):
        if self.stop_Lmin is not None and Lmin > self.stop_Lmin:
            raise WindowLimitReached(Lmin)
        u, p, ll, nc = SliceSampler.__next__(
            self, region, Lmin=Lmin, us=us, Ls=Ls, transform=transform,
            loglike=loglike, ndraw=ndraw, plot=plot, tregion=tregion)
        if u is not None:
            u = np.asarray(u, dtype=float)
            mpd = pdist(us).mean()
            order = int((np.asarray(Ls) < ll).sum())
            rjd = float(np.linalg.norm(u - self.last_jump[0]) / mpd)
            self.records.append((float(Lmin), order, rjd, int(len(Ls))))
        return u, p, ll, nc


# ---------------------------------------------------------------------------
# tree salvaging
# ---------------------------------------------------------------------------

def _node_point(sampler, node):
    """(u, p, logl) of a tree node, copied out of the point pile."""
    return (np.array(sampler.pointpile.getu(node.id)),
            np.array(sampler.pointpile.getp(node.id)), float(node.value))


def tree_live_points_above(sampler, Lmin):
    """Live points when processing was interrupted at threshold Lmin.

    Nodes are processed in ascending value order, so at the interrupt all
    processed nodes have values below the interrupting node's value. The
    live points are the childless nodes with value >= Lmin; childless
    nodes below are terminated tails (dead points).
    """
    out = []
    stack = list(sampler.root.children)
    while stack:
        node = stack.pop()
        if len(node.children) == 0:
            if node.value >= Lmin:
                out.append(node)
        else:
            stack.extend(node.children)
    return out


def population_at_border(sampler, L):
    """Live point population at the moment the likelihood threshold first
    crossed L: nodes above L whose parent was processed at or below L.

    Such points were created at a threshold below L (their creation Lmin
    is their parent's value) and have a likelihood above L, so they are
    suitable members of the live point population at threshold L.
    """
    out = []
    stack = [(child, -np.inf) for child in sampler.root.children]
    while stack:
        node, parent_value = stack.pop()
        if node.value > L:
            out.append(node)
        else:
            for child in node.children:
                stack.append((child, node.value))
    return out


def run_final_population(sampler):
    """Final live point population of a completed run: the childless nodes
    at or above the highest expanded node (the nodes processed after
    expansion stopped, which die without replacement)."""
    max_expanded = -np.inf
    childless = []
    stack = list(sampler.root.children)
    while stack:
        node = stack.pop()
        if len(node.children) > 0:
            max_expanded = max(max_expanded, node.value)
            stack.extend(node.children)
        else:
            childless.append(node)
    return [n for n in childless if n.value >= max_expanded]


# ---------------------------------------------------------------------------
# windows and segments
# ---------------------------------------------------------------------------

def volume_window_edges(vol_func, nwindows, vmin_frac):
    """Lmin edges of nwindows equal-volume windows spanning the prior
    volume fraction [vmin_frac, 1].

    Window k covers volume fractions [exp(-(k+1)*d), exp(-k*d)] with
    d = ln(1/vmin_frac)/nwindows. Returns (edges, d); edges has
    nwindows + 1 entries.
    """
    d = np.log(1.0 / vmin_frac) / nwindows
    fracs = np.exp(-np.arange(nwindows + 1) * d)
    edges = np.array([float(vol_func.Lmin_of_volume(max(fr, 1e-300)))
                      for fr in fracs])
    return edges, d


def window_indices(Lmins, edges):
    """Window index of each threshold Lmin, clipped to the window range."""
    return np.clip(np.searchsorted(edges, Lmins, side='right') - 1,
                   0, len(edges) - 2)


class WindowSegment:
    """One window's nested sampling segment from one run.

    records: list of (Lmin, insertion order, RJD, nlive), one per dead
        point, in processing order.
    start_pop: live points (u, p, logl) at the window's lower border,
        for resuming the population in a later trial.
    end_pop: live points at the window's upper border; for the last
        window this is the tail population used by the integrator.
    """

    def __init__(self, records, start_pop, end_pop, source, nsteps):
        self.records = records
        self.start_pop = start_pop
        self.end_pop = end_pop
        self.source = source
        self.nsteps = nsteps

    def mean_rjd(self):
        if len(self.records) == 0:
            return -np.inf
        return float(np.mean([r[2] for r in self.records]))


def segments_from_full_run(sampler, edges, nsteps, source):
    """Split a completed run into per-window segments.

    Each segment carries the window's iteration records and the live
    point populations at the window borders, extracted from the
    ReactiveNestedSampler tree.
    """
    records = sampler.stepsampler.records
    wins = window_indices([r[0] for r in records], edges)
    nwin = len(edges) - 1
    roots_pop = [_node_point(sampler, n) for n in sampler.root.children]
    tail_pop = [_node_point(sampler, n) for n in run_final_population(sampler)]

    segments = []
    for k in range(nwin):
        recs = [r for r, w in zip(records, wins) if w == k]
        if k == 0:
            # the initial live points are the population at the very start
            start_pop = roots_pop
        else:
            start_pop = [_node_point(sampler, n)
                         for n in population_at_border(sampler, edges[k])]
        if k == nwin - 1:
            # the last window ends at the run's final live points
            end_pop = tail_pop
        else:
            nodes = population_at_border(sampler, edges[k + 1])
            end_pop = ([_node_point(sampler, n) for n in nodes]
                       if len(nodes) >= 0.5 * len(roots_pop) else tail_pop)
        segments.append(WindowSegment(recs, start_pop, end_pop, source,
                                      nsteps))
    return segments


# ---------------------------------------------------------------------------
# combined run: stitching, evidence, window statistics
# ---------------------------------------------------------------------------

def combine_segments(segments, vol_func):
    """Stitch window segments into a combined nested sampling run.

    The combined run is integrated the way a nested sampling integrator
    would: the prior volume is tracked with the mean estimator (each dead
    point shrinks the remaining volume by 1/nlive), continuously from the
    full prior through the stitched dead point sequence, and the tail
    after the last dead point is integrated with the last segment's end
    population. For segments from runs with too-short chains the actual
    prior volume V(Lmin) of the constrained region lags this bookkeeping
    volume; the lnX drift ln(V(Lmin) / X_bookkeeping) quantifies that
    problematic behaviour (positive: the run is behind its own
    bookkeeping, its lnZ estimate is biased).

    Returns a dict with the dead point positions (bookkeeping and actual
    -lnX), the lnX drift, likelihoods, logz, logzerr and per-window
    statistics.
    """
    xs, xs_actual, Ls, logweights, nlives, drifts = [], [], [], [], [], []
    logZ_terms = []
    win_stats = []
    tail_Ls = None
    logZ_tail = None
    Vfull = float(vol_func.Vfull)
    logX_tracked = 0.0  # ln of the bookkeeping remaining volume
    # deepest likelihood covered by a tabulated volume function; the
    # drift is undefined (volume clamped) beyond it
    Lgrid = getattr(vol_func, 'Lgrid', None)
    L_covered = float(Lgrid[-1]) if Lgrid is not None else np.inf

    for k, seg in enumerate(segments):
        orders, rjds, Lmins = [], [], []
        win_nlive, win_xs, win_drifts = [], [], []
        for (Lmin, order, rjd, nlive) in seg.records:
            X_actual = max(float(vol_func(Lmin)) / Vfull, 1e-300)
            logleft = np.log1p(-np.exp(-1.0 / nlive))
            logweights.append(logX_tracked + logleft)
            logZ_terms.append(logX_tracked + logleft + Lmin)
            logX_tracked -= 1.0 / nlive
            x = -logX_tracked
            x_actual = -np.log(X_actual)
            xs.append(x)
            xs_actual.append(x_actual)
            drift = -x_actual + x  # ln(V(Lmin) / X_bookkeeping)
            if Lmin > L_covered:
                drift = np.nan
            drifts.append(drift)
            Ls.append(Lmin)
            orders.append(order)
            rjds.append(rjd)
            Lmins.append(Lmin)
            nlives.append(nlive)
            win_nlive.append(nlive)
            win_xs.append(x)
            win_drifts.append(drifts[-1])
        if k == len(segments) - 1 and len(seg.end_pop) > 0:
            tail_Ls = np.array([logl for (_, _, logl) in seg.end_pop])
            logZ_tail = (logX_tracked + logsumexp(tail_Ls)
                         - np.log(len(tail_Ls)))
            logZ_terms.append(logZ_tail)
        mean_nlive = float(np.mean(win_nlive)) if win_nlive else np.nan
        if len(orders) > 0:
            u_z = float(infinite_U_zscore(np.asarray(orders),
                                          int(round(mean_nlive))))
        else:
            u_z = np.nan
        arr_d = np.asarray(win_drifts, dtype=float)
        mean_drift = (float(np.nanmean(arr_d))
                      if np.isfinite(arr_d).any() else np.nan)
        win_stats.append(dict(
            window=k,
            x_center=float(np.mean(win_xs)) if win_xs else np.nan,
            source=seg.source,
            nsteps=seg.nsteps, niter=len(orders),
            mean_L=float(np.mean(Lmins)) if Lmins else np.nan,
            mean_nlive=mean_nlive,
            mean_rjd=float(np.mean(rjds)) if rjds else np.nan,
            mean_drift=mean_drift,
            u_z=u_z,
        ))

    logZ = float(logsumexp(np.array(logZ_terms)))
    logw_dead = np.asarray(logweights)
    Ls_arr = np.asarray(Ls)
    # information gain H = E_post[logL] - logZ, error ~ sqrt(H / nlive)
    eL = float((np.exp(logw_dead + Ls_arr - logZ) * Ls_arr).sum())
    if tail_Ls is not None and len(tail_Ls) > 0:
        q = np.exp(tail_Ls - logsumexp(tail_Ls))
        eL += float(np.exp(logZ_tail - logZ)) * float((q * tail_Ls).sum())
    H = eL - logZ
    nlive_mean = float(np.mean(nlives)) if nlives else 1.0
    logZerr = float(np.sqrt(max(H, 0.0) / nlive_mean))

    return dict(xs=np.asarray(xs), xs_actual=np.asarray(xs_actual),
                drift=np.asarray(drifts), Ls=Ls_arr, logz=logZ,
                logzerr=logZerr, win_stats=win_stats)


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------

class VariableSpeedNestedSampler:
    """Wrapper around ReactiveNestedSampler that adapts the number of
    step sampler steps per equal-volume lnX window.

    Set :py:attr:`stepsampler` to a template (its nsteps gives the pilot
    step count, its generate_direction is reused for all runs), then call
    :py:meth:`run`. See the module docstring for the algorithm.
    """

    def __init__(self, param_names, loglike, transform=None, vol_func=None,
                 nwindows=20, **kwargs):
        """Initialise.

        Parameters
        ----------
        param_names: list of str
            Names of the parameters.
        loglike: function
            log-likelihood function.
        transform: function
            parameter transform from unit cube to physical parameters.
        vol_func: object
            prior volume function V(Lmin) with an invertible
            ``Lmin_of_volume``, used to define the equal-volume windows
            (see perfectns.make_volume_function).
        nwindows: int
            number of equal-volume windows across the pilot lnX range.
        kwargs: dict
            further arguments passed to ReactiveNestedSampler.
        """
        assert vol_func is not None, "vol_func with Lmin_of_volume is required"
        self.init_args = dict(param_names=param_names, loglike=loglike,
                              transform=transform, **kwargs)
        self.vol_func = vol_func
        self.nwindows = nwindows
        self.stepsampler = None

    def _make_stepsampler(self, nsteps, stop_Lmin=None):
        return RecordingSliceSampler(
            nsteps=nsteps,
            generate_direction=self.stepsampler.generate_direction,
            stop_Lmin=stop_Lmin)

    def _run_full(self, nsteps, nlive, seed, run_kwargs):
        """Independent full run from the prior with the given nsteps.

        A single ReactiveNestedSampler pass (no improvement loops): the
        recorded iterations then form the dead point sequence of the final
        tree, which the per-window segmentation and combination rely on.
        """
        np.random.seed(seed)
        sampler = ReactiveNestedSampler(**self.init_args)
        sampler.stepsampler = self._make_stepsampler(nsteps)
        result = sampler.run(min_num_live_points=nlive,
                             max_num_improvement_loops=0, **run_kwargs)
        return sampler, result

    def _run_trial(self, nsteps, start_pop, stop_Lmin, seed, run_kwargs,
                   max_iters):
        """Run one window trial with the given number of steps.

        The live point population *start_pop* (the combined run's
        population at the window's lower border) is fed into a fresh
        ReactiveNestedSampler as the tree roots; points created below the
        border threshold with a likelihood above it remain valid. The
        step sampler raises WindowLimitReached when Lmin exceeds
        *stop_Lmin* (the window's upper edge); the tree is then salvaged:
        the live points at the border are the childless nodes at or above
        the crash threshold.

        Returns (records, end_pop), where records are the window's
        iteration records and end_pop the population at the window's
        upper border (None if the run finished without crossing it).
        """
        np.random.seed(seed)
        sampler = ReactiveNestedSampler(**self.init_args)
        sampler.stepsampler = self._make_stepsampler(nsteps, stop_Lmin=stop_Lmin)
        for u, v, logl in start_pop:
            sampler.root.children.append(sampler.pointpile.make_node(logl, u, v))
        end_pop = None
        try:
            sampler.run(min_num_live_points=max(len(start_pop), 40),
                        max_num_improvement_loops=0, max_iters=max_iters,
                        **run_kwargs)
        except WindowLimitReached as exc:
            end_pop = [_node_point(sampler, n)
                       for n in tree_live_points_above(sampler, exc.Lmin)]
        records = sampler.stepsampler.records
        if end_pop is None:
            # finished without crossing the border: take the population at it
            nodes = population_at_border(sampler, stop_Lmin)
            if len(nodes) > 0:
                end_pop = [_node_point(sampler, n) for n in nodes]
        return records, end_pop

    def run(self, nlive=400, seed=1, max_loops=5, **run_kwargs):
        """Run the pilot, the doubled-steps run and the improvement loop.

        Parameters
        ----------
        nlive: int
            number of live points.
        seed: int
            random seed (the doubled run and the trials use shifted seeds).
        max_loops: int
            maximum number of trial rounds after the first doubling.
        run_kwargs: dict
            arguments passed to ReactiveNestedSampler.run.

        Returns
        -------
        out: dict
            iterations: list of combined per-window segment lists, one
                per loop iteration (the first is the pilot run alone).
            edges, d: window edges and lnX window width.
            results: per-iteration dicts with logz and logzerr.
            sampler0, result0: pilot run and its result.
            sampler1, result1: doubled-steps run and its result.
        """
        assert self.stepsampler is not None, "set a template stepsampler first"
        nsteps0 = int(self.stepsampler.nsteps)

        print('pilot run: nlive=%d, nsteps=%d ...' % (nlive, nsteps0), flush=True)
        sampler0, result0 = self._run_full(nsteps0, nlive, seed, run_kwargs)
        recs0 = sampler0.stepsampler.records
        assert len(recs0) > 0, "pilot run produced no iterations"
        Lmax0 = max(r[0] for r in recs0)
        vmin_frac = float(np.clip(self.vol_func(Lmax0), 1e-300, 1.0))
        edges, d = volume_window_edges(self.vol_func, self.nwindows, vmin_frac)
        print('windows: %d x dlnX=%.3f over Lmin %.4f..%.4f' % (
            self.nwindows, d, edges[0], edges[-1]), flush=True)

        combined = segments_from_full_run(sampler0, edges, nsteps0, 'pilot')
        iterations = [list(combined)]

        nsteps = 2 * nsteps0
        print('independent run: nlive=%d, nsteps=%d ...' % (nlive, nsteps),
              flush=True)
        sampler1, result1 = self._run_full(nsteps, nlive, seed + 1, run_kwargs)
        segs1 = segments_from_full_run(sampler1, edges, nsteps,
                                       'double%d' % nsteps)
        replaced = []
        for k in range(self.nwindows):
            if segs1[k].mean_rjd() > combined[k].mean_rjd():
                combined[k] = segs1[k]
                replaced.append(k)
        print('iteration 1: replaced %d windows: %s' % (
            len(replaced), replaced), flush=True)
        iterations.append(list(combined))

        nsteps_try = 2 * nsteps
        for loop_j in range(2, 2 + max_loops):
            if len(replaced) == 0:
                print('no windows were improved in the last iteration, done.',
                      flush=True)
                break
            print('trial round %d: nsteps=%d in windows %s ...' % (
                loop_j, nsteps_try, replaced), flush=True)
            new_replaced = []
            new_combined = list(combined)
            max_iters = int(10 * nlive * d) + 1000
            for k in replaced:
                seg = combined[k]
                recs, end_pop = self._run_trial(
                    nsteps_try, seg.start_pop, edges[k + 1],
                    seed + 1000 * loop_j + k, run_kwargs, max_iters)
                if end_pop is None or len(recs) < 5:
                    print('  window %2d: trial unusable (%d iterations), kept'
                          % (k, len(recs)), flush=True)
                    continue
                trial = WindowSegment(recs, seg.start_pop, end_pop,
                                      'trial%d-w%d' % (nsteps_try, k),
                                      nsteps_try)
                if trial.mean_rjd() > seg.mean_rjd():
                    new_combined[k] = trial
                    new_replaced.append(k)
                    print('  window %2d: mean RJD %.3f -> %.3f: replaced' % (
                        k, seg.mean_rjd(), trial.mean_rjd()), flush=True)
                else:
                    print('  window %2d: mean RJD %.3f -> %.3f: kept' % (
                        k, seg.mean_rjd(), trial.mean_rjd()), flush=True)
            combined = new_combined
            iterations.append(list(combined))
            print('iteration %d: replaced %d windows: %s' % (
                loop_j, len(new_replaced), new_replaced), flush=True)
            replaced = new_replaced
            nsteps_try *= 2

        results = []
        for j, segs in enumerate(iterations):
            comb = combine_segments(segs, self.vol_func)
            results.append(dict(iteration=j, logz=comb['logz'],
                                logzerr=comb['logzerr']))
            print('combined lnZ after iteration %d: %.4f +- %.4f' % (
                j, comb['logz'], comb['logzerr']), flush=True)

        return dict(iterations=iterations, edges=edges, d=d, results=results,
                    sampler0=sampler0, result0=result0,
                    sampler1=sampler1, result1=result1)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_panels(iterations, vol_func, labels, outdir, title):
    """Four stacked panels versus the bookkeeping -lnX: the dead point
    log-likelihoods, the per-window insertion-order U test z-score, the
    per-window mean RJD, and the lnX drift (the actual prior volume
    V(Lmin) versus the bookkeeping volume); one colored line per loop
    iteration."""
    niter = len(iterations)
    colors = [plt.cm.plasma(f) for f in np.linspace(0.0, 0.85, niter)]
    fig, axes = plt.subplots(4, 1, figsize=(7, 13), sharex=True)

    xmax = 0.0
    for j, segs in enumerate(iterations):
        comb = combine_segments(segs, vol_func)
        c = colors[j]
        axes[0].plot(comb['xs'], comb['Ls'], '-', color=c, lw=0.8,
                     label=labels[j])
        xc = [s['x_center'] for s in comb['win_stats']]
        axes[1].plot(xc, [s['u_z'] for s in comb['win_stats']], 'o-',
                     color=c, ms=3, lw=0.8)
        axes[2].plot(xc, [s['mean_rjd'] for s in comb['win_stats']], 'o-',
                     color=c, ms=3, lw=0.8)
        axes[3].plot(comb['xs'], comb['drift'], '-', color=c, lw=0.8)
        xmax = max(xmax, float(np.max(comb['xs'])))

    axes[0].set_ylabel('lnL')
    axes[0].set_title(title, fontsize=11)
    axes[0].legend(fontsize=8, loc='lower right', framealpha=0.9)

    axes[1].axhline(0.0, color='k', lw=0.8, ls=':')
    for z in (-4, 4):
        axes[1].axhline(z, color='k', lw=0.8, ls='--')
    axes[1].set_ylabel('insertion order U test z-score')
    axes[1].set_title('U test per window (0 = uniform, +-4 = threshold)',
                      fontsize=11)

    axes[2].axhline(1.0, color='k', lw=0.8, ls=':')
    axes[2].set_ylabel('mean RJD per window')
    axes[2].set_title('chain relative jump distance (>1 = moved far enough)',
                      fontsize=11)

    axes[3].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[3].set_ylabel('lnX drift ln(V(Lmin) / X_bookkeeping)')
    axes[3].set_title('lnX drift (>0: actual volume lags bookkeeping, '
                      'lnZ biased)', fontsize=11)
    axes[3].set_xlabel('-ln(X) (bookkeeping)')
    axes[3].set_xlim(0, xmax)

    fig.tight_layout()
    filename = os.path.join(outdir, 'variablespeed_panels.png')
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


def plot_logz(results, references, outdir, title):
    """ln(Z) and its error versus loop iteration, with reference lines."""
    fig = plt.figure(figsize=(6, 4))
    x = [r['iteration'] for r in results]
    y = [r['logz'] for r in results]
    yerr = [r['logzerr'] for r in results]
    plt.errorbar(x, y, yerr=yerr, fmt='o-', color='k', capsize=3,
                 label='combined run')
    for (label, value) in references:
        if value is None or not np.isfinite(value):
            continue
        plt.axhline(value, lw=0.8, ls='--', label=label)
    plt.xlabel('loop iteration')
    plt.ylabel('ln(Z)')
    plt.title(title, fontsize=11)
    plt.legend(fontsize=8, loc='best', framealpha=0.9)
    fig.tight_layout()
    filename = os.path.join(outdir, 'variablespeed_logz.png')
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def make_problem(args):
    """Test problem with an analytically known prior volume function."""
    if args.problem == 'hyperpyramid':
        # constrained region is an axis-aligned hyperrectangle at every
        # threshold, with an exact volume function
        return HyperPyramidLRPS(args.dim)
    if args.problem == 'banana':
        # constant-Jacobian banana (Rosenbrock-like); volume function by
        # fixed-seed Monte Carlo over the prior
        return BananaGaussianLRPS.standard_problem(args.dim)
    if args.problem == 'funnel':
        return UncorrelatedFunnelLRPS(args.dim, trunc_sigmas=6.0)
    if args.problem == 'gauss':
        return NonIsotropicGaussianLRPS(args.dim)
    raise ValueError("Unknown problem=%s" % args.problem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['hyperpyramid', 'banana',
                                              'funnel', 'gauss'],
                        default='hyperpyramid')
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--nlive', type=int, default=400)
    parser.add_argument('--nwindows', type=int, default=20)
    parser.add_argument('--max-loops', type=int, default=5,
                        help='maximum number of trial rounds after the '
                             'first doubling')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--vol-ndraw', type=int, default=20000000,
                        help='prior draws for the Monte Carlo volume table '
                             '(banana, gauss)')
    parser.add_argument('--vol-nkeep', type=int, default=200001,
                        help='kept points of the volume table; sets the '
                             'depth coverage (X down to 1/(nkeep-1))')
    parser.add_argument('--outdir', default='evaluate/variablespeed_plots')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    problem = make_problem(args)
    if args.problem in ('gauss', 'banana'):
        # rank-spaced Monte Carlo volume table; nkeep sets the depth
        # coverage (X down to 1/(nkeep-1))
        vol_func = mc_prior_volume_table(
            problem, ndraw=args.vol_ndraw, nkeep=args.vol_nkeep)
    else:
        vol_func = problem.make_volume_function()

    wrapper = VariableSpeedNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vol_func=vol_func, nwindows=args.nwindows, vectorized=True)
    wrapper.stepsampler = SliceSampler(
        nsteps=1, generate_direction=generate_mixture_random_direction)

    out = wrapper.run(nlive=args.nlive, seed=args.seed,
                      max_loops=args.max_loops,
                      show_status=not args.quiet, viz_callback=False)

    os.makedirs(args.outdir, exist_ok=True)
    title = '%s, dim=%d, nlive=%d, %d windows' % (
        args.problem, args.dim, args.nlive, args.nwindows)

    labels = ['iteration 0 (pilot, nsteps=1)']
    for j, segs in enumerate(out['iterations'][1:], 1):
        nsteps_set = sorted(set(s.nsteps for s in segs))
        labels.append('iteration %d (nsteps=%s)' % (
            j, ','.join(str(n) for n in nsteps_set)))
    fn = plot_panels(out['iterations'], vol_func, labels,
                     args.outdir, title)
    print('wrote', fn, flush=True)

    try:
        analytic = float(problem.analytic_lnZ())
    except Exception:
        analytic = None
    references = [
        ('pilot run lnZ', float(out['result0']['logz'])),
        ('doubled run lnZ', float(out['result1']['logz'])),
        ('analytic lnZ', analytic),
    ]
    fn = plot_logz(out['results'], references, args.outdir, title)
    print('wrote', fn, flush=True)

    table_fn = os.path.join(args.outdir, 'variablespeed_table.txt')
    with open(table_fn, 'w') as f:
        f.write('# iteration window x_center source nsteps niter '
                'mean_L mean_nlive u_z mean_rjd mean_drift\n')
        for j, segs in enumerate(out['iterations']):
            comb = combine_segments(segs, vol_func)
            for st in comb['win_stats']:
                f.write('%d %d %.4f %s %d %d %.6f %.1f %.4f %.4f %.4f\n' % (
                    j, st['window'], st['x_center'], st['source'],
                    st['nsteps'], st['niter'], st['mean_L'],
                    st['mean_nlive'], st['u_z'], st['mean_rjd'],
                    st['mean_drift']))
            f.write('# iteration %d: lnz=%.6f logzerr=%.6f\n' % (
                j, comb['logz'], comb['logzerr']))
    print('wrote', table_fn, flush=True)


if __name__ == '__main__':
    main()
