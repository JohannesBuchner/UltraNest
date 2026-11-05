"""
Snowballing nested sampling driver
----------------------------------

Runs snowball rotations with the :class:`SnowballingNestedSampler`:
each rotation advances batches of `P` threads until the burial-gate
`frac_remain` criterion fires, integrates the buried dead points with
`1/K` strips, then widens K (doubling the threads with fresh prior
draws) for the next rotation.

The sampler logs through the ultranest-style logger; when `log_dir` is
given (``log_to_disk``), the growing HDF5 point store, the per-rotation
derived records, the plots and ``results.json`` are written to disk.
Without `log_dir`, the point store is kept in memory and the plots are
shown directly (e.g. inline in a jupyter notebook).
"""
import json
import logging
import os
import sys
import time

import numpy as np
from numpy import exp, log, logaddexp

import matplotlib.pyplot as plt

from ultranest.mlfriends import LocalAffineLayer, ScalingLayer
from ultranest.netiter import MultiCounter
from ultranest.utils import create_logger, resample_equal

from .state import SnowballState
from .utils import (CubeRegion, CovarianceRegion, RotationRecordStore,
                    binned_rjd, fit_gompertz, gompertz_decision,
                    lmin_bin_edges, segment_geomeans, segment_stats,
                    _gompertz)
from ultranest.netiter import TreeNode


class SnowballingNestedSampler:
    """Snowballing nested sampling with a configurable direction
    proposal, integrated by the custom `1/K`-strip integrator inside the
    state machine.

    Threads (lineages) of live points are advanced in batch advances:
    each batch advance replaces the P lowest-likelihood thread ends with
    a replacement walk each. Dead points are buried through the burial
    gate (released once every live point sits above their threshold) and
    integrated with 1/K strips. Each snowball rotation widens K (doubling
    the threads with fresh prior draws) for the next rotation.

    The lnZ error is estimated by K-folding instead of bootstrapping:
    alongside the integrator over all sequences, K fold integrators run,
    the ith integrating the sequences with j % K == i (one sequence
    each) as a 1-live-point run; the error of the fold mean is combined
    with the nominal sqrt(H/K) in quadrature.

    The RJD diagnostics normalize each jump by the region's own distance
    attribute: reused from region building when the region is
    MLFriends-family (sqrt(maxradiussq)), otherwise the mean pair
    distance of the live points computed at region build.

    Parameters
    ----------
    paramnames: list of str
        names of the parameters
    loglike: function
        loglikelihood function (vectorized if vectorized=True)
    transform: function, optional
        prior transform function (vectorized if vectorized=True)
    vectorized: bool
        whether loglike and transform accept arrays of points
    log_dir: str, optional
        output folder for the HDF5 point store, the rotation records,
        the plots, ``debug.log`` and ``results.json``. When not given
        (``log_to_disk == False``), the point store is kept in memory and
        the plots are shown directly (e.g. inline in a jupyter notebook).
    num_live_points_base: int
        initial number of threads, doubled by widening each rotation;
        default 50 (the proposal pool is padded with recent dead points
        to ``min_point_pool`` for diversity at small K)
    num_steps: int
        number of step-sampler steps per walk
    region_class: class
        region class built from the live points (default:
        :py:class:`CovarianceRegion`, which only determines the
        covariance of the live points -- no MLFriends, no
        bootstrapping). For step samplers that do not use the region
        axes, :py:class:`CubeRegion` (a no-op returning the unit cube
        axes) is allowed. RobustEllipsoidRegion (MLFriends machinery)
        can be passed for the ellipsoid axes as a stiff axis guess.
    stepsampler: function
        vectorized direction proposal ``direction(u, region, scale=1)
        -> w`` from ultranest.stepfuncs (e.g.
        ``generate_quadratic_slice_direction`` or
        ``generate_mixture_random_direction``). Sets ``self.stepsampler``;
        can be changed later by the user. Default: the quadratic slice
        direction.
    P: int
        number of threads advanced per batch advance; default: half the
        live points, capped at 100
    integrator: str
        'custom' (the 1/K-strip integrator inside the state machine) or
        'multicounter' (the graph-based MultiCounter)

    Attributes
    ----------
    logger: logging.Logger
        ultranest-style logger (stdout and, with log_dir, debug.log)
    log_to_disk: bool
        whether a log_dir was given and the point store, records, plots
        and results.json are written to disk
    K: int
        current number of live points (doubles with each rotation)
    P: int
        threads advanced per batch advance
    ncall: int
        cumulative number of likelihood function evaluations
    nrotations: int
        number of snowball rotations completed
    results: dict
        latest results: logz, logzerr, samples, weighted_samples,
        posterior, logvol, niter, ncall, converged, nlive, wall_time
    converged: bool or None
        whether the convergence criterion (the Gompertz looping
        criterion) fired; False if a termination budget or the
        live-points limit hit first; None before the first run
    rotation_summaries: list of dict
        per-rotation summaries (rotation, K, logz, logzerr, nburied,
        segment_geomeans)
    nlive_per_rotation: list of int
        the live points used in each rotation
    state: SnowballState
        the cython state machine (threads, walkers, burial, store)
    store: DeadPointStore or MemoryPointStore
        the growing point store behind the state machine
    Lmax_global: float
        the highest likelihood ever seen

    Usage::

        sampler = SnowballingNestedSampler(
            paramnames, loglike, transform=transform, vectorized=True,
            log_dir='chains/my-run')
        sampler.stepsampler = generate_quadratic_slice_direction
        result = sampler.run()
        sampler.plot_rotations()
    """

    def __init__(self, paramnames, loglike, transform=None, *,
                 vectorized=True, log_dir=None, num_live_points_base=50,
                 num_steps=1, region_class=CovarianceRegion,
                 stepsampler=None, P=None, integrator='multicounter'):
        if integrator not in ('multicounter', 'custom'):
            raise ValueError("integrator must be 'multicounter' or 'custom'")
        self.integrator = integrator
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        if not vectorized:
            from .utils import vectorize
            transform = vectorize(transform)
            loglike = vectorize(loglike)
        self.paramnames = list(paramnames)
        self.x_dim = len(self.paramnames)
        self.loglike = loglike
        self.transform = transform if transform is not None else (lambda x: x)
        self.log = True
        self.log_to_disk = log_dir is not None
        self.logger = create_logger(
            'ultranest.' + type(self).__name__, log_dir=log_dir)
        self.log_dir = log_dir
        self.store_path = (os.path.join(log_dir, 'snowball_points.h5')
                           if log_dir is not None else None)
        self.records_path = (os.path.join(log_dir, 'snowball_records.h5')
                             if log_dir is not None else None)
        self.num_live_points_base = num_live_points_base
        self.num_steps = num_steps
        # benchmarked on banana-2d (K=400, 8 rotations): P=50 and P=100
        # run 2.9x faster than P=20 with the vectorized proposals
        self.P = (min(max(num_live_points_base // 2, 1), 100)
                  if P is None else P)
        self.region_class = region_class
        if stepsampler is None:
            from ultranest.stepfuncs import \
                generate_quadratic_slice_direction as stepsampler
        self.stepsampler = stepsampler
        # tree: the tree root represents the prior volume, each thread
        # starts at a child of the root
        self.root = TreeNode(value=-np.inf, id=-1)
        self.thread_nodes = []   # per-thread leaf node
        self.chain_nodes = []    # (node, rootid) for every chain node
        self.K = num_live_points_base
        self.ncall = 0
        self.nrotations = 0
        self.Lmax_global = -np.inf
        self.results = None
        self.converged = None  # set by run(): did the criterion fire?
        self.bin_edges = None
        self.state = None
        self.main_iterator = None
        self.rotation_summaries = []
        self.nlive_per_rotation = []
        # (node, rootid) of the earlier rotations' dead points
        self.previous_nodes = []
        # (node, rootid) that died this rotation
        self._rotation_new_nodes = []
        self._prev_ptr = 0
        self._store_rows = []    # (row, x, rootid) of the latest rotation
        self._logweights = []    # latest rotation's strip weights
        self._logvols = []       # latest rotation's bookkeeping positions
        # per-rotation derived records, kept in memory when the records
        # are not written to disk
        self._rotation_records_memory = []
        self._rjd_ref_distance = None

    def _prior_draw(self, n):
        """Draw n prior points, evaluate and sanity-check them."""
        u = np.random.uniform(size=(n, self.x_dim))
        p = self.transform(u)
        logl = np.asarray(self.loglike(p), dtype=float).reshape(-1)
        assert p.shape == (n, self.x_dim), \
            "Error in transform function: returned shape %s" % (p.shape,)
        assert np.logical_and(u > 0, u < 1).all(), \
            "Error in transform function: u was modified!"
        assert logl.shape == (n,), \
            "Error in loglikelihood function: returned shape %s" % \
            (logl.shape,)
        return u, p, logl

    def _make_region(self, us):
        """Build the region from the live points, without bootstrapping.

        The ellipsoid axes are only used by some proposals as a stiff
        axis guess, so this is updated imperfectly and irregularly."""
        if self.region_class in (CubeRegion, CovarianceRegion):
            # the very simple region classes determine the axes
            # themselves from the live points
            return self.region_class(us)
        d = self.x_dim
        layer = LocalAffineLayer() if d > 1 else ScalingLayer()
        layer.optimize(us, us)
        region = self.region_class(us, transformLayer=layer)
        region.compute_enlargement(nbootstraps=1, minvol=1e-30)
        return region

    def _region_points(self, u_heads, min_point_pool):
        """The direction-sampling point pool: the live points, padded
        with the most recent dead points from the store to at least
        min_point_pool points (proposal diversity at small K). The pool
        is only used for the direction sampling; the region geometry is
        computed from the live points alone."""
        if self.K >= min_point_pool:
            return u_heads
        ndead_need = min_point_pool - self.K
        nrows = self.state.store.nrows
        if nrows <= 0:
            return u_heads
        rows = np.arange(max(0, nrows - ndead_need), nrows)
        d = self.x_dim
        data = self.state.store.read(rows)
        u_dead = np.asarray(data[:, 7:7 + d], dtype=float)
        return np.vstack([u_dead, np.asarray(u_heads, dtype=float)])

    def _rjd_reference_distance(self, region, u_heads):
        """The reference distance for the RJD diagnostics: the region's
        own distance attribute when it has one (sqrt(maxradiussq) for
        MLFriends-family regions, reused from region building), otherwise
        the mean pair distance of the live points computed here."""
        maxradiussq = getattr(region, 'maxradiussq', None)
        if maxradiussq is not None and maxradiussq > 0:
            from math import sqrt
            return float(sqrt(maxradiussq))
        from scipy.spatial.distance import pdist
        pts = np.asarray(u_heads, dtype=float)
        if len(pts) > 800:
            pts = pts[np.random.randint(0, len(pts), size=800)]
        if len(pts) < 2:
            return None
        return float(pdist(pts).mean())

    def _update_region(self, min_point_pool):
        """Rebuild the region from the live points (no bootstrapping),
        then replace its point pool with the live points padded with the
        most recent dead points to at least min_point_pool points: the
        pool feeds the direction sampling only, the region geometry is
        from the live points alone."""
        u_heads, _ = self.state.get_heads()
        region = self._make_region(u_heads)
        region.u = self._region_points(u_heads, min_point_pool)
        self._rjd_ref_distance = self._rjd_reference_distance(region, u_heads)
        self.state.set_region(region)
        return region

    def print_posterior(self, use_unicode=True):
        """Give summary of the marginal likelihood and parameter
        posteriors, as in ReactiveNestedSampler: the logZ estimate with
        error and a unicode histogram sketch per parameter.

        Parameters
        ----------
        use_unicode: bool
            Whether to print a unicode plot of the posterior distributions

        """
        if not self.log or self.results is None:
            return
        print()
        print('logZ = %(logz).3f +- %(logzerr).3f' % self.results)
        print()
        for i, p in enumerate(self.paramnames):
            v = self.results['samples'][:, i]
            sigma = v.std()
            med = v.mean()
            if sigma == 0:
                j = 3
            else:
                j = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % j
            try:
                if not use_unicode:
                    raise UnicodeEncodeError("")
                # make fancy terminal visualisation on a best-effort basis
                ' ▁▂▃▄▅▆▇██'.encode(sys.stdout.encoding)
                H, edges = np.histogram(v, bins=40)
                dist = ''.join([' ▁▂▃▄▅▆▇██'[k]
                                for k in np.ceil(H * 7 / H.max())
                                .astype(int)])
                lo, hi = edges[0], edges[-1]
                print('    %-20s: %-6s│%s│%-6s    %s +- %s'
                      % (p, fmt % lo, dist, fmt % hi, fmt % med,
                         fmt % sigma))
            except Exception:
                print('    %-20s: %s +- %s' % (p, fmt % med, fmt % sigma))
        print()

    def run_iter(self, *, frac_remain=0.5,
                 update_interval_ncall=None,
                 update_interval_volume_fraction=0.2, show_status=True,
                 viz_callback=None, min_vol_reduction=1.0,
                 min_num_iterations=None, min_point_pool=400,
                 num_volume_segments_for_convergence=20):
        """Run one snowball rotation: a new MultiCounter is created and
        gradually filled as the integrator walks through the entire new
        width, including the new and the earlier columns: the earlier
        rotations' dead points are inserted into the integration at their
        threshold positions, merged with the newly released dead points
        in threshold order, once every live point sits above them.

        Alongside the main integrator, K fold integrators (one per
        sequence) integrate their single sequence as a 1-live-point run;
        the lnZ error comes from the variance of the fold lnZ values (the
        error of their mean) combined with the nominal sqrt(H/K) in
        quadrature.

        The termination is the ReactiveNestedSampler-style criterion with
        the global L_max (the highest likelihood ever seen, kept across
        rotations): it does not terminate until the live point remainder
        integral L_max times the remaining volume drops below frac_remain
        times the total lnZ error. The region is rebuilt from the thread
        ends irregularly, based on where the integrator is. Returns the
        rotation summary dict.
        """
        if self.state is None:
            self._setup()
        self.nrotations += 1
        self.nlive_per_rotation.append(self.K)
        if min_num_iterations is None:
            min_num_iterations = self.K
        if self.integrator == 'custom':
            return self._run_iter_custom(
                frac_remain=frac_remain,
                update_interval_volume_fraction=update_interval_volume_fraction,
                show_status=show_status, viz_callback=viz_callback,
                min_point_pool=min_point_pool)
        main_iterator = MultiCounter(
            nroots=self.K, nbootstraps=0, random=False,
            check_insertion_order=True)
        self.main_iterator = main_iterator
        # K-folding error estimator: K fold integrators, the ith
        # integrating the sequences with j % K == i (one sequence each)
        # as a 1-live-point run
        folds = [MultiCounter(nroots=1, nbootstraps=0, random=False)
                 for _ in range(self.K)]
        self.folds = folds
        fold_rootids = np.arange(1)
        self.logzerr_folding = np.nan
        # the earlier rotations' dead points are inserted gradually, at
        # their threshold positions, merged with the newly released dead
        # points
        self.previous_nodes.extend(self._rotation_new_nodes)
        self.previous_nodes.sort(key=lambda c: c[0].value)
        self._rotation_new_nodes = []
        self._prev_ptr = 0
        self._store_rows = []
        self._logweights = []
        self._logvols = []
        next_update_interval_volume = main_iterator.logVolremaining \
            - update_interval_volume_fraction
        last_status_ncall = self.ncall
        vol_at_start = main_iterator.logVolremaining
        nburied_at_start = len(self._store_rows)

        while True:
            # region update, based on where the integrator is
            if main_iterator.logVolremaining < next_update_interval_volume:
                self._update_region(min_point_pool)
                next_update_interval_volume = \
                    main_iterator.logVolremaining - \
                    update_interval_volume_fraction
            self.state.start_round()
            while self.state.any_active():
                prop = self.state.propose()
                pnew = self.transform(prop)
                Lnew = np.asarray(self.loglike(pnew),
                                  dtype=float).reshape(-1)
                self.state.receive(Lnew)
                self.ncall += self.state.P
            # release the buried dead points and feed the integrator:
            # the earlier rotations' dead points below the front and the
            # newly released ones, merged in threshold order
            _, headL = self.state.get_heads()
            front = float(np.min(headL))
            threads, values, rows = self.state.release()
            headL = np.asarray(headL, dtype=float)
            rootids = np.arange(self.K)
            old_cands = []
            while self._prev_ptr < len(self.previous_nodes) \
                    and self.previous_nodes[self._prev_ptr][0].value < front:
                old_cands.append(self.previous_nodes[self._prev_ptr])
                self._prev_ptr += 1
            if len(threads):
                data = self.state.store.read(rows)
                walkend_L = np.asarray(data[:, 1], dtype=float)
            else:
                walkend_L = np.zeros(0)
            new_cands = sorted(zip(threads, values, rows, walkend_L),
                               key=lambda c: c[1])
            merged = []
            a = b = 0
            while a < len(old_cands) and b < len(new_cands):
                if old_cands[a][0].value < new_cands[b][1]:
                    merged.append(('old', old_cands[a]))
                    a += 1
                else:
                    merged.append(('new', new_cands[b]))
                    b += 1
            merged += [('old', c) for c in old_cands[a:]] \
                + [('new', c) for c in new_cands[b:]]
            for kind, c in merged:
                if kind == 'old':
                    node, rootid = c
                    main_iterator.passing_node(rootid, node, rootids, headL)
                    folds[rootid].passing_node(0, node, fold_rootids, headL)
                    # the store row of this death is the child created
                    # when the node died
                    row, t = node.children[-1].id, rootid
                else:
                    t, value, row, walkend = c
                    dead_node = self.thread_nodes[t]
                    child = TreeNode(value=float(walkend), id=int(row))
                    dead_node.children.append(child)
                    self.thread_nodes[t] = child
                    self.chain_nodes.append((child, int(t)))
                    # the dead point enters the re-walk set at its death
                    self._rotation_new_nodes.append((dead_node, int(t)))
                    main_iterator.passing_node(int(t), dead_node, rootids,
                                               headL)
                    folds[t].passing_node(0, dead_node, fold_rootids,
                                          headL)
                x = -main_iterator.logVolremaining
                self._store_rows.append((row, x, t))
                self._logweights.append(
                    float(main_iterator.logweights[-1][0]))
                self._logvols.append(x)
            # the global L_max: the highest likelihood ever seen
            self.Lmax_global = max(self.Lmax_global, float(np.max(headL)))
            nburied = len(self._store_rows) - nburied_at_start
            vol_reduced = vol_at_start - main_iterator.logVolremaining
            if show_status and (self.ncall - last_status_ncall
                                >= (update_interval_ncall or self.K)):
                last_status_ncall = self.ncall
                self.logger.info(
                    'Z=%.1f(%.2f%%) | Like=%.2f..%.2f | '
                    'it/evals=%d/%d N=%d',
                    main_iterator.logZ,
                    100 * (1 - main_iterator.remainder_fraction),
                    float(np.min(headL)), self.Lmax_global,
                    len(main_iterator.logweights), self.ncall, self.K)
                if viz_callback is not None:
                    viz_callback(dict(
                        it=len(main_iterator.logweights), ncall=self.ncall,
                        logz=main_iterator.logZ,
                        logz_remain=main_iterator.logZremain,
                        logvol=main_iterator.logVolremaining,
                        paramnames=self.paramnames))
            # the K-folding lnZ error: the variance of the per-sequence
            # fold integrators' lnZ (the error of their mean), combined
            # with the nominal sqrt(H/K) in quadrature
            logz_folds = np.array([f.logZ for f in folds])
            finite = np.isfinite(logz_folds)
            nfolds = int(finite.sum())
            if nfolds >= 2:
                logzerr_folds = float(np.sqrt(np.var(logz_folds[finite])
                                              / nfolds))
            else:
                logzerr_folds = 0.0
            H = main_iterator.all_H[0]
            nominal = float(np.sqrt(H / self.K)) \
                if np.isfinite(H) and H > 0 else 0.0
            self.logzerr_folding = float(np.sqrt(logzerr_folds ** 2
                                                 + nominal ** 2))
            self._logz_bs = float(np.mean(logz_folds[finite])) \
                if finite.any() else float('nan')
            self._logzerr_folds = logzerr_folds
            logzremain = self.Lmax_global + main_iterator.logVolremaining
            logzerr = self.logzerr_folding
            if (0 < logzerr < np.inf
                    and logzremain - main_iterator.logZ
                    < log(frac_remain * logzerr)
                    and vol_reduced >= min_vol_reduction
                    and nburied >= min_num_iterations):
                break
        # assemble the results for this rotation
        self._assemble_results(main_iterator)
        summary = dict(rotation=self.nrotations, K=self.K,
                       logz=self.results['logz'],
                       logzerr=self.results['logzerr'],
                       nburied=len(self._store_rows),
                       ncall=self.ncall)
        self.rotation_summaries.append(summary)
        return summary

    def _run_iter_custom(self, *, frac_remain=0.5,
                         update_interval_volume_fraction=0.2,
                         show_status=True, viz_callback=None,
                         min_point_pool=400):
        """One snowball rotation with the custom integrator: the state
        machine integrates the released dead points with 1/K strips; the
        termination does not fire until the live point remainder integral
        drops below frac_remain times the total lnZ error. Returns the
        rotation summary dict."""
        next_update_volume = -self.state.logVrem \
            + update_interval_volume_fraction
        last_status_ncall = self.ncall
        while True:
            # region update, based on where the integrator is
            if -self.state.logVrem > next_update_volume:
                self._update_region(min_point_pool)
                next_update_volume = -self.state.logVrem \
                    + update_interval_volume_fraction
            self.state.start_round()
            while self.state.any_active():
                prop = self.state.propose()
                pnew = self.transform(prop)
                Lnew = np.asarray(self.loglike(pnew),
                                  dtype=float).reshape(-1)
                self.state.receive(Lnew)
                self.ncall += self.state.P
            self.state.release()
            _, headL = self.state.get_heads()
            # the global L_max: the highest likelihood ever seen
            self.Lmax_global = max(self.Lmax_global, float(np.max(headL)))
            if show_status and (self.ncall - last_status_ncall >= self.K):
                last_status_ncall = self.ncall
                logzremain = self.Lmax_global + self.state.logVrem
                remfrac = 1.0 / (1.0 + np.exp(self.state.logZ - logzremain))
                self.logger.info(
                    'Z=%.1f(%.2f%%) | Like=%.2f..%.2f | '
                    'it/evals=%d/%d N=%d',
                    self.state.logZ, 100 * (1 - remfrac),
                    float(np.min(headL)), self.Lmax_global,
                    int(round(-self.state.logVrem * self.K)),
                    self.ncall, self.K)
                if viz_callback is not None:
                    viz_callback(dict(
                        it=int(round(-self.state.logVrem * self.K)),
                        ncall=self.ncall, logz=self.state.logZ,
                        logz_remain=self.Lmax + self.state.logVrem,
                        logvol=-self.state.logVrem,
                        paramnames=self.paramnames))
            logzremain = self.Lmax_global + self.state.logVrem
            logzerr = (max(self.state.h, 0.0) / self.K) ** 0.5
            if (logzerr > 0
                    and logzremain - self.state.logZ
                    < log(frac_remain * logzerr)):
                break
        self._assemble_results_custom()
        lmin = self.state.get_records()[0]
        summary = dict(rotation=self.nrotations, K=self.K,
                       logz=self.results['logz'],
                       logzerr=self.results['logzerr'],
                       nburied=len(lmin), ncall=self.ncall)
        self.rotation_summaries.append(summary)
        return summary

    def _assemble_results_custom(self):
        """Build the results dict from the custom integrator: the
        released dead points weighted by their 1/K strips plus the
        thread ends weighted by the remaining volume."""
        lmin, recL, recU, ou, oL, jump, order, x, thread = \
            self.state.get_records()
        n = len(lmin)
        u_heads, headL = self.state.get_heads()
        logz = float(logaddexp(self.state.logZ, self.state.logVrem
                               + logaddexp.reduce(headL)
                               - np.log(self.K)))
        logzerr = float(np.sqrt(max(self.state.h, 0.0) / self.K))
        weighted_u = np.vstack([recU, np.asarray(u_heads, dtype=float)])
        weights = np.concatenate([
            np.full(n, 1.0 / self.K),
            np.full(self.K, np.exp(self.state.logVrem) / self.K)])
        weights /= weights.sum()
        samples = resample_equal(weighted_u, weights)
        mean = (weighted_u * weights[:, None]).sum(axis=0)
        stdev = np.sqrt((weighted_u ** 2 * weights[:, None]).sum(axis=0)
                        - mean ** 2)
        self.results = dict(
            logz=logz, logzerr=logzerr,
            logz_bs=float('nan'), logzerr_bs=logzerr,
            H=float('nan'), samples=samples,
            weighted_samples=dict(points=weighted_u, weights=weights,
                                  logweights=np.log(weights)),
            weights=weights, logweights=np.log(weights),
            logvol=np.concatenate([x, np.full(self.K, -self.state.logVrem)]),
            paramnames=self.paramnames, niter=n, ncall=self.ncall,
            posterior=dict(mean=mean, stdev=stdev),
            rotation_summaries=list(self.rotation_summaries))

    def _feed_nodes(self, threads, values, rows, walkend_L, parallel_values):
        """Extend the tree with the released dead points and feed them to
        the MultiCounter in threshold order."""
        main_iterator = self.main_iterator
        rootids = np.arange(self.K)
        new_nodes = []
        for t, value, row, walkend in zip(threads, values, rows, walkend_L):
            dead_node = self.thread_nodes[t]
            child = TreeNode(value=float(walkend), id=int(row))
            dead_node.children.append(child)
            self.thread_nodes[t] = child
            self.chain_nodes.append((child, int(t)))
            main_iterator.passing_node(int(t), dead_node, rootids,
                                       parallel_values)
            new_nodes.append((child, int(t), int(row),
                              -main_iterator.logVolremaining,
                              float(main_iterator.logweights[-1][0])))
        return new_nodes

    def _assemble_results(self, main_iterator):
        """Build the results dict for the latest rotation: the buried
        dead points weighted by their strips plus the thread ends
        weighted by the remaining volume."""
        rows = [r for r, x, t in self._store_rows]
        data = self.state.store.read(rows) if rows else \
            np.zeros((0, self.state.store.ncols))
        d = self.x_dim
        us_dead = np.asarray(data[:, 7:7 + d], dtype=float)
        logw_dead = np.asarray(self._logweights, dtype=float)
        x_dead = np.asarray(self._logvols, dtype=float)
        u_heads, headL = self.state.get_heads()
        # thread ends weighted by the remaining volume per thread
        logw_live = np.full(self.K, main_iterator.logVolremaining
                            - np.log(self.K))
        weighted_u = np.vstack([us_dead, np.asarray(u_heads, dtype=float)])
        logw = np.concatenate([logw_dead, logw_live])
        weights = np.exp(logw - np.max(logw[np.isfinite(logw)]))
        weights /= weights.sum()
        samples = resample_equal(weighted_u, weights)
        mean = (weighted_u * weights[:, None]).sum(axis=0)
        stdev = np.sqrt((weighted_u ** 2 * weights[:, None]).sum(axis=0)
                        - mean ** 2)
        self.results = dict(
            logz=float(logaddexp(main_iterator.logZ,
                                 main_iterator.logZremain)),
            logzerr=float(self.logzerr_folding),
            logz_bs=float(self._logz_bs),
            logzerr_bs=float(self._logzerr_folds),
            H=float(main_iterator.all_H[0]),
            samples=samples,
            weighted_samples=dict(points=weighted_u, weights=weights,
                                  logweights=logw),
            weights=weights, logweights=logw,
            logvol=np.concatenate([x_dead, np.full(
                self.K, -main_iterator.logVolremaining)]),
            paramnames=self.paramnames, niter=len(self._store_rows),
            ncall=self.ncall,
            posterior=dict(mean=mean, stdev=stdev),
            rotation_summaries=list(self.rotation_summaries))

    def run(self, *, frac_remain=0.5, max_num_live_points=100000,
            min_point_pool=400, num_volume_segments_for_convergence=20,
            max_total_steps=None, max_ncalls=None,
            max_execution_time_in_s=None,
            convergence_gompertz_factor=0.5,
            update_interval_ncall=None, show_status=True,
            viz_callback=None, resume=True):
        """Run snowball rotations with widening until the Gompertz
        looping criterion, the max_num_live_points limit or a termination
        budget.

        Resumes from finished rotations stored in log_dir when present:
        the thread ends and the tree are rebuilt from the stored heads
        and the point store, and the run continues with the next
        rotation. Without log_dir, the run always starts fresh.

        Parameters
        ----------
        frac_remain: float
            termination criterion on the remaining live-point integral,
            as a fraction of the total lnZ error
        max_num_live_points: int
            stop at the next rotation boundary once the number of live
            points reaches this limit (the widening doubles K each
            rotation). Default 100000.
        min_point_pool: int
            minimum size of the direction-sampling point pool (the live
            points, padded with the most recent dead points when
            K < min_point_pool); default 400
        num_volume_segments_for_convergence: int
            number of equal-count likelihood-volume bins for the RJD
            diagnostics and the Gompertz looping criterion; default 20
        max_total_steps: int, optional
            stop at the next rotation boundary once the number of live
            points times nsteps reaches this limit
        max_ncalls: int, optional
            stop at the next rotation boundary once this many likelihood
            function evaluations have been made
        max_execution_time_in_s: float, optional
            stop at the next rotation boundary once this much wall time
            has passed
        convergence_gompertz_factor: float
            threshold of the Gompertz looping criterion -- keep widening
            only if the improvement the Gompertz fit predicts for the
            next rotation exceeds convergence_gompertz_factor times the
            change predicted by the linear trend slope of the per-segment
            geomean RJD values. Larger values keep looping longer.
            Default 0.5.
        update_interval_ncall: int, optional
            how often to print the status line
        show_status: bool
            whether to print the status line during the run
        viz_callback: function, optional
            called with a dict of the current state at each status update
        resume: bool
            whether to resume from stored rotations in log_dir

        Whether the convergence criterion was fulfilled is printed and
        stored as results['converged']: True if the Gompertz criterion
        fired, False if the run stopped at max_num_live_points or a
        budget first, None for stored results without a recorded status.
        """
        self._t_start = time.time()
        resumed = False
        keys = []
        if resume and self.records_path is not None \
                and os.path.exists(self.records_path):
            import h5py
            with h5py.File(self.records_path, 'r') as f:
                keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
            if keys:
                with h5py.File(self.records_path, 'r') as f:
                    last = f[keys[-1]]
                    heads_u = np.asarray(last['heads_u'])
                    heads_L = np.asarray(last['heads_L'])
                nrot = len(keys)
                self.K = len(heads_u)
                self.state = SnowballState(
                    self.K, P=min(self.P, self.K), d=self.x_dim,
                    nsteps=self.num_steps,
                    seed=np.random.randint(1, 2**31 - 1),
                    storepath=self.store_path, mode='a',
                    integrate=(self.integrator == 'custom'))
                self.state.set_heads(heads_u, heads_L)
                # rebuild the tree from the stored points
                self.root = TreeNode(value=-np.inf, id=-1)
                self.thread_nodes = []
                self.chain_nodes = []
                # the re-walk set: every stored dead point's node (all
                # thread nodes except the current thread ends)
                self.previous_nodes = []
                self._rotation_new_nodes = []
                self._prev_ptr = 0
                data = self.state.store.read_all()
                for t in range(self.K):
                    rows_t = np.nonzero(data[:, 5] == t)[0]
                    # order by storage row (the walk ends in walk order)
                    rows_t = rows_t[np.argsort(data[rows_t, 6])]
                    if len(rows_t):
                        # the head node's value is the first cell's
                        # threshold
                        head_node = TreeNode(
                            value=float(data[rows_t[0], 0]), id=-1)
                        self.root.children.append(head_node)
                        self.chain_nodes.append((head_node, t))
                        prev = head_node
                        dead_nodes = [head_node]
                        for r in rows_t:
                            node = TreeNode(value=float(data[r, 1]),
                                            id=int(r))
                            prev.children.append(node)
                            self.chain_nodes.append((node, t))
                            dead_nodes.append(node)
                            prev = node
                        self.thread_nodes.append(prev)
                        # all stored cells except the last are dead
                        # points whose nodes enter the re-walk set
                        self.previous_nodes.extend(
                            (n, t) for n in dead_nodes[:-1])
                self.nrotations = nrot
                resumed = True
                self.logger.info('resuming from rotation %d (K=%d)',
                                 nrot, self.K)
        if not resumed:
            # a fresh run: remove stale rotation records so the rotation
            # group names do not collide (the point store is truncated by
            # _setup)
            if self.records_path is not None \
                    and os.path.exists(self.records_path):
                os.remove(self.records_path)
            self._setup()
        if resumed and self.K >= max_num_live_points:
            # the stored run is already at the live points limit: report
            # the last stored rotation instead of running an empty loop
            import h5py
            with h5py.File(self.records_path, 'r') as f:
                last = f[keys[-1]]
                attrs = dict(last.attrs)
            self.results = dict(
                logz=float(attrs['logz']),
                logzerr=float(attrs['logzerr']),
                logz_bs=float('nan'),
                logzerr_bs=float(attrs['logzerr']),
                H=float('nan'), samples=None, weighted_samples=None,
                weights=None, logweights=None, logvol=None,
                paramnames=self.paramnames,
                niter=int(attrs['total']), ncall=self.ncall,
                posterior=None,
                rotation_summaries=list(self.rotation_summaries))
            self.converged = None  # unknown for stored results
        else:
            self.converged = False
        while True:
            summary = self.run_iter(
                frac_remain=frac_remain,
                update_interval_ncall=update_interval_ncall,
                show_status=show_status, viz_callback=viz_callback,
                min_point_pool=min_point_pool,
                num_volume_segments_for_convergence=20)
            self._store_rotation_records(
                summary, num_volume_segments_for_convergence)
            # the lnZ estimate with error; use %.1e when the values are
            # too large for %.4f
            if (abs(summary['logz']) > 1e9
                    or abs(summary['logzerr']) > 1e9):
                fmt = '%.1e'
            else:
                fmt = '%.4f'
            self.logger.info(
                'Snowball rotation %d[K=%d] iter %d '
                '(%d likelihood function evaluations), lnZ = ' + fmt
                + ' +- ' + fmt,
                summary['rotation'], self.K, summary['nburied'],
                self.ncall, summary['logz'], summary['logzerr'])
            # the posterior histogram, as in ReactiveNestedSampler
            self.print_posterior()
            # the Gompertz convergence status
            improving, nseg = self._gompertz_status(
                factor=convergence_gompertz_factor,
                num_volume_segments=num_volume_segments_for_convergence)
            if improving is None:
                self.logger.info(
                    'Gompertz convergence: needs at least 3 rotations')
            else:
                self.logger.info(
                    'Gompertz convergence: %d/%d segments predicted to '
                    'improve in the next rotation', improving, nseg)
            if improving is not None and improving == 0:
                self.converged = True
                self.logger.info(
                    'Gompertz criterion: no likelihood bin predicted to '
                    'improve; done.')
                break
            if max_num_live_points is not None \
                    and self.K >= max_num_live_points:
                self.logger.info(
                    'max_num_live_points=%d reached before convergence '
                    '(K=%d); stopping.',
                    max_num_live_points, self.K)
                break
            if max_total_steps is not None \
                    and self.K * self.num_steps >= max_total_steps:
                self.logger.info(
                    'max_total_steps=%d reached before convergence '
                    '(K=%d, nsteps=%d); stopping.',
                    max_total_steps, self.K, self.num_steps)
                break
            if max_ncalls is not None and self.ncall >= max_ncalls:
                self.logger.info(
                    'max_ncalls=%d reached before convergence '
                    '(ncall=%d); stopping.', max_ncalls, self.ncall)
                break
            if max_execution_time_in_s is not None \
                    and time.time() - self._t_start \
                    >= max_execution_time_in_s:
                self.logger.info(
                    'max_execution_time_in_s=%gs reached before '
                    'convergence; stopping.', max_execution_time_in_s)
                break
            # widen: double K with fresh prior draws for the next
            # snowball rotation
            K_old = self.K
            u_add = np.random.uniform(size=(K_old, self.x_dim))
            L_add = np.asarray(self.loglike(self.transform(u_add)),
                               dtype=float).reshape(-1)
            self.ncall += K_old
            self.state.widen(u_add, L_add)
            self._widen_tree(u_add, L_add)
            self.K = 2 * K_old
            self._store_heads()
        self.state.close()
        self.results['converged'] = self.converged
        self.logger.info('convergence criterion fulfilled: %s',
                         self.converged)
        self._write_summary()
        return self.results

    def _setup(self):
        """Draw the prior thread heads, build the tree and the state
        machine. The numpy generator is not seeded here: following the
        ultranest convention, the user sets np.random.seed() beforehand."""
        self.K = self.num_live_points_base
        u, p, logl = self._prior_draw(self.K)
        self.state = SnowballState(
            self.K, P=min(self.P, self.K), d=self.x_dim,
            nsteps=self.num_steps, seed=np.random.randint(1, 2**31 - 1),
            storepath=self.store_path,
            integrate=(self.integrator == 'custom'))
        self.state.set_heads(u, logl)
        self.root = TreeNode(value=-np.inf, id=-1)
        self.thread_nodes = []
        self.chain_nodes = []
        for t in range(self.K):
            node = TreeNode(value=float(logl[t]), id=-1)
            self.root.children.append(node)
            self.thread_nodes.append(node)
            self.chain_nodes.append((node, t))
        # a node enters the re-walk set when it becomes a dead point
        self.previous_nodes = []
        self._rotation_new_nodes = []
        self._prev_ptr = 0
        region = self._make_region(u)
        self._rjd_ref_distance = self._rjd_reference_distance(region, u)
        self.state.set_region(region)
        self.state.stepsampler = self.stepsampler
        return region

    def _widen_tree(self, u_add, L_add):
        """Append the widened threads as new root children."""
        for t in range(len(u_add)):
            node = TreeNode(value=float(L_add[t]), id=-1)
            self.root.children.append(node)
            self.thread_nodes.append(node)
            self.chain_nodes.append((node, self.K + t))

    def _gompertz_status(self, factor=0.5, num_volume_segments=20):
        """Gompertz convergence status on the per-segment geomean RJD
        sequences across rotations: (n_improving, n_segments), where
        n_improving counts the segments whose predicted next improvement
        exceeds factor times their linear trend slope. None when fewer
        than 3 rotations are collected."""
        if len(self.rotation_summaries) < 3:
            return None, num_volume_segments
        gm_history = []
        for summary in self.rotation_summaries:
            gm = summary.get('segment_geomeans')
            if gm is None:
                return None, num_volume_segments
            gm_history.append(gm)
        improving = 0
        for b in range(num_volume_segments):
            gms = np.asarray([h[b] for h in gm_history], dtype=float)
            if np.all(np.isfinite(gms)) and gompertz_decision(gms,
                                                              factor=factor):
                improving += 1
        return improving, num_volume_segments

    def _store_rotation_records(self, summary, num_volume_segments):
        """Write the rotation's per-cell records, the thread ends and the
        Gompertz per-segment geomeans to the rotation records file (when
        log_to_disk), and always keep them in memory for the plots."""
        u_heads, headL = self.state.get_heads()
        d = self.x_dim
        if self.integrator == 'custom':
            lmin, recL, recU, ou, oL, jump, order, x, thread = \
                self.state.get_records()
        else:
            rows = [r for r, x, t in self._store_rows]
            data = self.state.store.read(rows) if rows else \
                np.zeros((0, self.state.store.ncols))
            lmin = np.asarray(data[:, 0], dtype=float)
            recL = np.asarray(data[:, 1], dtype=float)
            ou = np.asarray(data[:, 7 + d:7 + 2 * d], dtype=float)
            jump = np.asarray(data[:, 3], dtype=float)
            order = np.asarray(data[:, 4], dtype=np.int64)
            thread = np.asarray(data[:, 5], dtype=np.int64)
            x = np.asarray([x for r, x, t in self._store_rows], dtype=float)
        if 'bin_edges' not in self.__dict__ or self.bin_edges is None:
            self.bin_edges = (lmin_bin_edges(lmin, num_volume_segments)
                              if len(lmin) else None)
        seg = (np.searchsorted(self.bin_edges, lmin)
               if self.bin_edges is not None
               else np.zeros(len(lmin), dtype=int))
        rjd = binned_rjd(lmin, ou, jump, seg, num_volume_segments,
                         ref_distance=self._rjd_ref_distance)
        gm = segment_geomeans(rjd, seg, num_volume_segments)
        summary['segment_geomeans'] = gm
        record_arrays = dict(Lmin=lmin, L=recL, rjd=rjd, order=order, x=x,
                             segment=seg, drift=np.full(len(lmin), np.nan),
                             thread=np.asarray(thread, dtype=np.int64))
        # per-cell dicts, the same representation as load_all
        record_cells = [dict(Lmin=float(lmin[i]), L=float(recL[i]),
                             rjd=float(rjd[i]), order=int(order[i]),
                             x=float(x[i]), segment=int(seg[i]),
                             drift=float('nan'), thread=int(thread[i]))
                        for i in range(len(lmin))]
        self._rotation_records_memory.append(record_cells)
        if self.log_to_disk:
            recstore = RotationRecordStore(self.records_path, mode='a')
            recstore.add_rotation(self.nrotations - 1, record_arrays,
                                  attrs=dict(K=self.K, logz=summary['logz'],
                                             logzerr=summary['logzerr'],
                                             total=len(lmin),
                                             nsteps=self.num_steps))
            recstore.close()

    def _store_heads(self):
        """Store the post-widening thread ends in the rotation records
        (for resuming)."""
        if not self.log_to_disk:
            return
        u_heads, headL = self.state.get_heads()
        import h5py
        with h5py.File(self.records_path, 'a') as f:
            g = f['rotation_%d' % (self.nrotations - 1)]
            g.create_dataset('heads_u', data=np.asarray(u_heads,
                                                        dtype=float))
            g.create_dataset('heads_L', data=np.asarray(headL,
                                                        dtype=float))

    def _load_rotation_records(self):
        """The per-rotation derived records: from the HDF5 records file
        when log_to_disk, otherwise from memory."""
        if self.log_to_disk:
            recstore = RotationRecordStore(self.records_path, mode='a')
            rotation_records, results = recstore.load_all()
            recstore.close()
            return rotation_records, results
        return (self._rotation_records_memory,
                self.rotation_summaries)

    def _write_summary(self):
        self.results['nlive'] = list(self.nlive_per_rotation)
        self.results['wall_time'] = time.time() - self._t_start
        if not self.log_to_disk:
            return
        with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
            json.dump(dict(
                logz=self.results['logz'], logzerr=self.results['logzerr'],
                logz_bs=self.results['logz_bs'],
                logzerr_bs=self.results['logzerr_bs'],
                converged=self.results.get('converged'),
                nrotations=self.nrotations, ncall=self.ncall,
                wall_time=self.results['wall_time'],
                rotation_summaries=[
                    dict(rotation=s['rotation'], K=s['K'],
                         logz=s['logz'], logzerr=s['logzerr'],
                         nburied=s['nburied'])
                    for s in self.rotation_summaries]), f, indent=1)

    def plot(self):
        """Make corner and trace plots.

        calls:

        * plot_corner()
        * plot_trace()
        """
        self.plot_corner()
        self.plot_trace()

    def plot_corner(self, filename='corner.pdf'):
        """Make corner plot.

        Writes corner.pdf to log_dir (or the given filename within
        log_dir), with the plotting code of ReactiveNestedSampler. When
        log_dir is not set, the plot is shown directly.
        """
        import matplotlib.pyplot as plt

        from ..plot import cornerplot
        if self.log_to_disk:
            self.logger.debug('Making corner plot ...')
        cornerplot(self.results)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.log_dir, filename),
                        bbox_inches='tight')
            plt.close()
            self.logger.debug('Making corner plot ... done')

    def plot_trace(self, filename='trace.pdf'):
        """Make trace plot.

        Writes trace.pdf to log_dir (or the given filename within
        log_dir), with the plotting code of ReactiveNestedSampler. When
        log_dir is not set, the plot is shown directly.
        """
        import matplotlib.pyplot as plt

        from ..plot import traceplot
        if self.log_to_disk:
            self.logger.debug('Making trace plot ...')
        traceplot(self.results)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.log_dir, filename),
                        bbox_inches='tight')
            plt.close()
            self.logger.debug('Making trace plot ... done')

    def _load_plot_data(self):
        """The per-rotation records and summaries for the plots."""
        rotation_records, results = self._load_rotation_records()
        gm_history = [s.get('segment_geomeans')
                      for s in self.rotation_summaries]
        return rotation_records, results, gm_history

    def plot_sequences(self, filename='sequences.pdf'):
        """Sequence plot: sequence id (y) versus -ln(volume) (x), one
        horizontal line per sequence with a marker at each dead point,
        colored by where the point sits in the posterior (its likelihood,
        as in the trace plots). A short lnL panel on top shares the
        x-axis. Writes <filename> to log_dir when log_to_disk, otherwise
        shows the plot directly."""
        rotation_records, results, gm_history = self._load_plot_data()
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.06)
        axL = fig.add_subplot(gs[0])
        axS = fig.add_subplot(gs[1], sharex=axL)
        xmax = 0.0
        sc = None
        for records in rotation_records:
            xs = np.asarray([r['x'] for r in records])
            Lmin = np.asarray([r['Lmin'] for r in records])
            threads = np.asarray([r['thread'] for r in records])
            for t in np.unique(threads):
                m = threads == t
                axS.plot(xs[m], np.full(int(m.sum()), t), '-',
                         color='0.85', lw=0.7, zorder=1)
            sc = axS.scatter(xs, threads, c=Lmin, cmap='viridis', s=6,
                             zorder=2)
            axL.plot(xs, Lmin, 'o', ms=1.5, color='0.4')
            xmax = max(xmax, float(xs.max()))
        axL.set_ylabel('lnL')
        plt.setp(axL.get_xticklabels(), visible=False)
        axS.set_xlabel('-ln(X) (bookkeeping)')
        axS.set_ylabel('sequence id')
        if sc is not None:
            cb = fig.colorbar(sc, ax=axS, pad=0.01)
            cb.set_label('lnL (posterior position)')
        if self.log_to_disk:
            fig.savefig(os.path.join(self.log_dir, filename),
                        bbox_inches='tight')
            plt.close(fig)
            self.logger.debug('wrote %s', filename)
        return fig

    def plot_panel_lnL(self, ax, rotation_records=None):
        """Panel: lnL versus -ln(X) trajectories, one line per rotation."""
        if rotation_records is None:
            rotation_records = self._load_plot_data()[0]
        colors = plt.cm.plasma(np.linspace(0.0, 0.85,
                                           len(rotation_records)))
        xmax = 0.0
        for j, records in enumerate(rotation_records):
            xs = np.asarray([r['x'] for r in records])
            Ls = np.asarray([r['Lmin'] for r in records])
            ax.plot(xs, Ls, '-', color=colors[j], lw=0.8,
                    label='rotation %d' % (j + 1))
            xmax = max(xmax, float(xs.max()))
        ax.set_ylabel('lnL')
        ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
        ax.set_xlim(0, xmax)
        return ax

    def plot_panel_nlive(self, ax):
        """Panel: live points (K) per rotation iteration."""
        nlive = np.asarray(self.nlive_per_rotation, dtype=float)
        its = np.arange(1, len(nlive) + 1)
        ax.step(its, nlive, where='post', color='k', lw=1)
        ax.plot(its, nlive, 'o', color='k', ms=4)
        ax.set_yscale('log', base=2)
        ticks = sorted(set(nlive[np.isfinite(nlive)].tolist()))
        if ticks:
            ax.set_yticks(ticks)
            ax.set_yticklabels(['%g' % t for t in ticks], fontsize=8)
        ax.set_ylabel('nlive (K)')
        ax.set_title('live points per rotation', fontsize=11)
        ax.set_xlabel('rotation iteration')
        ax.set_xlim(0.5, max(len(nlive), 1) + 0.5)
        return ax

    def plot_panel_utest(self, ax, rotation_records=None,
                         num_volume_segments=20):
        """Panel: insertion-order U test z-score per segment."""
        if rotation_records is None:
            rotation_records = self._load_plot_data()[0]
        colors = plt.cm.plasma(np.linspace(0.0, 0.85,
                                           len(rotation_records)))
        xmax = 0.0
        for j, records in enumerate(rotation_records):
            segs = segment_stats(records, num_volume_segments,
                                 len(records))
            ax.plot([s['x'] for s in segs], [s['u_z'] for s in segs],
                    'o-', color=colors[j], ms=3, lw=0.8)
            if segs:
                xmax = max(xmax, float(np.max([s['x'] for s in segs])))
        ax.axhline(0.0, color='k', lw=0.8, ls=':')
        ax.set_ylabel('insertion order U test z-score')
        ax.set_title('U test per segment (0 = uniform)', fontsize=11)
        ax.set_xlim(0, xmax)
        return ax

    def plot_panel_rjd(self, ax, rotation_records=None,
                       num_volume_segments=20):
        """Panel: mean relative jump distance per segment (one line per
        rotation; >1 = the walks moved far enough)."""
        if rotation_records is None:
            rotation_records = self._load_plot_data()[0]
        colors = plt.cm.plasma(np.linspace(0.0, 0.85,
                                           len(rotation_records)))
        xmax = 0.0
        for j, records in enumerate(rotation_records):
            segs = segment_stats(records, num_volume_segments,
                                 len(records))
            ax.plot([s['x'] for s in segs], [s['rjd'] for s in segs],
                    'o-', color=colors[j], ms=3, lw=0.8)
            if segs:
                xmax = max(xmax, float(np.max([s['x'] for s in segs])))
        ax.axhline(1.0, color='k', lw=0.8, ls=':')
        ax.set_ylabel('mean RJD per segment')
        ax.set_title('relative jump distance (>1 = moved far enough)',
                     fontsize=11)
        ax.set_xlim(0, xmax)
        return ax

    def plot_panel_drift(self, ax, rotation_records=None):
        """Panel: lnX drift ln(V(Lmin) / X_bookkeeping) versus -ln(X)."""
        if rotation_records is None:
            rotation_records = self._load_plot_data()[0]
        colors = plt.cm.plasma(np.linspace(0.0, 0.85,
                                           len(rotation_records)))
        xmax = 0.0
        for j, records in enumerate(rotation_records):
            xs = np.asarray([r['x'] for r in records])
            drifts = np.asarray([r['drift'] for r in records], dtype=float)
            m = np.isfinite(drifts)
            ax.plot(xs[m], drifts[m], '-', color=colors[j], lw=0.8)
            xmax = max(xmax, float(xs.max()))
        ax.axhline(0.0, color='k', lw=0.8, ls=':')
        ax.set_ylabel('lnX drift ln(V(Lmin) / X_bookkeeping)')
        ax.set_title('lnX drift (>0: actual volume lags bookkeeping, '
                     'lnZ biased)', fontsize=11)
        ax.set_xlabel('-ln(X) (bookkeeping)')
        ax.set_xlim(0, xmax)
        return ax

    def plot_panel_gompertz(self, ax, num_volume_segments=20):
        """Panel: the current Gompertz fits per segment (points: data,
        line: fit; one color per segment)."""
        rotation_records, results, gm_history = self._load_plot_data()
        segcolors = plt.cm.tab20(np.linspace(0.0, 1.0, num_volume_segments))
        nrot_gm = len(gm_history)
        for k in range(num_volume_segments):
            gms = np.asarray([h[k] if h is not None else np.nan
                              for h in gm_history], dtype=float)
            m = np.isfinite(gms)
            if not m.any():
                continue
            it = np.arange(1, nrot_gm + 1)[m]
            ax.plot(it, gms[m], 'o', color=segcolors[k], ms=3)
            if m.sum() >= 3:
                popt, ok = fit_gompertz(gms[m])
                if ok:
                    jj = np.linspace(0, m.sum() - 1, 100)
                    ax.plot(jj + 1, _gompertz(jj, *popt), '-',
                            color=segcolors[k], lw=1.0)
        ax.set_xlabel('rotation iteration')
        ax.set_ylabel('geometric-mean RJD')
        ax.set_title('current Gompertz fits per segment (points: data, '
                     'line: fit; one color per segment)', fontsize=11)
        ax.set_xticks(range(1, max(nrot_gm, 1) + 1))
        return ax

    def plot_panel_lnZ(self, ax, analytic=np.nan):
        """Panel: lnZ per rotation iteration with sqrt(H/nlive) error
        bars."""
        rotation_records, results, gm_history = self._load_plot_data()
        iters = np.arange(1, len(results) + 1)
        ax.errorbar(iters, [r['logz'] for r in results],
                    yerr=[r['logzerr'] for r in results],
                    fmt='o-', color='k', ms=4, lw=1, capsize=3)
        if np.isfinite(analytic):
            ax.axhline(analytic, color='r', lw=1, ls='--',
                       label='analytic')
            ax.legend(fontsize=8)
        ax.set_xlabel('rotation iteration')
        ax.set_ylabel('lnZ')
        ax.set_title('lnZ per rotation iteration (error bars: '
                     'sqrt(H/nlive))', fontsize=11)
        return ax

    def plot_rotations(self, num_volume_segments=20,
                       filename='snowballingns_panels.pdf'):
        """The rotation-convergence diagnostics: a panel figure with the
        lnL-vs-lnX trajectories, the live points per rotation, the
        per-segment U test, the per-segment mean RJD, the lnX drift, the
        current Gompertz fits, and lnZ per rotation iteration. Each panel
        is a plot_panel_* function; write to log_dir as PDF when
        log_to_disk, otherwise show the figure directly."""
        rotation_records, results, gm_history = self._load_plot_data()
        fig, axes = plt.subplots(7, 1, figsize=(7, 21))
        self.plot_panel_lnL(axes[0], rotation_records)
        self.plot_panel_nlive(axes[1])
        self.plot_panel_utest(axes[2], rotation_records,
                              num_volume_segments)
        self.plot_panel_rjd(axes[3], rotation_records, num_volume_segments)
        self.plot_panel_drift(axes[4], rotation_records)
        self.plot_panel_gompertz(axes[5], num_volume_segments)
        self.plot_panel_lnZ(axes[6])
        fig.tight_layout()
        if self.log_to_disk:
            fig.savefig(os.path.join(self.log_dir, filename),
                        bbox_inches='tight')
            plt.close(fig)
            self.logger.debug('wrote %s', filename)
        return fig
