"""
Snowballing nested sampling integration driver
----------------------------------------------

Runs snowball rotations over a :class:`ultranest.snowballingns.state.
SnowballState`: each rotation runs batch advances until the burial-gate
frac_remain criterion fires, then widens K for the next snowball
rotation.  Completed dead points are written once to the growing HDF5
store; the per-rotation derived records are appended to a second HDF5
file and the panel plots written to the output folder.
"""
import json
import os
import sys
import time

import numpy as np
from numpy import exp, log, logaddexp

from ultranest.mlfriends import LocalAffineLayer, ScalingLayer
from ultranest.netiter import MultiCounter
from ultranest.utils import resample_equal

from .state import SnowballState
from .utils import (CubeRegion, CovarianceRegion, RotationRecordStore,
                    binned_rjd, gompertz_decision, lmin_bin_edges,
                    make_region, plot_panels, plot_rjd_distributions,
                    plot_rotations_panel, segment_geomeans, subsample)
from ultranest.netiter import TreeNode


def run_rotation(state, transform, loglike, frac_remain, region):
    """Run batch advances until the burial-gate frac_remain criterion
    fires.

    Each batch advance the P lowest-likelihood thread ends are advanced;
    the state machine buries the dead points the front has passed after
    every likelihood function evaluation and evaluates the termination
    criterion on the incrementally maintained values. The region axes
    are refreshed every K//4 batch advances from the subsampled thread
    ends. Returns (nrounds, niter, region): likelihood function
    evaluations, batch advances, and the refreshed region.
    """
    d = state.d
    nrounds = 0
    niter = 0
    while True:
        u_heads, _ = state.get_heads()
        if niter % max(1, state.K // 4) == 0:
            region = make_region(subsample(u_heads), d)
            state.set_axes(region.transformLayer.axes)
        state.start_round()
        while state.any_active():
            prop = state.propose()
            pnew = transform(prop)
            Lnew = np.asarray(loglike(pnew), dtype=float).reshape(-1)
            state.receive(Lnew)
            nrounds += 1
        niter += 1
        if state.check(frac_remain):
            break
    return nrounds, niter, region


def run_snowballing(transform, loglike, d, K=400, P=20, nsteps=1,
                    frac_remain=0.5, max_rotations=8, seed=1,
                    outdir=None, store_path=None, records_path=None,
                    vol_func=None, make_region_fn=None):
    """Run a snowballing nested sampling run.

    Parameters
    ----------
    transform, loglike: functions
        prior transform and vectorized loglikelihood
    d: int
        dimensionality
    K: int
        initial number of threads (doubled by widening each rotation)
    P: int
        number of threads advanced per batch advance
    nsteps: int
        accepted slice steps per walk
    frac_remain: float
        termination criterion on the remaining volume fraction
    max_rotations: int
        maximum number of snowball rotations
    seed: int
        random seed (the prior thread draws)
    outdir, store_path, records_path: str
        output folder (panels plot) and HDF5 file paths; derived from
        outdir when not given
    vol_func: function
        X(Lmin) = P(logL >= Lmin) under the prior, for the lnX drift
        panel (optional)
    make_region_fn: function
        region builder (defaults to the MLFriends region)

    Returns
    -------
    dict with 'results' (per-rotation logz, logzerr, total), 'logz',
    'logzerr', 'nlive' (K per rotation), 'meta' (timings), 'nrounds'
    (total likelihood function evaluations).
    """
    if make_region_fn is None:
        make_region_fn = make_region
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        if store_path is None:
            store_path = os.path.join(outdir, 'snowball_points.h5')
        if records_path is None:
            records_path = os.path.join(outdir, 'snowball_records.h5')

    t_start = time.time()
    np.random.seed(seed)
    u = np.random.uniform(size=(K, d))
    L = np.asarray(loglike(transform(u)), dtype=float).flatten()
    state = SnowballState(K, P, d, nsteps=nsteps, seed=seed + 1,
                          storepath=store_path)
    state.set_heads(u, L)
    region = make_region_fn(u.copy(), d)
    state.set_axes(region.transformLayer.axes)
    recstore = (RotationRecordStore(records_path)
                if records_path is not None else None)

    nsegments = 20
    results = []
    gm_history = []
    nlive_per_rotation = []
    rotation_meta = []
    bin_edges = None
    for rotation in range(1, max_rotations + 1):
        t0 = time.time()
        nrounds, niter, region = run_rotation(state, transform, loglike,
                                              frac_remain, region)
        dt = time.time() - t0
        lmin, recL, recU, ou, oL, jump, order, x = state.get_records()
        narcs = len(lmin)
        # reported lnZ: the buried dead points + the thread ends'
        # remaining-volume term (significant at frac_remain=0.5)
        _, headL = state.get_heads()
        logz = float(np.logaddexp(state.logZ, state.logVrem
                                  + np.logaddexp.reduce(headL)
                                  - np.log(state.K)))
        logzerr = float(np.sqrt(max(state.h, 0.0) / state.K))
        # likelihood-threshold bins (from rotation 1, fixed across
        # rotations)
        if bin_edges is None:
            bin_edges = lmin_bin_edges(lmin, nsegments)
        seg = np.searchsorted(bin_edges, lmin)
        rjd = binned_rjd(lmin, ou, jump, seg, nsegments)
        drift = np.full(narcs, np.nan)
        if vol_func is not None:
            m = lmin <= getattr(vol_func, 'Lgrid', np.array([np.inf]))[-1]
            if m.any():
                try:
                    vv = np.asarray(vol_func(lmin[m]), dtype=float).reshape(-1)
                except Exception:
                    vv = np.array([float(vol_func(li)) for li in lmin[m]])
                drift[m] = log(np.maximum(vv, 1e-300)) + x[m]
        gm = float(exp(np.mean(np.log(rjd)))) if narcs else float('nan')
        if recstore is not None:
            recstore.add_rotation(rotation - 1,
                                  dict(Lmin=lmin, L=recL, rjd=rjd,
                                       order=order, x=x, segment=seg,
                                       drift=drift),
                                  attrs=dict(K=state.K, logz=logz,
                                             logzerr=logzerr, total=narcs,
                                             nsteps=nsteps))
        gm_history.append(segment_geomeans(rjd, seg, nsegments))
        results.append(dict(logz=logz, logzerr=logzerr, total=narcs,
                            K=state.K))
        nlive_per_rotation.append(state.K)
        rotation_meta.append(dict(dt=dt, nrounds=nrounds, niter=niter))
        print('rotation %d: K=%d, %d buried cells (%d likelihood '
              'evaluations in %d batch advances, %.1fs), lnZ = %.4f '
              '+- %.4f, geomean RJD = %.4f'
              % (rotation, state.K, narcs, nrounds, niter, dt, logz,
                 logzerr, gm), flush=True)
        if rotation >= 3:
            improve = [b for b in range(nsegments)
                       if np.all(np.isfinite([h[b] for h in gm_history]))
                       and gompertz_decision(np.array([h[b] for h in
                                                       gm_history]))]
            if not improve:
                print('Gompertz criterion: no likelihood bin predicted to '
                      'improve; done.', flush=True)
                break
            print('Gompertz criterion: %d/%d likelihood bins predicted to '
                  'improve; keep looping' % (len(improve), nsegments),
                  flush=True)
        # widen: double K with fresh prior draws as the new threads, for
        # the next snowball rotation
        K_old = state.K
        u_add = np.random.uniform(size=(K_old, d))
        L_add = np.asarray(loglike(transform(u_add)), dtype=float).flatten()
        state.widen(u_add, L_add)
        region = make_region_fn(subsample(state.get_heads()[0]), d)
        state.set_axes(region.transformLayer.axes)

    state.close()
    if recstore is not None:
        recstore.close()

    # the panels plot is rebuilt from the stored records (transiently)
    if outdir is not None and recstore is not None:
        title = ('snowballingns, dim=%d, K=%d->%d, P=%d, nsteps=%d, '
                 'frac_remain=%g' % (d, K, nlive_per_rotation[-1], P,
                                     nsteps, frac_remain))
        rotation_records, _ = RotationRecordStore(records_path,
                                                  mode='a').load_all()
        fn = plot_panels(rotation_records, results, nsegments, outdir,
                         title, gm_history=gm_history,
                         nlive_per_rotation=nlive_per_rotation)
        print('wrote', fn, flush=True)
        fn = plot_rjd_distributions(rotation_records, nsegments, outdir,
                                    title)
        print('wrote', fn, flush=True)

    return dict(results=results, logz=results[-1]['logz'],
                logzerr=results[-1]['logzerr'],
                nlive=nlive_per_rotation, meta=rotation_meta,
                nrounds=sum(m['nrounds'] for m in rotation_meta),
                wall_time=time.time() - t_start)


class SnowballingNestedSampler:
    """Snowballing nested sampling with a configurable direction
    proposal, integrated by the graph-based
    :py:class:`ultranest.netiter.MultiCounter`.

    Threads (lineages) of live points are advanced in batch advances:
    each batch advance replaces the P lowest-likelihood thread ends with
    a replacement walk each. The threads are branches of a tree: the
    tree root represents the prior volume, each thread starts at a child
    of the root, and every dead point extends its thread's branch. The
    dead points are fed to the :py:class:`ultranest.netiter.MultiCounter`
    in threshold order once every live point sits above them (the
    burial gate), which integrates the tree breadth-first.

    Each snowball rotation runs batch advances until the live point
    remainder integral drops below ``frac_remain`` times the total lnZ
    error, then widens K (doubling the threads with fresh prior draws)
    for the next rotation.
    The integral is re-materialized each rotation: the MultiCounter is
    reset and the whole tree re-passed at the new width. Completed dead
    points are written once to a growing HDF5 store; the per-rotation
    derived records and the thread ends are stored for resuming.

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
    num_live_points_base: int
        initial number of threads (doubled by widening each rotation)
    num_steps: int
        number of step-sampler steps per walk
    P: int
        number of threads advanced per batch advance; default: half the
        initial number of live points, capped at 100 (larger P
        vectorizes the proposals better; benchmarked 2.9x faster than
        P=20 on banana-2d with K=400). At K=10000 the measured optimum
        is P~250-500, ~15-20% faster than P=100 with the quadratic
        slice, and P~500-1000 with the mixture proposal (9% faster on
        gauss-50d, 37% faster on banana-20d; the default stays at the
        cap of 100, which is optimal at K=400; sweeps in
        evaluate/perfectns/benchmarks/)
    region_class: class
        region class built from the live points (default:
        :py:class:`CovarianceRegion`, which only determines the
        covariance of the live points -- no MLFriends, no
        bootstrapping). For step samplers that do not use the region
        axes, :py:class:`CubeRegion` (a no-op returning the unit cube
        axes) is allowed. RobustEllipsoidRegion (MLFriends machinery)
        can be passed for the ellipsoid axes as a stiff axis guess;
        they are updated imperfectly and irregularly, based on where
        the integrator is, to save compute time
    stepsampler: function
        vectorized direction proposal ``direction(u, region, scale=1)
        -> w`` from ultranest.stepfuncs (e.g.
        ``generate_quadratic_slice_direction`` or
        ``generate_mixture_random_direction``). Sets ``self.stepsampler``;
        can be changed later by the user. Default: the quadratic slice
        direction.
    log_dir: str
        output folder for the HDF5 point store, rotation records and
        plots (required)

    The lnZ error is estimated by K-folding instead of bootstrapping:
    alongside the integrator over all sequences, K fold integrators run,
    the ith integrating the sequences with j % K == i (one sequence
    each) as a 1-live-point run; the error of the fold mean is combined
    with the nominal sqrt(H/K) in quadrature.

    Usage::

        sampler = SnowballingNestedSampler(
            paramnames, loglike, transform=transform, vectorized=True,
            log_dir='chains/my-run')
        sampler.stepsampler = generate_quadratic_slice_direction
        result = sampler.run(frac_remain=0.5, max_rotations=8)
        sampler.plot_rotations()
    """

    def __init__(self, paramnames, loglike, transform=None, vectorized=True,
                 num_live_points_base=400, num_steps=1, P=None,
                 region_class=CovarianceRegion, stepsampler=None,
                 log_dir=None, draw_multiple=False,
                 integrator='multicounter'):
        if integrator not in ('multicounter', 'custom'):
            raise ValueError("integrator must be 'multicounter' or 'custom'")
        self.integrator = integrator
        if log_dir is None:
            raise ValueError("log_dir must be provided (the HDF5 point "
                             "store, rotation records and plots are "
                             "written there)")
        os.makedirs(log_dir, exist_ok=True)
        if not vectorized:
            from .utils import vectorize
            transform = vectorize(transform)
            loglike = vectorize(loglike)
        self.paramnames = list(paramnames)
        self.x_dim = len(self.paramnames)
        self.loglike = loglike
        self.transform = transform if transform is not None else (lambda x: x)
        self.log_dir = log_dir
        self.store_path = os.path.join(log_dir, 'snowball_points.h5')
        self.records_path = os.path.join(log_dir, 'snowball_records.h5')
        self.num_live_points_base = num_live_points_base
        self.num_steps = num_steps
        # benchmarked on banana-2d (K=400, 8 rotations): P=50 and P=100
        # run 2.9x faster than P=20 with the vectorized proposals
        self.P = (min(max(num_live_points_base // 2, 1), 100)
                  if P is None else P)
        self.region_class = region_class
        self.draw_multiple = draw_multiple
        self.folds = []              # per-sequence fold integrators
        self.logzerr_folding = np.nan
        if stepsampler is None:
            from ultranest.stepfuncs import generate_quadratic_slice_direction
            stepsampler = generate_quadratic_slice_direction
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
        self.bin_edges = None
        self.state = None
        self.main_iterator = None
        self.rotation_summaries = []
        self.nlive_per_rotation = []
        self.previous_nodes = []      # (node, rootid) of the earlier
                                      # rotations' dead points
        self._rotation_new_nodes = [] # (node, rootid) that died this
                                      # rotation
        self._prev_ptr = 0
        self._store_rows = []    # (row, x, rootid) of the latest rotation
        self._logweights = []    # latest rotation's strip weights
        self._logvols = []       # latest rotation's bookkeeping positions

    def _prior_draw(self, n, seed):
        """Draw n prior points, evaluate and sanity-check them."""
        u = np.random.uniform(size=(n, self.x_dim))
        p = self.transform(u)
        logl = np.asarray(self.loglike(p), dtype=float).reshape(-1)
        assert p.shape == (n, self.x_dim), \
            "Error in transform function: returned shape %s" % (p.shape,)
        assert np.logical_and(u > 0, u < 1).all(), \
            "Error in transform function: u was modified!"
        assert logl.shape == (n,), \
            "Error in loglikelihood function: returned shape %s" % (logl.shape,)
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

    def _setup(self, seed):
        """Draw the prior thread heads, build the tree and the state
        machine."""
        # the prior draws and the vectorized direction proposals use the
        # global numpy generator; seed it so runs are reproducible
        np.random.seed(seed)
        self.K = self.num_live_points_base
        u, p, logl = self._prior_draw(self.K, seed)
        self.state = SnowballState(self.K, P=min(self.P, self.K), d=self.x_dim,
                                   nsteps=self.num_steps, seed=seed + 1,
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
        self.state.set_region(region)
        self.state.stepsampler = self.stepsampler
        return region

    def _update_region(self):
        """Rebuild the region from the thread ends (no bootstrapping)."""
        u_heads, _ = self.state.get_heads()
        region = self._make_region(subsample(u_heads))
        self.state.set_region(region)
        return region

    def run_iter(self, frac_remain=0.5, update_interval_ncall=None,
                 update_interval_volume_fraction=0.2, show_status=True,
                 viz_callback=None, min_vol_reduction=1.0,
                 min_num_iterations=None, max_batches=None, max_dead=None):
        """Run one snowball rotation: a new MultiCounter is created and
        gradually filled as the integrator walks through the entire new
        width, including the new and the earlier columns: the earlier
        rotations' dead points are inserted into the integration at their
        threshold positions, merged with the newly released dead points
        in threshold order, once every live point sits above them.

        max_batches, max_dead: optional benchmark caps on the custom
        integrator path (batch advances / released dead points).

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
            self._setup(seed=1)
        self.nrotations += 1
        self.nlive_per_rotation.append(self.K)
        if min_num_iterations is None:
            min_num_iterations = self.K
        if self.integrator == 'custom':
            return self._run_iter_custom(
                frac_remain=frac_remain,
                update_interval_volume_fraction=update_interval_volume_fraction,
                show_status=show_status, viz_callback=viz_callback,
                max_batches=max_batches, max_dead=max_dead)
        d = self.x_dim
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
                region = self._update_region()
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
                    merged.append(('old', old_cands[a])); a += 1
                else:
                    merged.append(('new', new_cands[b])); b += 1
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
                sys.stdout.write(
                    'Z=%.1f(%.2f%%) | Like=%.2f..%.2f | '
                    'it/evals=%d/%d N=%d \r'
                    % (main_iterator.logZ,
                       100 * (1 - main_iterator.remainder_fraction),
                       float(np.min(headL)), self.Lmax_global,
                       len(main_iterator.logweights), self.ncall, self.K))
                sys.stdout.flush()
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

    def _run_iter_custom(self, frac_remain=0.5,
                         update_interval_volume_fraction=0.2,
                         show_status=True, viz_callback=None,
                         max_batches=None, max_dead=None):
        """One snowball rotation with the custom integrator: the state
        machine integrates the released dead points with 1/K strips; the
        termination does not fire until the live point remainder integral
        drops below frac_remain times the total lnZ error. Returns the
        rotation summary dict.

        max_batches, max_dead: optional benchmarks caps -- stop after
        that many batch advances / released dead points, whichever comes
        first (the rotation is cut short before its termination
        criterion fires)."""
        next_update_volume = -self.state.logVrem \
            + update_interval_volume_fraction
        last_status_ncall = self.ncall
        nbatches = 0
        ndeaths = 0
        # the earlier rotations' dead points inserted gradually
        self.previous_nodes.extend(self._rotation_new_nodes)
        self.previous_nodes.sort(key=lambda c: c[0].value)
        self._rotation_new_nodes = []
        self._prev_ptr = 0
        while True:
            # region update, based on where the integrator is
            if -self.state.logVrem > next_update_volume:
                self._update_region()
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
            released = self.state.release()
            nbatches += 1
            ndeaths += len(released[0])
            del released
            _, headL = self.state.get_heads()
            # the global L_max: the highest likelihood ever seen
            self.Lmax_global = max(self.Lmax_global, float(np.max(headL)))
            if show_status and (self.ncall - last_status_ncall
                                >= self.K):
                last_status_ncall = self.ncall
                logzremain = self.Lmax_global + self.state.logVrem
                remfrac = 1.0 / (1.0 + np.exp(self.state.logZ - logzremain))
                sys.stdout.write(
                    'Z=%.1f(%.2f%%) | Like=%.2f..%.2f | '
                    'it/evals=%d/%d N=%d \r'
                    % (self.state.logZ, 100 * (1 - remfrac),
                       float(np.min(headL)), self.Lmax_global,
                       int(round(-self.state.logVrem * self.K)),
                       self.ncall, self.K))
                sys.stdout.flush()
                if viz_callback is not None:
                    viz_callback(dict(
                        it=int(round(-self.state.logVrem * self.K)),
                        ncall=self.ncall, logz=self.state.logZ,
                        logz_remain=self.state.Lmax + self.state.logVrem,
                        logvol=-self.state.logVrem,
                        paramnames=self.paramnames))
            logzremain = self.Lmax_global + self.state.logVrem
            logzerr = (max(self.state.h, 0.0) / self.K) ** 0.5
            if (logzerr > 0
                    and logzremain - self.state.logZ
                    < log(frac_remain * logzerr)):
                break
            if max_batches is not None and nbatches >= max_batches:
                break
            if max_dead is not None and ndeaths >= max_dead:
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
        lmin, recL, recU, ou, oL, jump, order, x = self.state.get_records()
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
            weights=weights, logweights=np.log(weights), logvol=x,
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

    def run(self, frac_remain=0.5, max_rotations=8, seed=1,
            update_interval_ncall=None, show_status=True, viz_callback=None,
            resume=True, max_batches=None, max_dead=None):
        """Run snowball rotations with widening until max_rotations or
        the Gompertz looping criterion.

        Resumes from finished rotations stored in log_dir when present:
        the thread ends and the tree are rebuilt from the stored heads
        and the point store, and the run continues with the next
        rotation.

        max_batches, max_dead: optional benchmark caps applied per
        rotation (batch advances / released dead points), cutting each
        rotation short before its termination criterion fires.
        """
        import h5py
        self._t_start = time.time()
        start_rotation = 1
        resumed = False
        keys = []
        if resume and os.path.exists(self.records_path):
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
                    nsteps=self.num_steps, seed=seed + 1,
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
                d = self.x_dim
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
                    else:
                        # a thread without dead points: its head is the
                        # stored end
                        node = TreeNode(value=float(heads_L[t]), id=-1)
                        self.root.children.append(node)
                        self.thread_nodes.append(node)
                        self.chain_nodes.append((node, t))
                self.previous_nodes.sort(key=lambda c: c[0].value)
                region = self._make_region(heads_u)
                self.state.set_region(region)
                self.state.stepsampler = self.stepsampler
                start_rotation = nrot + 1
                self.nrotations = nrot
                resumed = True
                print('resuming from rotation %d (K=%d)'
                      % (nrot, self.K), flush=True)
        if not resumed:
            self._setup(seed)
        if resumed and start_rotation > max_rotations:
            # all rotations already stored: report the last stored
            # rotation instead of running an empty loop
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
        for rotation in range(start_rotation, max_rotations + 1):
            summary = self.run_iter(
                frac_remain=frac_remain,
                update_interval_ncall=update_interval_ncall,
                show_status=show_status, viz_callback=viz_callback,
                max_batches=max_batches, max_dead=max_dead)
            self._store_rotation_records(summary)
            print('rotation %d: K=%d, %d buried cells, lnZ = %.4f +- %.4f'
                  % (rotation, self.K, summary['nburied'],
                     summary['logz'], summary['logzerr']), flush=True)
            if rotation >= 3:
                if not self._gompertz_continue():
                    print('Gompertz criterion: no likelihood bin '
                          'predicted to improve; done.', flush=True)
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
            self._store_heads(u_add, L_add, K_old)
        self.state.close()
        self._write_summary()
        return self.results

    def _widen_tree(self, u_add, L_add):
        """Append the widened threads as new root children."""
        for t in range(len(u_add)):
            node = TreeNode(value=float(L_add[t]), id=-1)
            self.root.children.append(node)
            self.thread_nodes.append(node)
            self.chain_nodes.append((node, self.K + t))

    def _gompertz_continue(self):
        """Gompertz looping criterion on the per-segment geomean RJD
        sequences across rotations."""
        if len(self.rotation_summaries) < 3:
            return True
        nsegments = 20
        gm_history = []
        for summary in self.rotation_summaries:
            gm = summary.get('segment_geomeans')
            if gm is None:
                return True
            gm_history.append(gm)
        for b in range(nsegments):
            gms = np.asarray([h[b] for h in gm_history], dtype=float)
            if np.all(np.isfinite(gms)) and gompertz_decision(gms):
                return True
        return False

    def _store_rotation_records(self, summary):
        """Write the rotation's per-cell records, the thread ends and the
        Gompertz per-segment geomeans to the rotation records file."""
        u_heads, headL = self.state.get_heads()
        d = self.x_dim
        if self.integrator == 'custom':
            lmin, recL, recU, ou, oL, jump, order, x = \
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
            x = np.asarray([x for r, x, t in self._store_rows], dtype=float)
        if 'bin_edges' not in self.__dict__ or self.bin_edges is None:
            self.bin_edges = lmin_bin_edges(lmin, 20) if len(lmin) else None
        seg = (np.searchsorted(self.bin_edges, lmin)
               if self.bin_edges is not None else np.zeros(len(lmin),
                                                           dtype=int))
        rjd = binned_rjd(lmin, ou, jump, seg, 20)
        gm = segment_geomeans(rjd, seg, 20)
        summary['segment_geomeans'] = gm
        recstore = RotationRecordStore(self.records_path, mode='a')
        recstore.add_rotation(self.nrotations - 1,
                              dict(Lmin=lmin, L=recL, rjd=rjd, order=order,
                                   x=x, segment=seg, drift=np.full(
                                       len(lmin), np.nan)),
                              attrs=dict(K=self.K, logz=summary['logz'],
                                         logzerr=summary['logzerr'],
                                         total=len(lmin),
                                         nsteps=self.num_steps))
        recstore.close()

    def _store_heads(self, u_add, L_add, K_old):
        """Store the post-widening thread ends in the rotation records
        (for resuming)."""
        u_heads, headL = self.state.get_heads()
        import h5py
        with h5py.File(self.records_path, 'a') as f:
            g = f['rotation_%d' % (self.nrotations - 1)]
            g.create_dataset('heads_u', data=np.asarray(u_heads,
                                                        dtype=float))
            g.create_dataset('heads_L', data=np.asarray(headL,
                                                        dtype=float))

    def _write_summary(self):
        self.results['nlive'] = list(self.nlive_per_rotation)
        self.results['wall_time'] = time.time() - self._t_start
        with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
            json.dump(dict(
                logz=self.results['logz'], logzerr=self.results['logzerr'],
                logz_bs=self.results['logz_bs'],
                logzerr_bs=self.results['logzerr_bs'],
                nrotations=self.nrotations, ncall=self.ncall,
                rotation_summaries=[
                    dict(rotation=s['rotation'], K=s['K'],
                         logz=s['logz'], logzerr=s['logzerr'],
                         nburied=s['nburied'])
                    for s in self.rotation_summaries]), f, indent=1)

    def plot(self, **kwargs):
        """Corner plot of the latest result (see plot_corner)."""
        return self.plot_corner(**kwargs)

    def plot_corner(self, **kwargs):
        """Corner plot of the latest result, written to log_dir."""
        import ultranest.plot
        fig = ultranest.plot.cornerplot(self.results, **kwargs)
        if fig is not None:
            fig.savefig(os.path.join(self.log_dir, 'corner.png'), dpi=110)
        return fig

    def plot_trace(self, **kwargs):
        """Trace plot of the latest result, written to log_dir."""
        import ultranest.plot
        out = ultranest.plot.traceplot(self.results, **kwargs)
        fig = out[0] if isinstance(out, tuple) else out
        if fig is not None:
            fig.savefig(os.path.join(self.log_dir, 'trace.png'), dpi=110)
        return out

    def plot_rotations(self, nsegments=20, **kwargs):
        """The rotation-convergence diagnostics (the panel plot): how the
        rotations converge, from the stored rotation records."""
        import h5py
        from .utils import plot_rotations_panel
        recstore = RotationRecordStore(self.records_path, mode='a')
        rotation_records, results = recstore.load_all()
        recstore.close()
        gm_history = []
        for records in rotation_records:
            segs = np.asarray([r['segment'] for r in records])
            rjds = np.asarray([r['rjd'] for r in records])
            gm_history.append(segment_geomeans(rjds, segs, nsegments))
        nlive = [r['K'] for r in results]
        title = 'snowballingns, dim=%d, K=%d->%d' % (
            self.x_dim, self.num_live_points_base,
            nlive[-1] if nlive else self.K)
        fn = plot_rotations_panel(rotation_records, results, nsegments,
                                  self.log_dir, title, gm_history=gm_history,
                                  nlive_per_rotation=nlive)
        return fn
