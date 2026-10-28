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

from ultranest.mlfriends import (LocalAffineLayer, RobustEllipsoidRegion,
                                 ScalingLayer)
from ultranest.netiter import MultiCounter
from ultranest.utils import resample_equal

from .state import SnowballState
from .utils import (RotationRecordStore, binned_rjd, gompertz_decision,
                    lmin_bin_edges, make_region, plot_panels,
                    plot_rjd_distributions, plot_rotations_panel,
                    segment_geomeans, subsample)
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

    Each snowball rotation runs batch advances until the remainder
    fraction of the evidence drops below ``frac_remain``, then widens K
    (doubling the threads with fresh prior draws) for the next rotation.
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
    region_class: class
        region class built and updated from the live points
        (default: RobustEllipsoidRegion without bootstrapping; the
        ellipsoid axes are only used by some proposals as a stiff axis
        guess, so it is updated imperfectly and irregularly, based on
        where the integrator is, to save compute time)
    stepsampler: function
        vectorized direction proposal ``direction(u, region, scale=1)
        -> w`` from ultranest.stepfuncs (e.g.
        ``generate_quadratic_slice_direction`` or
        ``generate_mixture_random_direction``). Sets ``self.stepsampler``;
        can be changed later by the user. Default: the quadratic slice
        direction.
    num_bootstraps: int
        number of bootstrap rounds of the MultiCounter
    log_dir: str
        output folder for the HDF5 point store, rotation records and
        plots (required)

    Usage::

        sampler = SnowballingNestedSampler(
            paramnames, loglike, transform=transform, vectorized=True,
            log_dir='chains/my-run')
        sampler.stepsampler = generate_quadratic_slice_direction
        result = sampler.run(frac_remain=0.5, max_rotations=8)
        sampler.plot_rotations()
    """

    def __init__(self, paramnames, loglike, transform=None, vectorized=True,
                 num_live_points_base=400, num_steps=1,
                 region_class=RobustEllipsoidRegion, stepsampler=None,
                 num_bootstraps=10, log_dir=None, draw_multiple=False):
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
        self.region_class = region_class
        self.num_bootstraps = num_bootstraps
        self.draw_multiple = draw_multiple
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
        self.results = None
        self.bin_edges = None
        self.state = None
        self.main_iterator = None
        self.rotation_summaries = []
        self.nlive_per_rotation = []
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
        d = self.x_dim
        layer = LocalAffineLayer() if d > 1 else ScalingLayer()
        layer.optimize(us, us)
        region = self.region_class(us, transformLayer=layer)
        region.compute_enlargement(nbootstraps=1, minvol=1e-30)
        return region

    def _setup(self, seed):
        """Draw the prior thread heads, build the tree and the state
        machine."""
        self.K = self.num_live_points_base
        u, p, logl = self._prior_draw(self.K, seed)
        self.state = SnowballState(self.K, P=min(20, self.K), d=self.x_dim,
                                   nsteps=self.num_steps, seed=seed + 1,
                                   storepath=self.store_path)
        self.state.set_heads(u, logl)
        self.root = TreeNode(value=-np.inf, id=-1)
        self.thread_nodes = []
        self.chain_nodes = []
        for t in range(self.K):
            node = TreeNode(value=float(logl[t]), id=-1)
            self.root.children.append(node)
            self.thread_nodes.append(node)
            self.chain_nodes.append((node, t))
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
                 viz_callback=None):
        """Run one snowball rotation: batch advances until the remainder
        fraction of the evidence drops below ``frac_remain``.

        The MultiCounter is reset and the whole tree re-passed at the
        current width (the integral is re-materialized each rotation).
        The region is rebuilt from the thread ends irregularly, based on
        where the integrator is. Returns the rotation summary dict.
        """
        if self.state is None:
            self._setup(seed=1)
        self.nrotations += 1
        self.nlive_per_rotation.append(self.K)
        d = self.x_dim
        main_iterator = MultiCounter(
            nroots=self.K, nbootstraps=self.num_bootstraps,
            random=False, check_insertion_order=True)
        self.main_iterator = main_iterator
        # re-materialize: re-pass the whole tree at the current width
        leaves = set(id(n) for n in self.thread_nodes)
        to_pass = sorted((node, rootid) for node, rootid in self.chain_nodes
                         if id(node) not in leaves)
        u_heads, headL = self.state.get_heads()
        rootids = np.arange(self.K)
        for node, rootid in to_pass:
            main_iterator.passing_node(rootid, node, rootids,
                                       np.asarray(headL, dtype=float))
        self._store_rows = []
        self._logweights = []
        self._logvols = []
        ncall_at_start = self.ncall
        next_update_interval_volume = main_iterator.logVolremaining \
            + update_interval_volume_fraction
        last_status_ncall = self.ncall
        Lmax = float(np.max(headL))

        while True:
            # region update, based on where the integrator is
            if main_iterator.logVolremaining < next_update_interval_volume:
                region = self._update_region()
                self.next_update_interval_volume = \
                    main_iterator.logVolremaining + \
                    update_interval_volume_fraction
            self.state.start_round()
            while self.state.any_active():
                prop = self.state.propose()
                pnew = self.transform(prop)
                Lnew = np.asarray(self.loglike(pnew),
                                  dtype=float).reshape(-1)
                self.state.receive(Lnew)
                self.ncall += self.state.P
            # release the buried dead points and feed the integrator
            _, headL = self.state.get_heads()
            threads, values, rows = self.state.release()
            if len(threads):
                # the walk ends' likelihoods come from the store
                data = self.state.store.read(rows)
                walkend_L = np.asarray(data[:, 1], dtype=float)
                new_nodes = self._feed_nodes(threads, values, rows,
                                             walkend_L, np.asarray(headL))
                for child, t, row, x, logw in new_nodes:
                    self._store_rows.append((row, x, t))
                    self._logweights.append(logw)
                    self._logvols.append(x)
            Lmax = max(Lmax, float(np.max(headL)))
            if show_status and (self.ncall - last_status_ncall
                                >= (update_interval_ncall or self.K)):
                last_status_ncall = self.ncall
                sys.stdout.write(
                    'Z=%.1f(%.2f%%) | Like=%.2f..%.2f | '
                    'it/evals=%d/%d N=%d \r'
                    % (main_iterator.logZ,
                       100 * (1 - main_iterator.remainder_fraction),
                       float(np.min(headL)), Lmax,
                       len(main_iterator.logweights), self.ncall, self.K))
                sys.stdout.flush()
                if viz_callback is not None:
                    viz_callback(dict(
                        it=len(main_iterator.logweights), ncall=self.ncall,
                        logz=main_iterator.logZ,
                        logz_remain=main_iterator.logZremain,
                        logvol=main_iterator.logVolremaining,
                        paramnames=self.paramnames))
            if main_iterator.remainder_fraction < frac_remain:
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
            logzerr=float(main_iterator.logZerr_bs),
            logz_bs=float(main_iterator.logZ_bs),
            logzerr_bs=float(main_iterator.logZerr_bs),
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
            resume=True):
        """Run snowball rotations with widening until max_rotations or
        the Gompertz looping criterion.

        Resumes from finished rotations stored in log_dir when present:
        the thread ends and the tree are rebuilt from the stored heads
        and the point store, and the run continues with the next
        rotation.
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
                    self.K, P=min(20, self.K), d=self.x_dim,
                    nsteps=self.num_steps, seed=seed + 1,
                    storepath=self.store_path, mode='a')
                self.state.set_heads(heads_u, heads_L)
                # rebuild the tree from the stored points
                self.root = TreeNode(value=-np.inf, id=-1)
                self.thread_nodes = []
                self.chain_nodes = []
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
                        for r in rows_t:
                            node = TreeNode(value=float(data[r, 1]),
                                            id=int(r))
                            prev.children.append(node)
                            self.chain_nodes.append((node, t))
                            prev = node
                        self.thread_nodes.append(prev)
                    else:
                        # a thread without dead points: its head is the
                        # stored end
                        node = TreeNode(value=float(heads_L[t]), id=-1)
                        self.root.children.append(node)
                        self.thread_nodes.append(node)
                        self.chain_nodes.append((node, t))
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
        for rotation in range(start_rotation, max_rotations + 1):
            summary = self.run_iter(
                frac_remain=frac_remain,
                update_interval_ncall=update_interval_ncall,
                show_status=show_status, viz_callback=viz_callback)
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
        rows = [r for r, x, t in self._store_rows]
        data = self.state.store.read(rows) if rows else \
            np.zeros((0, self.state.store.ncols))
        d = self.x_dim
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
        nlive = [len(r) for r in rotation_records]
        title = 'snowballingns, dim=%d, K=%d->%d' % (
            self.x_dim, self.num_live_points_base,
            nlive[-1] if nlive else self.K)
        fn = plot_rotations_panel(rotation_records, results, nsegments,
                                  self.log_dir, title, gm_history=gm_history,
                                  nlive_per_rotation=nlive)
        return fn
