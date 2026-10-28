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
import os
import time

import numpy as np
from numpy import exp, log

from ultranest.mlfriends import MLFriends, LocalAffineLayer, ScalingLayer

from .state import SnowballState
from .utils import (RotationRecordStore, binned_rjd, gompertz_decision,
                    lmin_bin_edges, make_region, plot_panels,
                    plot_rjd_distributions, segment_geomeans, subsample)


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
