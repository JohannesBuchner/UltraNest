"""
Snowballing nested sampling utilities
--------------------------------------

Per-rotation record storage (growing HDF5), the RJD diagnostics, the
Gompertz looping criterion and the panel plots.
"""
import os

import numpy as np
from numpy import exp, log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ultranest.ordertest import infinite_U_zscore


class RotationRecordStore:
    """Growing HDF5 storage of the per-rotation derived records (one
    group per snowball rotation), following the growing approach of
    ultranest.store.HDF5PointStore."""
    FIELDS = ('Lmin', 'L', 'rjd', 'order', 'x', 'segment', 'drift')

    def __init__(self, filepath, mode='w'):
        import h5py
        self.fileobj = h5py.File(filepath, mode)

    def add_rotation(self, index, fields, attrs):
        """Append one rotation's per-cell record arrays and scalars."""
        g = self.fileobj.create_group('rotation_%d' % index)
        for key in self.FIELDS:
            g.create_dataset(key, data=np.asarray(fields[key]))
        for key, value in attrs.items():
            g.attrs[key] = value
        self.fileobj.flush()

    def load_all(self):
        """All rotations as (rotation_records, results) for the plot
        functions; the per-cell dict lists are materialised transiently
        (only here, never during the run)."""
        rotation_records = []
        results = []
        keys = sorted(self.fileobj.keys(),
                      key=lambda k: int(k.split('_')[1]))
        for key in keys:
            g = self.fileobj[key]
            cols = {k: np.asarray(g[k]) for k in self.FIELDS}
            nsteps = int(g.attrs.get('nsteps', 1))
            records = [
                dict(Lmin=float(cols['Lmin'][i]), L=float(cols['L'][i]),
                     rjd=float(cols['rjd'][i]), order=int(cols['order'][i]),
                     x=float(cols['x'][i]), segment=int(cols['segment'][i]),
                     nsteps=nsteps, drift=float(cols['drift'][i]))
                for i in range(len(cols['Lmin']))]
            rotation_records.append(records)
            results.append(dict(logz=float(g.attrs['logz']),
                                logzerr=float(g.attrs['logzerr']),
                                total=int(g.attrs['total']),
                                K=int(g.attrs.get('K', 0))))
        return rotation_records, results

    def close(self):
        self.fileobj.close()


def subsample(u, maxpts=4000):
    if len(u) > maxpts:
        return u[np.random.choice(len(u), maxpts, replace=False)]
    return u


class _FixedAxesLayer:
    """Minimal transform layer with fixed axes. The rows are basis
    vectors in u-space, as in ultranest.mlfriends.AffineLayer (whose
    axes are the principal axes of the sample covariance, scaled by the
    square-root eigenvalues)."""

    def __init__(self, axes):
        self.axes = axes


class CubeRegion:
    """No-op region for step samplers that do not use the region axes
    (e.g. the quadratic slice direction): it keeps the live points and
    returns the unit cube axes."""

    def __init__(self, us):
        self.u = np.ascontiguousarray(us, dtype=float)
        self.transformLayer = _FixedAxesLayer(np.eye(self.u.shape[1]))

    def compute_enlargement(self, nbootstraps=0, minvol=1e-30):
        pass


class CovarianceRegion:
    """Very simple region: the axes are the principal axes of the live
    points' covariance, scaled by the square-root eigenvalues (no
    MLFriends, no bootstrapping)."""

    def __init__(self, us):
        self.u = np.ascontiguousarray(us, dtype=float)
        d = self.u.shape[1]
        if len(self.u) < 2 or d == 1:
            std = np.maximum(self.u.std(axis=0), 1e-300)
            axes = np.diag(std)
        else:
            cov = np.cov(self.u, rowvar=False)
            eigval, eigvec = np.linalg.eigh(cov)
            # numerical stability, as in mlfriends.AffineLayer
            eigvalmin = max(eigval.max() * 1e-40, 1e-300)
            eigval[eigval < eigvalmin] = eigvalmin
            axes = (eigvec * eigval ** 0.5).T
        self.transformLayer = _FixedAxesLayer(axes)

    def compute_enlargement(self, nbootstraps=0, minvol=1e-30):
        pass


def make_region(us, d):
    """MLFriends region around the live points, enlarged."""
    from ultranest.mlfriends import MLFriends, LocalAffineLayer, ScalingLayer
    if d > 1:
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


def lmin_bin_edges(lmin_ref, nsegments):
    """Equal-count likelihood-threshold bin edges from a reference
    rotation (fixed across rotations; later arcs clip into the edge
    bins)."""
    return np.quantile(lmin_ref, np.linspace(0, 1, nsegments + 1))[1:-1]


def binned_rjd(lmin, ou, jump, seg, nsegments, maxpts=800):
    """RJD = raw jump / mean pair distance of the origin points in the
    same likelihood-threshold bin (the local live-point spacing)."""
    from scipy.spatial.distance import pdist
    rjd = np.full(len(jump), np.nan)
    for b in range(nsegments):
        m = seg == b
        n = int(m.sum())
        if n < 5:
            continue
        pts = ou[m]
        if n > maxpts:
            pts = pts[np.random.randint(0, n, size=maxpts)]
        mpd = float(pdist(pts).mean())
        rjd[m] = jump[m] / mpd
    return rjd


def _gompertz(x, A, b, c):
    return A * np.exp(-b * np.exp(-c * x))


def fit_gompertz(gms):
    """Fit a Gompertz curve g(j) = A exp(-b exp(-c j)) to the
    geometric-mean values at rotation indices 0..len-1.

    Initialises the asymptote A as an extension of the last two values
    (the last trend step doubled), solves the double-log-linearised
    problem ln(-ln(y/A)) = ln b - c j by linear least squares, and
    refines all three parameters with gradient-based optimisation of the
    non-linear least squares (analytic jacobian). Returns (popt, ok).
    """
    from scipy.optimize import least_squares
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
    """Whether a segment should be improved in the next rotation.

    Fits a Gompertz curve g(j) = A exp(-b exp(-c j)) to the geometric-mean
    RJD values (j = 0-based rotation index) and predicts the next
    rotation's improvement; improve only if it reaches `factor` times the
    change predicted by the linear trend slope fit to all collected
    values. Defaults to improving when fewer than 3 rotations are
    collected or the fit fails.
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


def segment_geomeans(rjd, seg, nsegments):
    """Geometric mean RJD per segment from arrays (nan where empty)."""
    out = np.full(nsegments, np.nan)
    for k in range(nsegments):
        vals = rjd[seg == k]
        if len(vals):
            out[k] = exp(np.mean(np.log(vals)))
    return out


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


def plot_panels(rotation_records, results, nsegments, outdir, title,
                analytic=np.nan, gm_history=None, nlive_per_rotation=None):
    """Panel figure: lnL-vs-lnX trajectories, the live points (nlive) per
    rotation, the per-segment U test, the per-segment mean RJD (one line
    per rotation), the lnX drift, the current Gompertz fits per segment
    (dataset+curve in one color per segment), and lnZ per rotation
    iteration. Writes <outdir>/snowballingns_panels.png."""
    nrot = len(rotation_records)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, nrot))
    segcolors = plt.cm.tab20(np.linspace(0.0, 1.0, nsegments))
    fig, axes = plt.subplots(7, 1, figsize=(7, 21))

    xmax = 0.0
    for j, records in enumerate(rotation_records):
        c = colors[j]
        xs = np.asarray([r['x'] for r in records])
        Ls = np.asarray([r['Lmin'] for r in records])
        axes[0].plot(xs, Ls, '-', color=c, lw=0.8,
                     label='rotation %d' % (j + 1))
        nl = 400 if nlive_per_rotation is None else int(nlive_per_rotation[j])
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

    if nlive_per_rotation is None:
        nlive_per_rotation = [r.get('K', np.nan) for r in results]
    nlive_per_rotation = np.asarray(nlive_per_rotation, dtype=float)
    its = np.arange(1, len(nlive_per_rotation) + 1)
    axes[1].step(its, nlive_per_rotation, where='post', color='k', lw=1)
    axes[1].plot(its, nlive_per_rotation, 'o', color='k', ms=4)
    axes[1].set_yscale('log', base=2)
    ticks = sorted(set(nlive_per_rotation[
        np.isfinite(nlive_per_rotation)].tolist()))
    if ticks:
        axes[1].set_yticks(ticks)
        axes[1].set_yticklabels(['%g' % t for t in ticks], fontsize=8)
    axes[1].set_ylabel('nlive (K)')
    axes[1].set_title('live points per rotation', fontsize=11)
    axes[1].set_xlabel('rotation iteration')
    axes[1].set_xlim(0.5, max(len(nlive_per_rotation), 1) + 0.5)

    axes[2].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[2].set_ylabel('insertion order U test z-score')
    axes[2].set_title('U test per segment (0 = uniform)', fontsize=11)

    axes[3].axhline(1.0, color='k', lw=0.8, ls=':')
    axes[3].set_ylabel('mean RJD per segment')
    axes[3].set_title('relative jump distance (>1 = moved far enough)',
                      fontsize=11)

    axes[4].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[4].set_ylabel('lnX drift ln(V(Lmin) / X_bookkeeping)')
    axes[4].set_title('lnX drift (>0: actual volume lags bookkeeping, '
                      'lnZ biased)', fontsize=11)
    axes[4].set_xlabel('-ln(X) (bookkeeping)')
    for ax in [axes[0]] + list(axes[2:5]):
        ax.set_xlim(0, xmax)

    if gm_history is not None:
        nrot_gm = len(gm_history)
        for k in range(nsegments):
            gms = np.asarray([h[k] for h in gm_history], dtype=float)
            m = np.isfinite(gms)
            if not m.any():
                continue
            it = np.arange(1, nrot_gm + 1)[m]
            axes[5].plot(it, gms[m], 'o', color=segcolors[k], ms=3)
            if m.sum() >= 3:
                popt, ok = fit_gompertz(gms[m])
                if ok:
                    jj = np.linspace(0, m.sum() - 1, 100)
                    axes[5].plot(jj + 1, _gompertz(jj, *popt), '-',
                                 color=segcolors[k], lw=1.0)
    axes[5].set_xlabel('rotation iteration')
    axes[5].set_ylabel('geometric-mean RJD')
    axes[5].set_title('current Gompertz fits per segment (points: data, '
                      'line: fit; one color per segment)', fontsize=11)
    if nrot >= 1:
        axes[5].set_xticks(range(1, max(nrot, 1) + 1))

    iters = np.arange(1, len(results) + 1)
    axes[6].errorbar(iters, [r['logz'] for r in results],
                     yerr=[r['logzerr'] for r in results],
                     fmt='o-', color='k', ms=4, lw=1, capsize=3)
    if np.isfinite(analytic):
        axes[6].axhline(analytic, color='r', lw=1, ls='--',
                        label='analytic')
        axes[6].legend(fontsize=8)
    axes[6].set_xlabel('rotation iteration')
    axes[6].set_ylabel('lnZ')
    axes[6].set_title('lnZ per rotation iteration (error bars: '
                      'sqrt(H/nlive))', fontsize=11)

    fig.tight_layout()
    filename = os.path.join(outdir, 'snowballingns_panels.png')
    fig.savefig(filename, dpi=110)
    plt.close(fig)
    return filename


def plot_rjd_distributions(rotation_records, nsegments, outdir, title,
                           width_frac=0.1, shift_frac=0.5):
    """Separate figure: one panel per segment with the RJD distributions
    at each rotation (log10-RJD histograms, colored by rotation), plus
    the per-segment geometric-mean RJD saturation analysis.

    Saturation of the geometric mean is identified two ways:
    (a) the rotation-to-rotation shift of the geometric mean is small
        relative to the width of the distribution:
        |shift| < width_frac * std(log rjd);
    (b) the most recent shift is small compared to the previous shifts:
        |shift| < shift_frac * median(previous |shifts|).

    Writes <outdir>/rjd_distributions.png and a per-segment table to
    <outdir>/rjd_distributions.txt; returns the filename.
    """
    nrot = len(rotation_records)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, nrot))
    ncols = 5
    nrows = int(np.ceil(nsegments / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.0 * nrows),
                             sharey=True)
    axes = np.asarray(axes).ravel()

    sat_a_all = np.full(nsegments, np.nan)
    sat_b_all = np.full(nsegments, np.nan)
    table = {}
    for k in range(nsegments):
        ax = axes[k]
        dists = [np.asarray([r['rjd'] for r in records
                             if r['segment'] == k], dtype=float)
                 for records in rotation_records]
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
        table[k] = np.exp(lgm)
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
                      label='rotation %d' % (j + 1)) for j in range(nrot)]
    fig.legend(handles=handles, loc='upper center',
               ncol=min(nrot, 11), fontsize=7, framealpha=0.9)
    fig.suptitle(title + ': RJD distributions per segment', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    filename = os.path.join(outdir, 'rjd_distributions.png')
    fig.savefig(filename, dpi=110)
    plt.close(fig)

    lines = ['%s: geometric-mean RJD per segment and rotation' % title,
             'saturation (a): |geomean shift| < %g * std(log rjd), '
             '(b): |shift| < %g * median(previous |shifts|)'
             % (width_frac, shift_frac), '']
    lines.append('segment ' + ' '.join('r%-5d' % (j + 1)
                                       for j in range(nrot)))
    for k in range(nsegments):
        lines.append('%-7d ' % k + ' '.join(
            '%-6.3f' % table[k][j] if np.isfinite(table[k][j]) else '-     '
            for j in range(nrot)))
    with open(os.path.join(outdir, 'rjd_distributions.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return filename


def plot_rotations_panel(rotation_records, results, nsegments, outdir, title,
                         gm_history=None, nlive_per_rotation=None):
    """Alias of :py:func:`plot_panels`: the rotation-convergence
    diagnostics (lnL-vs-lnX trajectories, live points per rotation,
    U test, RJD, lnX drift, Gompertz fits, lnZ per rotation)."""
    return plot_panels(rotation_records, results, nsegments, outdir, title,
                       gm_history=gm_history,
                       nlive_per_rotation=nlive_per_rotation)
