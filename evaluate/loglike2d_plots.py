"""
2D loglikelihood plots of the perfectns test problems
=====================================================

For each problem at d=2 (the full two-dimensional problem), two panels:
the loglikelihood over the entire prior space, and a zoom whose axis
range is chosen so that the Lmax - 3 contour spans 20% of the axis
(clipped to the prior range), with contour lines at Lmax - delta for
delta = 1, 2, 4, 10, 40, 100, 400, 1000, 4000, 10000, ... The funnel is
shown in its natural (s, w) coordinates at w2 = 0 (its sampling space is
(sigma, dp1, dp2)); the Gauss ellipsoid is invisible on the unit cube
(the log-spaced widths), so the zoom panel shows it.

Usage::

    python evaluate/loglike2d_plots.py [--dim 2] [--outdir evaluate/diagnostics_plots]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultranest.perfectns import (GaussContour, BananaContour,
    HyperPyramidContour, ShellContour, SplitEllipsoidContour)
from ultranest.perfectns import UncorrelatedFunnelLRPS


def delta_levels(lmax, lmin):
    """Contour offsets 1, 2, 4, 10, 40, 100, 400, ... up to the range."""
    deltas = [1.0, 2.0, 4.0]
    while deltas[-1] < lmax - lmin and len(deltas) < 30:
        deltas.append(deltas[-1] * (2.5 if len(deltas) % 2 == 0 else 4.0))
    return np.array([d for d in deltas if d < lmax - lmin])


def zoom_range(cx, cy, L, prior, frac=0.2, delta=3.0):
    """Axis ranges where the L >= Lmax - delta region spans frac of each
    axis, centered on that region, clipped to the prior range."""
    mask = L >= L.max() - delta
    if mask.sum() == 0:
        return prior
    (xlo, xhi), (ylo, yhi) = prior
    ex = cx[mask].max() - cx[mask].min()
    ey = cy[mask].max() - cy[mask].min()
    xm = 0.5 * (cx[mask].max() + cx[mask].min())
    ym = 0.5 * (cy[mask].max() + cy[mask].min())
    rx, ry = ex / frac / 2, ey / frac / 2
    return ((max(xlo, xm - rx), min(xhi, xm + rx)),
            (max(ylo, ym - ry), min(yhi, ym + ry)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--outdir', default='evaluate/diagnostics_plots')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = args.dim

    problems = {
        'gauss': (GaussContour(d), (0.0, 1.0), (0.0, 1.0)),
        'funnel': (UncorrelatedFunnelLRPS(d, trunc_sigmas=6.0),
                   (-10.0, 10.0), (-3.0, 3.0)),  # (w, s) prior ranges
        'banana': (BananaContour(d), (0.0, 1.0), (0.0, 1.0)),
        'hyperpyramid': (HyperPyramidContour(d), (0.0, 1.0), (0.0, 1.0)),
        'shell': (ShellContour(d), (0.0, 1.0), (0.0, 1.0)),
        'splitellipsoid': (SplitEllipsoidContour(d),
                           (0.0, 1.0), (0.0, 1.0)),
    }

    n = 600
    for name, (prob, (xlo, xhi), (ylo, yhi)) in problems.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        if name == 'funnel':
            # (s, w1) plane at w2 = 0 (the sampling space is (sigma, dp1, dp2))
            def evaluate(x0, x1):
                w1 = np.linspace(x0[0], x0[-1], n)
                s = np.linspace(x1[0], x1[-1], n)
                W, S = np.meshgrid(w1, s, indexing='ij')
                pts = np.concatenate([
                    np.exp(S.ravel())[:, None], W.ravel()[:, None],
                    np.zeros((S.size, d - 1))], axis=1)
                L = prob.loglike(pts).reshape(W.shape)
                return w1, s, L
            xlabel, ylabel = '$w_1$', '$s = \\log\\sigma$'
            title = 'funnel d=%d: loglikelihood in (s, w) at w2 = 0' % d
        else:
            def evaluate(x0, x1):
                xs = np.linspace(x0[0], x0[-1], n)
                ys = np.linspace(x1[0], x1[-1], n)
                X0, X1 = np.meshgrid(xs, ys, indexing='ij')
                cols = [X0.ravel(), X1.ravel()] + \
                    [np.full(X0.size, 0.5)] * (d - 2)
                pts = np.stack(cols, axis=1)
                L = prob.loglike(pts).reshape(X0.shape)
                return xs, ys, L
            xlabel, ylabel = '$x_1$', '$x_0$'
            title = '%s d=%d: loglikelihood' % (name, d)

        # second panel: zoom where the Lmax-3 contour spans 20% of the axis
        _, _, L_full = evaluate((xlo, xhi), (ylo, yhi))
        if name == 'funnel':
            # grid coordinates for the zoom computation
            zw1 = np.linspace(xlo, xhi, n)
            zs = np.linspace(ylo, yhi, n)
            cx, cy = np.meshgrid(zw1, zs, indexing='ij')
        else:
            zxs = np.linspace(xlo, xhi, n)
            zys = np.linspace(ylo, yhi, n)
            cx, cy = np.meshgrid(zxs, zys, indexing='ij')
        (zx0, zx1), (zy0, zy1) = zoom_range(cx, cy, L_full,
                                            ((xlo, xhi), (ylo, yhi)))

        for ax, (x0, x1), panel in zip(
                axes, [((xlo, xhi), (ylo, yhi)), ((zx0, zx1), (zy0, zy1))],
                ['entire prior space', 'zoom (Lmax-3 contour = 20%)']):
            w1, s, L = evaluate(x0, x1)
            im = ax.imshow(L, origin='lower', extent=[w1[0], w1[-1],
                                                      s[0], s[-1]],
                           aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax, label='loglikelihood')
            lmax = L.max()
            levels = np.sort(lmax - delta_levels(lmax, L.min()))
            if len(levels):
                cxx, cyy = np.meshgrid(w1, s, indexing='ij')
                cs = ax.contour(cxx, cyy, L, levels=levels, colors='white',
                                linewidths=0.9)
                ax.clabel(cs, fmt=lambda v: '$-%g$' % (lmax - v), fontsize=7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('%s: %s' % (title, panel), fontsize=10)
        fig.tight_layout()
        fn = os.path.join(args.outdir, 'loglike2d_%s.png' % name)
        fig.savefig(fn, dpi=120)
        plt.close(fig)
        print('wrote', fn, flush=True)


if __name__ == '__main__':
    main()
