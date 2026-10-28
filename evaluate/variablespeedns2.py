"""
Variable-speed nested sampling II: results-based step doubling
==============================================================

Runs one persistent nested sampling live-point population with a step
sampler at nsteps. At every dead-point replacement, the chain that
produced the replacement point is *continued* for another nsteps accepted
steps, giving an "alternative" replacement final point which is recorded
but not used. After a window of replacements, the paired likelihoods of
used and alternative replacement points are compared with a one-sided
Wilcoxon signed-rank test (alternative > used). If the test is
significant, the chain length systematically changes the results: the run
rewinds to the live point population at the window's start, stitches the
window's alternative points back in where they can validly replace a live
point (a point recorded at threshold Lmin may replace any live point with
L >= Lmin that it exceeds — the resume/point-stack logic), and resumes
with doubled nsteps, again building alternatives with the doubled
continuation. If the test is not significant, the window stands and the
run continues unchanged: high RJD is not chased when the results do not
change.

Unlike variablespeedns.py, this variant needs no knowledge of the true
prior volume V(Lmin): the windows are iteration-count windows and the
evidence is integrated with the usual 1/nlive bookkeeping, so it is
applicable to arbitrary problems.

Usage::

    python evaluate/variablespeedns2.py [--problem banana] [--dim 2]
        [--nlive 400] [--window-iters 10000] [--alpha 0.01]
"""
import argparse
import os

import numpy as np
from numpy import exp, log, log1p, logaddexp
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultranest.mlfriends import MLFriends, LocalAffineLayer, ScalingLayer
from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
from ultranest.perfectns import (
    BananaGaussianLRPS, HyperPyramidLRPS, NonIsotropicGaussianLRPS,
    UncorrelatedFunnelLRPS)


def make_problem(args):
    """Test problem (no volume function needed by this variant)."""
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


def run(problem, args):
    """Run the Wilcoxon-triggered variable-speed nested sampling.

    Returns a dict with the evidence, the nsteps progression and the
    per-window test results.
    """
    ndim = len(problem.paramnames)
    transform, loglike = problem.transform, problem.loglike
    nlive = args.nlive
    np.random.seed(args.seed)
    u = np.random.uniform(size=(nlive, ndim))
    L = np.asarray(loglike(transform(u)), dtype=float).flatten()
    assert L.shape == (nlive,)

    region = make_region(u, ndim)
    nsteps = args.nsteps
    sampler = SliceSampler(nsteps=nsteps,
                           generate_direction=generate_mixture_random_direction)

    logVrem = 0.0       # ln of bookkeeping remaining volume
    logZ = -1e300
    h = 0.0
    Lmax = float(L.max())

    pairs = []   # (L_used, L_alt) since the last window test
    alts = []    # (Lmin, u_alt, L_alt) since the last window test
    stack = []   # alternatives to stitch in (after a rewind)
    traj = []    # (Lmin, -lnX bookkeeping, nsteps) per dead point
    snap = (u.copy(), L.copy(), logVrem, h, logZ, 0)
    iter_since_test = 0
    windows = []
    nrewinds = 0

    total = 0
    while total < args.max_iters:
        worst = int(np.argmin(L))
        Lmin = float(L[worst])

        # 1. resume logic: the first usable alternative replaces the
        #    worst live point (a point recorded at threshold Lmin_r may
        #    replace a live point with L >= Lmin_r that it exceeds)
        used_stack = False
        for j, row in enumerate(stack):
            if row[0] <= Lmin and row[2] > Lmin:
                u[worst] = row[1]
                L[worst] = row[2]
                stack.pop(j)
                used_stack = True
                break

        if not used_stack:
            # 2. dual chain: the used replacement from an nsteps chain,
            #    the alternative from continuing it another nsteps steps
            while True:
                u1, p1, L1, nc = sampler.__next__(
                    region, Lmin=Lmin, us=u, Ls=L,
                    transform=transform, loglike=loglike)
                if u1 is not None:
                    break
            while True:
                u2, p2, L2, nc = sampler.__next__(
                    region, Lmin=Lmin, us=u, Ls=L,
                    transform=transform, loglike=loglike)
                if u2 is not None:
                    break
            u1 = np.asarray(u1, dtype=float).flatten()
            L1 = float(np.asarray(L1).flatten()[0])
            u2 = np.asarray(u2, dtype=float).flatten()
            L2 = float(np.asarray(L2).flatten()[0])
            u[worst] = u1
            L[worst] = L1
            pairs.append((L1, L2))
            alts.append((Lmin, u2, L2))

        # bookkeeping: the dead point is the replaced worst point
        logwidth = logVrem + log1p(-exp(-1.0 / nlive))
        logwt = logwidth + Lmin
        logZ_new = logaddexp(logZ, logwt)
        h = (exp(logwt - logZ_new) * Lmin
             + exp(logZ - logZ_new) * (h + logZ) - logZ_new)
        logZ = logZ_new
        logVrem -= 1.0 / nlive
        traj.append((Lmin, -logVrem, nsteps))
        Lmax = max(Lmax, float(L[worst]))
        total += 1
        iter_since_test += 1

        # keep the region informed about the new live point
        region.u[worst] = u[worst]
        region.unormed[worst] = region.transformLayer.transform(u[worst])
        region.ellipsoid_center = np.mean(region.u, axis=0)
        if total % nlive == 0:
            region = make_region(u, ndim)

        # window boundary: results-based test
        if iter_since_test >= args.window_iters:
            d = np.asarray([a - b for b, a in pairs], dtype=float)
            mean_d = float(d.mean()) if len(d) else np.nan
            mean_L = (float(np.mean([b for b, a in pairs]))
                      if len(pairs) else np.nan)
            significant = False
            p = np.nan
            p_ad = np.nan
            if len(d) >= args.min_pairs:
                try:
                    stat, p = stats.wilcoxon(d, alternative='greater')
                    p = float(p)
                    significant = p < args.alpha
                except ValueError:
                    p = np.nan
                try:
                    ad = stats.anderson_ksamp(
                        [np.asarray([b for b, a in pairs]),
                         np.asarray([a for b, a in pairs])])
                    p_ad = float(np.clip(ad.pvalue, 1e-4, 1.0))
                except (ValueError, np.linalg.LinAlgError):
                    p_ad = np.nan
            if significant:
                # rewind to the window start, stitch the alternatives in,
                # resume with doubled nsteps
                u[:] = snap[0]
                L[:] = snap[1]
                logVrem, h, logZ = snap[2], snap[3], snap[4]
                del traj[snap[5]:]
                stack = alts
                nsteps *= 2
                sampler = SliceSampler(
                    nsteps=nsteps,
                    generate_direction=generate_mixture_random_direction)
                region = make_region(u, ndim)
                pairs, alts = [], []
                iter_since_test = 0
                nrewinds += 1
                print('window %d: %d pairs, '
                      'mean(L_alt-L_used)=%+.4f, Wilcoxon p=%.3g, '
                      'Anderson-Darling p=%.3g -> REWIND: '
                      'stitching %d alternatives, nsteps -> %d' % (
                          len(windows) + 1, len(d), mean_d, p, p_ad,
                          len(stack), nsteps), flush=True)
            else:
                # the window stands
                snap = (u.copy(), L.copy(), logVrem, h, logZ, len(traj))
                pairs, alts = [], []
                iter_since_test = 0
                print('window %d: %d pairs, '
                      'mean(L_alt-L_used)=%+.4f, Wilcoxon p=%s, '
                      'Anderson-Darling p=%s -> continue '
                      '(nsteps=%d)' % (
                          len(windows) + 1, len(d),
                          mean_d, '%.3g' % p if np.isfinite(p) else 'nan',
                          '%.3g' % p_ad if np.isfinite(p_ad) else 'nan',
                          nsteps), flush=True)
            windows.append(dict(
                total=total, npairs=int(len(d)), mean_d=mean_d, p=p, p_ad=p_ad,
                significant=bool(significant), nsteps=nsteps, mean_L=mean_L))

        # convergence
        logZremain = Lmax + logVrem
        if logZremain - logZ < log(args.frac_remain) \
                or (Lmax - float(L.min())) < args.Lepsilon:
            break

    logzerr = float(np.sqrt(max(h, 0.0) / nlive))
    return dict(logz=logZ, logzerr=logzerr, h=h, nsteps=nsteps,
                total=total, ndead=total, windows=windows,
                nrewinds=nrewinds, L=L, u=u, traj=traj)


def plot_panels(traj, windows, outdir, title):
    """Four panels: the dead point trajectory (lnL vs bookkeeping -lnX),
    the window test p-values at the lnL where they were tested
    (color-coded by window), the window mean likelihood differences, and
    the nsteps adaptation along the run."""
    traj = np.asarray(traj, dtype=float)
    nwin = max(len(windows), 1)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, nwin))
    fig, axes = plt.subplots(4, 1, figsize=(7, 13))

    ax = axes[0]
    ax.plot(traj[:, 1], traj[:, 0], '-', color='k', lw=0.8)
    ax.set_ylabel('lnL')
    ax.set_title(title, fontsize=11)

    ax = axes[1]
    for k, w in enumerate(windows):
        c = colors[min(k, nwin - 1)]
        if np.isfinite(w['p']):
            ax.semilogy(w['mean_L'], max(w['p'], 1e-7), 'o', color=c, ms=7)
        if np.isfinite(w.get('p_ad', np.nan)):
            ax.semilogy(w['mean_L'], max(w['p_ad'], 1e-7), '^',
                        color=c, ms=7, mfc='none')
    ax.axhline(0.01, color='k', ls='--', lw=0.8)
    ax.set_ylabel('test p')
    ax.set_title('window test p-values at the tested lnL '
                 '(color: window; circle Wilcoxon, triangle Anderson-Darling)',
                 fontsize=11)
    ax.set_ylim(1e-7, 2)

    ax = axes[2]
    for k, w in enumerate(windows):
        if np.isfinite(w['mean_d']):
            ax.plot(w['mean_L'], w['mean_d'], 'o',
                    color=colors[min(k, nwin - 1)], ms=7)
    ax.axhline(0.0, color='k', ls=':', lw=0.8)
    ax.set_ylabel('mean(L_alt - L_used)')
    ax.set_title('window mean likelihood difference', fontsize=11)

    ax = axes[3]
    ax.step(traj[:, 1], traj[:, 2], '-', color='k', lw=1.2, where='post')
    ax.set_ylabel('nsteps')
    ax.set_xlabel('-ln(X) (bookkeeping)')
    ax.set_title('step count adaptation', fontsize=11)

    fig.tight_layout()
    filename = os.path.join(outdir, 'variablespeedns2_panels.png')
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['banana', 'hyperpyramid',
                                              'funnel', 'gauss'],
                        default='banana')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--nlive', type=int, default=400)
    parser.add_argument('--nsteps', type=int, default=1,
                        help='initial number of steps')
    parser.add_argument('--window-iters', type=int, default=10000,
                        help='replacements per test window')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='one-sided Wilcoxon significance threshold')
    parser.add_argument('--min-pairs', type=int, default=10,
                        help='minimum number of pairs to run the test')
    parser.add_argument('--frac-remain', type=float, default=0.01)
    parser.add_argument('--Lepsilon', type=float, default=0.001)
    parser.add_argument('--max-iters', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()

    problem = make_problem(args)
    out = run(problem, args)

    print()
    print('converged: %d replacements (incl. rewound work), '
          'final nsteps=%d, %d windows tested, %d rewinds' % (
              out['total'], out['nsteps'], len(out['windows']),
              out['nrewinds']))
    print('lnZ = %.4f +- %.4f' % (out['logz'], out['logzerr']))
    try:
        print('analytic lnZ = %.4f' % problem.analytic_lnZ())
    except Exception:
        pass

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        title = '%s, dim=%d, nlive=%d, windows of %d' % (
            args.problem, args.dim, args.nlive, args.window_iters)
        fn = plot_panels(out['traj'], out['windows'], args.outdir, title)
        print('wrote', fn, flush=True)
        fn = os.path.join(args.outdir, 'variablespeedns2_windows.txt')
        with open(fn, 'w') as f:
            f.write('# window total npairs mean_d p_wilcoxon p_anderson '
                    'significant nsteps\n')
            for k, win in enumerate(out['windows'], 1):
                f.write('%d %d %d %+.5f %s %s %d %d\n' % (
                    k, win['total'], win['npairs'], win['mean_d'],
                    '%.4g' % win['p'] if np.isfinite(win['p']) else 'nan',
                    ('%.4g' % win['p_ad'])
                    if np.isfinite(win.get('p_ad', np.nan)) else 'nan',
                    win['significant'] * 1, win['nsteps']))
        print('wrote', fn, flush=True)


if __name__ == '__main__':
    main()
