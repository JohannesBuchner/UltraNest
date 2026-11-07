"""Benchmark the projection-based stepping-out bracketing.

Compares the default adaptive bracket (walk origin +- scale, scale
adapted from walk to walk) against the optional live-point-projection
bracket (``bracket_live_points=s``: the walk bracket starts from the
extreme projections of the live points onto the proposed slice
direction, padded by s times the projected span, scale fixed), over
test distributions and dimensions.

The metric is likelihood evaluations per buried dead point over three
rotations (K = 50 -> 200): it measures the bracketing cost (stepping-out
doublings plus bisection rejections) per walk.
"""
import time

import numpy as np

from ultranest.perfectns import NonIsotropicGaussianLRPS
from ultranest.snowballingns import SnowballingNestedSampler


def make_rosenbrock2d():
    """2d Rosenbrock function on [-10, 10]^2 with a flat prior."""

    def loglike(theta):
        a = theta[:, :-1]
        b = theta[:, 1:]
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum(axis=1)

    def transform(u):
        return u * 20 - 10

    return ['param1', 'param2'], transform, loglike


def make_sine_problem():
    """The docs notebook problem: two sinusoids + background, seeded."""

    def sine_model2(t, B, A1, P1, t1, A2, P2, t2):
        return A1 * np.sin((t / P1 + t1) * 2 * np.pi) \
            + A2 * np.sin((t / P2 + t2) * 2 * np.pi) + B

    np.random.seed(42)
    t = np.random.uniform(0, 5, size=50)
    y = np.random.normal(sine_model2(t, B=1.0, A1=4.2, P1=3, t1=0,
                                     A2=1.2, P2=1.2, t2=1.2), 1.0)

    def transform(cube):
        params = cube.copy()
        params[:, 0] = cube[:, 0] * 20 - 10
        params[:, 1] = 10**(cube[:, 1] * 3 - 1)
        params[:, 2] = 10**(cube[:, 2] * 2)
        params[:, 3] = cube[:, 3]
        params[:, 4] = 10**(cube[:, 4] * 3 - 1)
        params[:, 5] = 10**(cube[:, 5] * 2)
        params[:, 6] = cube[:, 6]
        return params

    def loglike(params):
        B, A1, P1, t1, A2, P2, t2 = params.transpose()
        y_model = sine_model2(
            t[:, None], B=B[None, :], A1=A1[None, :], P1=P1[None, :],
            t1=t1[None, :], A2=A2[None, :], P2=P2[None, :], t2=t2[None, :])
        loglike = -0.5 * (((y_model - y[:, None]) / 1.0)**2).sum(axis=0)
        swap = P1 < P2
        loglike[swap] = -1e300 * np.abs(P1[swap] - P2[swap])
        return loglike

    return ['B', 'A1', 'P1', 't1', 'A2', 'P2', 't2'], transform, loglike


def benchmark_bracket(num_live_points_base=50, max_num_live_points=200):
    """Sweep the bracketing modes over test distributions/dimensions.

    Runs three widening rotations from *num_live_points_base* up to
    *max_num_live_points* (e.g. K = 50 -> 200, or one rotation at
    K = 100 with num_live_points_base = max_num_live_points = 100).
    """
    problems = [
        ('gauss-2d', NonIsotropicGaussianLRPS(2)),
        ('gauss-10d', NonIsotropicGaussianLRPS(10)),
        ('rosenbrock-2d', None),
        ('sine-7d', None),
    ]
    configs = [('adaptive', {})] + [
        ('lp%.2f' % s, dict(bracket_live_points=s))
        for s in (0.05, 0.1, 0.2, 0.5, 1.0)] + [
        ('pad-adapt pw1.1', dict(bracket_live_points=0.1,
                                 bracket_pad_adaptive=True)),
        ('pad-adapt glob1.1', dict(bracket_live_points=0.1,
                                   bracket_pad_adaptive=True,
                                   bracket_pad_global=True))]

    for pname, problem in problems:
        if problem is None:
            if pname == 'rosenbrock-2d':
                paramnames, transform, loglike = make_rosenbrock2d()
            else:
                paramnames, transform, loglike = make_sine_problem()
        else:
            paramnames = problem.paramnames
            transform = problem.transform
            loglike = problem.loglike
        print('=== %s (K=%d) ===' % (pname, num_live_points_base))
        for cname, kw in configs:
            np.random.seed(1)
            t0 = time.time()
            sampler = SnowballingNestedSampler(
                paramnames, loglike, transform=transform, vectorized=True,
                num_steps=1, num_live_points_base=num_live_points_base,
                **kw)
            result = sampler.run(
                frac_remain=0.5, resume=False,
                max_num_live_points=max_num_live_points,
                show_status=False)
            duration = time.time() - t0
            sampler.state.close()
            print('  %-9s: %7d evals %5d dead  %5.1f evals/dead  %4.1fs  '
                  'logz=%8.3f+-%.3f' % (
                      cname, result['ncall'], result['niter'],
                      result['ncall'] / max(result['niter'], 1), duration,
                      result['logz'], result['logzerr']))


if __name__ == '__main__':
    benchmark_bracket()
