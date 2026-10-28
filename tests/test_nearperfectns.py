import numpy as np
import pytest
from numpy.testing import assert_allclose

from ultranest import ReactiveNestedSampler
from ultranest.nearperfectns import (
    NearPerfectStepsampler, AsymGauss, Funnel, BetaProduct, BoxGaussianLRPS, box_chord_interval)


def run_with_sampler(problem, nlive=100, seed=1, **kwargs):
    np.random.seed(seed)
    sampler = ReactiveNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True)
    sampler.stepsampler = problem.create_stepsampler()
    result = sampler.run(min_num_live_points=nlive, **kwargs)
    return sampler, result


def test_funnel_analytic_values():
    """Funnel lnZ and H against independent quadrature over s."""
    from scipy import integrate, special
    problem = Funnel(2)
    nd, tau, D = problem.ndim, problem.tau, problem.D
    K = 0.5 * (nd * np.log(2 * np.pi) + problem.logdetM)
    qint = np.sqrt(2 * np.pi) * (2 * special.ndtr(D) - 1)

    def weight(s):
        return np.exp(-nd * s - K) * qint ** nd / (2 * D) ** nd \
            * np.exp(-0.5 * (s / tau) ** 2) / (tau * np.sqrt(2 * np.pi))

    m2 = 1 - 2 * D * np.exp(-0.5 * D ** 2) / np.sqrt(2 * np.pi) \
        / (2 * special.ndtr(D) - 1)
    Z, _ = integrate.quad(weight, -20, 20)
    H, _ = integrate.quad(lambda s: weight(s) * (-0.5 * nd * m2 - nd * s - K), -20, 20)
    assert_allclose(np.log(Z), problem.analytic_lnZ(), atol=1e-5)
    assert_allclose(H / Z - np.log(Z), problem.analytic_H(), atol=1e-5)


def test_asymgauss_analytic_values():
    """AsymGauss lnZ ~ 0 (normalized Gaussian in unit cube)."""
    problem = AsymGauss(2)
    assert abs(problem.analytic_lnZ()) < 1e-4
    assert problem.analytic_H() > 0


def test_beta_analytic_values():
    """Beta product has lnZ = 0 and positive information gain."""
    problem = BetaProduct(2)
    assert problem.analytic_lnZ() == 0.0
    assert problem.analytic_H() > 0


def test_box_analytic_values():
    """BoxGaussian lnZ against 2d numerical integration."""
    from scipy import integrate
    problem = BoxGaussianLRPS(2)

    def integrand(y, x):
        return np.exp(problem.loglike(np.array([[x, y]]))[0])

    Z, _ = integrate.dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)
    assert_allclose(np.log(Z), problem.analytic_lnZ(), atol=1e-3)


def test_beta_chord_intervals():
    """Beta chord intervals match a brute-force likelihood scan."""
    problem = BetaProduct(2)
    rng = np.random.RandomState(1)
    nmismatch = 0
    for trial in range(50):
        x = rng.uniform(0.01, 0.99, size=2)
        v = rng.normal(size=2)
        v /= np.linalg.norm(v)
        Lmin = problem.loglike(x[None, :])[0] - rng.exponential()
        intervals = problem.chord_intervals(x, v, Lmin)
        t_lo, t_hi = box_chord_interval(x, v, 0.0, 1.0)
        ts = np.linspace(t_lo, t_hi, 5001)
        inside = np.array(
            [problem._line_loglike(x, v, t) >= Lmin for t in ts])
        grid_measure = inside.mean() * (t_hi - t_lo)
        chord_measure = sum(hi - lo for lo, hi in intervals)
        if abs(grid_measure - chord_measure) > 0.01 * (t_hi - t_lo):
            nmismatch += 1
    assert nmismatch == 0


def test_funnel_chain_constrained_prior():
    """Funnel hit-and-run chain stays inside the constrained region and
    reproduces the exact s-marginal."""
    problem = Funnel(2)
    Lmin = -6.0
    C = -2.0 * (Lmin + 0.5 * (problem.ndim * np.log(2 * np.pi) + problem.logdetM))
    s, w = 0.3, np.array([2.0, -1.5])
    samples_s, samples_w = [], []
    for i in range(20000):
        s, w = problem.step((s, w), Lmin, np.random)
        if i >= 2000:
            samples_s.append(s)
            samples_w.append(w)
    S = np.array(samples_s)
    W = np.array(samples_w)
    # every point satisfies the likelihood threshold
    assert ((W ** 2).sum(axis=1) + 2 * problem.ndim * S <= C + 1e-8).all()
    # the s-marginal matches an exact Monte-Carlo estimate
    D, tau = problem.D, problem.tau
    wmc = np.random.RandomState(42).uniform(-D, D, size=(400000, problem.ndim))
    grid = np.linspace(-2, 1.2, 40)
    exact = np.array([
        np.exp(-0.5 * (s / tau) ** 2)
        * ((wmc ** 2).sum(axis=1) <= C - 2 * problem.ndim * s).mean()
        for s in grid])
    exact = exact / exact.sum()
    hist = np.histogram(S, bins=40, range=(-2, 1.2))[0].astype(float)
    hist = hist / hist.sum()
    # chain has ~18k samples, so ~1% binomial noise is expected
    assert np.abs(hist - exact).max() < 0.02


@pytest.mark.parametrize("problem_factory", [
    lambda: AsymGauss(2),
    lambda: Funnel(2),
    lambda: BetaProduct(2),
    lambda: BoxGaussianLRPS(2),
], ids=["asymgauss", "funnel", "beta", "box"])
def test_runs_recover_analytic_values(problem_factory):
    """ReactiveNestedSampler with the near-perfect sampler recovers the
    analytic lnZ, information gain and passes the jump diagnostics."""
    problem = problem_factory()
    sampler, result = run_with_sampler(problem, nlive=100, frac_remain=0.001)
    lnz_true, H_true = problem.analytic_lnZ(), problem.analytic_H()
    # the sampler is nearly exact, so deviations are the nested sampling
    # statistical error; allow generous 4-sigma
    assert abs(result['logz'] - lnz_true) < 4 * result['logzerr'] + 0.3, \
        (result['logz'], result['logzerr'], lnz_true)
    assert abs(result['H'] - H_true) < 4 * result['Herr'] + 0.5, \
        (result['H'], result['Herr'], H_true)
    # diagnostics of the step sampler
    print(problem, 'logz=%.4f+-%.4f (true %.4f), H=%.4f+-%.4f (true %.4f), '
          'far_enough=%.2f' % (
              result['logz'], result['logzerr'], lnz_true,
              result['H'], result['Herr'], H_true,
              sampler.stepsampler.far_enough_fraction))
    assert sampler.stepsampler.far_enough_fraction > 0.2
