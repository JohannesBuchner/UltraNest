import numpy as np
import pytest
from numpy.testing import assert_allclose

from ultranest import ReactiveNestedSampler
from ultranest.perfectns import (
    PerfectRegion, NonIsotropicGaussianLRPS, UncorrelatedFunnelLRPS,
    BoxGaussianLRPS, BananaGaussianLRPS, HyperPyramidLRPS,
    GaussContour, HyperPyramidContour, ShellContour, BananaContour,
    SplitEllipsoidContour, mc_information_gain)


def run_with_region_class(problem, nlive=200, seed=1, **kwargs):
    np.random.seed(seed)
    sampler = ReactiveNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True)
    result = sampler.run(min_num_live_points=nlive,
                         region_class=PerfectRegion.with_problem(problem),
                         **kwargs)
    return sampler, result


def test_funnel_analytic_values():
    """Funnel lnZ and H against independent quadrature over s."""
    from scipy import integrate, special
    problem = UncorrelatedFunnelLRPS(2)
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


def test_box_analytic_values():
    """BoxGaussian lnZ against 2d numerical integration."""
    from scipy import integrate
    problem = BoxGaussianLRPS(2)

    def integrand(y, x):
        return np.exp(problem.loglike(np.array([[x, y]]))[0])

    Z, _ = integrate.dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)
    assert_allclose(np.log(Z), problem.analytic_lnZ(), atol=1e-3)


def test_gauss_analytic_values():
    problem = NonIsotropicGaussianLRPS.standard_problem(2)
    assert abs(problem.analytic_lnZ()) < 1e-4
    assert problem.analytic_H() > 0


def test_banana_reference_values_converge():
    """The grid-based reference values are converged."""
    problem = BananaGaussianLRPS.standard_problem(2)
    lnz1, H1 = problem._reference_values(ngrid=500)
    problem._reference_cache = None
    lnz2, H2 = problem._reference_values(ngrid=1500)
    assert abs(lnz1 - lnz2) < 5e-3
    assert abs(H1 - H2) < 5e-3


def test_hyperpyramid_reference_values():
    """Hyperpyramid lnZ/H from the layer-cake identity against a
    brute-force 2d grid, for both sigma variants."""
    for sigma in ('ones', 'anisotropic'):
        problem = HyperPyramidLRPS(2, s=100.0, sigma=sigma)
        n = 3000
        axes = np.linspace(0, 1, n)
        X, Y = np.meshgrid(axes, axes, indexing='ij')
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        logl = problem.loglike(pts)
        w = np.exp(logl - logl.max())
        lnZ_grid = np.log(w.mean()) + logl.max()
        e_logl = (w * logl).sum() / w.sum()
        assert abs(problem.analytic_lnZ() - lnZ_grid) < 1e-4
        assert abs(problem.analytic_H() - (e_logl - lnZ_grid)) < 1e-6


def test_hyperpyramid_proposal_exact():
    """The direct analytic proposal has acceptance rate 1 and draws from
    the constrained region at any threshold, including d=20."""
    problem = HyperPyramidLRPS(20)
    rng = np.random.RandomState(42)
    for Lmin in (-0.99, -0.5, -1e-3):
        candidates, accepted = problem.proposal(Lmin, rng, 50000)
        assert accepted.all()
        rho = (-Lmin) ** problem.s
        in_region = np.all(
            np.abs(candidates - 0.5) <= problem.sigma * rho + 1e-15, axis=1)
        assert in_region.all()
        assert problem.loglike(candidates).min() >= Lmin - 1e-12


def test_gauss_proposal_matches_exact_ball():
    """Accepted gauss proposals are uniform in the whitened ball."""
    problem = NonIsotropicGaussianLRPS.standard_problem(2)
    Lmin = -10.0 - problem._lognorm()  # whitened radius^2 = 20
    rng = np.random.RandomState(42)
    for method in ('bbox', 'ellipsoid'):
        s = problem.sample(rng, 100000, Lmin, method)
        zw = (s - problem.mu) / problem.sigma
        rad2 = (zw ** 2).sum(axis=1)
        assert problem.loglike(s).min() >= Lmin - 1e-8
        # radius^2 / radius_max^2 ~ Uniform for uniform-in-ball
        quant = np.percentile(rad2 / 20.0, [25, 50, 75])
        assert_allclose(quant, [0.25, 0.5, 0.75], atol=0.02)


def test_banana_proposal_matches_exact_body():
    """Accepted banana proposals are uniform in the banana body."""
    problem = BananaGaussianLRPS.standard_problem(2)
    Lmin = -10.0
    rng = np.random.RandomState(42)
    for method in ('banana', 'bbox'):
        s = problem.sample(rng, 100000, Lmin, method)
        z = problem.to_z(s)
        rad2 = (z ** 2).sum(axis=1)
        assert problem.loglike(s).min() >= Lmin - 1e-8
        assert s.min() > 0 and s.max() < 1
        quant = np.percentile(rad2 / 20.0, [25, 50, 75])
        assert_allclose(quant, [0.25, 0.5, 0.75], atol=0.02)


def test_funnel_proposal_matches_exact_rejection():
    """Funnel proposals reproduce the exact constrained s-marginal."""
    problem = UncorrelatedFunnelLRPS(2)
    Lmin = -6.0
    C = problem._contour_value_C(Lmin)
    rng = np.random.RandomState(42)
    # exact reference by rejection from the prior
    s0 = rng.normal(size=2000000) * problem.tau
    w0 = rng.uniform(-problem.D, problem.D, size=(2000000, 2))
    ok = (w0 ** 2).sum(axis=1) + 2 * problem.ndim * s0 <= C
    exact_s = np.percentile(s0[ok], [25, 50, 75])
    for method in ('prior', 'unboxed'):
        samples = problem.sample(rng, 100000, Lmin, method)
        S, W = samples[:, 0], samples[:, 1:]
        Cv = (W ** 2).sum(axis=1) + 2 * problem.ndim * S
        assert (Cv <= C + 1e-8).all()
        assert_allclose(np.percentile(S, [25, 50, 75]), exact_s, atol=0.02)


def test_perfect_region_fallback_without_problem():
    """Without an attached problem, PerfectRegion behaves like SimpleRegion."""
    from ultranest.mlfriends import AffineLayer
    np.random.seed(1)
    us = np.random.uniform(0.2, 0.8, size=(30, 2))
    layer = AffineLayer()
    layer.optimize(us, us)
    region = PerfectRegion(us, layer)
    r, f = region.compute_enlargement(nbootstraps=5)
    region.maxradiussq = r
    region.enlarge = f
    region.create_ellipsoid()
    samples = region.sample(nsamples=10)
    assert len(samples) > 0
    assert np.all((samples > 0) & (samples < 1))


@pytest.mark.parametrize("problem_factory", [
    lambda: NonIsotropicGaussianLRPS.standard_problem(2),
    lambda: UncorrelatedFunnelLRPS(2),
    lambda: BoxGaussianLRPS(2),
    lambda: BananaGaussianLRPS.standard_problem(2),
    lambda: HyperPyramidLRPS(20, sigma='ones'),
], ids=['gauss', 'funnel', 'box', 'banana', 'hyperpyramid'])
def test_region_class_runs_recover_reference_values(problem_factory):
    """The PerfectRegion region_class path (no stepsampler) recovers the
    reference lnZ and information gain; surplus accepted hits are cached
    by the integrator's point stack."""
    problem = problem_factory()
    np.random.seed(1)
    sampler = ReactiveNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True)
    result = sampler.run(min_num_live_points=200, frac_remain=0.001,
                         region_class=PerfectRegion.with_problem(problem))
    lnz_true, H_true = problem.analytic_lnZ(), problem.analytic_H()
    print(problem, 'logz=%.4f+-%.4f (true %.4f), H=%.4f+-%.4f (true %.4f)' % (
        result['logz'], result['logzerr'], lnz_true,
        result['H'], result['Herr'], H_true))
    assert abs(result['logz'] - lnz_true) < 4 * result['logzerr'] + 0.5
    assert abs(result['H'] - H_true) < 4 * result['Herr'] + 1.0


def test_funnel_importance_sampler_truncated():
    """The exact posterior-shape importance sampler gives the definitive
    truncated lnZ (zero-variance weights) and exact posterior samples."""
    from ultranest.perfectns import funnel_importance_sampler
    problem = UncorrelatedFunnelLRPS(20, trunc_sigmas=6.0)
    analytic = problem.analytic_lnZ()
    u, lnZ, n = funnel_importance_sampler(problem, 200000, seed=3)
    assert n > 0
    # the weights are constant, so lnZ must equal the analytic formula
    assert abs(lnZ - analytic) < 1e-6, (lnZ, analytic)
    # the samples are exact posterior draws: s inside the truncated support
    from scipy.special import ndtri
    svals = problem.tau * ndtri(np.clip(u[:, 0], 1e-300, 1 - 1e-300))
    assert svals.min() >= problem.s_lower - 1e-9
    assert svals.max() <= problem.s_upper + 1e-9
    assert np.isfinite(problem.loglike(problem.transform(u))).all()


def test_funnel_truncated_run_agrees():
    """With the truncated s-prior, the nested sampling run recovers the
    truncated analytic lnZ at d=20 (the ridge is reachable)."""
    problem = UncorrelatedFunnelLRPS(20, trunc_sigmas=6.0)
    np.random.seed(1)
    sampler = ReactiveNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True)
    result = sampler.run(min_num_live_points=400, frac_remain=0.001,
                         region_class=PerfectRegion.with_problem(problem),
                         max_ncalls=100000)
    print('truncated funnel d=20: lnZ=%.4f+-%.4f (true %.4f)' % (
        result['logz'], result['logzerr'], problem.analytic_lnZ()))
    assert abs(result['logz'] - problem.analytic_lnZ()) < \
        4 * result['logzerr'] + 0.5


# ---------------------------------------------------------------------------
# contour-parameterized problems (GaussContour, HyperPyramidContour,
# ShellContour, BananaContour, SplitEllipsoidContour): agreement with a
# fine grid in the 2d case
# ---------------------------------------------------------------------------

CONTOUR_PROBLEMS = {
    'gauss': GaussContour,
    'hyperpyramid': HyperPyramidContour,
    'shell': ShellContour,
    'banana': BananaContour,
    'splitellipsoid': SplitEllipsoidContour,
}


def _grid_reference(problem, n=1200):
    """Fine-grid reference values in the 2d case: lnZ and H by
    importance sampling with the uniform prior over a fine grid (each
    cell carries equal prior mass, so the estimator is exact up to the
    grid discretization), plus the sorted likelihoods for the empirical
    survival function."""
    from scipy.special import logsumexp
    xs = np.linspace(0.0, 1.0, n)
    X0, X1 = np.meshgrid(xs, xs)
    pts = np.stack([X0.ravel(), X1.ravel()], axis=-1)
    L = problem.loglike(pts)
    lnZ = float(logsumexp(L) - np.log(L.size))
    w = np.exp(L - L.max())
    H = float((L * w).sum() / w.sum()) - lnZ
    return L, lnZ, H


@pytest.mark.parametrize('name', sorted(CONTOUR_PROBLEMS))
def test_contour_grid_volume(name):
    """Volume function of each contour problem against a fine 2d grid."""
    problem = CONTOUR_PROBLEMS[name](2)
    L, lnZ, H = _grid_reference(problem)
    vol = problem.make_volume_function(ndraw=int(1e7))
    Ls = np.sort(L.ravel())
    ncell = L.size
    for frac in [0.3, 0.05, 0.005]:
        # level where frac of the grid cells are above
        lmin = float(Ls[int(round(frac * ncell))])
        v_grid = float((L >= lmin).mean())
        assert_allclose(float(vol(lmin)), v_grid, rtol=0.05)
        if hasattr(problem, 'prior_volume'):
            assert_allclose(float(problem.prior_volume(lmin)), v_grid,
                            rtol=0.05)


@pytest.mark.parametrize('name', sorted(CONTOUR_PROBLEMS))
def test_contour_grid_H(name):
    """Information gain H of each contour problem against a fine 2d
    grid. The default gauss and hyperpyramid widths (down to 1e-4) are
    far sharper than any affordable grid, so those two use wider
    sigmas, where the grid is a valid independent reference. The
    splitellipsoid likelihood is nearly flat (H ~ 0.01 nats), so an
    absolute tolerance is used there."""
    if name == 'gauss':
        problem = GaussContour(2, sigma=0.05)
    elif name == 'hyperpyramid':
        problem = HyperPyramidContour(2, sigma=0.05)
    else:
        problem = CONTOUR_PROBLEMS[name](2)
    L, lnZ, H = _grid_reference(problem)
    try:
        H_mod = float(problem.analytic_H())
    except NotImplementedError:
        H_mod = float(mc_information_gain(problem))
    assert_allclose(H_mod, H, rtol=0.05, atol=0.005)


@pytest.mark.parametrize('name', sorted(CONTOUR_PROBLEMS))
def test_contour_grid_proposal(name):
    """Proposal draws are uniform in the constrained region: the
    survival of L among accepted draws matches the volume curve."""
    problem = CONTOUR_PROBLEMS[name](2)
    L, lnZ, H = _grid_reference(problem)
    vol = problem.make_volume_function(ndraw=int(1e7))
    Ls = np.sort(L.ravel())
    ncell = L.size
    lmin = float(Ls[int(round(0.05 * ncell))])
    rng = np.random.RandomState(1)
    x, ok = problem.proposal(lmin, rng, 200000)
    assert ok.sum() > 1000
    samples = problem.to_u_batch(x[ok])
    Lp = problem.loglike(samples)
    assert (Lp >= lmin - 1e-9).all()
    vol_lmin = float(vol(lmin))
    for frac in [0.03, 0.01]:
        l = float(Ls[int(round(frac * ncell))])
        expected = float(vol(l)) / vol_lmin
        assert_allclose(float((Lp >= l).mean()), expected, rtol=0.1)


def test_splitellipsoid_symmetry():
    """The two split-ellipsoid components are exact mirror images."""
    problem = SplitEllipsoidContour(2)
    xs = np.linspace(0.0, 1.0, 401)
    X0, X1 = np.meshgrid(xs, xs)
    pts = np.stack([X0.ravel(), X1.ravel()], axis=-1)
    L = problem.loglike(pts)
    assert np.abs(L - problem.loglike(1.0 - pts)).max() < 1e-10


def test_funnel_volume_grid():
    """Funnel prior volume against a fine (s, w1, w2) grid."""
    from scipy import stats
    problem = UncorrelatedFunnelLRPS(2, trunc_sigmas=6.0)
    N, tau, D = problem.ndim, problem.tau, problem.D
    logK = 0.5 * (N * np.log(2 * np.pi) + problem.logdetM)
    ss = np.linspace(problem.s_lower, problem.s_upper, 201)
    ws = np.linspace(-D, D, 201)
    P_s = stats.truncnorm(problem.s_lower / tau, problem.s_upper / tau,
                          scale=tau)
    W1, W2 = np.meshgrid(ws, ws)
    dw2 = (ws[1] - ws[0]) ** 2
    for Lmin in [-50.0, -20.0, -10.0, -5.0]:
        w2max = -2.0 * (Lmin + logK) - 2 * N * ss  # |w|^2 budget per s
        A = np.array([
            ((W1 ** 2 + W2 ** 2 <= wm) * dw2).sum() if wm > 0 else 0.0
            for wm in w2max])
        v_grid = float(np.trapezoid(P_s.pdf(ss) * A, ss))
        assert_allclose(float(problem.prior_volume(Lmin)), v_grid,
                        rtol=0.02)
