"""
Perfect nested sampling test problems with direct rejection sampling.

MCMC-free validation problems: at each likelihood constraint, candidates
are drawn in batches from an analytically known region that encloses the
constrained prior, and the points passing the likelihood threshold are
kept. Surplus accepted points are cached and served at subsequent
iterations by the integrator's point stack: a uniform sample of a set
remains an exact uniform sample of the set restricted by a raised
threshold, so cached points stay valid after filtering.

Problems: gauss, funnel, box, banana, hyperpyramid. Each provides an exact
enclosing proposal built from the known geometry (``proposal``), reference
values (``analytic_lnZ``, ``analytic_H``) and the usual ``paramnames``,
``transform``, ``loglike`` interface. The funnel optionally truncates its
Gaussian log(sigma) prior (``trunc_sigmas``) and comes with an exact
posterior-shape importance sampler (``funnel_importance_sampler``) that
gives a definitive lnZ and posterior at any dimension. (The beta product's
divergent spikes cannot be enclosed by these proposals and is not
supported.)

Usage, through the standard region-based sampling path::

    from ultranest import ReactiveNestedSampler
    from ultranest.perfectns import NonIsotropicGaussianLRPS, PerfectRegion

    problem = NonIsotropicGaussianLRPS.standard_problem(2)
    sampler = ReactiveNestedSampler(problem.paramnames, problem.loglike,
                                    transform=problem.transform,
                                    vectorized=True)
    result = sampler.run(region_class=PerfectRegion.with_problem(problem))

    print(result['logz'], problem.analytic_lnZ())

The samples come from ``problem.proposal``; the integrator evaluates the
likelihood and caches surplus accepted hits. Without an attached problem,
:class:`PerfectRegion` behaves like :py:class:`SimpleRegion`; problems
without a known enclosing geometry can use the default region classes.
"""
import functools

import numpy as np
from scipy import integrate, optimize, special

from .mlfriends import SimpleRegion

# ---------------------------------------------------------------------------
# exact prior volumes of the constrained region
# ---------------------------------------------------------------------------


class BoxBallVolume:
    """Exact volume of the N-ball of radius r clipped to the box.

    |w_i| <= D, as a fraction of the box volume:

        F(t) = P(sum_i u_i^2 <= t),   t = (r/D)^2,  u_i ~ U[-1, 1]

    a sum of N iid Beta(1/2, 1) variables. min(box, ball) is NOT enough:
    in the crossover regime D < r < D*sqrt(N) neither shape contains the
    other and the box corners stick out of the ball. The CDF is
    precomputed once by discrete convolution of the exact per-bin masses
    of u^2, P(x < u^2 <= x+dx) = sqrt(x+dx) - sqrt(x) (sampling the
    1/sqrt(x) density on a grid is inaccurate); the left-edge bin
    placement shifts the sum down by N*dx/2, compensated exactly.
    """

    def __init__(self, N, D, ngrid=None):
        self.N = int(N)
        self.D = float(D)
        if ngrid is None:
            ngrid = max(16384, 1024 * self.N)
        self.t = np.linspace(0.0, float(self.N), ngrid)
        dx = self.t[1] - self.t[0]
        upper = np.minimum(self.t + dx, 1.0)
        p = np.sqrt(upper) - np.sqrt(self.t)
        p = np.maximum(p, 0.0)
        p *= 1.0 / p.sum()
        nfft = 2 * ngrid
        massN = np.fft.irfft(np.fft.rfft(p, nfft) ** self.N, nfft)[:ngrid]
        massN = np.maximum(massN, 0.0)
        F = np.cumsum(massN)
        F *= 1.0 / F[-1]
        self.shift = 0.5 * self.N * dx
        self.F = F

    def frac(self, r2):
        """Volume fraction of the box inside the ball of radius sqrt(r2),.

        vectorized over r2.
        """
        t = np.asarray(r2, dtype=float) / self.D ** 2
        return np.interp(t - self.shift, self.t, self.F, left=0.0, right=1.0)


_BBVOL_CACHE = {}


def box_ball_volume(N, D):
    """Return a cached BoxBallVolume for the given (N, D)."""
    key = (int(N), round(float(D), 12))
    if key not in _BBVOL_CACHE:
        _BBVOL_CACHE[key] = BoxBallVolume(N, D)
    return _BBVOL_CACHE[key]


class _FunnelVolumeKernel:
    """Exact prior volume V(Lmin) of {loglike >= Lmin} for the (truncated).

    funnel: an s-grid trapezoid rule (~400x faster than quadrature;
    accuracy ~1e-5 relative), with the exact ball-box crossover fraction.
    """

    def __init__(self, problem, ngrid=800):
        N, tau, D = problem.ndim, problem.tau, problem.D
        self.logK = 0.5 * (N * np.log(2 * np.pi) + problem.logdetM)
        self.s = np.linspace(problem.s_lower, problem.s_upper, ngrid)
        P_a = (2 * special.ndtr(problem.trunc_sigmas) - 1
               if problem.trunc_sigmas else 1.0)
        self.log_prior = (-0.5 * (self.s / tau) ** 2
                          - np.log(tau * np.sqrt(2 * np.pi) * P_a))
        self.two_N_s = 2 * N * self.s
        self.log_box = N * np.log(2 * D)
        self.bbvol = box_ball_volume(N, D)

    def __call__(self, Lmin):
        Lmin = np.asarray(Lmin, dtype=float)
        C = -2.0 * (Lmin[..., None] + self.logK)   # (..., 1)
        w2max = C - self.two_N_s                    # (..., ngrid)
        ok = w2max > 0
        log_vol_d = np.where(ok, self.log_box + np.log(np.maximum(
            self.bbvol.frac(np.where(ok, w2max, 1.0)), 1e-300)), -np.inf)
        V = np.trapezoid(np.exp(self.log_prior + log_vol_d), self.s)
        return float(V) if Lmin.ndim == 0 else V


class TabulatedVolume:
    """Tabulated exact prior volume V(Lmin) on an Lmin grid,.

    log-interpolated, with the exact saturated volume Vfull and an
    invertible Lmin_of_volume; vectorized over Lmin.
    """

    def __init__(self, Lgrid, lnV):
        self.Lgrid = np.asarray(Lgrid, dtype=float)
        self.lnV = np.asarray(lnV, dtype=float)
        self.lnV_exp = self.lnV[::-1]      # ascending in V
        self.Lgrid_asc = self.Lgrid[::-1]  # descending in L
        self.Vfull = float(np.exp(self.lnV[0]))  # saturated grid end

    def __call__(self, Lmin):
        return np.exp(np.interp(Lmin, self.Lgrid, self.lnV))

    def Lmin_of_volume(self, V):
        """Invert V(Lmin): the Lmin threshold at the given prior volume."""
        return float(np.interp(np.log(V), self.lnV_exp, self.Lgrid_asc))


class ExactVolume:
    """Exact per-call volume function with bisective inversion, for.

    problems whose volume is cheap to evaluate.
    """

    def __init__(self, problem, L_lo=-1.0, L_hi=0.0):
        self.problem = problem
        self.Vfull = problem.Vfull
        self.L_lo, self.L_hi = L_lo, L_hi  # V(L_lo) >= V >= V(L_hi)

    def __call__(self, Lmin):
        return self.problem.prior_volume(Lmin)

    def Lmin_of_volume(self, V):
        """Return the likelihood threshold enclosing the given volume."""
        lo, hi = self.L_lo, self.L_hi
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self(mid) >= V:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def mc_prior_volume_table(problem, ndraw=100000000, seed=0, nkeep=200001):
    """Tabulate X(Lmin) = P(logL >= Lmin) under the prior by Monte Carlo.

    with a fixed seed (exact in expectation, reproducible): the empirical
    survival function of the likelihood over ndraw prior draws, kept at
    nkeep rank-spaced likelihoods. For problems whose clipped
    constrained-region volume has no closed form (gauss, banana).
    """
    rng = np.random.RandomState(seed)
    Ls = np.empty(ndraw)
    # keep the per-chunk arrays ~<200 MB at high dimension
    chunk = max(100000, min(1000000, 200000000 // (8 * problem.ndim)))
    for i in range(0, ndraw, chunk):
        m = min(chunk, ndraw - i)
        us = rng.uniform(size=(m, problem.ndim))
        Ls[i:i + m] = problem.loglike(problem.transform(us))
    Ls.sort()
    step = max(1, ndraw // (nkeep - 1))
    keep = np.arange(0, ndraw, step)
    Lgrid = Ls[keep]
    X = 1.0 - (keep + 0.5) / ndraw
    del Ls
    return TabulatedVolume(Lgrid, np.log(np.maximum(X, 1e-300)))


def mc_information_gain(problem, ndraw=1000000, seed=0):
    """Monte-Carlo information gain H = E_post[logL] - lnZ for problems.

    without an analytic value. For constant-Jacobian problems
    (``from_z``), draws from the Gaussian posterior shape in z-space and
    keeps the preimage inside the cube; otherwise falls back to
    importance sampling with the prior (reliable only if the posterior
    occupies a large prior fraction).
    """
    rng = np.random.RandomState(seed)
    if hasattr(problem, 'from_z'):
        z = rng.normal(size=(ndraw, problem.ndim))
        x = problem.from_z(z)
        ok = np.all((x > 0) & (x < 1), axis=1)
        lnZ = (np.log(ok.mean()) + 0.5 * problem.ndim * np.log(2 * np.pi)
               + np.log(problem.sigma).sum())
        Epost = -0.5 * (z[ok] ** 2).sum(axis=1).mean()
        return Epost - lnZ
    us = rng.uniform(size=(ndraw, problem.ndim))
    L = problem.loglike(problem.transform(us))
    w = np.exp(L - L.max())
    lnZ = np.log(w.sum()) - np.log(ndraw) + L.max()
    return (w * L).sum() / w.sum() - lnZ


__all__ = [
    'PerfectRegion',
    'NonIsotropicGaussianLRPS',
    'UncorrelatedFunnelLRPS',
    'BoxGaussianLRPS',
    'BananaGaussianLRPS',
    'HyperPyramidLRPS',
    'funnel_importance_sampler',
]

_EPS = 1e-14


# ============================================================================
# Helpers
# ============================================================================

def sample_uniform_ball(rng, nsamples, ndim, radius):
    """Batched uniform draws from the N-ball ||z|| <= radius."""
    direction = rng.normal(size=(nsamples, ndim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radial = radius * rng.uniform(size=nsamples) ** (1.0 / ndim)
    return direction * np.asarray(radial).reshape((-1, 1))


def _normpdf(z):
    return np.exp(-0.5 * np.asarray(z) ** 2) / np.sqrt(2 * np.pi)


def _truncated_second_moment(lo, hi):
    """E[z^2 | lo <= z <= hi] for a standard normal z."""
    p = special.ndtr(hi) - special.ndtr(lo)
    return 1.0 + (lo * _normpdf(lo) - hi * _normpdf(hi)) / p


# ============================================================================
# Region class for the standard region-based sampling path
# ============================================================================

class PerfectRegion(SimpleRegion):
    """Axis-aligned region (like :py:class:`SimpleRegion`) that draws its.

    samples from the analytically known enclosing region of an attached
    problem.

    Use it through the standard region-based sampling path (no stepsampler
    needed)::

        from ultranest.perfectns import PerfectRegion
        sampler.run(region_class=PerfectRegion.with_problem(problem))

    :py:meth:`sample` draws candidates from ``problem.proposal`` (uniform
    in an enclosing region of the constrained prior), converts them to
    u-space and returns those inside the unit cube. The integrator
    evaluates the likelihood on them and stacks all accepted points, so
    surplus hits are cached by the integrator's point stack for subsequent
    iterations, exactly as in the default region-based path - the exact
    proposals merely make the acceptance rate much higher.

    The base region machinery is only used for the integrator's region
    tracking; the sampling itself bypasses it, so the cheap SimpleRegion
    geometry suffices and no MLFriends region building is required.

    Without an attached problem, behaves exactly like SimpleRegion.
    """

    def __init__(self, u, transformLayer, problem=None):
        SimpleRegion.__init__(self, u, transformLayer)
        self.problem = problem

    @classmethod
    def with_problem(cls, problem):
        """Return a region_class constructor that uses this problem's.

        enclosing proposals.
        """
        return functools.partial(cls, problem=problem)

    def _live_points_Lmin(self):
        """Return the current likelihood threshold: the worst live point's.

        likelihood (the integrator replaces that point next).
        """
        return float(np.min(self.problem.loglike(self.problem.transform(self.u))))

    def sample(self, nsamples=100):
        """Draw exact constrained-prior samples."""
        if self.problem is None:
            return SimpleRegion.sample(self, nsamples=nsamples)
        Lmin = self._live_points_Lmin()
        candidates, ok = self.problem.proposal(Lmin, np.random, nsamples)
        if not ok.any():
            return candidates[:0, :].reshape((0, self.u.shape[1]))
        u = self.problem.to_u_batch(candidates[ok, :])
        inside = np.all((u > 0) & (u < 1), axis=1)
        return u[inside, :]


# ============================================================================
# 1. Non-isotropic Gaussian likelihood on unit cube ("gauss")
# ============================================================================

class NonIsotropicGaussianLRPS:
    """Flat prior on [0,1]^N and likelihood.

        logL(x) = -0.5 sum_i ((x_i - mu_i)/sigma_i)^2
                  - 0.5 sum_i log(2 pi sigma_i^2)

    i.e. a normalized Gaussian truncated by the cube, so lnZ and H are
    analytic. The widths span 10^max(-1 - 53/ndim, -7) to 0.1.

    Enclosing proposals (all give independent, uniform draws from
    [0,1]^N intersected with {logL >= Lmin}):

    * ``bbox``: uniform in the coordinate bounds of the likelihood
      ellipsoid clipped to the cube; accept with the likelihood.
    * ``ellipsoid``: uniform in the likelihood ellipsoid; accept inside
      the cube.

    ``method='auto'`` picks whichever envelope is smaller.
    """

    def __init__(self, ndim, mu=0.5):
        log10_sigmamin = max(-1 - 53.0 / ndim, -7)
        self.sigma = np.logspace(log10_sigmamin, -1, ndim)
        self.mu = np.broadcast_to(mu, (ndim,)).astype(float)
        self.ndim = int(ndim)

    @classmethod
    def standard_problem(cls, ndim, mu=0.5):
        """Construct the standard problem configuration."""
        return cls(ndim=ndim, mu=mu)

    @property
    def paramnames(self):
        """Return the parameter names."""
        return ['param%d' % (i + 1) for i in range(self.ndim)]

    def __str__(self):
        """Return the string representation."""
        return 'NonIsotropicGaussianLRPS(ndim=%d)' % self.ndim

    def transform(self, x):
        """Transform the unit cube to the parameters."""
        return np.array(x)

    def to_natural(self, u):
        """Return the natural parameter space coordinates."""
        return np.asarray(u, dtype=float)

    def to_u(self, x):
        """Convert the natural parameters back to the unit cube."""
        return np.asarray(x, dtype=float)

    def to_u_batch(self, x):
        """Convert a batch of natural parameters to the unit cube."""
        return np.asarray(x, dtype=float)

    def loglike(self, theta):
        """Return the log-likelihood."""
        theta = np.asarray(theta, dtype=float)
        z = (theta - self.mu) / self.sigma
        like = -0.5 * (z ** 2).sum(axis=1) \
            - 0.5 * np.log(2 * np.pi * self.sigma ** 2).sum()
        return np.where(np.isfinite(like), like, -1e300)

    def _lognorm(self):
        return 0.5 * np.log(2 * np.pi * self.sigma ** 2).sum()

    def radius(self, Lmin):
        """Whitened radius of the likelihood ellipsoid {logL >= Lmin}."""
        return np.sqrt(max(0.0, -2.0 * (Lmin + self._lognorm())))

    def bbox(self, Lmin):
        """Return the bounding box of the likelihood constraint."""
        r = self.radius(Lmin)
        lower = np.maximum(0.0, self.mu - r * self.sigma)
        upper = np.minimum(1.0, self.mu + r * self.sigma)
        return lower, upper

    def proposal_volumes(self, Lmin):
        """Volumes of the enclosing envelopes; the smaller gives the.

        higher acceptance rate.
        """
        r = self.radius(Lmin)
        lower, upper = self.bbox(Lmin)
        ellipsoid_volume = np.exp(
            0.5 * self.ndim * np.log(np.pi)
            - special.gammaln(0.5 * self.ndim + 1)
            + self.ndim * np.log(r) + np.log(self.sigma).sum())
        return {
            'bbox': np.prod(upper - lower),
            'ellipsoid': ellipsoid_volume,
        }

    def proposal(self, Lmin, rng, nsamples, method='auto'):
        """Draw candidates from an enclosing region.

        Returns (candidates, accepted) with candidates in u-space and
        accepted indicating membership of the constrained region.
        """
        if method == 'auto':
            volumes = self.proposal_volumes(Lmin)
            method = min(volumes, key=lambda name: volumes[name])
        if method == 'bbox':
            lower, upper = self.bbox(Lmin)
            candidates = rng.uniform(lower, upper, size=(nsamples, self.ndim))
            return candidates, self.loglike(candidates) >= Lmin
        if method == 'ellipsoid':
            r = self.radius(Lmin)
            z = sample_uniform_ball(rng, nsamples, self.ndim, r)
            candidates = self.mu[None, :] + z * self.sigma[None, :]
            inside_cube = np.all((candidates > 0) & (candidates < 1), axis=1)
            return candidates, inside_cube
        raise ValueError("Unknown method=%s" % method)

    def sample(self, rng, nsamples, Lmin, method='auto'):
        """Draw exact constrained-prior samples (rejection, batched)."""
        out = []
        while sum(len(o) for o in out) < nsamples:
            candidates, accepted = self.proposal(Lmin, rng, nsamples, method)
            out.append(candidates[accepted, :])
        return np.concatenate(out, axis=0)[:nsamples]

    def analytic_lnZ(self):
        """Return the analytic log-evidence."""
        lo = (0.0 - self.mu) / self.sigma
        hi = (1.0 - self.mu) / self.sigma
        return np.log(special.ndtr(hi) - special.ndtr(lo)).sum()

    def analytic_H(self):
        """Return the analytic information gain."""
        lo = (0.0 - self.mu) / self.sigma
        hi = (1.0 - self.mu) / self.sigma
        m2 = _truncated_second_moment(lo, hi).sum()
        # H = E_posterior[logL] - lnZ
        return -0.5 * m2 - self._lognorm() - self.analytic_lnZ()

    @property
    def Vfull(self):
        """Full prior volume (flat prior on the unit cube)."""
        return 1.0

    def make_volume_function(self, vmin_frac=None, ngrid=None,
                             ndraw=100000000, seed=0):
        """Return a tabulated interpolation of X(Lmin) = P(logL >= Lmin) under the prior,.

        by fixed-seed Monte Carlo: the likelihood ellipsoid clipped to
        the cube has no closed-form volume.
        """
        return mc_prior_volume_table(self, ndraw=ndraw, seed=seed)


# ============================================================================
# 2. Funnel: s = log(sigma) ~ Normal(0, tau^2), w_i ~ Uniform[-D, D]
# ============================================================================

class UncorrelatedFunnelLRPS:
    """Funnel in the coordinates s = log(sigma), w_i.

    Prior: s ~ Normal(0, tau^2), w_i ~ Uniform[-D, D]; the nested sampling
    parameters are (sigma, dp_1..dp_N) with dp = L w, where L is the
    Cholesky matrix of M, the correlation matrix with off-diagonals gamma.
    Likelihood:

        logL = -0.5 dp' M^-1 dp - N log(sigma)
               - 0.5 (N log(2 pi) + log det M)

    In (s, w) coordinates the likelihood contour is the paraboloid

        |w|^2 + 2 N s <= C,   |w_i| <= D.

    Enclosing proposals:

    * ``prior``: draw from the prior, accept the contour constraint.
    * ``unboxed``: draw s from the analytically known constrained
      s-marginal (tabulated inverse CDF of N(s;0,tau^2)(C-2Ns)^(N/2))
      and w uniform in the ball of radius sqrt(C - 2 N s); accept
      w inside its prior box. Much more efficient at low thresholds.

    ``method='unboxed'`` (default) is recommended.

    An optional hard cutoff ``trunc_sigmas`` truncates the Gaussian prior
    on s to |s| <= trunc_sigmas * tau. With trunc_sigmas=6 the posterior
    ridge at s = -N tau^2 stays inside the prior support (reachable), and
    the analytic lnZ becomes the truncated-Gaussian formula below.
    """

    def __init__(self, ndim, tau=0.5, D=10.0, gamma=0.0, trunc_sigmas=None):
        assert ndim > 0
        self.ndim = int(ndim)  # number of dp parameters; total dim = ndim + 1
        self.tau = float(tau)
        self.D = float(D)
        self.trunc_sigmas = float(trunc_sigmas) if trunc_sigmas is not None else None
        M = np.full((ndim, ndim), gamma, dtype=float)
        np.fill_diagonal(M, 1.0)
        self.M = M
        self.Minv = np.linalg.inv(M)
        self.L = np.linalg.cholesky(M)
        self.logdetM = float(np.linalg.slogdet(M)[1])
        self._s_cdf_cache = None

    @property
    def s_lower(self):
        """Lower prior support of s (truncated Gaussian, or -inf)."""
        return -self.trunc_sigmas * self.tau if self.trunc_sigmas else -np.inf

    @property
    def s_upper(self):
        """Upper prior support of s (truncated Gaussian, or +inf)."""
        return self.trunc_sigmas * self.tau if self.trunc_sigmas else np.inf

    @property
    def paramnames(self):
        """Return the parameter names."""
        return ['sigma'] + ['dp%d' % i for i in range(self.ndim)]

    @property
    def Vfull(self):
        """Full prior volume (s-prior normalized, w-box (2D)^N)."""
        return (2 * self.D) ** self.ndim

    def prior_volume(self, Lmin):
        """Exact prior volume of {loglike >= Lmin}: the s-grid trapezoid.

        of prior(s) * (w-ball at level s) intersected with the w-box,
        with the exact ball-box crossover fraction (BoxBallVolume).
        Vectorized over Lmin.
        """
        if not hasattr(self, '_vol_kernel'):
            self._vol_kernel = _FunnelVolumeKernel(self)
        return self._vol_kernel(Lmin)

    def make_volume_function(self, vmin_frac=1e-4, ngrid=1500,
                             ndraw=None, seed=0):
        """Return a tabulated interpolation of the exact prior volume over the range.

        [lowest supported likelihood - 10, the vmin_frac level + 5]:
        microsecond log-interpolated evaluation, exact Vfull and
        Lmin_of_volume for defining equal-volume windows.
        """
        K = 0.5 * (self.ndim * np.log(2 * np.pi) + self.logdetM)
        # lowest likelihood on the support: s = s_upper, |w|^2 maximal
        L_support_lo = (-0.5 * self.ndim * self.D ** 2
                        - self.ndim * self.s_upper - K)
        s_mode = np.clip(-self.ndim * self.tau ** 2,
                         self.s_lower, self.s_upper)
        L_top = -self.ndim * s_mode - K  # posterior-mode likelihood
        target = vmin_frac * self.Vfull
        L_hi = optimize.brentq(lambda L: self.prior_volume(L) - target,
                               L_top + 5.0, L_support_lo - 10.0, xtol=1e-2)
        Ls = np.linspace(L_support_lo - 10.0, L_hi + 5.0, ngrid)
        vals = np.array([self.prior_volume(L) for L in Ls])
        pos = vals > 0
        return TabulatedVolume(Ls[pos], np.log(vals[pos]))

    def __str__(self):
        """Return the string representation."""
        return 'UncorrelatedFunnelLRPS(ndim=%d, tau=%g, D=%g)' % (
            self.ndim, self.tau, self.D)

    def transform(self, u):
        """Transform the unit cube to the parameters."""
        u = np.asarray(u, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = self.tau * special.ndtri(u[:, 0])
            w = u[:, 1:] * 2 * self.D - self.D
            d = w @ self.L.T
            return np.concatenate([np.exp(s)[:, None], d], axis=1)

    def to_natural(self, u):
        """Return the natural parameter space coordinates."""
        u = np.asarray(u, dtype=float)
        return self.tau * special.ndtri(u[0]), 2 * self.D * u[1:] - self.D

    def to_u(self, state):
        """Convert the natural parameters back to the unit cube."""
        s, w = state
        u0 = special.ndtr(np.asarray(s, dtype=float) / self.tau)
        ui = (np.asarray(w, dtype=float) + self.D) / (2 * self.D)
        return np.concatenate([np.atleast_1d(u0), np.atleast_1d(ui)])

    def to_u_batch(self, x):
        """Batched (s, w) -> u-space conversion."""
        x = np.asarray(x, dtype=float)
        u0 = special.ndtr(x[:, 0] / self.tau)
        ui = (x[:, 1:] + self.D) / (2 * self.D)
        return np.concatenate([u0[:, None], ui], axis=1)

    def loglike(self, theta):
        """Return the log-likelihood."""
        theta = np.asarray(theta, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.log(theta[:, 0])
            d = theta[:, 1:]
            r = np.einsum('ij,jk,ik->i', d, self.Minv, d)
            like = -0.5 * (r + 2 * self.ndim * s
                           + self.ndim * np.log(2 * np.pi) + self.logdetM)
        return np.where(np.isfinite(like), like, -1e300)

    def _contour_value_C(self, Lmin):
        return -2.0 * (Lmin + 0.5 * (self.ndim * np.log(2 * np.pi) + self.logdetM))

    def _s_mode(self, C):
        """Mode of log p(s) = -s^2/(2 tau^2) + N/2 log(C - 2 N s)."""
        N = self.ndim
        return (C - np.sqrt(C * C + 8.0 * N ** 3 * self.tau ** 2)) / (4.0 * N)

    def make_unboxed_s_inverse_cdf(self, C, ngrid=16384, tail_sigmas=14.0):
        """Tabulated inverse CDF of p(s) ~ N(s;0,tau^2) (C-2Ns)^(N/2).

        on s < C/(2N), respecting the optional prior truncation.
        Returns (grid, cdf).
        """
        N = self.ndim
        s_hi = min(C / (2.0 * N), self.s_upper)
        s_lo = max(self.s_lower,
                   min(self._s_mode(C) - tail_sigmas * self.tau, -tail_sigmas * self.tau))
        s_hi = np.nextafter(s_hi, -np.inf)
        grid = np.linspace(s_lo, s_hi, ngrid)
        u = C - 2.0 * N * grid
        with np.errstate(divide='ignore'):
            logpdf = -0.5 * (grid / self.tau) ** 2 + 0.5 * N * np.log(u)
        logpdf -= logpdf.max()
        pdf = np.exp(logpdf)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * (grid[1] - grid[0]))])
        cdf /= cdf[-1]
        cdf[-1] = 1.0
        return grid, cdf

    def proposal(self, Lmin, rng, nsamples, method='unboxed'):
        """Draw candidates from an enclosing region in (s, w) coordinates.

        Returns (candidates, accepted) where candidates are (s, w) pairs
        and accepted indicates membership of the constrained region.
        """
        C = self._contour_value_C(Lmin)
        if method == 'prior':
            s = rng.normal(size=nsamples) * self.tau
            w = rng.uniform(-self.D, self.D, size=(nsamples, self.ndim))
            trunc_ok = np.abs(s) <= self.trunc_sigmas * self.tau \
                if self.trunc_sigmas else np.ones(nsamples, dtype=bool)
            contour_ok = (w ** 2).sum(axis=1) + 2.0 * self.ndim * s <= C
            candidates = np.concatenate([s[:, None], w], axis=1)
            return candidates, contour_ok & trunc_ok
        if method == 'unboxed':
            if self._s_cdf_cache is None or abs(self._s_cdf_cache[2] - C) > 1e-10:
                self._s_cdf_cache = self.make_unboxed_s_inverse_cdf(C) + (C,)
            s_grid, s_cdf, _ = self._s_cdf_cache
            u = rng.uniform(size=nsamples)
            s = np.interp(u, s_cdf, s_grid)
            radius = np.sqrt(np.maximum(C - 2.0 * self.ndim * s, 0.0))
            w = sample_uniform_ball(rng, nsamples, self.ndim, radius)
            box_ok = np.all(np.abs(w) <= self.D, axis=1)
            candidates = np.concatenate([s[:, None], w], axis=1)
            return candidates, box_ok
        raise ValueError("Unknown method=%s" % method)

    def sample(self, rng, nsamples, Lmin, method='unboxed'):
        """Draw exact constrained-prior samples (rejection, batched)."""
        out = []
        while sum(len(o) for o in out) < nsamples:
            candidates, accepted = self.proposal(Lmin, rng, nsamples, method)
            out.append(candidates[accepted, :])
        return np.concatenate(out, axis=0)[:nsamples]

    def analytic_lnZ(self):
        """Return the analytic log-evidence."""
        nd = self.ndim
        q = 2 * special.ndtr(self.D) - 1  # P(|z| <= D)
        # s-integral over the prior support [s_lower, s_upper] of
        # N(s;0,tau^2) exp(-N s): completing the square around -N tau^2,
        # the tau sqrt(2 pi) of the Gaussian normalization cancels:
        # = exp(N^2 tau^2 / 2) [Phi(u) - Phi(l)]
        # with u = N tau + s_upper/tau, l = N tau + s_lower/tau
        usr = nd * self.tau + self.s_upper / self.tau
        lsr = nd * self.tau + self.s_lower / self.tau
        log_phidiff = special.log_ndtr(usr) + np.log1p(
            -np.exp(special.log_ndtr(lsr) - special.log_ndtr(usr)))
        # w-integral: [sqrt(2 pi) (2 Phi(D)-1)]^N cancels the likelihood's
        # (2 pi)^(-N/2):
        return 0.5 * nd ** 2 * self.tau ** 2 + log_phidiff \
            - 0.5 * self.logdetM + nd * np.log(q / (2 * self.D))

    def analytic_H(self):
        """Return the analytic information gain."""
        nd = self.ndim
        q = 2 * special.ndtr(self.D) - 1
        m2 = 1.0 - 2 * self.D * _normpdf(self.D) / q  # E[z^2 | |z| <= D]
        lognorm = 0.5 * (nd * np.log(2 * np.pi) + self.logdetM)
        # posterior factorises: s ~ Normal(-N tau^2, tau^2) truncated to the
        # prior support (mean from the truncated-normal formula),
        # d_i ~ Normal(0,1) truncated; E_post[s]:
        mu_t = -nd * self.tau ** 2
        if self.trunc_sigmas:
            alpha = (self.s_lower - mu_t) / self.tau
            beta = (self.s_upper - mu_t) / self.tau
            e_s = mu_t + self.tau * (_normpdf(alpha) - _normpdf(beta)) \
                / (special.ndtr(beta) - special.ndtr(alpha))
        else:
            e_s = mu_t
        e_logl = -0.5 * nd * m2 - nd * e_s - lognorm
        return e_logl - self.analytic_lnZ()


def funnel_importance_sampler(problem, nsamples, rng=None, seed=None):
    """Exact posterior-shape importance sampler for the (truncated) funnel.

    The funnel posterior factorises exactly:

        s ~ exp(-N s - s^2/(2 tau^2)) on the prior support   (a tilted
        truncated Gaussian),   w_i ~ N(0,1) truncated to |w_i| <= D,

    independent of each other. This sampler draws both exactly, converts
    to u-space (rejecting proposals outside the unit hypercube), and
    returns the posterior samples together with the importance weights.
    Because the proposal equals the posterior shape, all weights are
    constant, so lnZ has zero Monte-Carlo error and equals the analytic
    truncated-Gaussian formula, and the samples are exact posterior draws
    regardless of dimension.

    Returns (u_samples, lnZ, n_evaluated)
    """
    N, tau, D = problem.ndim, problem.tau, problem.D
    a = problem.trunc_sigmas
    assert a is not None, "funnel_importance_sampler requires trunc_sigmas"
    rng = rng if rng is not None else np.random.RandomState(seed)

    # s ~ Normal(-N tau^2, tau^2) truncated to [s_lower, s_upper]: the exact
    # posterior shape of s, sampled by inverse CDF (the tilt center can lie
    # outside the truncation window, so rejection would be infeasible)
    mu_t = -N * tau ** 2
    alpha = (problem.s_lower - mu_t) / tau
    beta = (problem.s_upper - mu_t) / tau
    u_s = rng.uniform(special.ndtr(alpha), special.ndtr(beta), size=nsamples)
    s = mu_t + tau * special.ndtri(u_s)

    # w_i ~ Normal(0,1) truncated to |w_i| <= D, sampled by inverse CDF
    u_w = rng.uniform(special.ndtr(-D), special.ndtr(D), size=(nsamples, N))
    w = special.ndtri(u_w)

    # u-space points; reject outside the unit hypercube
    u = problem.to_u_batch(np.concatenate([s[:, None], w], axis=1))
    ok = np.all((u > 0) & (u < 1), axis=1)
    u = u[ok, :]

    # constant importance weight: prior * lnL / q
    # = exp(N^2 tau^2/2 - K) (2 pi)^(N/2) (2D)^(-N) (2 Phi(D)-1)^N
    #   * [Phi(N tau + a) - Phi(N tau - a)] / [2 Phi(a) - 1]
    # (the tilted-Gaussian truncation normalization Phi(N tau +- a) accounts
    #  for p~ being the normalized posterior-shape proposal of s)
    q = 2 * special.ndtr(D) - 1
    usr = N * tau + a
    lsr = N * tau - a
    log_phidiff = special.log_ndtr(usr) + np.log1p(
        -np.exp(special.log_ndtr(lsr) - special.log_ndtr(usr)))
    lnZ = 0.5 * N ** 2 * tau ** 2 - 0.5 * (N * np.log(2 * np.pi) + problem.logdetM) \
        + 0.5 * N * np.log(2 * np.pi) + N * np.log(q / (2 * D)) \
        + log_phidiff - np.log(2 * special.ndtr(a) - 1)
    return u, lnZ, len(u)


# ============================================================================
# 3. Gaussian bump with box bonus ("box")
# ============================================================================

class BoxGaussianLRPS:
    """Flat prior on [0,1]^N and likelihood.

        logL(x) = -0.5 sum_i (x_i/sigma)^2 + logbonus * [max_i x_i < sigma]

    a Gaussian of width sigma centered on the cube corner with a large
    constant bonus inside the box [0, sigma]^N. The constrained region
    changes character with the threshold (ball / box / cube / union of
    pieces); the evidence and information gain are analytic.

    Enclosing proposals (method='auto' picks per threshold):

    * ``orthant-ball``: in the ball phase (Lmin >= logbonus - 0.5) and the
      deep tail (Lmin <= -ndim/2) the region is the positive-orthant part
      of a ball, sampled exactly through Dirichlet-distributed squared
      direction components (a bbox would waste a factor 2^-ndim V_ndim).
    * ``bbox``: in the box/mixed/shallow-tail phases, uniform in the
      coordinate extent [0, U]^N; in the dominant box phase the envelope
      coincides with the region (acceptance rate 1).
    """

    def __init__(self, ndim, sigma=0.1, logbonus=100.0):
        self.ndim = int(ndim)
        self.sigma = float(sigma)
        self.logbonus = float(logbonus)

    @property
    def paramnames(self):
        """Return the parameter names."""
        return ['param%d' % (i + 1) for i in range(self.ndim)]

    def __str__(self):
        """Return the string representation."""
        return 'BoxGaussianLRPS(ndim=%d)' % self.ndim

    def transform(self, x):
        """Transform the unit cube to the parameters."""
        return np.array(x)

    def to_natural(self, u):
        """Return the natural parameter space coordinates."""
        return np.asarray(u, dtype=float)

    def to_u(self, x):
        """Convert the natural parameters back to the unit cube."""
        return np.asarray(x, dtype=float)

    def to_u_batch(self, x):
        """Convert a batch of natural parameters to the unit cube."""
        return np.asarray(x, dtype=float)

    def loglike(self, theta):
        """Return the log-likelihood."""
        theta = np.asarray(theta, dtype=float)
        delta = theta.max(axis=1)
        return -0.5 * ((theta / self.sigma) ** 2).sum(axis=1) \
            + self.logbonus * (delta < self.sigma)

    def _bbox_upper(self, Lmin):
        """Largest coordinate value reached by the constrained region."""
        if Lmin >= self.logbonus - 0.5:
            # ball phase: region is the positive-orthant part of a ball
            return self.sigma * np.sqrt(max(0.0, 2.0 * (self.logbonus - Lmin)))
        # box/mixed/tail phases: box extent and outside-ball extent
        return max(self.sigma,
                   min(1.0, self.sigma * np.sqrt(max(0.0, -2.0 * Lmin))))

    def _proposal_orthant_ball(self, Lmin, rng, nsamples):
        """Uniform draws from the positive-orthant part of the whitened.

        ball of radius rho, intersected with the cube.

        In the ball phase (Lmin >= logbonus - 0.5) the constrained region
        is exactly the orthant-ball of radius sqrt(2 (logbonus - Lmin));
        in the deep tail (Lmin <= -ndim/2) it is the orthant-ball of
        radius sqrt(-2 Lmin) intersected with the cube. Squared direction
        components of a uniform positive-orthant-cap direction follow
        Dirichlet(1/2, ..., 1/2), so the orthant-ball is directly
        sampleable without rejection (acceptance 1 when the ball lies
        inside the cube, which a single bbox cannot achieve: it wastes a
        factor 2^-ndim V_ndim).
        """
        if Lmin >= self.logbonus - 0.5:
            rho = np.sqrt(max(0.0, 2.0 * (self.logbonus - Lmin)))
        else:
            rho = np.sqrt(max(0.0, -2.0 * Lmin))
        d = rng.dirichlet(np.full(self.ndim, 0.5), size=nsamples)
        radial = rho * rng.uniform(size=nsamples) ** (1.0 / self.ndim)
        candidates = self.sigma * radial[:, None] * np.sqrt(d)
        inside_cube = np.all(candidates < 1.0, axis=1)
        return candidates, inside_cube

    def proposal(self, Lmin, rng, nsamples, method='auto'):
        """Draw candidates from an enclosing region; returns.

        (candidates, accepted).

        method='auto' picks the orthant-ball proposal in the ball phase
        and the deep tail (where a bbox wastes a factor 2^-ndim V_ndim),
        and the bbox envelope in the box/mixed/shallow-tail phases.
        """
        if method == 'bbox' or (method == 'auto'
                                and -0.5 * self.ndim < Lmin < self.logbonus - 0.5):
            upper = self._bbox_upper(Lmin)
            candidates = rng.uniform(0.0, upper, size=(nsamples, self.ndim))
            return candidates, self.loglike(candidates) >= Lmin
        if method in ('auto', 'orthant-ball'):
            return self._proposal_orthant_ball(Lmin, rng, nsamples)
        raise ValueError("Unknown method=%s" % method)

    def sample(self, rng, nsamples, Lmin, method='auto'):
        """Draw exact constrained-prior samples (rejection, batched)."""
        out = []
        while sum(len(o) for o in out) < nsamples:
            candidates, accepted = self.proposal(Lmin, rng, nsamples, method)
            out.append(candidates[accepted, :])
        return np.concatenate(out, axis=0)[:nsamples]

    def analytic_lnZ(self):
        """Return the analytic log-evidence."""
        N = self.ndim
        A = self.sigma * np.sqrt(2 * np.pi)
        q0 = special.ndtr(1.0) - 0.5               # integral over [0, sigma] / A
        q1 = special.ndtr(1.0 / self.sigma) - 0.5  # integral over [0, 1] / A
        # Z = A^N [q0^N e^logbonus + q1^N - q0^N]
        return N * np.log(A) + np.log(
            q0 ** N * np.exp(self.logbonus) + q1 ** N - q0 ** N)

    def analytic_H(self):
        """Return the analytic information gain."""
        N = self.ndim
        A = self.sigma * np.sqrt(2 * np.pi)
        q0 = special.ndtr(1.0) - 0.5
        q1 = special.ndtr(1.0 / self.sigma) - 0.5
        lnZ = self.analytic_lnZ()
        m2_in = 1.0 - _normpdf(1.0) / q0
        hi = 1.0 / self.sigma
        m2_out = 1.0 - hi * _normpdf(hi) / q1
        # integral of L * logL over the box and over cube\box
        I_box = np.exp(self.logbonus) * (A * q0) ** N * (
            self.logbonus - 0.5 * N * m2_in)
        I_out = -0.5 * N * ((A * q1) ** N * m2_out - (A * q0) ** N * m2_in)
        Z = A ** N * (q0 ** N * np.exp(self.logbonus) + q1 ** N - q0 ** N)
        return (I_box + I_out) / Z - lnZ

    @property
    def Vfull(self):
        """Full prior volume (flat prior on the unit cube)."""
        return 1.0

    def prior_volume(self, Lmin):
        """Exact prior volume fraction of {loglike >= Lmin} in the deep.

        tail (Lmin <= -ndim/2, where nested sampling runs live): the
        constrained region is exactly the positive-orthant part of the
        ball of whitened radius sqrt(-2 Lmin) (the bonus box lies inside
        it), so the cube fraction inside is the exact Beta-sum crossover
        fraction (BoxBallVolume). Not valid in the box/mixed phases.
        """
        Lmin = np.asarray(Lmin, dtype=float)
        if np.any(Lmin > -0.5 * self.ndim):
            raise ValueError("prior_volume implements only the deep tail")
        r2 = -2.0 * Lmin * self.sigma ** 2   # physical radius^2
        return box_ball_volume(self.ndim, 1.0).frac(r2)

    def make_volume_function(self, vmin_frac=None, ngrid=None,
                             ndraw=None, seed=0):
        """Return the exact prior volume over the deep-tail range."""
        d = self.ndim
        t_hi = d / 3.0 + 8.0 * 0.353 * np.sqrt(d)   # X ~ 1 there
        L_lo = -t_hi / (2.0 * self.sigma ** 2)
        return ExactVolume(self, L_lo=L_lo, L_hi=-0.5 * d)


# ============================================================================
# 4. Constant-Jacobian banana ("banana")
# ============================================================================

class BananaGaussianLRPS:
    """Triangular constant-Jacobian banana likelihood.

        z_0 = (x_0 - mu_0) / sigma_0
        z_i = (x_i - centreline_i) / sigma_i,
        centreline_i = mu_i + bend_{i-1} ((x_{i-1}-mu_{i-1})^2 - offset_{i-1})

    logL(x) = -0.5 ||z||^2.

    The map x -> z is triangular with constant Jacobian determinant
    prod_i 1/sigma_i, so a uniform draw in the ball ||z|| <= r maps to a
    uniform draw in the banana body {logL >= -r^2/2}. Reference values
    lnZ and H are computed by numerical integration over the cube.

    Enclosing proposals:

    * ``banana``: uniform in the banana body; accept inside the cube.
    * ``bbox``: uniform in a conservative coordinate envelope of the
      banana body clipped to the cube; accept with the likelihood.

    ``method='auto'`` picks whichever envelope is smaller.
    """

    def __init__(self, sigma, bend, mu=0.5, offset=None):
        sigma = np.asarray(sigma, dtype=float)
        self.ndim = int(sigma.shape[0])
        self.sigma = sigma
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), sigma.shape)
        bend = np.asarray(bend, dtype=float)
        self.bend = np.full((self.ndim - 1,), bend) if bend.ndim == 0 else bend
        if offset is None:
            offset = sigma[:-1] ** 2
        offset = np.asarray(offset, dtype=float)
        self.offset = np.full((self.ndim - 1,), offset) if offset.ndim == 0 \
            else offset
        self._reference_cache = None

    @classmethod
    def standard_problem(cls, ndim=2, sigma=0.1, bend=1.0, mu=0.5):
        """Construct the standard problem configuration."""
        assert ndim >= 2
        return cls(sigma=np.full(ndim, sigma), bend=bend, mu=mu)

    @property
    def paramnames(self):
        """Return the parameter names."""
        return ['param%d' % (i + 1) for i in range(self.ndim)]

    def __str__(self):
        """Return the string representation."""
        return 'BananaGaussianLRPS(ndim=%d)' % self.ndim

    def transform(self, x):
        """Transform the unit cube to the parameters."""
        return np.array(x)

    def to_natural(self, u):
        """Return the natural parameter space coordinates."""
        return np.asarray(u, dtype=float)

    def to_u(self, x):
        """Convert the natural parameters back to the unit cube."""
        return np.asarray(x, dtype=float)

    def to_u_batch(self, x):
        """Convert a batch of natural parameters to the unit cube."""
        return np.asarray(x, dtype=float)

    def to_z(self, x):
        """Map to the whitened z-space."""
        x = np.asarray(x, dtype=float)
        z0 = (x[..., 0] - self.mu[0]) / self.sigma[0]
        zs = [z0]
        for i in range(1, self.ndim):
            previous = x[..., i - 1] - self.mu[i - 1]
            centreline = self.mu[i] + self.bend[i - 1] \
                * (previous ** 2 - self.offset[i - 1])
            zs.append((x[..., i] - centreline) / self.sigma[i])
        return np.stack(zs, axis=-1)

    def from_z(self, z):
        """Map from the whitened z-space back to the parameters."""
        z = np.asarray(z, dtype=float)
        xs = [self.mu[0] + self.sigma[0] * z[..., 0]]
        for i in range(1, self.ndim):
            previous = xs[-1] - self.mu[i - 1]
            xi = self.mu[i] + self.bend[i - 1] \
                * (previous ** 2 - self.offset[i - 1]) \
                + self.sigma[i] * z[..., i]
            xs.append(xi)
        return np.stack(xs, axis=-1)

    def loglike(self, theta):
        """Return the log-likelihood."""
        z = self.to_z(theta)
        return -0.5 * (z ** 2).sum(axis=-1)

    def radius(self, Lmin):
        """Return the likelihood radius at the given threshold."""
        return np.sqrt(max(0.0, -2.0 * np.asarray(Lmin, dtype=float)))

    def conservative_bbox(self, Lmin):
        """Conservative recursive coordinate envelope of the banana body.

        (not the exact bounding box, but it safely encloses the body).
        """
        r = self.radius(Lmin)
        lower = [self.mu[0] - r * self.sigma[0]]
        upper = [self.mu[0] + r * self.sigma[0]]
        for i in range(1, self.ndim):
            previous_lo = lower[-1] - self.mu[i - 1]
            previous_hi = upper[-1] - self.mu[i - 1]
            sq_min = np.where((previous_lo <= 0) & (previous_hi >= 0), 0.0,
                              np.minimum(previous_lo ** 2, previous_hi ** 2))
            sq_max = np.maximum(previous_lo ** 2, previous_hi ** 2)
            b, off = self.bend[i - 1], self.offset[i - 1]
            curve_lo = np.where(b >= 0, b * (sq_min - off), b * (sq_max - off))
            curve_hi = np.where(b >= 0, b * (sq_max - off), b * (sq_min - off))
            lower.append(self.mu[i] + curve_lo - r * self.sigma[i])
            upper.append(self.mu[i] + curve_hi + r * self.sigma[i])
        lower = np.maximum(0.0, np.stack(lower))
        upper = np.minimum(1.0, np.stack(upper))
        return lower, upper

    def proposal_volumes(self, Lmin):
        """Volumes of the enclosing envelopes; the smaller gives the.

        higher acceptance rate.
        """
        r = self.radius(Lmin)
        with np.errstate(divide='ignore'):
            body_volume = np.exp(
                0.5 * self.ndim * np.log(np.pi)
                - special.gammaln(0.5 * self.ndim + 1)
                + self.ndim * np.log(r) + np.log(self.sigma).sum())
        lower, upper = self.conservative_bbox(Lmin)
        return {'banana': body_volume, 'bbox': np.prod(upper - lower)}

    def proposal(self, Lmin, rng, nsamples, method='auto'):
        """Draw candidates from an enclosing region; returns.

        (candidates, accepted) in u-space.
        """
        if method == 'auto':
            volumes = self.proposal_volumes(Lmin)
            method = min(volumes, key=lambda name: volumes[name])
        if method == 'banana':
            r = self.radius(Lmin)
            z = sample_uniform_ball(rng, nsamples, self.ndim, r)
            candidates = self.from_z(z)
            inside_cube = np.all((candidates > 0) & (candidates < 1), axis=1)
            return candidates, inside_cube
        if method == 'bbox':
            lower, upper = self.conservative_bbox(Lmin)
            candidates = rng.uniform(lower, upper, size=(nsamples, self.ndim))
            return candidates, self.loglike(candidates) >= Lmin
        raise ValueError("Unknown method=%s" % method)

    def sample(self, rng, nsamples, Lmin, method='auto'):
        """Draw exact constrained-prior samples (rejection, batched)."""
        out = []
        while sum(len(o) for o in out) < nsamples:
            candidates, accepted = self.proposal(Lmin, rng, nsamples, method)
            out.append(candidates[accepted, :])
        return np.concatenate(out, axis=0)[:nsamples]

    def analytic_lnZ(self, ngrid=1000):
        """Return the reference evidence, by numerical integration over the cube."""
        return self._reference_values(ngrid)[0]

    def analytic_H(self, ngrid=1000):
        """Return the reference information gain, by numerical integration."""
        return self._reference_values(ngrid)[1]

    @property
    def Vfull(self):
        """Full prior volume (flat prior on the unit cube)."""
        return 1.0

    def make_volume_function(self, vmin_frac=None, ngrid=None,
                             ndraw=100000000, seed=0):
        """Return a tabulated interpolation of X(Lmin) = P(logL >= Lmin) under the prior,.

        by fixed-seed Monte Carlo: the banana body clipped to the cube
        has no closed-form volume.
        """
        return mc_prior_volume_table(self, ndraw=ndraw, seed=seed)

    def _reference_values(self, ngrid=1000):
        if self._reference_cache is not None \
                and self._reference_cache[0] == ngrid:
            return self._reference_cache[1], self._reference_cache[2]
        axes = [np.linspace(0.0, 1.0, ngrid)] * self.ndim
        if self.ndim != 2:
            raise ValueError(
                "Numerical reference values are only implemented for ndim=2")
        X, Y = np.meshgrid(axes[0], axes[1], indexing='ij')
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        logl = self.loglike(pts)
        # posterior is exp(loglike), prior is uniform on the cube
        w = np.exp(logl - logl.max())
        lnZ = np.log(w.mean()) + logl.max()  # mean over the unit cube
        e_logl = (w * logl).sum() / w.sum()
        H = e_logl - lnZ
        self._reference_cache = (ngrid, lnZ, H)
        return lnZ, H


# ============================================================================
# 5. Hyperpyramid: flat likelihood with a razor-thin spike
# ============================================================================

class HyperPyramidLRPS:
    """Hyperpyramid likelihood.

        logL(x) = -(max_i |x_i - 1/2| / sigma_i)^(1/s)

    with exponent 1/s (default s=100) and per-dimension widths sigma_i,
    either all 1 (default) or 10^(-3 i / ndim).

    The likelihood is extremely flat: it varies by only ~1 nat over the
    whole cube, with a razor-thin spike at x = 1/2.

    The constrained region {logL >= Lmin} is the axis-aligned box

        |x_i - 1/2| <= sigma_i (-Lmin)^s

    clipped to the unit cube, so the direct analytic proposal draws
    uniform samples from the clipped box with acceptance rate 1.

    Reference lnZ and H follow from the layer-cake identity: with
    V(l) = vol{logL >= l} = prod_i width_i((-l)^s) the exact region
    volume at likelihood level l,

        Z = int V(l) e^l dl,
        E_post[logL] = int (l + 1) V(l) e^l dl / Z,

    both 1d integrals evaluated by quadrature.
    """

    def __init__(self, ndim, s=100.0, sigma='ones'):
        self.ndim = int(ndim)
        self.s = float(s)
        if isinstance(sigma, str):
            if sigma == 'ones':
                sigma = np.ones(self.ndim)
            elif sigma == 'anisotropic':
                sigma = 10.0 ** (-3.0 * np.arange(self.ndim) / self.ndim)
            else:
                raise ValueError("Unknown sigma=%s" % sigma)
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float),
                                     (self.ndim,)).astype(float)
        self._reference_cache = None

    @classmethod
    def standard_problem(cls, ndim, s=100.0, sigma='ones'):
        """Construct the standard problem configuration."""
        return cls(ndim=ndim, s=s, sigma=sigma)

    @property
    def paramnames(self):
        """Return the parameter names."""
        return ['param%d' % (i + 1) for i in range(self.ndim)]

    @property
    def Vfull(self):
        """Full prior volume (flat prior on the unit cube)."""
        return 1.0

    def prior_volume(self, Lmin):
        """Exact prior volume fraction of {loglike >= Lmin}: the clipped.

        axis-aligned box |x_i - 1/2| <= sigma_i (-Lmin)^s.
        Vectorized over Lmin.
        """
        Lmin = np.asarray(Lmin, dtype=float)
        with np.errstate(over='ignore'):
            rho = np.maximum(0.0, -Lmin) ** self.s
        if Lmin.ndim == 0:
            return float(self._region_volume(float(rho)))
        return np.array([self._region_volume(float(r)) for r in rho])

    def make_volume_function(self, vmin_frac=None, ngrid=None,
                             ndraw=None, seed=0):
        """Return the exact prior volume: per-call exact volume with bisective.

        inversion over [-1, 0].
        """
        return ExactVolume(self, L_lo=-1.0, L_hi=0.0)

    def __str__(self):
        """Return the string representation."""
        return 'HyperPyramidLRPS(ndim=%d, s=%g)' % (self.ndim, self.s)

    def transform(self, x):
        """Transform the unit cube to the parameters."""
        return np.array(x)

    def to_natural(self, u):
        """Return the natural parameter space coordinates."""
        return np.asarray(u, dtype=float)

    def to_u(self, x):
        """Convert the natural parameters back to the unit cube."""
        return np.asarray(x, dtype=float)

    def to_u_batch(self, x):
        """Convert a batch of natural parameters to the unit cube."""
        return np.asarray(x, dtype=float)

    def loglike(self, theta):
        """Return the log-likelihood."""
        theta = np.asarray(theta, dtype=float)
        m = np.max(np.abs(theta - 0.5) / self.sigma, axis=1)
        return -(m ** (1.0 / self.s))

    def _region_bounds(self, rho):
        """Clipped box bounds of {max_i |x_i-1/2|/sigma_i <= rho}."""
        lower = np.maximum(0.0, 0.5 - self.sigma * rho)
        upper = np.minimum(1.0, 0.5 + self.sigma * rho)
        return lower, upper

    def _region_volume(self, rho):
        lower, upper = self._region_bounds(rho)
        return np.prod(upper - lower)

    def proposal(self, Lmin, rng, nsamples, method='box'):
        """Direct analytic proposal: uniform draws from the constrained.

        region (the clipped box |x_i - 1/2| <= sigma_i (-Lmin)^s);
        acceptance rate 1.
        """
        if method != 'box':
            raise ValueError("Unknown method=%s" % method)
        with np.errstate(over='ignore'):
            rho = max(0.0, -Lmin) ** self.s
        lower, upper = self._region_bounds(rho)
        candidates = rng.uniform(lower, upper, size=(nsamples, self.ndim))
        return candidates, np.ones(nsamples, dtype=bool)

    def sample(self, rng, nsamples, Lmin, method='box'):
        """Draw exact constrained-prior samples (acceptance rate 1)."""
        candidates, _ = self.proposal(Lmin, rng, nsamples, method)
        return candidates

    def analytic_lnZ(self, points=None):
        """Return the reference evidence from the layer-cake identity."""
        def integrand(logrho):
            """Return the integrand."""
            with np.errstate(over='ignore'):
                rho = np.float64(-logrho) ** self.s
            return self._region_volume(rho) * np.exp(logrho)
        Z, _ = integrate.quad(integrand, -np.inf, 0.0, limit=200,
                              points=points)
        return np.log(Z)

    def analytic_H(self, points=None):
        """Return the reference information gain from the layer-cake identity."""
        def integrand(logrho):
            """Return the integrand."""
            with np.errstate(over='ignore'):
                rho = np.float64(-logrho) ** self.s
            return (logrho + 1.0) * self._region_volume(rho) * np.exp(logrho)
        I, _ = integrate.quad(integrand, -np.inf, 0.0, limit=200,
                              points=points)
        Z = np.exp(self.analytic_lnZ(points=points))
        E_post_logL = I / Z
        return E_post_logL - self.analytic_lnZ(points=points)


# ---------------------------------------------------------------------------
# contour-parameterized problems: the constrained region is exactly
# {t(x) <= t_min}, and the proposal draws uniformly from the region
# itself (acceptance 1 up to cube clipping)
# ---------------------------------------------------------------------------

def logspace_sigma(ndim, lo=0.1, hi=1e-4):
    """Per-axis widths logarithmically spaced between lo and hi, in.

    unit-hypercube coordinates.
    """
    return np.logspace(np.log10(lo), np.log10(hi), int(ndim))


class ContourProblem:
    """Base class for contour-parameterized test problems.

    The constrained region {loglike >= Lmin} equals {t(x) <= t_min} with
    t_min = t_of_L(Lmin), and loglike(x) = L_of_t(t(x)).
    """

    def loglike(self, theta):
        """Return the log-likelihood."""
        return self.L_of_t(self.t(theta))

    def t_of_L(self, Lmin):
        """Return the volume-like coordinate t at the likelihood threshold."""
        raise NotImplementedError

    def L_of_t(self, t):
        """Return the likelihood threshold at the volume-like coordinate t."""
        raise NotImplementedError

    def t(self, x):
        """Return the volume-like coordinate t of the given point."""
        raise NotImplementedError

    def proposal(self, Lmin, rng, nsamples):
        """Exact uniform draws from {t(x) <= t_min}; returns.

        (candidates, inside) with candidates in u-space.
        """
        raise NotImplementedError

    def transform(self, x):
        """Transform the unit cube to the parameters."""
        return np.array(x)

    def to_natural(self, u):
        """Return the natural parameter space coordinates."""
        return np.asarray(u, dtype=float)

    def to_u(self, x):
        """Convert the natural parameters back to the unit cube."""
        return np.asarray(x, dtype=float)

    def to_u_batch(self, x):
        """Convert a batch of natural parameters to the unit cube."""
        return np.asarray(x, dtype=float)

    @property
    def paramnames(self):
        """Return the parameter names."""
        return ['param%d' % (i + 1) for i in range(self.ndim)]

    @property
    def Vfull(self):
        """Full prior volume."""
        return 1.0

    def sample_prior(self, rng, nsamples):
        """Uniform draws from the prior (default: the unit cube)."""
        return rng.uniform(size=(nsamples, self.ndim))

    def analytic_H(self):
        """Return the analytic information gain."""
        raise NotImplementedError("no analytic information gain")

    def make_volume_function(self, vmin_frac=None, ngrid=None,
                             ndraw=100000000, seed=0):
        """Return a tabulated interpolation of the prior volume fraction X(Lmin), by.

        fixed-seed Monte Carlo over the prior (exact in expectation).
        """
        return mc_prior_volume_table(self, ndraw=ndraw, seed=seed)


# ============================================================================
# 1. Gauss: ellipsoidal contours shrinking toward mu
# ============================================================================

class GaussContour(ContourProblem):
    """Flat prior on [0,1]^N with ellipsoid contours shrinking toward mu.

        t(x) = sum_i ((x_i - mu_i)/sigma_i)^2

        t(x) = sum_i ((x_i - mu_i)/sigma_i)^2
        L(t) = -t/2 - sum_i log(sqrt(2 pi) sigma_i)

    The proposal draws uniformly from the ellipsoid {t <= t_min} and
    keeps the part inside the cube (acceptance ~1 for sigma small
    enough that the deep contours lie inside the cube). The default
    widths are logarithmically spaced between 0.1 and 1e-4 in
    unit-hypercube coordinates, giving an information gain of
    H = ndim * (5/2 ln 10 - 1/2 - 1/2 ln(2 pi)) ~ 4.34 ndim nats.
    """

    def __init__(self, ndim, mu=0.5, sigma='logspace'):
        self.ndim = int(ndim)
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float),
                                  (self.ndim,)).astype(float)
        if isinstance(sigma, str) and sigma == 'logspace':
            sigma = logspace_sigma(self.ndim)
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float),
                                     (self.ndim,)).astype(float)

    def t(self, x):
        """Return the volume-like coordinate t of the given point."""
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sigma
        return (z ** 2).sum(axis=-1)

    def _lognorm(self):
        return 0.5 * np.log(2 * np.pi * self.sigma ** 2).sum()

    def L_of_t(self, t):
        """Return the likelihood threshold at the volume-like coordinate t."""
        return -0.5 * np.asarray(t, dtype=float) - self._lognorm()

    def t_of_L(self, Lmin):
        """Return the volume-like coordinate t at the likelihood threshold."""
        return -2.0 * (Lmin + self._lognorm())

    def proposal_volumes(self, Lmin):
        """Volumes of the enclosing envelopes; the smaller gives the.

        higher acceptance rate.
        """
        r = np.sqrt(self.t_of_L(Lmin))
        lower = np.maximum(0.0, self.mu - r * self.sigma)
        upper = np.minimum(1.0, self.mu + r * self.sigma)
        ellipsoid_volume = np.exp(
            0.5 * self.ndim * np.log(np.pi)
            - special.gammaln(0.5 * self.ndim + 1)
            + self.ndim * np.log(r) + np.log(self.sigma).sum())
        return {'bbox': np.prod(upper - lower),
                'ellipsoid': ellipsoid_volume}

    def proposal(self, Lmin, rng, nsamples, method='auto'):
        """Draws from the smaller of two enclosing envelopes: the.

        coordinate bounding box of the ellipsoid clipped to the cube
        (efficient at shallow levels, where the ellipsoid covers the
        cube and its in-cube fraction is ~1e-6), or the ellipsoid
        itself (efficient at deep levels); accepted by likelihood.
        """
        t_min = self.t_of_L(Lmin)
        if method == 'auto':
            volumes = self.proposal_volumes(Lmin)
            method = min(volumes, key=lambda name: volumes[name])
        if method == 'bbox':
            r = np.sqrt(t_min)
            lower = np.maximum(0.0, self.mu - r * self.sigma)
            upper = np.minimum(1.0, self.mu + r * self.sigma)
            x = rng.uniform(lower, upper, size=(nsamples, self.ndim))
            return x, self.loglike(x) >= Lmin
        z = sample_uniform_ball(rng, nsamples, self.ndim, np.sqrt(t_min))
        x = self.mu[None, :] + z * self.sigma[None, :]
        ok = np.all((x > 0) & (x < 1), axis=1)
        return x, ok

    def analytic_H(self):
        """Return the analytic information gain."""
        lo = (0.0 - self.mu) / self.sigma
        hi = (1.0 - self.mu) / self.sigma
        m2 = _truncated_second_moment(lo, hi).sum()
        lnZ = np.log(special.ndtr(hi) - special.ndtr(lo)).sum()
        return -0.5 * m2 - self._lognorm() - lnZ


# ============================================================================
# 2. HyperPyramid: box contours shrinking toward 0.5
# ============================================================================

class HyperPyramidContour(ContourProblem):
    """Flat prior on [0,1]^N with axis-aligned box contours.

        t(x) = max_i |x_i - 0.5| / sigma_i
        L(t) = -t / t0            (schedule='linear', default)
             = -t^(1/s)           (schedule='power')

    The proposal draws uniformly from the box clipped to the cube with
    acceptance rate 1; the volume is analytic.

    The widths are logarithmically spaced between 0.1 and 1e-4: the max
    over the normalized distances is dominated by the narrowest axis.
    With the original power schedule the likelihood varies by only ~1
    nat over the cube (information gain ~0.0005 nats, nearly
    independent of dimension) -- much shallower than a Gaussian. The
    linear schedule instead gives L in [-t_max/t0, 0] over the cube and
    the prior volume V(l) = prod min(1, 2 sigma_i t0 (-l)) declines as
    |l|^-N in the deep tail; t0 is solved numerically (brentq) so that
    the layer-cake information gain matches that of a Gaussian with
    the same log-spaced widths, H = N(5/2 ln 10 - 1/2 - 1/2 ln 2pi)
    ~ 4.34 N nats, making the integration depth comparable.
    """

    def __init__(self, ndim, s=100.0, sigma='logspace', schedule='linear',
                 H_target=None):
        self.ndim = int(ndim)
        self.s = float(s)
        self.schedule = schedule
        if isinstance(sigma, str):
            if sigma == 'ones':
                sigma = np.ones(self.ndim)
            elif sigma == 'logspace':
                sigma = logspace_sigma(self.ndim)
            elif sigma == 'anisotropic':
                sigma = 10.0 ** (-3.0 * np.arange(self.ndim) / self.ndim)
            else:
                raise ValueError("Unknown sigma=%s" % sigma)
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float),
                                     (self.ndim,)).astype(float)
        # t_max: largest t over the cube (narrowest axis boundary)
        self.t_max = 0.5 / self.sigma.min()
        if schedule == 'linear':
            if H_target is None:
                H_target = self.ndim * (2.5 * np.log(10) - 0.5
                                        - 0.5 * np.log(2 * np.pi))
            # H(t0) is monotone decreasing in t0 (larger t0 -> shallower
            # likelihood); bracket between a steep (large H) and a flat
            # (H ~ 0 at t0 = t_max, where the whole cube is above the
            # schedule's L at the prior boundary) choice
            from scipy.optimize import brentq

            def f(logt0):
                """Return the f."""
                """Return the f."""
                return self._H_of_t0(np.exp(logt0)) - H_target
            lo, hi = np.log(self.t_max), np.log(self.t_max * 1e-6)
            self.t0 = np.exp(brentq(f, lo, hi, xtol=1e-6, rtol=1e-10))
        else:
            self.t0 = 1.0

    def _H_of_t0(self, t0):
        """Layer-cake information gain for the linear schedule L=-t/t0,.

        on a composite grid: linear near l=0, log-spaced in the deep
        |l| tail (the per-axis clipping transitions span
        1/(2 sigma_max t0) .. 1/(2 sigma_min t0)).
        """
        l_edge = 1.0 / (2 * self.sigma.max() * t0)
        l_lo = 1.0 / (2 * self.sigma.min() * t0)
        parts = [np.linspace(-50.0 * max(l_edge, 1.0), 0.0, 100001)]
        if l_lo > 50.0 * max(l_edge, 1.0):
            parts.append(-np.exp(np.linspace(np.log(l_lo * 10),
                                             np.log(50.0 * max(l_edge, 1.0)),
                                             100001)))
        ls = np.concatenate(parts)
        t_min = np.maximum(0.0, -ls) * t0
        vol = np.prod(np.minimum(1.0, 2 * np.outer(t_min, self.sigma)),
                      axis=-1)
        w = np.exp(ls)
        Z = np.trapezoid(vol * w, ls)
        info = np.trapezoid(vol * (ls + 1.0) * w, ls)
        return info / Z - np.log(Z)

    def t(self, x):
        """Return the volume-like coordinate t of the given point."""
        x = np.asarray(x, dtype=float)
        return (np.abs(x - 0.5) / self.sigma[None, :]).max(axis=-1)

    def L_of_t(self, t):
        """Return the likelihood threshold at the volume-like coordinate t."""
        t = np.asarray(t, dtype=float)
        if self.schedule == 'linear':
            return -t / self.t0
        with np.errstate(over='ignore'):
            return -t ** (1.0 / self.s)

    def t_of_L(self, Lmin):
        """Return the volume-like coordinate t at the likelihood threshold."""
        if self.schedule == 'linear':
            return np.maximum(0.0, -np.asarray(Lmin, dtype=float)) * self.t0
        with np.errstate(over='ignore'):
            return max(0.0, -Lmin) ** self.s

    def proposal(self, Lmin, rng, nsamples):
        """Draw exact constrained-prior samples from the proposal."""
        t_min = self.t_of_L(Lmin)
        lower = np.maximum(0.0, 0.5 - t_min * self.sigma)
        upper = np.minimum(1.0, 0.5 + t_min * self.sigma)
        x = rng.uniform(lower, upper, size=(nsamples, self.ndim))
        return x, np.ones(nsamples, dtype=bool)

    def prior_volume(self, Lmin):
        """Exact prior volume of {loglike >= Lmin} (the clipped box)."""
        t_min = self.t_of_L(Lmin)
        return float(np.prod(np.minimum(1.0, 2 * t_min * self.sigma)))

    def analytic_H(self):
        """Layer-cake information gain, integrated on a grid in l: the.

        volume fraction V(l) transitions from ~1 to ~0 over a narrow l
        range (the prior boundary), which adaptive quadrature over an
        infinite interval misses.
        """
        if self.schedule == 'linear':
            return self._H_of_t0(self.t0)
        # power schedule L = -t^(1/s): V(l) = prod min(1, 2 sigma_i (-l)^s)
        # informative range: from well below the prior boundary to 0
        l_lo = -(1.0 / (2 * self.sigma.min())) ** (1.0 / self.s) - 10.0
        ls = np.linspace(l_lo, 0.0, 100001)
        with np.errstate(over='ignore'):
            t_min = np.maximum(0.0, -ls) ** self.s
        vol = np.prod(np.minimum(1.0, 2 * np.outer(t_min, self.sigma)),
                      axis=-1)
        w = np.exp(ls)
        Z = np.trapezoid(vol * w, ls)
        info = np.trapezoid(vol * (ls + 1.0) * w, ls)
        return info / Z - np.log(Z)


# ============================================================================
# 3. Shell: spherical annuli around a shell of radius r
# ============================================================================

class ShellContour(ContourProblem):
    """Flat prior on [0,1]^N; contours are spherical annuli around the.

    shell of radius r centered on mu:

        t(x) = | ||x - mu|| - r |
        L(t) = -t^2 / (2 sigma_shell^2)

    The proposal draws the radius exactly uniformly from the annulus
    {r - t_min <= |x - mu| <= r + t_min} and keeps the part inside the
    cube.

    Dimensional scaling: the distribution of the distance ||x - mu||
    for a uniform point in the unit cube is centered at
    sqrt(ndim/12) -- the cube's volume moves away from the center as
    dimension grows, and an inscribed ring of fixed radius occupies a
    vanishing volume fraction (at d=100 a ball of radius 0.3 fills
    ~1e-75 of the cube, so a fixed-radius ring becomes unreachable and
    the likelihood flat). The *absolute width* of that radial
    distribution is however dimension-independent, std ~ 0.129
    (a 1-D projection effect). So the ring is specified through volume
    fractions rather than a radius ratio: r = sqrt(ndim/12) puts it
    where the cube's radial density peaks, and an absolute width
    (sigma_shell ~ fractions of 0.129) keeps the ring's volume fraction
    of the cube -- and with it the difficulty -- constant with
    dimension. (Keeping the ring's volume fraction of its *enclosing
    ball* constant instead would require w/r ~ 1/ndim; the cube
    fraction is the operative one since the prior is the cube.)
    """

    def __init__(self, ndim, mu=0.5, r=0.1, sigma_shell=0.02):
        self.ndim = int(ndim)
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float),
                                  (self.ndim,)).astype(float)
        self.r = float(r)
        self.sigma_shell = float(sigma_shell)

    def t(self, x):
        """Return the volume-like coordinate t of the given point."""
        x = np.asarray(x, dtype=float)
        dist = np.linalg.norm(x - self.mu, axis=-1)
        return np.abs(dist - self.r)

    def L_of_t(self, t):
        """Return the likelihood threshold at the volume-like coordinate t."""
        return -0.5 * (np.asarray(t, dtype=float) / self.sigma_shell) ** 2

    def t_of_L(self, Lmin):
        """Return the volume-like coordinate t at the likelihood threshold."""
        return self.sigma_shell * np.sqrt(max(0.0, -2.0 * Lmin))

    def proposal(self, Lmin, rng, nsamples):
        """Draw exact constrained-prior samples from the proposal."""
        t_min = self.t_of_L(Lmin)
        rho_lo = max(0.0, self.r - t_min)
        rho_hi = self.r + t_min
        direction = rng.normal(size=(nsamples, self.ndim))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radial = (rho_lo ** self.ndim
                  + rng.uniform(size=nsamples)
                  * (rho_hi ** self.ndim - rho_lo ** self.ndim)) \
            ** (1.0 / self.ndim)
        x = self.mu[None, :] + radial[:, None] * direction
        ok = np.all((x > 0) & (x < 1), axis=1)
        return x, ok


# ============================================================================
# 4. Banana: ball contours under a triangular warping map
# ============================================================================

class BananaContour(ContourProblem):
    """Flat prior on [0,1]^N; contours are the image of Euclidean balls.

    under the triangular constant-Jacobian warping map:

        z_0 = (x_0 - mu_0) / sigma_0
        z_i = (x_i - mu_i - bend (x_{i-1} - mu_{i-1})^2
               + bend offset) / sigma_i
        t(x) = sum_i z_i^2,   L(t) = -t/2

    The proposal draws uniformly in the ball {t <= t_min}, maps it
    through the inverse warping and keeps the part inside the cube. The
    default values give the classic Rosenbrock banana in the
    [-10, 10] coordinates u = 20 x - 10: with bend = 20 the valley is
    the parabola u1 = u0^2, with the valley width 1/sqrt(200) ~ 0.07
    (sigma = 0.0035 as a fraction of [0, 1]) and the vertex at the
    origin (offset = sigma^2).
    """

    def __init__(self, ndim, sigma=0.0035, bend=20.0, mu=0.5, offset=None):
        sigma = np.asarray(sigma, dtype=float)
        self.ndim = int(sigma.shape[0]) if sigma.ndim else int(ndim)
        self.sigma = np.broadcast_to(sigma, (self.ndim,)).astype(float)
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float),
                                  (self.ndim,)).astype(float)
        bend = np.asarray(bend, dtype=float)
        self.bend = np.full((self.ndim - 1,), bend) if bend.ndim == 0 else bend
        if offset is None:
            offset = self.sigma[:-1] ** 2
        offset = np.asarray(offset, dtype=float)
        self.offset = np.full((self.ndim - 1,), offset) if offset.ndim == 0 \
            else offset

    def to_z(self, x):
        """Map to the whitened z-space."""
        x = np.asarray(x, dtype=float)
        zs = [(x[..., 0] - self.mu[0]) / self.sigma[0]]
        for i in range(1, self.ndim):
            previous = x[..., i - 1] - self.mu[i - 1]
            centreline = self.mu[i] + self.bend[i - 1] \
                * (previous ** 2 - self.offset[i - 1])
            zs.append((x[..., i] - centreline) / self.sigma[i])
        return np.stack(zs, axis=-1)

    def from_z(self, z):
        """Map from the whitened z-space back to the parameters."""
        z = np.asarray(z, dtype=float)
        xs = [self.mu[0] + self.sigma[0] * z[..., 0]]
        for i in range(1, self.ndim):
            previous = xs[-1] - self.mu[i - 1]
            xs.append(self.mu[i] + self.bend[i - 1]
                      * (previous ** 2 - self.offset[i - 1])
                      + self.sigma[i] * z[..., i])
        return np.stack(xs, axis=-1)

    def t(self, x):
        """Return the volume-like coordinate t of the given point."""
        z = self.to_z(x)
        return (z ** 2).sum(axis=-1)

    def L_of_t(self, t):
        """Return the likelihood threshold at the volume-like coordinate t."""
        return -0.5 * np.asarray(t, dtype=float)

    def t_of_L(self, Lmin):
        """Return the volume-like coordinate t at the likelihood threshold."""
        return max(0.0, -2.0 * Lmin)

    def proposal_volumes(self, Lmin):
        """Volumes of the enclosing envelopes; the smaller gives the.

        higher acceptance rate (the region is contained in both).
        """
        r = np.sqrt(self.t_of_L(Lmin))
        with np.errstate(over='ignore'):
            ball_volume = _ball_volume(self.ndim) * float(np.exp(
                self.ndim * np.log(r) + np.log(self.sigma).sum()))
        return {'ball': ball_volume, 'cube': 1.0}

    def proposal(self, Lmin, rng, nsamples, method='auto'):
        """Draws from the smaller of two envelopes: the image of the.

        z-space ball {t <= t_min} under the inverse warping (efficient
        at deep levels, where the ball image lies inside the cube
        preimage), or the unit cube itself (acceptance = the region
        volume fraction, robust at shallow levels where the ball image
        is enormous and its in-cube fraction vanishes). The envelope
        volume heuristic ignores the parabola chain constraint: at high
        dimension the ball image can have a vanishing in-cube fraction
        even when its volume is small, so the ball batch is probed and
        the cube used as a fallback when nothing is accepted.
        """
        t_min = self.t_of_L(Lmin)
        auto = method == 'auto'
        if auto:
            volumes = self.proposal_volumes(Lmin)
            method = min(volumes, key=lambda name: volumes[name])
        if method == 'ball':
            with np.errstate(over='ignore', invalid='ignore'):
                z = sample_uniform_ball(rng, nsamples, self.ndim,
                                        np.sqrt(t_min))
                x = self.from_z(z)
                ok = np.all((x > 0) & (x < 1), axis=1)
            if ok.any() or not auto:
                return x, ok
            method = 'cube'
        x = rng.uniform(size=(nsamples, self.ndim))
        return x, self.loglike(x) >= Lmin

    def analytic_H(self):
        """Information gain by importance sampling in z-space (the.

        posterior is the standard normal in z-space restricted to the
        cube preimage).
        """
        return mc_information_gain(self, ndraw=1000000, seed=0)


# ============================================================================
# 5. SplitEllipsoid: two-mode problem with a two-phase contour family
# ============================================================================

def _ball_volume(ndim):
    return np.exp(0.5 * ndim * np.log(np.pi)
                  - special.gammaln(0.5 * ndim + 1))


class SplitEllipsoidContour(ContourProblem):
    """Two-mode problem with the unit hypercube as prior.

    The constrained region is the intersection of the cube with the
    union of two balls that start large enough to cover the whole cube
    (radius sigma_start at s = 0, so the prior is the unit hypercube)
    and shrink toward their centers mu +- delta e0 (e0 = the first unit
    axis). With the draft's contour parameter s in [0, 1]:

    Phase 1 (s < t_split): centers fixed, radius
        sigma(s) = sigma_start - (sigma_start - sigma_split) s / t_split,
        s_pm(x) = t_split (sigma_start - r_pm) / (sigma_start - sigma_split),
        r_pm = |x - (mu pm delta e0)|,
    Phase 2 (s >= t_split): the balls are separated (sigma_split <=
        delta) and fully inside the cube (sigma_split <= 0.5 - delta);
        inner edges pinned at C_pm = mu pm delta e0 mp sigma_split e0,
        radius shrinking to zero at s = 1 (the modes become the points
        C_pm):
        sigma(s) = sigma_split (1 - s) / (1 - t_split),
        mu_pm(s) = C_pm pm sigma(s) e0,
        s_pm(x) = 1 - sum_i a_i^2 / (2 sum_i a_i b_i),
        a = x - C_pm, b = sigma_split/(1 - t_split) e0,
    with s(x) = max(s_plus, s_minus) and s_pm = -1 beyond a branch's
    reach, so {s(x) >= s_min} is the union of the two branch regions.
    The interface coordinate is t(x) = -s(x) (region {t(x) <= t_min}),
    and the likelihood schedule is decreasing in t: linear L = beta s
    (bounded by beta at the mode points) or divergent
    L = alpha s / (1 - s). Points outside the cube have likelihood
    L(s = -1), below the prior support.

    The proposal is rejection sampling from the unit hypercube or from
    the balls, whichever envelope is smaller (the balls are only used
    as an envelope once they are separated, sigma <= delta, so branch
    picking is exact); once the balls no longer intersect the cube
    boundary (sigma <= 0.5 - delta) the ball envelope has acceptance
    rate 1. The prior volume has no closed form (ball-cube clipping)
    and is Monte-Carlo tabulated by ``make_volume_function``.
    """

    def __init__(self, ndim, mu=0.5, delta=0.2, sigma_split=0.1,
                 t_split=0.5, schedule='linear', beta=1.0, alpha=1.0,
                 code_version=2):
        # code_version: bumped when the s(x) geometry changes so that
        # joblib.Memory cache entries keyed on instance state are not
        # reused across behavior changes
        self.code_version = int(code_version)
        self.ndim = int(ndim)
        self.mu = float(mu)
        self.delta = float(delta)
        self.sigma_split = float(sigma_split)
        self.t_split = float(t_split)
        if not 0 < self.t_split < 1:
            raise ValueError("0 < t_split < 1 required")
        if self.sigma_split > min(self.delta, 0.5 - self.delta):
            raise ValueError("sigma_split <= min(delta, 0.5 - delta) "
                             "required (balls separated and inside the "
                             "cube at the end of Phase 1)")
        self.schedule = schedule
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.e0 = np.zeros(self.ndim)
        self.e0[0] = 1.0
        # radius at s = 0: large enough that one ball covers the whole
        # cube (the farthest cube-corner distance from a branch center)
        self.sigma_start = float(np.sqrt((0.5 + self.delta) ** 2
                                         + 0.25 * (self.ndim - 1)))

    # -- geometry ----------------------------------------------------------

    def sigma_radius(self, s):
        """Branch radius at contour parameter s."""
        if s <= self.t_split:
            return self.sigma_start - (self.sigma_start - self.sigma_split) \
                * s / self.t_split
        return self.sigma_split * (1.0 - s) / (1.0 - self.t_split)

    def _branch_s(self, branch, x):
        """Draft contour parameter s_pm of x for one branch; -1 beyond.

        the branch's reach (r > sigma_start).
        """
        x = np.asarray(x, dtype=float)
        branch = np.asarray(branch, dtype=float)
        if branch.ndim == 0:
            m = self.mu + branch * self.delta * self.e0
        else:
            m = self.mu + branch[:, None] * self.delta * self.e0
        r = np.linalg.norm(x - m, axis=-1)
        s_phase1 = self.t_split * (self.sigma_start - r) \
            / (self.sigma_start - self.sigma_split)
        # relative tolerance: points at r == sigma_split (the exact mode
        # neighborhoods C_pm) must classify as Phase 2 consistently for
        # both branches despite float round-off in r
        tol = 1e-12
        if np.ndim(r) == 0:
            if r > self.sigma_start * (1 + tol):
                return -1.0
            if r > self.sigma_split * (1 + tol):
                return float(s_phase1)
            # Phase 2 (r <= sigma_split): inner edge pinned, center
            # slides along the branch-signed direction
            C = m - branch * self.sigma_split * self.e0
            a = x - C
            b = branch * (self.sigma_split / (1.0 - self.t_split)) * self.e0
            ab = float(np.dot(a, b))
            anorm2 = float((a ** 2).sum())
            if anorm2 == 0.0:
                # the tangent point C_pm lies in every sliding ball
                return 1.0
            if ab <= 0.0:
                # inside the sigma_split-ball but outside every sliding
                # ball: plateau at the phase boundary
                return float(self.t_split)
            return 1.0 - anorm2 / (2 * ab)
        # vectorized (branch may be scalar with a batch of points)
        branch_col = branch.reshape((-1, 1)) if branch.ndim else branch
        s = np.where(r > self.sigma_start * (1 + tol), -1.0, s_phase1)
        C = m - branch_col * self.sigma_split * self.e0
        a = x - C
        b = branch_col * (self.sigma_split / (1.0 - self.t_split)) * self.e0
        ab = (a * b).sum(axis=-1)
        anorm2 = (a ** 2).sum(axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s_phase2 = 1.0 - anorm2 / (2 * ab)
        s_phase2 = np.where(anorm2 == 0.0, 1.0,
                            np.where(ab > 0, s_phase2, self.t_split))
        inner = r <= self.sigma_split * (1 + tol)
        return np.where(inner, s_phase2, s)

    def s_of_x(self, x):
        """Draft contour parameter s(x) = max(s_plus, s_minus) in [0, 1].

        for points inside the cube; -1 outside the reach.
        """
        return np.maximum(self._branch_s(1.0, x), self._branch_s(-1.0, x))

    def t(self, x):
        """Return the interface contour coordinate of the point.

        t(x) = -s(x), so the region is {t(x) <= t_min}, t in [-1, 0].
        """
        return -self.s_of_x(x)

    # -- likelihood schedule ------------------------------------------------

    def L_of_t(self, t):
        """Return the likelihood threshold at the volume-like coordinate t."""
        s = -np.asarray(t, dtype=float)
        if self.schedule == 'linear':
            return self.beta * s
        if self.schedule == 'divergent':
            return self.alpha * s / (1.0 - s)
        raise ValueError("Unknown schedule=%s" % self.schedule)

    def t_of_L(self, Lmin):
        """Return the volume-like coordinate t at the likelihood threshold."""
        Lmin = np.asarray(Lmin, dtype=float)
        if self.schedule == 'linear':
            t = -Lmin / self.beta
        elif self.schedule == 'divergent':
            t = -Lmin / (self.alpha + Lmin)
        else:
            raise ValueError("Unknown schedule=%s" % self.schedule)
        return float(t) if Lmin.ndim == 0 else t

    def loglike(self, theta):
        """Return the log-likelihood."""
        # the prior is the unit hypercube: points outside have likelihood
        # below the prior support (the s = -1 schedule value)
        theta = np.asarray(theta, dtype=float)
        inside = np.all((theta > 0) & (theta < 1), axis=-1)
        with np.errstate(invalid='ignore'):
            L = self.L_of_t(self.t(np.where(inside[..., None]
                                            if theta.ndim > 1 else inside,
                                            theta, 0.5)))
        if theta.ndim == 1:
            return float(L) if inside else float(self.L_of_t(1.0))
        return np.where(inside, L, self.L_of_t(1.0))

    # -- proposals -----------------------------------------------------------

    def _branch_sliding_center(self, branch, s):
        """Phase-2 branch center: the inner edge stays pinned at.

        C_pm = mu pm delta e0 mp sigma_split e0 while the radius
        shrinks, so the center slides from mu pm delta e0 (at
        s = t_split) toward C_pm (at s = 1).
        """
        branch = np.asarray(branch, dtype=float)
        if branch.ndim == 0:
            m = self.mu + branch * self.delta * self.e0
            C = m - branch * self.sigma_split * self.e0
            return C + branch * self.sigma_radius(s) * self.e0
        m = self.mu + branch[:, None] * self.delta * self.e0
        C = m - branch[:, None] * self.sigma_split * self.e0
        return C + branch[:, None] * self.sigma_radius(s) * self.e0

    def proposal(self, Lmin, rng, nsamples):
        """Draw exact constrained-prior samples from the proposal."""
        s_min = min(1.0, max(0.0, -self.t_of_L(Lmin)))
        sigma = self.sigma_radius(s_min)
        if s_min <= self.t_split:
            # Phase 1: the region is the union of the two fixed-center
            # balls clipped to the cube. Envelope: the balls if they are
            # separated (sigma <= delta, so branch picking is exact) and
            # their union envelope is smaller than the cube; else the
            # cube itself. Once sigma <= 0.5 - delta the balls no longer
            # intersect the cube boundary and the acceptance is 1.
            V_balls = 2.0 * _ball_volume(self.ndim) * sigma ** self.ndim
            if sigma <= self.delta and V_balls < 1.0:
                branch = np.where(rng.uniform(size=nsamples) < 0.5,
                                  1.0, -1.0)
                z = sample_uniform_ball(rng, nsamples, self.ndim, sigma)
                centers = self.mu + branch[:, None] * self.delta * self.e0
                x = centers + z
                if sigma <= 0.5 - self.delta:
                    return x, np.ones(nsamples, dtype=bool)
                return x, np.all((x > 0) & (x < 1), axis=1)
            x = rng.uniform(size=(nsamples, self.ndim))
            s = self.s_of_x(x)
            return x, s >= s_min
        # Phase 2: the region is exactly the two sliding balls
        # (C_pm pm sigma(s_min) e0, sigma(s_min)) -- the union of the
        # later contours lies inside the s_min-ball -- so the draws are
        # direct with acceptance 1 (the inner edge stays pinned at
        # C_pm >= 2 delta > 0 and the outer edge at C_pm + 2 sigma_split
        # <= 1, so the balls lie inside the cube)
        branch = np.where(rng.uniform(size=nsamples) < 0.5, 1.0, -1.0)
        centers = self._branch_sliding_center(branch, s_min)
        z = sample_uniform_ball(rng, nsamples, self.ndim, sigma)
        return centers + z, np.ones(nsamples, dtype=bool)

    # -- volume ---------------------------------------------------------------

    @property
    def Vfull(self):
        """Full prior volume: the unit hypercube."""
        return 1.0

    def sample_prior(self, rng, nsamples):
        """Uniform draws from the prior (the unit hypercube)."""
        return rng.uniform(size=(nsamples, self.ndim))
