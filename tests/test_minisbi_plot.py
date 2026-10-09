"""Tests for the SBI validation diagnostics in ultranest.simbase.minisbi.plot."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tqdm")
pytest.importorskip("joblib")
matplotlib = pytest.importorskip("matplotlib")
stats = pytest.importorskip("scipy.stats")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from ultranest.simbase.minisbi.plot import parameter_coverage_test, posterior_predictive_check, rank_histogram  # noqa: E402

LOGISTIC_NOISE_SCALE = 0.1
# the diagnostics evaluate the posterior CDF from single-precision network outputs (measured error ~1e-5)
CDF_ATOL = 1e-4


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened."""
    yield
    plt.close('all')


def _draw_klp(rng, loc, scale, a, b):
    """Draw from the documented generative model: v ~ TruncatedLogistic(loc, scale) on (0, 1), u = 1 - (1 - v^a)^b (scipy based)."""
    p0 = stats.logistic.cdf(0.0, loc, scale)
    p1 = stats.logistic.cdf(1.0, loc, scale)
    v = stats.logistic.ppf(p0 + rng.uniform(size=np.broadcast(loc, scale).shape) * (p1 - p0), loc, scale)
    return 1.0 - (1.0 - v ** a) ** b


def _klp_cdf(x, loc, scale, a, b):
    """CDF of the same generative model, P(1 - (1 - v^a)^b <= x), computed with scipy."""
    v = (1.0 - (1.0 - x) ** (1.0 / b)) ** (1.0 / a)
    p0, p1, pv = (stats.logistic.cdf(t, loc, scale) for t in (0.0, 1.0, v))
    return (pv - p0) / (p1 - p0)


class EchoPosterior(torch.nn.Module):
    """Fake NPE network: the data vector is (loc, scale, a, b) of the posterior, optionally distorted."""

    def __init__(self, n_params, scale_factor=1.0, loc_shift=0.0):
        """Store the number of parameters and the distortion applied to the posterior."""
        super().__init__()
        self.n_params = n_params
        self.scale_factor = scale_factor
        self.loc_shift = loc_shift

    def forward(self, x):
        """Split the data vector into the four distribution parameters."""
        loc, scale, a, b = torch.split(x, self.n_params, dim=1)
        return loc + self.loc_shift, scale * self.scale_factor, a, b


class LogisticNoisePosterior(torch.nn.Module):
    """Exact posterior of x = u + Logistic(0, s) noise under a uniform prior: TruncatedLogistic(x, s) with a = b = 1."""

    def __init__(self, scale_factor=1.0):
        """Store a factor by which the reported posterior scale is distorted."""
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """Return loc = x, scale = s, a = b = 1."""
        ones = torch.ones_like(x)
        return x, LOGISTIC_NOISE_SCALE * self.scale_factor * ones, ones, ones


class RecordingPosterior(torch.nn.Module):
    """Wrap a fake posterior network and keep the (loc, scale, a, b) it returned."""

    def __init__(self, inner):
        """Store the wrapped network."""
        super().__init__()
        self.inner = inner
        self.outputs = []

    def forward(self, x):
        """Call the wrapped network and record its output as float64 arrays."""
        out = self.inner(x)
        self.outputs.append([t.numpy().astype(float) for t in out])
        return out


def make_oracle_generator(calls=None):
    """Simulator whose true unit-cube parameters are drawn from exactly the Kumaraswamy-Logistic posterior the data encode."""
    def generate_noiseless_batch(*, batch_idx, n_sim, seed, n_params):
        if calls is not None:
            calls.append(dict(batch_idx=batch_idx, n_sim=n_sim, seed=seed, n_params=n_params))
        rng = np.random.default_rng([seed + 1000, batch_idx + 1000])
        shape = (n_sim, n_params)
        loc = rng.uniform(0.1, 0.9, size=shape).astype(np.float32)
        scale = rng.uniform(0.05, 0.3, size=shape).astype(np.float32)
        a = np.exp(rng.uniform(-1, 1, size=shape)).astype(np.float32)
        b = np.exp(rng.uniform(-1, 1, size=shape)).astype(np.float32)
        u = _draw_klp(rng, loc.astype(float), scale.astype(float), a.astype(float), b.astype(float))
        params = np.concatenate([loc, scale, a, b], axis=1)
        return {
            'u_samples': np.clip(u, 1e-7, 1 - 1e-7).astype(np.float32),
            'mean_props': [{'params': p} for p in params],
        }
    return generate_noiseless_batch


def echo_noise(props, rng):
    """Return the encoded posterior parameters as the observed data, without noise."""
    return props['params']


def make_uniform_prior_generator():
    """Simulator drawing the true unit-cube parameters from the uniform prior."""
    def generate_noiseless_batch(*, batch_idx, n_sim, seed, n_params):
        rng = np.random.default_rng([seed + 1000, batch_idx + 1000])
        u = rng.uniform(size=(n_sim, n_params)).astype(np.float32)
        return {'u_samples': u, 'mean_props': [{'u': ui.astype(float)} for ui in u]}
    return generate_noiseless_batch


def logistic_noise(props, rng):
    """Observe each unit-cube parameter with additive logistic noise."""
    return props['u'] + rng.logistic(0.0, LOGISTIC_NOISE_SCALE, size=len(props['u']))


SETUPS = {
    # model factory, generator factory, noise injector, number of parameters
    'oracle_kuma_logistic': (lambda **kw: EchoPosterior(2, **kw), make_oracle_generator, echo_noise, 2),
    'exact_bayes_logistic_noise': (lambda **kw: LogisticNoisePosterior(**kw), make_uniform_prior_generator, logistic_noise, 3),
}


def _run(function, setup, tmp_path, model_kwargs=None, **kwargs):
    """Run rank_histogram or parameter_coverage_test on one of the synthetic SETUPS."""
    make_model, make_generator, inject_noise, n_params = SETUPS[setup]
    return function(
        model=make_model(**(model_kwargs or {})), generate_noiseless_batch=make_generator(), inject_noise=inject_noise,
        n_params=n_params, folder=str(tmp_path), **kwargs)


def _run_with_true_cdf(function, setup, tmp_path, **kwargs):
    """Run a diagnostic on 300 test simulations; also return F(u_true), F being the CDF of the posterior the network predicted."""
    make_model, make_generator, inject_noise, n_params = SETUPS[setup]
    model, generate, batches = RecordingPosterior(make_model()), make_generator(), []

    def recording_generate(**generate_kwargs):
        batches.append(generate(**generate_kwargs))
        return batches[-1]

    result, _, _ = function(model=model, generate_noiseless_batch=recording_generate, inject_noise=inject_noise,
                            n_params=n_params, folder=str(tmp_path), n_test=300, **kwargs)
    (batch,), (posterior,) = batches, model.outputs
    return result, _klp_cdf(batch['u_samples'].astype(float), *posterior)


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_rank_histogram_ranks_are_discretised_posterior_cdf(setup, tmp_path):
    """Each rank is floor(n_posterior_samples * F(u_true)), F being the CDF of the predicted posterior."""
    n_samples = 200
    ranks, cdf = _run_with_true_cdf(rank_histogram, setup, tmp_path, n_posterior_samples=n_samples)
    # a truth within rounding error of a rank boundary may land on either side of it
    ambiguous = np.abs(n_samples * cdf - np.round(n_samples * cdf)) < n_samples * CDF_ATOL
    assert ambiguous.mean() < 0.1
    assert np.all((ranks == np.floor(n_samples * cdf)) | ambiguous)


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_coverage_is_fraction_of_truths_in_central_interval(setup, tmp_path):
    """Coverage at level L is the fraction of (1 - L) / 2 <= F(u_true) <= (1 + L) / 2, at the default levels linspace(0.05, 0.99, 20)."""
    coverage, cdf = _run_with_true_cdf(parameter_coverage_test, setup, tmp_path)
    levels = np.linspace(0.05, 0.99, 20)[:, None, None]
    lower, upper = (1 - levels) / 2, (1 + levels) / 2
    inside = (cdf >= lower) & (cdf <= upper)
    ambiguous = (np.abs(cdf - lower) < CDF_ATOL) | (np.abs(cdf - upper) < CDF_ATOL)
    assert ambiguous.mean() < 0.01
    assert coverage.shape == (20, cdf.shape[1])
    assert np.all(np.abs(coverage - inside.mean(axis=1)) <= ambiguous.mean(axis=1))


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_rank_histogram_calibrated_posterior_gives_uniform_ranks(setup, tmp_path):
    """When the predicted posterior is the true conditional distribution, the ranks are uniform over [0, n_posterior_samples)."""
    n_test, n_samples = 1000, 100
    ranks, fig, axes = _run(rank_histogram, setup, tmp_path, n_test=n_test, n_posterior_samples=n_samples, n_bins=10)
    n_params = SETUPS[setup][3]
    assert ranks.shape == (n_test, n_params)
    assert np.issubdtype(ranks.dtype, np.integer)
    assert ranks.min() >= 0 and ranks.max() <= n_samples
    for p in range(n_params):
        counts = np.bincount(ranks[:, p] * 10 // n_samples, minlength=11)
        assert counts[10] <= 1
        assert stats.chisquare(counts[:10]).pvalue > 1e-3, counts
        assert stats.kstest((ranks[:, p] + 0.5) / n_samples, 'uniform').pvalue > 1e-3
        mean_rank_sd = n_samples / np.sqrt(12 * n_test)
        assert abs(ranks[:, p].mean() - (n_samples - 1) / 2) < 4 * mean_rank_sd


@pytest.mark.parametrize("setup", sorted(SETUPS))
@pytest.mark.parametrize("scale_factor,shape", [(0.3, 'U'), (3.0, 'hump')])
def test_rank_histogram_detects_miscalibrated_width(setup, scale_factor, shape, tmp_path):
    """An over-confident posterior gives a U-shaped rank histogram, an under-confident one a hump."""
    n_test, n_samples = 2000, 100
    ranks, _, _ = _run(rank_histogram, setup, tmp_path, model_kwargs=dict(scale_factor=scale_factor),
                       n_test=n_test, n_posterior_samples=n_samples)
    # fraction of ranks in the lowest and highest 10%: 0.2 for a calibrated posterior
    outer = np.mean((ranks < n_samples // 10) | (ranks >= n_samples - n_samples // 10), axis=0)
    margin = 3 * np.sqrt(0.2 * 0.8 / n_test)
    if shape == 'U':
        assert np.all(outer > 0.2 + margin), outer
    else:
        assert np.all(outer < 0.2 - margin), outer


@pytest.mark.parametrize("loc_shift", [-0.1, 0.1])
def test_rank_histogram_detects_biased_posterior(loc_shift, tmp_path):
    """A posterior shifted to larger values places the truth at low ranks, and vice versa."""
    n_test, n_samples = 1000, 100
    ranks, _, _ = _run(rank_histogram, 'oracle_kuma_logistic', tmp_path, model_kwargs=dict(loc_shift=loc_shift),
                       n_test=n_test, n_posterior_samples=n_samples)
    # standardised deviation of the mean rank from its value under calibration
    z = (ranks.mean(axis=0) - (n_samples - 1) / 2) / (n_samples / np.sqrt(12 * n_test))
    if loc_shift > 0:
        assert np.all(z < -5), z
    else:
        assert np.all(z > 5), z


@pytest.mark.parametrize("function", [rank_histogram, parameter_coverage_test])
@pytest.mark.parametrize("n_params,grid", [(1, (1, 1)), (3, (1, 3)), (5, (2, 4))])
def test_diagnostic_figure_layout(function, n_params, grid, tmp_path):
    """One panel per parameter on a grid of at most four columns, unused panels hidden, titled by parameter name."""
    calls = []
    model = EchoPosterior(n_params).train()
    result, fig, axes = function(
        model=model, generate_noiseless_batch=make_oracle_generator(calls), inject_noise=echo_noise,
        n_params=n_params, folder=str(tmp_path), n_test=20, seed=42)
    assert not model.training
    assert calls == [dict(batch_idx=-1, n_sim=20, seed=42, n_params=n_params)]
    assert result.shape[1] == n_params
    assert axes.shape == grid
    assert fig.axes == list(axes.flat)
    assert [ax.get_visible() for ax in axes.flat] == [True] * n_params + [False] * (grid[0] * grid[1] - n_params)
    assert [ax.get_title() for ax in axes.flat[:n_params]] == [f"param_{p}" for p in range(n_params)]

    names = [f"theta{p}" for p in range(n_params)]
    _, _, axes = function(
        model=model, generate_noiseless_batch=make_oracle_generator(), inject_noise=echo_noise,
        n_params=n_params, folder=str(tmp_path), n_test=20, param_names=names)
    assert [ax.get_title() for ax in axes.flat[:n_params]] == names


def test_rank_histogram_figure_shows_rank_density(tmp_path):
    """Each panel draws the rank density as a step histogram of n_bins bins over [0, n_posterior_samples], with a flat-density reference line."""
    n_test, n_samples, n_bins = 300, 200, 8
    ranks, fig, axes = _run(rank_histogram, 'oracle_kuma_logistic', tmp_path, n_test=n_test, n_posterior_samples=n_samples, n_bins=n_bins)
    edges = np.linspace(0, n_samples, n_bins + 1)
    for p, ax in enumerate(axes.flat):
        counts = np.bincount(np.minimum(ranks[:, p] * n_bins // n_samples, n_bins - 1), minlength=n_bins)
        density = counts / (n_test * n_samples / n_bins)
        # step outline without its two baseline corners: (left edge, height), (right edge, height) for each bin
        outline = ax.patches[0].get_xy()[1:-1]
        assert_allclose(outline[:, 0], np.repeat(edges, 2)[1:-1])
        assert_allclose(outline[:, 1], np.repeat(density, 2))
        assert_allclose(ax.lines[0].get_ydata(), 1.0 / n_samples)
        assert ax.get_xlabel() == "rank"


@pytest.mark.parametrize("function", [rank_histogram, parameter_coverage_test])
def test_diagnostic_single_test_simulation(function, tmp_path):
    """A test set of a single simulation (batch size 1) works."""
    result, fig, axes = _run(function, 'oracle_kuma_logistic', tmp_path, n_test=1)
    if function is rank_histogram:
        assert result.shape == (1, 2)
        assert np.all((result >= 0) & (result <= 200))
    else:
        # one truth is either inside a central interval or not, and once inside it stays inside the wider ones
        assert result.shape == (20, 2)
        assert set(np.unique(result)) <= {0.0, 1.0}
        assert np.all(np.diff(result, axis=0) >= 0)


@pytest.mark.parametrize("function", [rank_histogram, parameter_coverage_test])
def test_diagnostics_are_deterministic_for_a_seed(function, tmp_path):
    """Repeated calls with the same seed give identical results, a different seed a different test set."""
    first, _, _ = _run(function, 'exact_bayes_logistic_noise', tmp_path, n_test=50, seed=5)
    second, _, _ = _run(function, 'exact_bayes_logistic_noise', tmp_path, n_test=50, seed=5)
    other, _, _ = _run(function, 'exact_bayes_logistic_noise', tmp_path, n_test=50, seed=6)
    assert_array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_coverage_calibrated_posterior_matches_nominal(setup, tmp_path):
    """With the true posterior, the empirical coverage of every central credible interval matches its level."""
    n_test = 1000
    coverage, fig, axes = _run(parameter_coverage_test, setup, tmp_path, n_test=n_test)
    levels = np.linspace(0.05, 0.99, 20)
    n_params = SETUPS[setup][3]
    assert coverage.shape == (len(levels), n_params)
    assert np.all(np.diff(coverage, axis=0) >= 0)
    tolerance = 4 * np.sqrt(levels * (1 - levels) / n_test) + 1.0 / n_test
    assert np.all(np.abs(coverage - levels[:, None]) <= tolerance[:, None]), coverage - levels[:, None]


@pytest.mark.parametrize("setup", sorted(SETUPS))
@pytest.mark.parametrize("scale_factor", [0.3, 3.0])
def test_coverage_detects_miscalibrated_width(setup, scale_factor, tmp_path):
    """Too narrow posteriors under-cover, too wide posteriors over-cover."""
    n_test = 2000
    levels = np.array([0.2, 0.4, 0.6, 0.8])
    coverage, _, _ = _run(parameter_coverage_test, setup, tmp_path, model_kwargs=dict(scale_factor=scale_factor),
                          n_test=n_test, credible_levels=levels.tolist())
    margin = 3 * np.sqrt(levels * (1 - levels) / n_test)[:, None]
    if scale_factor < 1:
        assert np.all(coverage < levels[:, None] - margin), coverage
    else:
        assert np.all(coverage > levels[:, None] + margin), coverage


def test_coverage_edge_and_unsorted_levels(tmp_path):
    """Level 1 covers everything, level 0 nothing, and rows follow the order of the given levels."""
    levels = [0.9, 0.0, 1.0, 0.5]
    coverage, _, _ = _run(parameter_coverage_test, 'oracle_kuma_logistic', tmp_path, n_test=200, credible_levels=levels)
    assert coverage.shape == (4, 2)
    assert_array_equal(coverage[2], 1.0)
    assert_array_equal(coverage[1], 0.0)
    sorted_coverage, _, _ = _run(parameter_coverage_test, 'oracle_kuma_logistic', tmp_path, n_test=200, credible_levels=sorted(levels))
    assert_array_equal(coverage[np.argsort(levels)], sorted_coverage)


def test_coverage_figure_shows_coverage_curve(tmp_path):
    """Each panel plots the returned coverage against the levels next to the ideal diagonal."""
    levels = np.linspace(0.1, 0.9, 5)
    coverage, fig, axes = _run(parameter_coverage_test, 'oracle_kuma_logistic', tmp_path, n_test=100, credible_levels=levels.tolist())
    for p, ax in enumerate(axes.flat):
        empirical, ideal = ax.lines
        assert_allclose(empirical.get_xdata(), levels)
        assert_allclose(empirical.get_ydata(), coverage[:, p])
        assert_allclose(ideal.get_xdata(), [0, 1])
        assert_allclose(ideal.get_ydata(), [0, 1])
        assert ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1)
    assert "coverage" in fig.get_suptitle()


def test_diagnostics_with_npe_network(tmp_path):
    """The diagnostics accept an (untrained) NPENetwork and a joblib-cached simulator from minisbi.utils."""
    pytest.importorskip("torchinfo")
    from ultranest.simbase.minisbi.npe import NPENetwork
    from ultranest.simbase.minisbi.utils import make_cached_generate, make_memory

    xg = np.linspace(0, 1, 4)

    def generate_mean_and_noise(theta, idx=0, seed=0):
        return {'mean': theta[0] + theta[1] * xg, 'noise_std': 0.1 + 0 * xg}

    def inject_noise(props, rng):
        return rng.normal(props['mean'], props['noise_std'])

    generate = make_cached_generate(make_memory(str(tmp_path / "cache")), lambda u: 4 * u - 2, generate_mean_and_noise)
    torch.manual_seed(0)
    model = NPENetwork(n_data=4, n_params=2, depth=2, width=8, activation_name='ReLU', layer_shape='rectangular')
    common = dict(model=model, generate_noiseless_batch=generate, inject_noise=inject_noise, n_params=2, folder=str(tmp_path), n_test=30)
    ranks, _, _ = rank_histogram(**common, n_posterior_samples=50)
    coverage, _, _ = parameter_coverage_test(**common, credible_levels=[0.5, 1.0])
    assert ranks.shape == (30, 2) and ranks.min() >= 0 and ranks.max() <= 50
    assert coverage.shape == (2, 2) and np.all(np.isfinite(coverage))
    assert_array_equal(coverage[1], 1.0)


# ---------------------------------------------------------------------------
# posterior_predictive_check
# ---------------------------------------------------------------------------

def make_linear_model(xg, noise_std, calls=None):
    """Return a straight-line generate_mean_and_noise recording its calls."""
    def generate_mean_and_noise(idx, seed, theta):
        if calls is not None:
            calls.append((idx, seed, np.array(theta)))
        return {'mean': theta[0] + theta[1] * xg, 'noise_std': noise_std + 0 * xg}
    return generate_mean_and_noise


def gaussian_noise(props, rng):
    """Add Gaussian noise of the given standard deviation."""
    return rng.normal(props['mean'], props['noise_std'])


def noiseless(props, rng):
    """Return the model mean unchanged."""
    return props['mean']


def test_ppc_returns_realisations_and_follows_call_contract(tmp_path):
    """Mean curves use distinct posterior samples with idx 10000+k, realisations idx 20000+k, both with the given seed."""
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(30, 2))
    xg = np.linspace(0, 1, 7)
    calls = []
    realisations, fig, axes = posterior_predictive_check(
        posterior_samples_theta=theta, observed_data=np.zeros(7), generate_mean_and_noise=make_linear_model(xg, 0.1, calls),
        inject_noise=noiseless, folder=str(tmp_path), n_mean_curves=10, n_realisation_curves=5, seed=99)
    assert realisations.shape == (5, 7) and realisations.dtype == np.float64
    assert len(axes) == 2 and fig.axes == list(axes)
    assert [c[0] for c in calls] == list(range(10000, 10010)) + list(range(20000, 20005))
    assert {c[1] for c in calls} == {99}
    used = [int(np.flatnonzero((theta == c[2]).all(axis=1))[0]) for c in calls]
    assert len(set(used[:10])) == 10 and len(set(used[10:])) == 5
    # without noise, each realisation is the model curve of the posterior sample it was drawn for
    assert_allclose(realisations, theta[used[10:], :1] + theta[used[10:], 1:] * xg)
    mean_lines = [line.get_ydata() for line in axes[0].lines]
    assert len(mean_lines) == 11
    assert_allclose(mean_lines[:-1], theta[used[:10], :1] + theta[used[:10], 1:] * xg)
    assert_allclose(mean_lines[-1], np.mean(mean_lines[:-1], axis=0))
    assert len(axes[1].lines) == 6
    assert_allclose(axes[1].lines[-1].get_ydata(), realisations.mean(axis=0))
    assert axes[1].get_title() == "Posterior data realisations  (n=5)"


def test_ppc_caps_curve_counts_at_number_of_samples(tmp_path):
    """With fewer posterior samples than requested curves, every sample is used exactly once."""
    theta = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]])
    xg = np.linspace(-1, 1, 5)
    calls = []
    realisations, fig, axes = posterior_predictive_check(
        posterior_samples_theta=theta, observed_data=np.ones(5), generate_mean_and_noise=make_linear_model(xg, 0.1, calls),
        inject_noise=noiseless, folder=str(tmp_path))
    assert realisations.shape == (3, 5)
    assert len(calls) == 6
    assert len(axes[0].lines) == 4 and len(axes[1].lines) == 4
    # the model curves of the three samples start at -1, 1 and 3 respectively
    assert_allclose(realisations[np.argsort(realisations[:, 0])], theta[:, :1] + theta[:, 1:] * xg)


def test_ppc_observed_data_and_x_coords(tmp_path):
    """Observed data are scattered at np.linspace(-5, 5, n_data) by default, or at the given x coordinates."""
    theta = np.random.default_rng(4).normal(size=(8, 2))
    observed = np.arange(6.0)
    xg = np.linspace(0, 1, 6)
    kwargs = dict(posterior_samples_theta=theta, observed_data=observed, generate_mean_and_noise=make_linear_model(xg, 0.1),
                  inject_noise=gaussian_noise, folder=str(tmp_path), n_mean_curves=3, n_realisation_curves=3)
    _, _, axes = posterior_predictive_check(**kwargs)
    for ax in axes:
        assert_allclose(ax.collections[0].get_offsets(), np.column_stack([np.linspace(-5, 5, 6), observed]))
        assert_allclose(ax.lines[0].get_xdata(), np.linspace(-5, 5, 6))
    x_coords = np.geomspace(1, 100, 6)
    _, _, axes = posterior_predictive_check(x_coords=x_coords, **kwargs)
    for ax in axes:
        assert_allclose(ax.collections[0].get_offsets(), np.column_stack([x_coords, observed]))
        assert_allclose(ax.lines[-1].get_xdata(), x_coords)


def test_ppc_is_deterministic_for_a_seed(tmp_path):
    """The noisy realisations depend only on the seed."""
    theta = np.random.default_rng(5).normal(size=(20, 2))
    kwargs = dict(posterior_samples_theta=theta, observed_data=np.zeros(4), generate_mean_and_noise=make_linear_model(np.arange(4.0), 0.3),
                  inject_noise=gaussian_noise, folder=str(tmp_path), n_mean_curves=5, n_realisation_curves=8)
    first, _, _ = posterior_predictive_check(seed=1, **kwargs)
    second, _, _ = posterior_predictive_check(seed=1, **kwargs)
    other, _, _ = posterior_predictive_check(seed=2, **kwargs)
    assert_array_equal(first, second)
    assert not np.allclose(first, other)


def test_ppc_without_mean_key_skips_mean_curves(tmp_path):
    """A simulator whose properties have no 'mean' entry still yields realisations, with only the data in the top panel."""
    def generate_rates(idx, seed, theta):
        return {'rate': np.full(5, theta[0])}

    def poisson_noise(props, rng):
        return rng.poisson(props['rate'])

    theta = np.array([[2.0], [3.0], [4.0], [5.0]])
    realisations, _, axes = posterior_predictive_check(
        posterior_samples_theta=theta, observed_data=np.full(5, 3.0), generate_mean_and_noise=generate_rates,
        inject_noise=poisson_noise, folder=str(tmp_path), n_realisation_curves=4)
    assert realisations.shape == (4, 5)
    assert np.all(realisations >= 0) and np.all(realisations == np.round(realisations))
    assert len(axes[0].lines) == 0 and len(axes[0].collections) == 1
    assert len(axes[1].lines) == 5


@pytest.mark.parametrize("shift,well_calibrated", [(0.0, True), (1.5, False)])
def test_ppc_realisations_bracket_data_from_correct_posterior(shift, well_calibrated, tmp_path):
    """With the exact conjugate posterior of a straight-line fit, about 90% of data points fall in the central 90% of realisations."""
    rng = np.random.default_rng(6)
    n_data, noise_std = 200, 0.5
    xg = np.linspace(-1, 1, n_data)
    design = np.column_stack([np.ones(n_data), xg])
    observed = design @ np.array([1.0, -2.0]) + rng.normal(0, noise_std, n_data)
    theta_hat = np.linalg.lstsq(design, observed, rcond=None)[0]
    cov = noise_std ** 2 * np.linalg.inv(design.T @ design)
    theta = rng.multivariate_normal(theta_hat + [shift, 0.0], cov, size=400)
    realisations, _, _ = posterior_predictive_check(
        posterior_samples_theta=theta, observed_data=observed, generate_mean_and_noise=make_linear_model(xg, noise_std),
        inject_noise=gaussian_noise, folder=str(tmp_path), n_mean_curves=20, n_realisation_curves=400, seed=8)
    assert realisations.shape == (400, n_data)
    assert_allclose(realisations.mean(axis=0), design @ (theta_hat + [shift, 0.0]), atol=0.15)
    assert_allclose(realisations.std(axis=0).mean(), noise_std, rtol=0.05)
    pit = (realisations < observed).mean(axis=0)
    inside = np.mean((pit > 0.05) & (pit < 0.95))
    if well_calibrated:
        assert abs(inside - 0.9) < 0.06, inside
    else:
        assert inside < 0.3, inside
