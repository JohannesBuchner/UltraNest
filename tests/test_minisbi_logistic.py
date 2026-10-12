"""Tests for the Kumaraswamy-Logistic distribution helpers of minisbi."""
import itertools

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from ultranest.simbase.minisbi.logistic import (  # noqa: E402
    _logistic_log_normaliser, _logistic_log_prob, _logistic_log_sf,
    kuma_icdf, kuma_icdf_np, kuma_log_prob, kuma_logistic_cdf,
    kuma_logistic_cdf_vec, kuma_logistic_icdf_vec, kuma_logistic_logpdf_vec,
    logistic_log_cdf, logit, nll_kuma_logistic_product,
    sample_kuma_logistic_product, sigmoid)

# Moderate per-dimension parameters (loc, scale, a, b): a peaked, a skewed, a nearly flat and an edge-piled dimension.
LOC = np.array([0.3, 0.75, 0.5, 0.05])
SCALE = np.array([0.08, 0.2, 2.0, 0.03])
A = np.array([1.5, 0.7, 1.0, 3.0])
B = np.array([2.0, 0.9, 1.0, 0.5])
PARAMS = (LOC, SCALE, A, B)
# the first three dimensions keep (almost) all of their mass inside [1e-7, 1 - 1e-7]
MODERATE = (LOC[:3], SCALE[:3], A[:3], B[:3])

# corners of the parameter ranges produced by the NPE network (loc via sigmoid, the others via exp of clamped logs)
NETWORK_CORNERS = list(itertools.product([1e-4, 0.9999], [np.exp(-8), np.exp(4)], [np.exp(-4), np.exp(4)], [np.exp(-4), np.exp(4)]))


def _dim(j, params=PARAMS):
    """Return the parameters of dimension ``j`` as length-1 arrays."""
    return tuple(p[j:j + 1] for p in params)


def _batch(params, n, dtype=torch.float64):
    """Broadcast per-dimension numpy parameters to ``(n, d)`` tensors."""
    return tuple(torch.tensor(np.tile(p, (n, 1)), dtype=dtype) for p in params)


def _simpson(y, x):
    """Integrate samples ``y`` on the uniform grid ``x`` (odd length) with the composite Simpson rule."""
    h = x[1] - x[0]
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def test_sigmoid_values_and_symmetry():
    """Sigmoid matches 1 / (1 + exp(-x)), is 0.5 at zero and satisfies sigmoid(-x) = 1 - sigmoid(x)."""
    x = np.linspace(-30, 30, 121)
    assert_allclose(sigmoid(x), 1.0 / (1.0 + np.exp(-x)), rtol=1e-12)
    assert_allclose(sigmoid(-x), 1.0 - sigmoid(x), atol=1e-15)
    assert sigmoid(0.0) == 0.5
    assert sigmoid(np.float64(2.0)) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_sigmoid_saturates_without_overflow():
    """Very large inputs are clipped before exponentiation: no floating point errors and results stay in [0, 1]."""
    x = np.array([-1e4, -600.0, 600.0, 1e4])
    with np.errstate(all="raise"):
        out = sigmoid(x)
    assert np.all(np.isfinite(out))
    assert 0.0 < out[0] < 1e-200 and out[0] == out[1]
    assert out[2] == 1.0 and out[3] == 1.0


def test_logit_inverts_sigmoid():
    """Logit is the inverse of sigmoid, antisymmetric around 0.5 and zero at 0.5."""
    x = np.linspace(-20, 20, 81)
    assert_allclose(logit(sigmoid(x)), x, atol=1e-6)
    p = np.linspace(0.01, 0.99, 99)
    assert_allclose(sigmoid(logit(p)), p, rtol=1e-12)
    assert_allclose(logit(1.0 - p), -logit(p), atol=1e-12)
    assert logit(0.5) == 0.0


def test_logit_clips_boundaries():
    """Probabilities of exactly 0 and 1 are clipped, so logit stays finite (about -/+34.5)."""
    with np.errstate(all="raise"):
        out = logit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(out))
    assert_allclose(out, [-34.54, 34.54], rtol=1e-3)


@pytest.mark.parametrize("dtype,rtol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_kuma_log_prob_matches_torch_distribution(dtype, rtol):
    """kuma_log_prob agrees with torch.distributions.Kumaraswamy for interior points and keeps shape and dtype."""
    a = torch.tensor([[0.5, 1.0, 2.5], [7.0, 0.3, 1.2]], dtype=dtype)
    b = torch.tensor([[0.7, 1.0, 3.0], [0.4, 2.0, 1.2]], dtype=dtype)
    u = torch.tensor([[0.1, 0.5, 0.3], [0.9, 0.05, 0.62]], dtype=dtype)
    out = kuma_log_prob(u, a, b)
    assert out.shape == (2, 3) and out.dtype == dtype
    expected = torch.distributions.Kumaraswamy(a, b).log_prob(u)
    assert_allclose(out.numpy(), expected.numpy(), rtol=rtol, atol=rtol)


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 5.0), (3.5, 1.5)])
def test_kuma_log_prob_integrates_to_one(a, b):
    """The Kumaraswamy density integrates to one over the unit interval."""
    u = torch.linspace(0.0, 1.0, 20001, dtype=torch.float64)
    dens = kuma_log_prob(u, torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)).exp()
    assert torch.trapezoid(dens, u).item() == pytest.approx(1.0, abs=1e-5)


def test_kuma_log_prob_finite_at_boundaries():
    """Points exactly at 0 and 1 are clamped to 1e-7 and 1 - 1e-7, so the log-density stays finite for any shape parameters."""
    u = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    a = torch.tensor([0.3, 0.3, 4.0, 4.0], dtype=torch.float64)
    b = torch.tensor([0.4, 0.4, 5.0, 5.0], dtype=torch.float64)
    out = kuma_log_prob(u, a, b)
    assert torch.isfinite(out).all()
    x = np.array([1e-7, 1 - 1e-7, 1e-7, 1 - 1e-7])
    an, bn = a.numpy(), b.numpy()
    expected = np.log(an * bn) + (an - 1) * np.log(x) + (bn - 1) * np.log1p(-x ** an)
    assert_allclose(out.numpy(), expected, rtol=1e-6)


@pytest.mark.parametrize("a,b", [(0.5, 0.7), (1.0, 1.0), (2.5, 3.0), (7.0, 0.4)])
def test_kuma_icdf_inverts_kumaraswamy_cdf(a, b):
    """kuma_icdf is increasing, maps into (0, 1), inverts the CDF 1 - (1 - v^a)^b and gives the closed-form median."""
    u = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64)
    bt = torch.tensor(b, dtype=torch.float64)
    v = kuma_icdf(u, at, bt)
    assert v.shape == u.shape and v.dtype == torch.float64
    assert ((v > 0) & (v < 1)).all()
    assert (v[1:] > v[:-1]).all()
    assert_allclose((1 - (1 - v ** at) ** bt).numpy(), u.numpy(), rtol=1e-10)
    median = kuma_icdf(torch.tensor(0.5, dtype=torch.float64), at, bt).item()
    assert median == pytest.approx((1 - 2 ** (-1 / b)) ** (1 / a), rel=1e-12)


def test_kuma_icdf_numpy_matches_torch():
    """The NumPy and PyTorch quantile functions agree, including on clamped boundary points."""
    rng = np.random.default_rng(3)
    u = rng.uniform(size=(3, 4))
    u[0, :2] = [0.0, 1.0]
    a = rng.uniform(0.2, 5.0, size=(3, 4))
    b = rng.uniform(0.2, 5.0, size=(3, 4))
    out_np = kuma_icdf_np(u, a, b)
    out_t = kuma_icdf(torch.tensor(u), torch.tensor(a), torch.tensor(b))
    assert out_np.shape == (3, 4)
    assert_allclose(out_np, out_t.numpy(), rtol=1e-8)
    assert np.all(np.isfinite(out_np)) and np.all((out_np > 0) & (out_np <= 1))
    assert kuma_icdf_np(0.5, 2.0, 3.0) == pytest.approx((1 - 0.5 ** (1 / 3)) ** 0.5)


def test_kuma_icdf_clamps_boundaries():
    """Inputs of exactly 0 and 1 are clamped to the open interval before inversion."""
    u = torch.tensor([0.0, 1.0], dtype=torch.float64)
    a = torch.tensor([2.0, 2.0], dtype=torch.float64)
    b = torch.tensor([1.5, 1.5], dtype=torch.float64)
    v = kuma_icdf(u, a, b)
    assert ((v > 0) & (v < 1)).all()
    assert_allclose(v.numpy(), kuma_icdf_np(np.array([1e-7, 1 - 1e-7]), 2.0, 1.5), rtol=1e-12)


def test_logistic_log_prob_closed_form_and_normalised():
    """The logistic log-density equals log(sigmoid(z) sigmoid(-z) / scale) per element and integrates to one."""
    rng = np.random.default_rng(4)
    loc = torch.tensor(rng.uniform(-1, 2, size=(2, 3)))
    scale = torch.tensor(rng.uniform(0.05, 3, size=(2, 3)))
    x = torch.tensor(rng.uniform(-2, 3, size=(2, 3)))
    out = _logistic_log_prob(x, loc, scale)
    assert out.shape == (2, 3) and out.dtype == torch.float64
    z = (x - loc) / scale
    logsig = torch.nn.functional.logsigmoid
    assert_allclose(out.numpy(), (logsig(z) + logsig(-z) - torch.log(scale)).numpy(), rtol=1e-12)

    grid = torch.linspace(-40.0, 40.0, 40001, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    dens = _logistic_log_prob(0.7 + 0.5 * grid, torch.tensor(0.7, dtype=torch.float64), 0.5 * one).exp()
    assert torch.trapezoid(dens, 0.7 + 0.5 * grid).item() == pytest.approx(1.0, abs=1e-8)


def test_logistic_log_prob_stable_in_tails():
    """Far in both tails the log-density approaches -|z| - log(scale) without overflow."""
    scale = torch.tensor([0.01, 0.01], dtype=torch.float64)
    loc = torch.tensor([0.5, 0.5], dtype=torch.float64)
    x = loc + scale * torch.tensor([-500.0, 500.0], dtype=torch.float64)
    out = _logistic_log_prob(x, loc, scale)
    assert_allclose(out.numpy(), -500.0 - np.log(0.01), rtol=1e-12)


def test_logistic_log_cdf_derivative_is_density():
    """The derivative of exp(logistic_log_cdf) with respect to x equals the logistic density."""
    loc = torch.tensor([0.2, 0.6, -0.3], dtype=torch.float64)
    scale = torch.tensor([0.1, 0.7, 2.0], dtype=torch.float64)
    x = torch.tensor([0.25, -0.4, 1.3], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(logistic_log_cdf(x, loc, scale).exp().sum(), x)
    assert_allclose(grad.numpy(), _logistic_log_prob(x, loc, scale).exp().detach().numpy(), rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_logistic_log_cdf_and_log_sf(dtype):
    """CDF and survival function sum to one, are mirror images around loc and stay finite deep in the tails."""
    loc = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3], dtype=dtype)
    scale = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1], dtype=dtype)
    x = torch.tensor([-0.5, 0.1, 0.3, 0.45, 1.2], dtype=dtype)
    log_cdf = logistic_log_cdf(x, loc, scale)
    log_sf = _logistic_log_sf(x, loc, scale)
    assert log_cdf.dtype == dtype and log_sf.shape == x.shape
    assert_allclose((log_cdf.exp() + log_sf.exp()).numpy(), 1.0, rtol=1e-6)
    assert_allclose(log_sf.numpy(), logistic_log_cdf(2 * loc - x, loc, scale).numpy(), rtol=1e-6)
    assert log_cdf[2].item() == pytest.approx(np.log(0.5))

    tail = torch.tensor([-200.0, 200.0], dtype=dtype)
    zero = torch.zeros(2, dtype=dtype)
    one = torch.ones(2, dtype=dtype)
    assert logistic_log_cdf(tail, zero, one)[0].item() == pytest.approx(-200.0)
    assert _logistic_log_sf(tail, zero, one)[1].item() == pytest.approx(-200.0)
    assert logistic_log_cdf(tail, zero, one)[1].item() == pytest.approx(0.0, abs=1e-30)


def test_logistic_log_normaliser_is_mass_in_unit_interval():
    """The log normaliser is the log of the logistic probability mass inside (0, 1), per element."""
    loc = torch.tensor([[0.3, 0.75, 0.5], [0.05, -0.2, 1.4]], dtype=torch.float64)
    scale = torch.tensor([[0.08, 0.2, 2.0], [0.03, 0.5, 0.3]], dtype=torch.float64)
    log_z = _logistic_log_normaliser(loc, scale)
    assert log_z.shape == (2, 3) and log_z.dtype == torch.float64
    expected = np.log(sigmoid((1 - loc.numpy()) / scale.numpy()) - sigmoid(-loc.numpy() / scale.numpy()))
    assert_allclose(log_z.numpy(), expected, rtol=1e-12)
    assert (log_z <= 0).all()

    grid = torch.linspace(0.0, 1.0, 20001, dtype=torch.float64)
    for i, j in itertools.product(range(2), range(3)):
        mass = torch.trapezoid(_logistic_log_prob(grid, loc[i, j], scale[i, j]).exp(), grid).item()
        assert mass == pytest.approx(log_z[i, j].exp().item(), rel=1e-5)


def test_logistic_log_normaliser_tiny_mass_is_finite():
    """A logistic with essentially no mass in (0, 1) still yields a finite, very negative log normaliser."""
    out = _logistic_log_normaliser(torch.tensor([[50.0]], dtype=torch.float64), torch.tensor([[0.01]], dtype=torch.float64))
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()
    assert out.item() < np.log(1e-25)


def test_nll_is_mean_over_batch_of_negative_logpdf():
    """The NLL is a scalar equal to minus the batch mean of the per-sample log-density summed over dimensions."""
    u = sample_kuma_logistic_product(*PARAMS, 6, np.random.default_rng(5)).astype(np.float64)
    nll = nll_kuma_logistic_product(torch.tensor(u), *_batch(PARAMS, 6))
    assert nll.shape == () and nll.dtype == torch.float64
    expected = -np.mean([kuma_logistic_logpdf_vec(row, *PARAMS) for row in u])
    assert nll.item() == pytest.approx(expected, rel=1e-10)


def test_nll_batch_of_one_duplicates_and_dimension_sum():
    """A batch of one works, duplicated rows do not change the mean, and dimensions add up."""
    u = np.array([[0.2, 0.7, 0.5, 0.01]])
    single = nll_kuma_logistic_product(torch.tensor(u), *_batch(PARAMS, 1))
    repeated = nll_kuma_logistic_product(torch.tensor(np.repeat(u, 3, axis=0)), *_batch(PARAMS, 3))
    assert single.item() == pytest.approx(repeated.item(), rel=1e-12)
    parts = sum(
        nll_kuma_logistic_product(torch.tensor(u[:, j:j + 1]), *_batch(_dim(j), 1)).item()
        for j in range(4))
    assert single.item() == pytest.approx(parts, rel=1e-12)


def test_nll_unit_kumaraswamy_is_truncated_logistic():
    """With a = b = 1 the Kumaraswamy map is the identity and the NLL is that of the truncated logistic."""
    rng = np.random.default_rng(6)
    u = torch.tensor(rng.uniform(0.01, 0.99, size=(5, 3)))
    loc, scale = _batch(MODERATE[:2], 5)
    ones = torch.ones(5, 3, dtype=torch.float64)
    nll = nll_kuma_logistic_product(u, loc, scale, ones, ones)
    expected = -(_logistic_log_prob(u, loc, scale) - _logistic_log_normaliser(loc, scale)).sum(dim=-1).mean()
    assert nll.item() == pytest.approx(expected.item(), rel=1e-12)


def test_nll_float32_training_path():
    """With float32 network-like outputs the NLL matches float64 and back-propagates finite, non-zero gradients."""
    rng = np.random.default_rng(7)
    u = rng.uniform(0.01, 0.99, size=(16, 4))
    params32 = [p.clone().requires_grad_(True) for p in _batch(PARAMS, 16, dtype=torch.float32)]
    nll32 = nll_kuma_logistic_product(torch.tensor(u, dtype=torch.float32), *params32)
    nll64 = nll_kuma_logistic_product(torch.tensor(u), *_batch(PARAMS, 16))
    assert nll32.dtype == torch.float32
    assert nll32.item() == pytest.approx(nll64.item(), rel=1e-4)
    nll32.backward()
    for p in params32:
        assert torch.isfinite(p.grad).all()
        assert (p.grad != 0).any()


def test_nll_is_lowest_at_true_parameters():
    """Samples from the distribution have a lower NLL under the generating parameters than under perturbed ones."""
    true = (np.array([0.35, 0.6]), np.array([0.1, 0.25]), np.array([1.3, 0.8]), np.array([1.7, 1.2]))
    u = torch.tensor(sample_kuma_logistic_product(*true, 3000, np.random.default_rng(8)).astype(np.float64))
    best = nll_kuma_logistic_product(u, *_batch(true, 3000)).item()
    for k, factor in itertools.product(range(4), [0.7, 1.4]):
        perturbed = list(true)
        perturbed[k] = true[k] * factor
        assert nll_kuma_logistic_product(u, *_batch(perturbed, 3000)).item() > best


def test_sample_shape_dtype_range_and_determinism():
    """Samples have shape (n_samples, n_params), dtype float32, lie in (0, 1), are reproducible and leave inputs untouched."""
    copies = [p.copy() for p in PARAMS]
    s = sample_kuma_logistic_product(*PARAMS, 7, np.random.default_rng(9))
    assert s.shape == (7, 4) and s.dtype == np.float32
    assert np.all((s > 0) & (s < 1))
    assert_array_equal(s, sample_kuma_logistic_product(*PARAMS, 7, np.random.default_rng(9)))
    for p, c in zip(PARAMS, copies):
        assert_array_equal(p, c)
    one = sample_kuma_logistic_product(*_dim(0), 1, np.random.default_rng(9))
    assert one.shape == (1, 1) and one.dtype == np.float32


def test_sample_is_inverse_cdf_of_uniform_draws():
    """The sampler applies the quantile function to the generator's uniform draws."""
    n = 50
    s = sample_kuma_logistic_product(*MODERATE, n, np.random.default_rng(10))
    w = np.random.default_rng(10).uniform(0.0, 1.0, size=(n, 3))
    expected = np.array([kuma_logistic_icdf_vec(row, *MODERATE) for row in w])
    assert_allclose(s, expected.astype(np.float32), rtol=1e-6)


def test_sample_follows_cdf_kolmogorov_smirnov():
    """Samples follow kuma_logistic_cdf according to a Kolmogorov-Smirnov test with a fixed seed."""
    n = 2000
    s = sample_kuma_logistic_product(*MODERATE, n, np.random.default_rng(42)).astype(np.float64)
    for j in range(3):
        x = np.sort(s[:, j])
        cdf = np.array([kuma_logistic_cdf(xi, MODERATE[0][j], MODERATE[1][j], MODERATE[2][j], MODERATE[3][j]) for xi in x])
        stat = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
        # critical value at the 1% significance level
        assert stat < 1.63 / np.sqrt(n)


def test_kuma_logistic_cdf_scalar_matches_vector():
    """The scalar CDF returns a float equal to the element-wise vectorised CDF."""
    for x in [0.0, 0.03, 0.2, 0.5, 0.81, 1.0]:
        vec = kuma_logistic_cdf_vec(np.full(4, x), *PARAMS)
        for j in range(4):
            val = kuma_logistic_cdf(x, LOC[j], SCALE[j], A[j], B[j])
            assert type(val) is float
            assert val == pytest.approx(vec[j], rel=1e-8, abs=1e-15)


def test_cdf_vec_monotone_bounded_with_limits():
    """The CDF has shape (d,), is non-decreasing, stays in [0, 1] and goes from about 0 to about 1."""
    grid = np.linspace(0.0, 1.0, 201)
    vals = np.array([kuma_logistic_cdf_vec(np.full(4, x), *PARAMS) for x in grid])
    assert kuma_logistic_cdf_vec(np.full(4, 0.5), *PARAMS).shape == (4,)
    assert np.all((vals >= 0) & (vals <= 1))
    assert np.all(np.diff(vals, axis=0) >= 0)
    assert np.all(vals[0, :3] < 1e-5)
    assert np.all(vals[-1] > 1 - 1e-5)


def test_cdf_unit_kumaraswamy_is_truncated_logistic():
    """With a = b = 1 the CDF is the truncated logistic CDF."""
    loc, scale = MODERATE[:2]
    ones = np.ones(3)
    p0 = sigmoid(-loc / scale)
    p1 = sigmoid((1 - loc) / scale)
    for x in np.linspace(0.01, 0.99, 25):
        expected = (sigmoid((x - loc) / scale) - p0) / (p1 - p0)
        assert_allclose(kuma_logistic_cdf_vec(np.full(3, x), loc, scale, ones, ones), expected, rtol=1e-10)


def test_cdf_with_flat_base_is_kumaraswamy_quantile():
    """A very wide logistic base is uniform on (0, 1), so P(U <= x) reduces to the Kumaraswamy quantile function."""
    a = np.array([0.5, 1.0, 2.5, 7.0])
    b = np.array([0.7, 1.0, 3.0, 0.4])
    loc = np.full(4, 0.5)
    scale = np.full(4, 1e4)
    for x in np.linspace(0.01, 0.99, 25):
        assert_allclose(kuma_logistic_cdf_vec(np.full(4, x), loc, scale, a, b), kuma_icdf_np(np.full(4, x), a, b), atol=1e-8)


def test_icdf_and_cdf_are_inverse():
    """The quantile function and the CDF invert each other on the interior of the unit interval."""
    for t in np.linspace(0.001, 0.999, 101):
        u = kuma_logistic_icdf_vec(np.full(3, t), *MODERATE)
        assert u.shape == (3,)
        assert_allclose(kuma_logistic_cdf_vec(u, *MODERATE), t, atol=1e-9)
    for x in np.linspace(0.05, 0.95, 19):
        t = kuma_logistic_cdf_vec(np.full(4, x), *PARAMS)
        keep = (t > 1e-6) & (t < 1 - 1e-6)
        assert_allclose(kuma_logistic_icdf_vec(t, *PARAMS)[keep], x, rtol=1e-8)


def test_icdf_bounded_and_monotone():
    """Quantiles increase with the level and stay within [1e-7, 1 - 1e-7], even for levels 0 and 1."""
    levels = [0.0, 1e-12, 0.1, 0.5, 0.9, 1 - 1e-12, 1.0]
    out = np.array([kuma_logistic_icdf_vec(np.full(4, t), *PARAMS) for t in levels])
    assert np.all(np.isfinite(out))
    assert np.all((out >= 1e-7) & (out <= 1 - 1e-7))
    assert np.all(np.diff(out, axis=0) >= 0)


def test_logpdf_is_derivative_of_cdf():
    """exp(logpdf) equals the numerical derivative of the CDF."""
    h = 1e-6
    for j in range(3):
        params = _dim(j)
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            slope = (kuma_logistic_cdf_vec(np.array([x + h]), *params) - kuma_logistic_cdf_vec(np.array([x - h]), *params))[0] / (2 * h)
            assert np.exp(kuma_logistic_logpdf_vec(np.array([x]), *params)) == pytest.approx(slope, rel=1e-6)


def test_logpdf_integral_matches_cdf_difference():
    """Integrating exp(logpdf) between two points gives the CDF difference."""
    x = np.linspace(0.05, 0.95, 2001)
    for j in range(3):
        params = _dim(j)
        dens = np.exp([kuma_logistic_logpdf_vec(np.array([xi]), *params) for xi in x])
        diff = kuma_logistic_cdf_vec(np.array([0.95]), *params)[0] - kuma_logistic_cdf_vec(np.array([0.05]), *params)[0]
        assert _simpson(dens, x) == pytest.approx(diff, rel=1e-7)


def test_logpdf_sums_over_dimensions():
    """The log-density is a float equal to the sum of the one-dimensional log-densities."""
    u = np.array([0.2, 0.7, 0.5, 0.01])
    total = kuma_logistic_logpdf_vec(u, *PARAMS)
    assert type(total) is float
    assert total == pytest.approx(sum(kuma_logistic_logpdf_vec(u[j:j + 1], *_dim(j)) for j in range(4)), rel=1e-12)


def test_logpdf_is_density_of_quantile_map():
    """By change of variables, logpdf at u = icdf(t) equals log|dt/du|, i.e. minus the summed log-derivative of the quantile map."""
    h = 1e-6
    for t in [np.array([0.1, 0.5, 0.9]), np.array([0.4, 0.05, 0.7]), np.array([0.97, 0.3, 0.2])]:
        u = kuma_logistic_icdf_vec(t, *MODERATE)
        dudt = (kuma_logistic_icdf_vec(t + h, *MODERATE) - kuma_logistic_icdf_vec(t - h, *MODERATE)) / (2 * h)
        assert kuma_logistic_logpdf_vec(u, *MODERATE) == pytest.approx(-np.sum(np.log(dudt)), rel=1e-6)


def test_functions_at_unit_interval_boundaries():
    """At u = 0 and u = 1 the CDF is close to 0 and 1, the log-density is finite, and the scalar CDF agrees."""
    zeros = np.zeros(3)
    ones = np.ones(3)
    assert np.all(kuma_logistic_cdf_vec(zeros, *MODERATE) < 1e-5)
    assert np.all(kuma_logistic_cdf_vec(ones, *MODERATE) > 1 - 1e-5)
    assert np.isfinite(kuma_logistic_logpdf_vec(zeros, *MODERATE))
    assert np.isfinite(kuma_logistic_logpdf_vec(ones, *MODERATE))
    assert kuma_logistic_cdf(0.0, 0.3, 0.08, 1.5, 2.0) < 1e-5
    assert kuma_logistic_cdf(1.0, 0.3, 0.08, 1.5, 2.0) > 1 - 1e-5


@pytest.mark.parametrize("loc,scale,a,b", NETWORK_CORNERS)
def test_extreme_network_parameters_stay_finite(loc, scale, a, b):
    """At the corners of the NPE output ranges all functions stay finite and consistent with each other."""
    pts = np.array([0.0, 1e-9, 1e-7, 0.2, 0.5, 0.9, 1 - 1e-7, 1.0])
    params = tuple(np.full(len(pts), v) for v in (loc, scale, a, b))
    one_dim = tuple(p[:1] for p in params)
    cdf = kuma_logistic_cdf_vec(pts, *params)
    assert np.all((cdf >= 0) & (cdf <= 1)) and np.all(np.diff(cdf) >= 0)
    assert_allclose([kuma_logistic_cdf(x, loc, scale, a, b) for x in pts], cdf, rtol=1e-8, atol=0)
    quantiles = kuma_logistic_icdf_vec(pts, *params)
    assert np.all((quantiles >= 1e-7) & (quantiles <= 1 - 1e-7)) and np.all(np.diff(quantiles) >= 0)
    assert all(np.isfinite(kuma_logistic_logpdf_vec(pts[i:i + 1], *one_dim)) for i in range(len(pts)))

    # float32 training path: finite loss for boundary points, finite gradients for interior points
    nll = nll_kuma_logistic_product(
        torch.tensor(pts[:, None], dtype=torch.float32),
        *(torch.full((len(pts), 1), v, dtype=torch.float32) for v in (loc, scale, a, b)))
    assert torch.isfinite(nll)
    net_out = [torch.full((3, 1), v, dtype=torch.float32, requires_grad=True) for v in (loc, scale, a, b)]
    nll_kuma_logistic_product(torch.tensor([[0.2], [0.5], [0.9]], dtype=torch.float32), *net_out).backward()
    assert all(torch.isfinite(p.grad).all() for p in net_out)

    # the sampler agrees with the quantile function up to its wider [0, 1] clipping
    s = sample_kuma_logistic_product(*one_dim, 20, np.random.default_rng(11))
    assert s.shape == (20, 1) and np.all((s >= 0) & (s <= 1))
    w = np.random.default_rng(11).uniform(0.0, 1.0, size=(20, 1))
    assert_allclose(s, [kuma_logistic_icdf_vec(row, *one_dim) for row in w], rtol=1e-6, atol=2e-7)
