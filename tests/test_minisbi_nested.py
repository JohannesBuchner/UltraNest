"""Tests for the nested sampling helpers of the minisbi simulation-based inference module."""
import json

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402
from scipy import integrate, stats  # noqa: E402

from ultranest.simbase.minisbi.logistic import nll_kuma_logistic_product, sample_kuma_logistic_product  # noqa: E402
from ultranest.simbase.minisbi.nested import KLPTransform, get_distribution_parameters  # noqa: E402

# (loc, scale, a, b) for one dimension; all of them put negligible mass
# within 1e-7 of the unit interval edges, where the implementation clips.
PARAMS_1D = [
    (0.5, 0.1, 1.0, 1.0),     # plain truncated logistic
    (0.3, 0.05, 2.0, 0.7),
    (0.65, 0.2, 0.8, 1.5),
    (0.1, 1.0, 1.5, 1.5),     # wide
    (0.5, 1e-3, 1.0, 1.0),    # very narrow
]

# loc, scale, a, b for a three-dimensional transform
PARAMS_3D = dict(
    loc=[0.5, 0.3, 0.65],
    scale=[0.1, 0.05, 0.2],
    a=[1.0, 2.0, 0.8],
    b=[1.0, 0.7, 1.5],
)


def _klp1(params):
    """Build a one-dimensional transform from a (loc, scale, a, b) tuple."""
    loc, scale, a, b = params
    return KLPTransform([loc], [scale], [a], [b])


def _transform1(klp, t):
    """Apply a one-dimensional transform to each point of ``t``."""
    return np.array([klp.transform(np.array([ti]))[0] for ti in t])


def _cdf1(klp, u):
    """Evaluate a one-dimensional CDF at each point of ``u``."""
    return np.array([klp.cdf(np.array([ui]))[0] for ui in u])


def _pdf1(klp, u):
    """Evaluate a one-dimensional density at a scalar point ``u``."""
    return np.exp(klp.logpdf(np.array([u])))


class _TinyKLPNet(torch.nn.Module):
    """Single linear layer returning ``(loc, scale, a, b)`` and recording its inputs."""

    def __init__(self, n_data, n_params):
        """Initialise with fixed random weights."""
        super().__init__()
        torch.manual_seed(42)
        self.n_params = n_params
        self.linear = torch.nn.Linear(n_data, 4 * n_params)
        self.calls = []

    def forward(self, x):
        """Map ``x`` of shape (batch, n_data) to four (batch, n_params) tensors."""
        self.calls.append((x.detach().clone(), torch.is_grad_enabled()))
        out = self.linear(x)
        n = self.n_params
        loc = torch.sigmoid(out[:, :n])
        scale = torch.exp(out[:, n:2 * n])
        a = torch.exp(out[:, 2 * n:3 * n])
        b = torch.exp(out[:, 3 * n:])
        return loc, scale, a, b


def test_get_distribution_parameters_matches_model_output():
    """Return one list of Python floats per parameter, equal to the network output for the observed data."""
    model = _TinyKLPNet(n_data=5, n_params=3)
    observed = np.random.default_rng(1).normal(size=5)

    params = get_distribution_parameters(model, observed)

    assert sorted(params) == ['a', 'b', 'loc', 'scale']
    with torch.no_grad():
        expected = model(torch.tensor(observed, dtype=torch.float32)[None, :])
    for key, value in zip(['loc', 'scale', 'a', 'b'], expected):
        assert isinstance(params[key], list)
        assert len(params[key]) == 3
        assert all(isinstance(v, float) for v in params[key])
        assert_array_equal(params[key], value[0].numpy())


def test_get_distribution_parameters_input_contract():
    """Call the model once, without gradients, on a float32 tensor of shape (1, n_data)."""
    model = _TinyKLPNet(n_data=4, n_params=2)
    observed = [0.25, -1.5, 3.0, 1e-3]

    get_distribution_parameters(model, observed)

    assert len(model.calls) == 1
    x, grad_enabled = model.calls[0]
    assert not grad_enabled
    assert x.dtype == torch.float32
    assert tuple(x.shape) == (1, 4)
    assert_array_equal(x[0].numpy(), np.array(observed, dtype=np.float32))


def test_get_distribution_parameters_single_parameter_and_datum():
    """Keep the parameter axis when the model has one parameter and the data one value."""
    model = _TinyKLPNet(n_data=1, n_params=1)

    params = get_distribution_parameters(model, np.array([0.7]))

    for key in ['loc', 'scale', 'a', 'b']:
        assert isinstance(params[key], list)
        assert len(params[key]) == 1
        assert np.isfinite(params[key][0])


def test_get_distribution_parameters_json_roundtrip_feeds_klptransform():
    """Store the parameters as JSON and rebuild an identical KLPTransform, as the tutorial does."""
    model = _TinyKLPNet(n_data=3, n_params=2)
    params = get_distribution_parameters(model, np.array([0.1, 0.2, -0.3]))

    reloaded = json.loads(json.dumps(params))
    assert reloaded == params

    klp = KLPTransform(**params)
    klp_reloaded = KLPTransform(**reloaded)
    t = np.array([0.2, 0.9])
    assert_array_equal(klp.transform(t), klp_reloaded.transform(t))
    assert klp.log_jacobian(t) == klp_reloaded.log_jacobian(t)


def test_get_distribution_parameters_with_npe_network():
    """Produce valid KLP parameters from an untrained NPENetwork."""
    pytest.importorskip("joblib")
    pytest.importorskip("torchinfo")
    from ultranest.simbase.minisbi.npe import NPENetwork

    torch.manual_seed(3)
    model = NPENetwork(n_data=6, n_params=2, depth=1, width=8, activation_name='ReLU', layer_shape='rectangular')
    model.eval()
    observed = np.random.default_rng(5).normal(size=6)

    params = get_distribution_parameters(model, observed)

    assert all(len(params[key]) == 2 for key in ['loc', 'scale', 'a', 'b'])
    assert all(0 < v < 1 for v in params['loc'])
    assert all(v > 0 for key in ['scale', 'a', 'b'] for v in params[key])
    klp = KLPTransform(**params)
    for t in [np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([0.99, 0.01])]:
        u = klp.transform(t)
        assert np.all((u > 0) & (u < 1))
        assert np.isfinite(klp.log_jacobian(t))


def test_klptransform_shapes_and_dtypes():
    """Return float32 coordinates of shape (n_params,), CDF values of the same shape that map back to t, and scalar log-densities."""
    klp = KLPTransform(**PARAMS_3D)
    t = np.array([0.1, 0.5, 0.95])
    t_orig = t.copy()

    u = klp.transform(t)
    assert u.shape == (3,)
    assert u.dtype == np.float32
    assert_array_equal(t, t_orig)

    t_back = klp.cdf(u.astype(float))
    assert t_back.shape == (3,)
    assert_allclose(t_back, t, atol=1e-5)

    assert isinstance(klp.logpdf(u), float)
    assert isinstance(klp.log_jacobian(t), float)


@pytest.mark.parametrize("params", PARAMS_1D)
def test_transform_is_monotone_inverse_of_cdf(params):
    """Map t monotonically to u such that cdf(transform(t)) recovers t."""
    klp = _klp1(params)
    t = np.linspace(1e-3, 1 - 1e-3, 999)

    u = _transform1(klp, t)

    assert np.all((u > 0) & (u < 1))
    assert np.all(np.diff(u) >= 0)
    assert u[-1] > u[0]
    assert_allclose(_cdf1(klp, u.astype(float)), t, atol=1e-4)


@pytest.mark.parametrize("params", PARAMS_1D)
def test_cdf_is_monotone_and_transform_inverts_it(params):
    """Keep the CDF monotone in [0, 1] with limits 0 and 1, and recover u from transform(cdf(u)) in the bulk."""
    klp = _klp1(params)
    u = np.linspace(0, 1, 2001)

    cdf = _cdf1(klp, u)

    assert np.all((cdf >= 0) & (cdf <= 1))
    assert np.all(np.diff(cdf) >= 0)
    assert_allclose(cdf[[0, -1]], [0, 1], atol=1e-4)

    bulk = (cdf > 1e-3) & (cdf < 1 - 1e-3)
    assert bulk.sum() >= 3
    assert_allclose(_transform1(klp, cdf[bulk]), u[bulk], atol=1e-6)


@pytest.mark.parametrize("params", PARAMS_1D)
def test_logpdf_is_derivative_of_cdf(params):
    """Match exp(logpdf(u)) with a central finite difference of the CDF."""
    klp = _klp1(params)
    h = 1e-7
    for q in [0.1, 0.5, 0.9]:
        u = float(klp.transform(np.array([q]))[0])
        fd = (klp.cdf(np.array([u + h]))[0] - klp.cdf(np.array([u - h]))[0]) / (2 * h)
        assert_allclose(_pdf1(klp, u), fd, rtol=1e-6)


@pytest.mark.parametrize("params", PARAMS_1D)
def test_pdf_integrates_to_cdf_differences(params):
    """Integrate the density over intervals to the CDF difference, and over (0, 1) to one."""
    klp = _klp1(params)
    knots = [float(klp.transform(np.array([q]))[0]) for q in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]]

    total, _ = integrate.quad(lambda x: _pdf1(klp, x), 0, 1, points=knots, limit=500)
    assert_allclose(total, 1, atol=5e-5)

    for lo, hi in [(knots[1], knots[3]), (knots[2], knots[5]), (knots[3], knots[4])]:
        mass, _ = integrate.quad(lambda x: _pdf1(klp, x), lo, hi, limit=200)
        assert_allclose(mass, klp.cdf(np.array([hi]))[0] - klp.cdf(np.array([lo]))[0], rtol=1e-6)


def test_dimensions_are_independent():
    """Treat each dimension separately: transform and cdf per axis, logpdf and log_jacobian summed over axes."""
    klp = KLPTransform(**PARAMS_3D)
    marginals = [_klp1(p) for p in zip(*PARAMS_3D.values())]
    t = np.array([0.05, 0.6, 0.97])
    u = klp.transform(t)

    assert_array_equal(u, [m.transform(t[i:i + 1])[0] for i, m in enumerate(marginals)])
    assert_allclose(klp.cdf(u.astype(float)), [m.cdf(u[i:i + 1].astype(float))[0] for i, m in enumerate(marginals)])
    assert_allclose(klp.logpdf(u), sum(m.logpdf(u[i:i + 1]) for i, m in enumerate(marginals)))
    assert_allclose(klp.log_jacobian(t), sum(m.log_jacobian(t[i:i + 1]) for i, m in enumerate(marginals)))


def test_logpdf_is_minus_log_derivative_of_transform():
    """The density at u = transform(t) equals dt/du, i.e. minus the log of the transform's derivative (central differences)."""
    klp = KLPTransform(**PARAMS_3D)
    h = 1e-3
    for t in [np.array([0.2, 0.5, 0.8]), np.array([0.6, 0.1, 0.4])]:
        dudt = (klp.transform(t + h).astype(float) - klp.transform(t - h).astype(float)) / (2 * h)
        assert np.all(dudt > 0)
        assert_allclose(klp.logpdf(klp.transform(t)), -np.sum(np.log(dudt)), atol=1e-3)


def test_dividing_by_density_recovers_uniform_prior_integrals():
    """Reweight t-space by 1 / q(transform(t)) to recover integrals under the original uniform prior on u.

    With u = transform(t), weighting the likelihood by exp(-logpdf(u)) gives the same evidence
    and posterior as the likelihood under a flat prior on u.
    """
    klp = _klp1((0.4, 0.15, 1.3, 1.6))
    n = 4000
    t = (np.arange(n) + 0.5) / n
    u = _transform1(klp, t).astype(float)
    weights = np.exp([-klp.logpdf(np.array([ui])) for ui in u])

    # flat likelihood: evidence equals the prior volume
    assert_allclose(weights.mean(), 1, rtol=1e-4)

    # Gaussian likelihood in u: compare evidence and posterior mean with the direct integral
    mu, sigma = 0.6, 0.05
    like = np.exp(-0.5 * ((u - mu) / sigma) ** 2)
    expected_z = np.sqrt(2 * np.pi) * sigma * (stats.norm.cdf((1 - mu) / sigma) - stats.norm.cdf(-mu / sigma))
    assert_allclose(np.mean(like * weights), expected_z, rtol=1e-5)
    assert_allclose(np.sum(u * like * weights) / np.sum(like * weights), mu, atol=1e-6)


def test_boundaries_and_extreme_parameters_stay_finite():
    """Keep transform inside (0, 1) and log_jacobian finite at t = 0 and 1, across the parameter range an NPENetwork can output."""
    for loc in [1e-6, 0.5, 1 - 1e-6]:
        for scale in [np.exp(-8), np.exp(4)]:
            for a in [np.exp(-4), np.exp(4)]:
                for b in [np.exp(-4), np.exp(4)]:
                    klp = KLPTransform([loc], [scale], [a], [b])
                    for t in [0.0, 1e-12, 0.5, 1 - 1e-12, 1.0]:
                        u = klp.transform(np.array([t]))
                        assert 0 < u[0] < 1, (loc, scale, a, b, t, u)
                        assert np.isfinite(klp.log_jacobian(np.array([t]))), (loc, scale, a, b, t)
                        assert 0 <= klp.cdf(u.astype(float))[0] <= 1


@pytest.mark.parametrize("params", PARAMS_1D)
def test_matches_closed_form_reference(params):
    """Agree with the generative model v ~ Logistic(loc, scale) truncated to (0, 1), u = 1 - (1 - v^a)^b, written with scipy.stats."""
    loc, scale, a, b = params
    klp = _klp1(params)
    logistic = stats.logistic(loc, scale)
    lo, mass = logistic.cdf(0), logistic.cdf(1) - logistic.cdf(0)
    grid = np.linspace(1e-3, 1 - 1e-3, 999)

    # quantile function: invert the truncated logistic, then apply the Kumaraswamy CDF (float32 output)
    v = logistic.ppf(lo + grid * mass)
    assert_allclose(_transform1(klp, grid), 1 - (1 - v ** a) ** b, rtol=1e-6)

    # CDF and density: map u back with the Kumaraswamy quantile function
    v = (1 - (1 - grid) ** (1 / b)) ** (1 / a)
    assert_allclose(_cdf1(klp, grid), (logistic.cdf(v) - lo) / mass, rtol=1e-9, atol=1e-12)
    log_kuma = np.log(a * b) + (a - 1) * np.log(v) + (b - 1) * np.log1p(-v ** a)
    assert_allclose([klp.logpdf(np.array([x])) for x in grid], logistic.logpdf(v) - np.log(mass) - log_kuma, rtol=1e-9, atol=1e-9)


def test_cdf_matches_npe_sampler():
    """Agree, dimension by dimension, with the sampler used to draw NPE posterior samples (KS test)."""
    arrays = {key: np.array(value) for key, value in PARAMS_3D.items()}
    samples = sample_kuma_logistic_product(n_samples=2000, rng=np.random.default_rng(99), **arrays)
    assert samples.shape == (2000, 3)

    for i, params in enumerate(zip(*PARAMS_3D.values())):
        klp = _klp1(params)
        result = stats.kstest(samples[:, i].astype(float), lambda x: _cdf1(klp, x))
        assert result.pvalue > 0.01, (i, result)


@pytest.mark.parametrize("batch", [1, 4])
def test_logpdf_agrees_with_torch_training_loss(batch):
    """Equal minus the batch mean of logpdf to the torch NLL used to train the network."""
    klp = KLPTransform(**PARAMS_3D)
    u = np.random.default_rng(batch).uniform(0.05, 0.95, size=(batch, 3))
    tensors = [torch.tensor(np.tile(value, (batch, 1)), dtype=torch.float64) for value in PARAMS_3D.values()]

    nll = nll_kuma_logistic_product(torch.tensor(u, dtype=torch.float64), *tensors)

    assert_allclose(nll.item(), -np.mean([klp.logpdf(row) for row in u]), rtol=1e-9)
