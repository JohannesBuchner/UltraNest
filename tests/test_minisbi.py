"""Unit tests for the simulation-based inference helpers (ultranest.simbase.minisbi)."""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import scipy.stats
import torch
import torch.nn as nn
from numpy.testing import assert_allclose

from ultranest.simbase.minisbi import logistic, nested, norm, npe, plot, utils

NOISE_STD = 0.1
# linear mapping from 2 model parameters to 3 data dimensions
A = np.array([[1.0, 0.2], [0.0, 1.0], [0.3, -0.2]])


def generate_mean_and_noise(idx, seed, theta):
	return {'mean': A @ np.asarray(theta, dtype=np.float64)}


def inject_noise(props, rng):
	return np.asarray(props['mean']) + rng.normal(0, NOISE_STD, size=A.shape[0])


def generate_batch(batch_idx, n_sim, seed, n_params):
	rng = np.random.default_rng(seed + batch_idx)
	u_samples = np.stack([utils.sample_prior_u(rng, n_params) for _ in range(n_sim)])
	mean_props = [
		generate_mean_and_noise(idx=batch_idx * n_sim + i, seed=seed, theta=u_samples[i])
		for i in range(n_sim)
	]
	return {'u_samples': u_samples, 'mean_props': mean_props}


def prior_transform(u):
	return u


class FixedParamsModel(nn.Module):
	"""Mock NPE network returning fixed Kumaraswamy-Logistic parameters."""

	def __init__(self, loc, scale, a, b):
		super().__init__()
		for name, value in [('loc', loc), ('scale', scale), ('a', a), ('b', b)]:
			self.register_buffer(name, torch.tensor(np.asarray(value, dtype=np.float32)))

	def forward(self, x):
		n = x.shape[0]
		return (self.loc.unsqueeze(0).expand(n, -1), self.scale.unsqueeze(0).expand(n, -1),
		        self.a.unsqueeze(0).expand(n, -1), self.b.unsqueeze(0).expand(n, -1))


# ---------------------------------------------------------------------------
# logistic.py
# ---------------------------------------------------------------------------

def test_sigmoid():
	assert logistic.sigmoid(0.0) == 0.5
	# monotonic (saturates to 0/1 in float64 beyond |x| ~ 37)
	x = np.linspace(-50, 50, 1001)
	s = logistic.sigmoid(x)
	assert (np.diff(s) >= 0).all()
	assert (np.diff(logistic.sigmoid(np.linspace(-30, 30, 1001))) > 0).all()
	# extreme values do not overflow
	s = logistic.sigmoid(np.array([-1e10, 1e10]))
	assert np.isfinite(s).all()
	assert_allclose(s, [0.0, 1.0], atol=1e-10)


def test_logit():
	assert logistic.logit(0.5) == 0.0
	# roundtrip logit(sigmoid(x)) == x
	x = np.linspace(-20, 20, 41)
	assert_allclose(logistic.logit(logistic.sigmoid(x)), x, atol=1e-6)


def test_kuma_icdf_identity():
	# Kumaraswamy(1, 1) is uniform, so the icdf is the identity
	u = np.linspace(0.01, 0.99, 50)
	assert_allclose(logistic.kuma_icdf_np(u, 1.0, 1.0), u, atol=1e-6)
	tu = torch.tensor(u)
	assert_allclose(logistic.kuma_icdf(tu, torch.tensor(1.0), torch.tensor(1.0)).numpy(), u, atol=1e-6)


def test_kuma_icdf_roundtrip():
	# icdf inverts the Kumaraswamy CDF u = 1 - (1 - v^a)^b
	v = np.linspace(0.01, 0.99, 50)
	a, b = 2.0, 3.0
	u = 1.0 - (1.0 - v ** a) ** b
	assert_allclose(logistic.kuma_icdf_np(u, a, b), v, atol=1e-6)


def test_sample_kuma_logistic_product():
	rng = np.random.default_rng(42)
	loc = np.array([0.4, 0.6])
	scale = np.array([0.25, 0.2])
	a = np.array([2.0, 1.5])
	b = np.array([3.0, 0.8])
	n = 40000
	samples = logistic.sample_kuma_logistic_product(loc, scale, a, b, n, rng)
	assert samples.shape == (n, 2)
	assert samples.dtype == np.float32
	# exactly 0 or 1 can occur through floating point rounding
	assert (samples >= 0).all() and (samples <= 1).all()
	# empirical CDF agrees with the analytic Kumaraswamy-Logistic CDF
	u_test = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]])
	cdf = logistic.kuma_logistic_cdf_vec(u_test, loc, scale, a, b)
	for i in range(len(u_test)):
		empirical = (samples < u_test[i]).mean(axis=0)
		assert_allclose(empirical, cdf[i], atol=0.02)


def test_sample_kuma_logistic_product_deterministic():
	loc = np.array([0.5]); scale = np.array([0.2]); a = np.array([2.0]); b = np.array([2.0])
	s1 = logistic.sample_kuma_logistic_product(loc, scale, a, b, 10, np.random.default_rng(7))
	s2 = logistic.sample_kuma_logistic_product(loc, scale, a, b, 10, np.random.default_rng(7))
	assert_allclose(s1, s2)


def test_kuma_logistic_cdf_trunclogistic():
	# for a = b = 1 the Kumaraswamy step is the identity, so the CDF
	# reduces to the truncated logistic CDF on (0, 1)
	loc, scale = 0.3, 0.25
	u = np.array([0.05, 0.3, 0.5, 0.8, 0.95])
	cdf = logistic.kuma_logistic_cdf_vec(u, np.array([loc]), np.array([scale]), np.array([1.0]), np.array([1.0]))
	p0 = scipy.stats.logistic.cdf(0, loc, scale)
	p1 = scipy.stats.logistic.cdf(1, loc, scale)
	expected = (scipy.stats.logistic.cdf(u, loc, scale) - p0) / (p1 - p0)
	assert_allclose(cdf, expected, atol=1e-6)


def test_kuma_logistic_cdf_monotonic_bounded():
	loc = np.array([0.5, 0.2]); scale = np.array([0.3, 0.1])
	a = np.array([2.0, 0.7]); b = np.array([3.0, 1.4])
	grid = np.linspace(0.0, 1.0, 101)
	# evaluate per dimension: the helpers are element-wise over equal-length vectors
	cdf = np.column_stack([
		logistic.kuma_logistic_cdf_vec(grid, loc[i:i+1], scale[i:i+1], a[i:i+1], b[i:i+1])
		for i in range(2)
	])
	assert (np.diff(cdf, axis=0) >= 0).all()
	assert (cdf >= 0).all() and (cdf <= 1).all()
	# scalar and vector versions agree
	for uk in [0.1, 0.5, 0.9]:
		scalar = logistic.kuma_logistic_cdf(uk, loc[0], scale[0], a[0], b[0])
		assert_allclose(scalar, cdf[int(round(uk * 100))][0], atol=1e-8)


def test_kuma_logistic_cdf_icdf_roundtrip():
	loc = np.array([0.4, 0.6]); scale = np.array([0.2, 0.3])
	a = np.array([2.0, 1.5]); b = np.array([3.0, 0.8])
	u = np.array([0.1, 0.9])
	t = logistic.kuma_logistic_cdf_vec(u, loc, scale, a, b)
	u2 = logistic.kuma_logistic_icdf_vec(t, loc, scale, a, b)
	assert_allclose(u2, u, atol=1e-5)
	t2 = logistic.kuma_logistic_cdf_vec(u2, loc, scale, a, b)
	assert_allclose(t2, t, atol=1e-5)


def test_kuma_logistic_logpdf_normalised():
	# the density integrates to 1 in each dimension
	# (a, b < 1 give a pdf bounded at both cube edges; larger values
	# create integrable singularities where uniform grids converge slowly)
	loc, scale, a, b = 0.4, 0.3, 0.5, 0.5
	grid = np.linspace(1e-6, 1.0 - 1e-6, 20001)
	logpdf = np.array([
		logistic.kuma_logistic_logpdf_vec(np.array([u]), np.array([loc]), np.array([scale]),
		                                  np.array([a]), np.array([b]))
		for u in grid
	])
	assert np.isfinite(logpdf).all()
	assert_allclose(np.exp(logpdf).mean(), 1.0, atol=1e-3)
	# numerical derivative of the CDF matches the density
	h = 1e-4
	u0 = 0.5
	dcdf = (logistic.kuma_logistic_cdf(u0 + h, loc, scale, a, b)
	        - logistic.kuma_logistic_cdf(u0 - h, loc, scale, a, b)) / (2 * h)
	pdf = np.exp(logpdf[np.searchsorted(grid, u0)])
	assert_allclose(dcdf, pdf, rtol=1e-3)


def test_nll_kuma_logistic_product_trunclogistic():
	# for a = b = 1, the NLL per dimension is the truncated logistic NLL
	u = torch.tensor([[0.2, 0.7]])
	loc = torch.tensor([[0.3, 0.6]])
	scale = torch.tensor([[0.25, 0.4]])
	a = torch.ones_like(loc)
	b = torch.ones_like(loc)
	nll = logistic.nll_kuma_logistic_product(u, loc, scale, a, b)
	p0 = scipy.stats.logistic.cdf(0, 0.3, 0.25)
	p1 = scipy.stats.logistic.cdf(1, 0.3, 0.25)
	expected0 = -np.log(scipy.stats.logistic.pdf(0.2, 0.3, 0.25) / (p1 - p0))
	p0 = scipy.stats.logistic.cdf(0, 0.6, 0.4)
	p1 = scipy.stats.logistic.cdf(1, 0.6, 0.4)
	expected1 = -np.log(scipy.stats.logistic.pdf(0.7, 0.6, 0.4) / (p1 - p0))
	assert_allclose(nll.item(), expected0 + expected1, rtol=1e-5)


def test_nll_matches_numpy_logpdf():
	# the torch NLL (sum over dims, mean over batch) and the numpy
	# log-density must agree for a single sample
	u = np.array([0.35, 0.8])
	loc = np.array([0.4, 0.6]); scale = np.array([0.25, 0.2])
	a = np.array([2.0, 1.5]); b = np.array([3.0, 0.8])
	nll = logistic.nll_kuma_logistic_product(
		torch.tensor(u[None]), torch.tensor(loc[None]), torch.tensor(scale[None]),
		torch.tensor(a[None]), torch.tensor(b[None]))
	logpdf = logistic.kuma_logistic_logpdf_vec(u, loc, scale, a, b)
	assert_allclose(nll.item(), -logpdf, rtol=1e-4)


def test_nll_gradients():
	# autograd gradients match finite differences
	u = torch.tensor([[0.3, 0.6]], dtype=torch.float64)
	loc = torch.tensor([[0.4, 0.5]], dtype=torch.float64, requires_grad=True)
	scale = torch.tensor([[0.3, 0.2]], dtype=torch.float64, requires_grad=True)
	a = torch.tensor([[2.0, 1.5]], dtype=torch.float64, requires_grad=True)
	b = torch.tensor([[3.0, 0.8]], dtype=torch.float64, requires_grad=True)
	loss = logistic.nll_kuma_logistic_product(u, loc, scale, a, b)
	grads = torch.autograd.grad(loss, [loc, scale, a, b])
	for g in grads:
		assert torch.isfinite(g).all()

	def nll_of(values):
		loss = logistic.nll_kuma_logistic_product(
			u,
			torch.tensor(values[None, :2]),
			torch.tensor(values[None, 2:4]),
			torch.tensor(values[None, 4:6]),
			torch.tensor(values[None, 6:8]))
		return loss.item()

	values = np.array([0.4, 0.5, 0.3, 0.2, 2.0, 1.5, 3.0, 0.8])
	h = 1e-6
	num = np.zeros(8)
	for i in range(8):
		vp = values.copy(); vp[i] += h
		vm = values.copy(); vm[i] -= h
		num[i] = (nll_of(vp) - nll_of(vm)) / (2 * h)
	analytic = np.concatenate([g.detach().numpy().flatten() for g in grads])
	assert_allclose(analytic, num, rtol=1e-4, atol=1e-6)


def test_nll_penalises_wrong_parameters():
	# samples from the true distribution fit the true parameters better
	rng = np.random.default_rng(3)
	loc = np.array([0.4, 0.6]); scale = np.array([0.25, 0.2])
	a = np.array([2.0, 1.5]); b = np.array([3.0, 0.8])
	u = logistic.sample_kuma_logistic_product(loc, scale, a, b, 2000, rng)
	loc_t = torch.tensor(loc[None]); scale_t = torch.tensor(scale[None])
	a_t = torch.tensor(a[None]); b_t = torch.tensor(b[None])
	u_t = torch.tensor(u)
	nll_true = logistic.nll_kuma_logistic_product(u_t, loc_t, scale_t, a_t, b_t).item()
	nll_wrong = logistic.nll_kuma_logistic_product(
		u_t, loc_t + 0.5, scale_t * 2.0, a_t * 3.0, b_t / 2.0).item()
	assert nll_wrong > nll_true


# ---------------------------------------------------------------------------
# norm.py
# ---------------------------------------------------------------------------

def test_zscorenorm_fit_and_transform():
	layer = norm.ZScoreNorm(3)
	# unfitted layer returns input unchanged
	x = torch.randn(5, 3)
	assert torch.equal(layer(x), x)

	rng = np.random.default_rng(4)
	data = rng.normal(10.0, 2.0, size=(1000, 3))
	layer.fit(data)
	assert layer.fitted
	assert_allclose(layer.mean_t.numpy(), data.mean(axis=0), atol=1e-4)
	assert_allclose(layer.std_t.numpy(), data.std(axis=0), atol=1e-4)

	transformed = layer(torch.tensor(data, dtype=torch.float32)).numpy()
	assert transformed.shape == data.shape
	assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-3)
	assert_allclose(transformed.std(axis=0), 1.0, atol=1e-2)


def test_zscorenorm_constant_feature():
	layer = norm.ZScoreNorm(2)
	data = np.column_stack([np.linspace(0, 1, 100), np.full(100, 5.0)])
	layer.fit(data)
	# zero-variance feature keeps std = 1, so it is only centred
	assert_allclose(layer.std_t.numpy()[1], 1.0)
	transformed = layer(torch.tensor(data, dtype=torch.float32)).numpy()
	assert_allclose(transformed[:, 1], 0.0, atol=1e-6)


def test_zscorenorm_wrong_features():
	layer = norm.ZScoreNorm(3)
	with pytest.raises(AssertionError):
		layer.fit(np.zeros((10, 4)))


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------

def test_sample_prior_u():
	rng = np.random.default_rng(1)
	u = utils.sample_prior_u(rng, 4)
	assert u.shape == (4,)
	assert u.dtype == np.float32
	assert (u >= 0.0).all() and (u < 1.0).all()


def test_cached_generate(tmp_path):
	memory = utils.make_memory(str(tmp_path))
	calls = []

	def gen(idx, seed, theta):
		calls.append(idx)
		return np.asarray(theta) * 2.0

	generate = utils.make_cached_generate(memory, prior_transform, gen)
	b1 = generate(batch_idx=0, n_sim=4, seed=1, n_params=2)
	assert b1['u_samples'].shape == (4, 2)
	assert b1['u_samples'].dtype == np.float32
	assert len(b1['mean_props']) == 4
	assert len(calls) == 4
	# same arguments are served from disk cache
	b1b = generate(batch_idx=0, n_sim=4, seed=1, n_params=2)
	assert len(calls) == 4
	assert_allclose(b1['u_samples'], b1b['u_samples'])
	assert_allclose(np.array(b1['mean_props'], dtype=np.float64),
	                np.array(b1b['mean_props'], dtype=np.float64))
	# different batch index recomputes with independent draws
	b2 = generate(batch_idx=1, n_sim=4, seed=1, n_params=2)
	assert len(calls) == 8
	assert not np.allclose(b1['u_samples'], b2['u_samples'])


def test_inject_noise_batch():
	n_sim = 3
	batch = {
		'u_samples': np.random.default_rng(5).uniform(size=(n_sim, 2)).astype(np.float32),
		'mean_props': [{'mean': np.array([1.0, 2.0, 3.0])}] * n_sim,
	}
	u_samples, raw_data = utils.inject_noise_batch(batch, np.random.default_rng(6), inject_noise)
	assert_allclose(u_samples, batch['u_samples'])
	assert raw_data.shape == (n_sim, 3)
	assert raw_data.dtype == np.float32
	# rows match fresh noise draws with the same rng state
	rng = np.random.default_rng(6)
	for i in range(n_sim):
		expected = batch['mean_props'][i]['mean'] + rng.normal(0, NOISE_STD, size=3)
		assert_allclose(raw_data[i], expected, atol=1e-6)


def test_random_derangement():
	for n in [0, 2, 3, 50]:
		perm = utils.random_derangement(n)
		assert perm.shape == (n,)
		assert sorted(perm.tolist()) == list(range(n))
		if n > 0:
			assert not torch.any(perm == torch.arange(n))


# ---------------------------------------------------------------------------
# nested.py
# ---------------------------------------------------------------------------

MOCK_PARAMS = dict(
	loc=[0.4, 0.6], scale=[0.2, 0.3], a=[2.0, 1.5], b=[3.0, 0.8],
)


def test_get_distribution_parameters():
	model = FixedParamsModel(**MOCK_PARAMS)
	params = nested.get_distribution_parameters(model, np.zeros(5))
	assert sorted(params.keys()) == ['a', 'b', 'loc', 'scale']
	for key, value in MOCK_PARAMS.items():
		assert len(params[key]) == 2
		assert_allclose(params[key], value, rtol=1e-6)
		assert all(isinstance(v, float) for v in params[key])


def test_klp_transform_roundtrip():
	t = KLP = nested.KLPTransform(**MOCK_PARAMS)
	# transform maps the nested sampler cube to the prior cube, in float32
	u = t.transform(np.array([0.2, 0.8]))
	assert u.dtype == np.float32
	assert (u > 0).all() and (u < 1).all()
	# cdf(transform(t)) == t and transform(cdf(u)) == u
	t_back = t.cdf(u)
	assert_allclose(t_back, np.array([0.2, 0.8]), atol=1e-5)
	u_back = t.transform(t.cdf(u))
	assert_allclose(u_back, u, atol=1e-5)


def test_klp_transform_jacobian_normalised():
	# (a, b < 1 keep the pdf bounded; see test_kuma_logistic_logpdf_normalised)
	t = nested.KLPTransform(loc=[0.4], scale=[0.3], a=[0.5], b=[0.5])
	# log_jacobian(t) is the KLP log-density at u = transform(t)
	u = t.transform(np.array([0.5]))
	assert_allclose(t.log_jacobian(np.array([0.5])), t.logpdf(u), rtol=1e-6)
	# the density exp(logpdf) integrates to 1 over the unit cube and is
	# the derivative of the CDF
	grid = np.linspace(1e-6, 1.0 - 1e-6, 20001)
	logpdf = np.array([t.logpdf(np.array([x])) for x in grid])
	assert np.isfinite(logpdf).all()
	assert_allclose(np.exp(logpdf).mean(), 1.0, atol=1e-3)
	h = 1e-5
	u0 = 0.5
	dcdf = (t.cdf(np.array([u0 + h])) - t.cdf(np.array([u0 - h]))) / (2 * h)
	assert_allclose(dcdf, np.exp(t.logpdf(np.array([u0]))), rtol=1e-3)


def test_klp_transform_maps_uniform_to_klp():
	# uniform draws from the nested sampler cube map to KLP-distributed
	# samples in the prior cube
	t = nested.KLPTransform(loc=[0.4], scale=[0.3], a=[0.5], b=[0.5])
	rng = np.random.default_rng(11)
	n = 40000
	t_samples = rng.uniform(size=(n, 1)).astype(np.float32)
	u_samples = t.transform(t_samples)
	assert (u_samples >= 0).all() and (u_samples <= 1).all()
	for u_test in [0.2, 0.5, 0.8]:
		analytic = t.cdf(np.array([u_test]))
		empirical = (u_samples < u_test).mean()
		assert_allclose(empirical, analytic, atol=0.01)


def test_klp_transform_jacobian_vectorised():
	# log_jacobian of independent dimensions adds up
	t = nested.KLPTransform(**MOCK_PARAMS)
	t1 = nested.KLPTransform(loc=[MOCK_PARAMS['loc'][0]], scale=[MOCK_PARAMS['scale'][0]],
	                         a=[MOCK_PARAMS['a'][0]], b=[MOCK_PARAMS['b'][0]])
	t2 = nested.KLPTransform(loc=[MOCK_PARAMS['loc'][1]], scale=[MOCK_PARAMS['scale'][1]],
	                         a=[MOCK_PARAMS['a'][1]], b=[MOCK_PARAMS['b'][1]])
	x = np.array([0.3, 0.7])
	assert_allclose(t.log_jacobian(x), t1.log_jacobian(x[:1]) + t2.log_jacobian(x[1:]), rtol=1e-6)
	assert_allclose(t.logpdf(x), t1.logpdf(x[:1]) + t2.logpdf(x[1:]), rtol=1e-6)


# ---------------------------------------------------------------------------
# npe.py
# ---------------------------------------------------------------------------

def test_arch_suffix():
	assert npe._arch_suffix(2, 128, 'ReLU') == '128_128_ReLU'
	assert npe._arch_suffix(3, 64, nn.Tanh) == '64_64_64_Tanh'


def test_build_layers_rectangular():
	layers = npe._build_layers(4, 2, depth=2, width=8, activation_cls=nn.ReLU)
	linear = [l for l in layers if isinstance(l, nn.Linear)]
	activations = [l for l in layers if isinstance(l, nn.ReLU)]
	assert len(linear) == 3 and len(activations) == 2
	assert linear[0].in_features == 4 and linear[0].out_features == 8
	assert linear[1].in_features == 8 and linear[1].out_features == 8
	assert linear[2].in_features == 8 and linear[2].out_features == 2


def test_build_layers_triangular():
	layers = npe._build_layers(4, 4, depth=4, width=32, activation_cls=nn.ReLU, shape='triangular')
	widths = [l.out_features for l in layers if isinstance(l, nn.Linear)][:-1]
	assert widths == sorted(widths, reverse=True)
	assert min(widths) >= 4


def test_build_layers_cascade():
	layers = npe._build_layers(4, 2, depth=4, width=16, activation_cls=nn.ReLU, shape='cascade')
	assert len(layers) == 1 and isinstance(layers[0], npe.CascadeNet)


def test_build_layers_unknown_shape():
	with pytest.raises(ValueError):
		npe._build_layers(4, 2, depth=2, width=8, activation_cls=nn.ReLU, shape='hexagon')


def test_cascadenet_forward():
	for hidden_widths in ([8, 6], [7], [16, 8, 5]):
		net = npe.CascadeNet(4, 3, hidden_widths, nn.ReLU)
		x = torch.randn(5, 4)
		out = net(x)
		assert out.shape == (5, 3)
	# gradients flow through the skip connections
	net = npe.CascadeNet(4, 2, [8, 6], nn.ReLU)
	out = net(torch.randn(3, 4))
	out.sum().backward()
	assert all(p.grad is not None for p in net.parameters())


def test_npenetwork_forward():
	net = npe.NPENetwork(n_data=3, n_params=2, depth=2, width=8,
	                     activation_name='ReLU', layer_shape='rectangular')
	x = torch.randn(7, 3)
	loc, scale, a, b = net(x)
	assert loc.shape == scale.shape == a.shape == b.shape == (7, 2)
	# sigmoid location is in (0, 1); scale and shapes are positive
	assert (loc > 0).all() and (loc < 1).all()
	assert (scale > 0).all() and (a > 0).all() and (b > 0).all()
	# normalisation layer can be fitted and applied
	data = np.random.default_rng(8).normal(3.0, 0.5, size=(100, 3))
	net.fit_norm(data)
	assert net.norm.fitted
	loc, scale, a, b = net(x)
	assert torch.isfinite(loc).all()


def test_sample_posterior():
	model = FixedParamsModel(**MOCK_PARAMS)
	u_samples, theta_samples = npe.sample_posterior(
		model, np.zeros(5), prior_transform, 2, 500)
	assert u_samples.shape == (500, 2)
	assert theta_samples.shape == (500, 2)
	assert (u_samples >= 0).all() and (u_samples <= 1).all()
	# identity prior transform keeps theta == u
	assert_allclose(theta_samples, u_samples)


def train_toy_npe(folder, fresh_example_fraction):
	torch.manual_seed(1)
	return npe.train_npe(
		folder=str(folder),
		generate_noiseless_batch=generate_batch,
		inject_noise=inject_noise,
		n_params=2,
		fresh_sim_batch_size=32,
		npe_lr=1e-2,
		base_seed=1,
		max_model_evals=512,
		npe_batches_epoch=2,
		npe_width=16,
		npe_depth=2,
		val_size=32,
		norm_size=32,
		layer_shape='rectangular',
		fresh_example_fraction=fresh_example_fraction,
	)


@pytest.mark.parametrize("fresh_example_fraction", [1.0, 0.5])
def test_train_npe(tmp_path, fresh_example_fraction):
	model = train_toy_npe(tmp_path, fresh_example_fraction)
	assert isinstance(model, npe.NPENetwork)
	assert not model.training
	# model was persisted
	model_path = tmp_path / "minisbi_npe_3d_16_16_ReLU_KLP_rectangular.pt"
	assert model_path.exists()
	# posterior samples for a mock observation are finite and inside the cube
	data = A @ np.array([0.3, 0.7]) + np.random.default_rng(42).normal(0, NOISE_STD, 3)
	u_samples, theta_samples = npe.sample_posterior(model, data, prior_transform, 2, 200)
	assert np.isfinite(u_samples).all()
	assert (u_samples >= 0).all() and (u_samples <= 1).all()
	# the network learned the linear-Gaussian problem: the posterior mean
	# concentrates near the true parameters
	assert np.linalg.norm(u_samples.mean(axis=0) - np.array([0.3, 0.7])) < 0.25


def test_train_npe_reloads_saved_model(tmp_path, capsys):
	model = train_toy_npe(tmp_path, 1.0)
	# a second call with the same folder loads the stored model
	model2 = train_toy_npe(tmp_path, 1.0)
	assert "Loading trained NPE model" in capsys.readouterr().out
	for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
		assert k1 == k2
		assert torch.allclose(v1, v2)


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_train_npe_invalid_fresh_example_fraction(tmp_path, fraction):
	with pytest.raises(ValueError, match="fresh_example_fraction"):
		train_toy_npe(tmp_path, fraction)


# ---------------------------------------------------------------------------
# plot.py
# ---------------------------------------------------------------------------

def test_rank_histogram(tmp_path):
	model = FixedParamsModel(**MOCK_PARAMS)
	ranks, fig, axes = plot.rank_histogram(
		model=model,
		generate_noiseless_batch=generate_batch,
		inject_noise=inject_noise,
		n_params=2,
		folder=str(tmp_path),
		n_test=50,
		n_posterior_samples=100,
		prior_transform=prior_transform,
	)
	assert ranks.shape == (50, 2)
	assert ranks.dtype == np.int32
	assert (ranks >= 0).all() and (ranks < 100).all()
	fig.savefig(tmp_path / "rank_histograms.pdf")
	matplotlib.pyplot.close(fig)


def test_parameter_coverage_test(tmp_path):
	model = FixedParamsModel(**MOCK_PARAMS)
	levels = np.linspace(0.05, 0.95, 10).tolist()
	coverage, fig, axes = plot.parameter_coverage_test(
		model=model,
		generate_noiseless_batch=generate_batch,
		inject_noise=inject_noise,
		n_params=2,
		folder=str(tmp_path),
		n_test=50,
		credible_levels=levels,
		prior_transform=prior_transform,
	)
	assert coverage.shape == (len(levels), 2)
	assert (coverage >= 0).all() and (coverage <= 1).all()
	# coverage increases monotonically with the credible level
	assert (np.diff(coverage, axis=0) >= 0).all()
	fig.savefig(tmp_path / "coverage_test.pdf")
	matplotlib.pyplot.close(fig)


def test_posterior_predictive_check(tmp_path):
	posterior = np.random.default_rng(9).uniform(size=(10, 2))
	observed = A @ np.array([0.5, 0.5]) + np.random.default_rng(10).normal(0, NOISE_STD, 3)
	realisations, fig, axes = plot.posterior_predictive_check(
		posterior_samples_theta=posterior,
		observed_data=observed,
		generate_mean_and_noise=generate_mean_and_noise,
		inject_noise=inject_noise,
		folder=str(tmp_path),
		n_mean_curves=5,
		n_realisation_curves=5,
	)
	assert realisations.shape == (5, 3)
	assert np.isfinite(realisations).all()
	matplotlib.pyplot.close(fig)
