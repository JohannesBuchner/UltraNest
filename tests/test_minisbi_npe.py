"""Tests for the neural posterior estimation (NPE) part of ultranest.simbase.minisbi."""
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchinfo")
pytest.importorskip("joblib")

import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from torch import nn  # noqa: E402

from ultranest.simbase.minisbi.logistic import nll_kuma_logistic_product  # noqa: E402
from ultranest.simbase.minisbi.npe import CascadeNet, NPENetwork, _arch_suffix, _build_layers, sample_posterior, train_npe  # noqa: E402
from ultranest.simbase.minisbi.utils import make_cached_generate, make_memory  # noqa: E402

# Toy problem: two parameters with a uniform prior on [-1, 1],
# observed through three noisy linear summaries.
NOISE_STD = 0.1
N_PARAMS = 2
N_DATA = 3
U_TRUE = np.array([0.3, 0.75])


def toy_prior_transform(u):
    """Map the unit cube to [-1, 1]^2."""
    return 2.0 * u - 1.0


def toy_mean(theta, idx=0, seed=0):
    """Return the noiseless summaries of the toy model."""
    return np.array([theta[0], theta[1], theta[0] + theta[1]])


def _make_counting_simulator(calls):
    """Return a noiseless batch generator and a noise injector that record their calls in ``calls``."""
    def generate_noiseless_batch(batch_idx, n_sim, seed, n_params):
        """Draw ``n_sim`` prior samples and their noiseless summaries."""
        calls['batches'].append((batch_idx, n_sim))
        rng = np.random.default_rng(seed + batch_idx)
        u = rng.uniform(size=(n_sim, n_params)).astype(np.float32)
        return {'u_samples': u, 'mean_props': [toy_mean(toy_prior_transform(ui)) for ui in u]}

    def inject_noise(props, rng):
        """Add Gaussian noise to the summaries."""
        calls['noise'] += 1
        return rng.normal(props, NOISE_STD)

    return generate_noiseless_batch, inject_noise


def _final_linear(model):
    """Return the output layer of an NPENetwork backbone."""
    last = model.backbone[-1]
    return last.final_layer if isinstance(last, CascadeNet) else last


def _set_constant_output(model, loc, scale, a, b):
    """Make ``model`` predict the given distribution parameters for any input."""
    final = _final_linear(model)
    bias = np.concatenate([np.log(loc) - np.log1p(-np.asarray(loc)), np.log(scale), np.log(a), np.log(b)])
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor(bias, dtype=torch.float32))


def _reference_cdf(x, loc, scale, a, b):
    """CDF of ``u = 1 - (1 - v**a)**b`` with ``v`` a logistic variable truncated to (0, 1)."""
    v = (1.0 - (1.0 - x) ** (1.0 / b)) ** (1.0 / a)
    base = scipy.stats.logistic(loc=loc, scale=scale)
    return (base.cdf(v) - base.cdf(0.0)) / (base.cdf(1.0) - base.cdf(0.0))


def _train_kwargs(folder, generate_noiseless_batch, inject_noise, **overrides):
    """Return keyword arguments for a tiny, fast ``train_npe`` run."""
    kwargs = dict(
        folder=str(folder), generate_noiseless_batch=generate_noiseless_batch, inject_noise=inject_noise,
        n_params=N_PARAMS, fresh_sim_batch_size=16, npe_lr=1e-2, base_seed=7, max_model_evals=16 * 2 * 3,
        npe_batches_epoch=2, npe_width=8, npe_depth=1, val_size=20, norm_size=32,
        layer_shape='rectangular', fresh_example_fraction=0.5, patience=100,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("depth,width,activation,expected", [
    (2, 128, 'ReLU', '128_128_ReLU'),
    (3, 64, nn.Tanh, '64_64_64_Tanh'),
    (1, 8, nn.ReLU, '8_ReLU'),
])
def test_arch_suffix(depth, width, activation, expected):
    """The suffix lists the width once per layer, then the activation name (class or string)."""
    assert _arch_suffix(depth, width, activation) == expected


def test_arch_suffix_class_and_name_agree():
    """Passing the activation class or its name gives the same suffix."""
    assert _arch_suffix(4, 32, nn.GELU) == _arch_suffix(4, 32, 'GELU')


def test_build_layers_rectangular():
    """A rectangular MLP has ``depth`` hidden layers of ``width`` neurons, each followed by its own activation."""
    layers = _build_layers(5, 8, depth=3, width=16, activation_cls=nn.Tanh, shape='rectangular')
    assert len(layers) == 2 * 3 + 1
    linears = layers[0::2]
    activations = layers[1::2]
    assert all(isinstance(layer, nn.Linear) for layer in linears)
    assert all(isinstance(act, nn.Tanh) for act in activations)
    assert len({id(act) for act in activations}) == 3
    assert [(lin.in_features, lin.out_features) for lin in linears] == [(5, 16), (16, 16), (16, 16), (16, 8)]
    assert nn.Sequential(*layers)(torch.zeros(4, 5)).shape == (4, 8)


@pytest.mark.parametrize("depth,width,output_dim,expected_hidden", [
    (4, 20, 8, [20, 16, 12, 8]),
    (3, 64, 4, [64, 34, 4]),
    (1, 20, 8, [20]),
    (3, 4, 8, [8, 8, 8]),
])
def test_build_layers_triangular(depth, width, output_dim, expected_hidden):
    """A triangular MLP narrows linearly from ``width`` to ``output_dim`` and is never narrower than the output."""
    layers = _build_layers(3, output_dim, depth=depth, width=width, activation_cls=nn.ReLU, shape='triangular')
    linears = [layer for layer in layers if isinstance(layer, nn.Linear)]
    assert [lin.out_features for lin in linears[:-1]] == expected_hidden
    assert linears[0].in_features == 3
    assert linears[-1].out_features == output_dim
    for prev, nxt in zip(linears[:-1], linears[1:]):
        assert prev.out_features == nxt.in_features
    assert nn.Sequential(*layers)(torch.zeros(2, 3)).shape == (2, output_dim)


@pytest.mark.parametrize("depth,width,expected_hidden", [
    (6, 64, [64, 32, 16, 8]),
    (2, 64, [64, 32]),
    (3, 100, [100, 50, 25]),
    (5, 8, [8]),
])
def test_build_layers_cascade(depth, width, expected_hidden):
    """A cascade halves the width per layer and stops before layers would get narrower than 8 neurons."""
    layers = _build_layers(3, 4, depth=depth, width=width, activation_cls=nn.ReLU, shape='cascade')
    assert len(layers) == 1
    net = layers[0]
    assert isinstance(net, CascadeNet)
    assert [lin.out_features for lin in net.hidden_layers] == expected_hidden
    assert len(net.hidden_layers) <= depth
    last = expected_hidden[-1]
    assert net.final_layer.in_features == (last - last // 2) + sum(w // 2 for w in expected_hidden)
    assert net.final_layer.out_features == 4


def test_build_layers_unknown_shape():
    """An unknown layer shape is rejected."""
    with pytest.raises(ValueError, match="Unknown layer shape"):
        _build_layers(3, 4, depth=2, width=16, activation_cls=nn.ReLU, shape='hexagonal')


def test_cascadenet_layer_sizes_with_odd_widths():
    """Each layer passes ``w - w // 2`` neurons on, and the final layer sees the last pass-through plus all skips."""
    net = CascadeNet(7, 3, [9, 5], nn.ReLU)
    assert [(lin.in_features, lin.out_features) for lin in net.hidden_layers] == [(7, 9), (5, 5)]
    assert net.final_layer.in_features == 3 + 4 + 2
    assert len({id(act) for act in net.activations}) == 2


@pytest.mark.parametrize("shape", [(1, 7), (6, 7), (2, 3, 7)])
def test_cascadenet_forward_shape(shape):
    """The forward pass maps ``(..., input_dim)`` to ``(..., output_dim)``."""
    torch.manual_seed(0)
    net = CascadeNet(7, 3, [16, 8], nn.ReLU)
    out = net(torch.randn(*shape))
    assert out.shape == shape[:-1] + (3,)
    assert out.dtype == torch.float32


def test_cascadenet_matches_reference_forward():
    """The output equals a hand-written cascade: first half of each layer skips to the output, second half feeds the next layer."""
    torch.manual_seed(1)
    net = CascadeNet(4, 2, [10, 6, 4], nn.Tanh)
    x = torch.randn(5, 4)
    current, skips = x, []
    for lin in net.hidden_layers:
        h = torch.tanh(current @ lin.weight.T + lin.bias)
        skips.append(h[:, :h.shape[1] // 2])
        current = h[:, h.shape[1] // 2:]
    expected = torch.cat([current] + skips, dim=-1) @ net.final_layer.weight.T + net.final_layer.bias
    with torch.no_grad():
        assert torch.allclose(net(x), expected, atol=1e-6)


def test_cascadenet_skip_connection_bypasses_deeper_layers():
    """With all deeper layers zeroed, the output still depends on the input through the first layer's skip neurons."""
    torch.manual_seed(2)
    net = CascadeNet(4, 2, [16, 8, 4], nn.Tanh)
    with torch.no_grad():
        for lin in net.hidden_layers[1:]:
            lin.weight.zero_()
            lin.bias.zero_()
        out = net(torch.stack([torch.zeros(4), torch.ones(4)]))
    assert not torch.allclose(out[0], out[1])

    # a plain MLP with a zeroed second layer has a constant output
    mlp = nn.Sequential(*_build_layers(4, 2, depth=2, width=16, activation_cls=nn.Tanh))
    with torch.no_grad():
        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        out = mlp(torch.stack([torch.zeros(4), torch.ones(4)]))
    assert torch.allclose(out[0], out[1])


@pytest.mark.parametrize("layer_shape", ['rectangular', 'triangular', 'cascade'])
@pytest.mark.parametrize("batch", [1, 5])
def test_npe_network_output_shapes_and_constraints(layer_shape, batch):
    """Forward returns loc in (0, 1) and positive scale, a, b, each of shape (batch, n_params)."""
    torch.manual_seed(3)
    model = NPENetwork(n_data=N_DATA, n_params=N_PARAMS, depth=3, width=16, activation_name='ReLU', layer_shape=layer_shape)
    with torch.no_grad():
        outputs = model(torch.randn(batch, N_DATA))
    assert len(outputs) == 4
    loc, scale, a, b = outputs
    for t in outputs:
        assert t.shape == (batch, N_PARAMS)
        assert t.dtype == torch.float32
        assert torch.isfinite(t).all()
    assert ((loc > 0) & (loc < 1)).all()
    assert (scale > 0).all()
    assert (a > 0).all()
    assert (b > 0).all()


@pytest.mark.parametrize("layer_shape", ['rectangular', 'cascade'])
def test_npe_network_output_order(layer_shape):
    """The 4 * n_params backbone outputs are read as blocks of loc (sigmoid), log scale, log a, log b."""
    model = NPENetwork(n_data=N_DATA, n_params=N_PARAMS, depth=2, width=16, activation_name='ReLU', layer_shape=layer_shape)
    _set_constant_output(model, loc=[0.2, 0.9], scale=[0.05, 0.3], a=[0.5, 2.0], b=[3.0, 1.5])
    with torch.no_grad():
        loc, scale, a, b = model(torch.randn(3, N_DATA))
    assert_allclose(loc.numpy(), [[0.2, 0.9]] * 3, rtol=1e-5)
    assert_allclose(scale.numpy(), [[0.05, 0.3]] * 3, rtol=1e-5)
    assert_allclose(a.numpy(), [[0.5, 2.0]] * 3, rtol=1e-5)
    assert_allclose(b.numpy(), [[3.0, 1.5]] * 3, rtol=1e-5)


@pytest.mark.parametrize("raw,log_scale,log_shape", [(1e4, 4.0, 4.0), (-1e4, -8.0, -4.0)])
def test_npe_network_extreme_outputs_are_clamped(raw, log_scale, log_shape):
    """Huge backbone outputs are clamped (log scale to [-8, 4], log a and log b to [-4, 4]), so the NLL stays finite."""
    model = NPENetwork(n_data=N_DATA, n_params=N_PARAMS, depth=1, width=8, activation_name='ReLU', layer_shape='rectangular')
    final = _final_linear(model)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(raw)
        loc, scale, a, b = model(torch.zeros(2, N_DATA))
        nll = nll_kuma_logistic_product(torch.full((2, N_PARAMS), 0.5), loc, scale, a, b)
    assert ((loc >= 0) & (loc <= 1)).all()
    assert_allclose(scale.numpy(), np.exp(log_scale), rtol=1e-6)
    assert_allclose(a.numpy(), np.exp(log_shape), rtol=1e-6)
    assert_allclose(b.numpy(), np.exp(log_shape), rtol=1e-6)
    assert torch.isfinite(nll)


def test_npe_network_fit_norm():
    """After fit_norm the network sees z-scored inputs; constant features are left unscaled."""
    rng = np.random.default_rng(4)
    x_np = np.column_stack([
        rng.normal(10.0, 2.0, size=500),
        rng.normal(-3.0, 0.5, size=500),
        np.full(500, 7.0),
    ])
    torch.manual_seed(5)
    model = NPENetwork(n_data=3, n_params=N_PARAMS, depth=2, width=16, activation_name='ReLU', layer_shape='cascade')
    reference = NPENetwork(n_data=3, n_params=N_PARAMS, depth=2, width=16, activation_name='ReLU', layer_shape='cascade')
    reference.load_state_dict(model.state_dict())

    assert model.fit_norm(x_np) is None
    assert_allclose(model.norm.mean_t.numpy(), x_np.mean(axis=0), rtol=1e-5)
    assert_allclose(model.norm.std_t.numpy(), [x_np[:, 0].std(), x_np[:, 1].std(), 1.0], rtol=1e-5)

    x = torch.tensor(x_np[:8], dtype=torch.float32)
    z = (x - torch.tensor(x_np.mean(axis=0), dtype=torch.float32)) / model.norm.std_t
    with torch.no_grad():
        for got, expected in zip(model(x), reference(z)):
            assert torch.isfinite(got).all()
            assert torch.allclose(got, expected, atol=1e-5)


def test_npe_network_fit_norm_rejects_wrong_width():
    """fit_norm refuses data with a different number of features than n_data."""
    model = NPENetwork(n_data=3, n_params=N_PARAMS, depth=1, width=8, activation_name='ReLU', layer_shape='rectangular')
    with pytest.raises(AssertionError):
        model.fit_norm(np.zeros((10, 4)))


@pytest.mark.parametrize("layer_shape", ['rectangular', 'triangular', 'cascade'])
def test_npe_network_gradients_reach_all_parameters(layer_shape):
    """The NLL of the network output back-propagates finite, non-zero gradients to every weight."""
    torch.manual_seed(6)
    model = NPENetwork(n_data=N_DATA, n_params=N_PARAMS, depth=3, width=16, activation_name='Tanh', layer_shape=layer_shape)
    x = torch.randn(32, N_DATA)
    u = torch.rand(32, N_PARAMS)
    loss = nll_kuma_logistic_product(u, *model(x))
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


@pytest.mark.parametrize("n_samples", [1, 37])
def test_sample_posterior_shapes_and_prior_transform(n_samples):
    """Samples have the documented shapes, lie in the unit cube, and theta is prior_transform applied row by row."""
    model = NPENetwork(n_data=N_DATA, n_params=N_PARAMS, depth=1, width=8, activation_name='ReLU', layer_shape='rectangular')
    _set_constant_output(model, loc=[0.4, 0.6], scale=[0.1, 0.2], a=[1.0, 2.0], b=[1.0, 0.5])
    seen = []

    def prior_transform(u):
        """Record the call and map to [-1, 1]."""
        seen.append(u.shape)
        return toy_prior_transform(u)

    u, theta = sample_posterior(model, np.zeros(N_DATA), prior_transform, N_PARAMS, n_samples)
    assert isinstance(u, np.ndarray) and isinstance(theta, np.ndarray)
    assert u.shape == (n_samples, N_PARAMS)
    assert theta.shape == (n_samples, N_PARAMS)
    assert u.dtype == np.float32
    assert np.all((u >= 0) & (u <= 1))
    assert seen == [(N_PARAMS,)] * n_samples
    assert_allclose(theta, toy_prior_transform(u))


def test_sample_posterior_matches_predicted_distribution():
    """Posterior samples follow the Kumaraswamy-Logistic distribution predicted by the network (KS test)."""
    model = NPENetwork(n_data=N_DATA, n_params=N_PARAMS, depth=2, width=8, activation_name='ReLU', layer_shape='cascade')
    _set_constant_output(model, loc=[0.3, 0.6], scale=[0.1, 0.2], a=[1.0, 2.0], b=[1.0, 0.5])
    observed = np.array([0.1, -0.2, 0.3])
    with torch.no_grad():
        params = [t.numpy()[0].astype(np.float64) for t in model(torch.tensor(observed[None, :], dtype=torch.float32))]

    u, _ = sample_posterior(model, observed, toy_prior_transform, N_PARAMS, 2000)
    for d in range(N_PARAMS):
        loc, scale, a, b = (p[d] for p in params)
        assert scipy.stats.kstest(u[:, d], lambda x: _reference_cdf(x, loc, scale, a, b)).pvalue > 0.01
        # the test has power: a shifted location is clearly rejected
        assert scipy.stats.kstest(u[:, d], lambda x: _reference_cdf(x, loc + 0.1, scale, a, b)).pvalue < 1e-3


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_train_npe_rejects_invalid_fresh_example_fraction(tmp_path, fraction):
    """fresh_example_fraction outside (0, 1] is rejected before any simulation is run."""
    calls = {'batches': [], 'noise': 0}
    generate, inject = _make_counting_simulator(calls)
    with pytest.raises(ValueError, match="fresh_example_fraction"):
        train_npe(**_train_kwargs(tmp_path, generate, inject, fresh_example_fraction=fraction))
    assert calls == {'batches': [], 'noise': 0}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.25])
def test_train_npe_replay_follows_fresh_example_fraction(tmp_path, fraction):
    """Each training step injects noise into the fresh simulations plus the replayed ones given by the documented formula."""
    torch.manual_seed(8)
    calls = {'batches': [], 'noise': 0}
    generate, inject = _make_counting_simulator(calls)
    # a budget of 101 simulations allows 6 full training batches of 16, not 7
    kwargs = _train_kwargs(tmp_path, generate, inject, fresh_example_fraction=fraction, max_model_evals=16 * 6 + 5, npe_batches_epoch=1)
    train_steps = 6
    model = train_npe(**kwargs)
    assert isinstance(model, NPENetwork)

    n_fresh = kwargs['fresh_sim_batch_size']
    replay = int(round(n_fresh * (1 - fraction) / fraction))
    norm_batches = -(-kwargs['norm_size'] // n_fresh)
    val_batches = -(-kwargs['val_size'] // n_fresh)
    # probe + normalisation + validation sets, then fresh + replayed examples per step (replay limited by the history)
    expected = n_fresh * (1 + norm_batches + val_batches)
    expected += sum(n_fresh + min(replay, step * n_fresh) for step in range(train_steps))
    assert calls['noise'] == expected

    # every simulator batch is requested once, with its own index and the configured size
    assert sorted(idx for idx, _ in calls['batches']) == list(range(norm_batches + val_batches + train_steps))
    assert all(n_sim == n_fresh for _, n_sim in calls['batches'])


@pytest.mark.parametrize("patience", [1, 2])
def test_train_npe_early_stopping(tmp_path, patience):
    """Without any improvement, training stops after ``patience`` epochs past the first one."""
    torch.manual_seed(9)
    calls = {'batches': [], 'noise': 0}
    generate, inject = _make_counting_simulator(calls)
    # a zero learning rate keeps the validation loss constant, so only the first epoch counts as an improvement
    kwargs = _train_kwargs(tmp_path, generate, inject, npe_lr=0.0, patience=patience, max_model_evals=16 * 2 * 10,
                           fresh_example_fraction=1.0)
    model = train_npe(**kwargs)
    assert not model.training

    n_fresh = kwargs['fresh_sim_batch_size']
    prefix = n_fresh * (1 + -(-kwargs['norm_size'] // n_fresh) + -(-kwargs['val_size'] // n_fresh))
    epochs_run = 1 + patience
    assert calls['noise'] == prefix + epochs_run * kwargs['npe_batches_epoch'] * n_fresh


def test_train_npe_restores_best_epoch(tmp_path, capsys):
    """When later epochs make the validation loss worse, the weights of the best epoch are returned, not the last ones."""
    calls = {'batches': [], 'noise': 0}
    clean_generate, inject = _make_counting_simulator(calls)
    n_fresh, per_epoch, val_size = 16, 4, 128
    # batches 0 to 8 are the normalisation and validation sets; corrupt everything after 4 clean training epochs
    first_bad_batch = 1 + val_size // n_fresh + 4 * per_epoch

    def generate(batch_idx, n_sim, seed, n_params):
        """Simulate correctly, but squeeze the parameters of late batches towards 0 so they no longer match the data."""
        batch = clean_generate(batch_idx, n_sim, seed, n_params)
        if batch_idx >= first_bad_batch:
            batch['u_samples'] = 0.05 * batch['u_samples']
        return batch

    kwargs = _train_kwargs(
        tmp_path, generate, inject, fresh_sim_batch_size=n_fresh, npe_batches_epoch=per_epoch, max_model_evals=n_fresh * per_epoch * 8,
        val_size=val_size, norm_size=n_fresh, npe_width=16, npe_depth=2, npe_lr=0.03, fresh_example_fraction=1.0,
    )
    torch.manual_seed(15)
    model = train_npe(**kwargs)
    log = capsys.readouterr().out
    val_losses = [float(v) for v in re.findall(r"Epoch +\d+/\d+ .*val_loss=(-?\d+\.\d+)", log)]
    assert len(val_losses) == 8
    best = min(val_losses)
    # the corrupted epochs clearly hurt, so returning the last weights would be wrong
    assert val_losses[-1] > best + 1.0
    assert re.findall(r"Restored best model \(val_loss=(-?\d+\.\d+)\)", log) == [f"{best:.4f}"]

    # on fresh clean data the returned model scores like the best epoch, not like the last one
    rng = np.random.default_rng(16)
    u = rng.uniform(size=(512, N_PARAMS))
    x = np.array([rng.normal(toy_mean(toy_prior_transform(ui)), NOISE_STD) for ui in u])
    with torch.no_grad():
        nll = nll_kuma_logistic_product(torch.tensor(u, dtype=torch.float32), *model(torch.tensor(x, dtype=torch.float32))).item()
    assert abs(nll - best) < 0.25


@pytest.fixture(scope="module")
def trained_toy(tmp_path_factory):
    """Train a small cascade NPE on the toy problem, using the cached simulator as in the tutorial."""
    folder = tmp_path_factory.mktemp("npe")
    generate = make_cached_generate(make_memory(str(folder)), toy_prior_transform, toy_mean)
    calls = {'batches': [], 'noise': 0}
    _, inject = _make_counting_simulator(calls)
    kwargs = _train_kwargs(
        folder, generate, inject, fresh_sim_batch_size=64, npe_batches_epoch=4, max_model_evals=64 * 4 * 30,
        npe_width=32, npe_depth=3, layer_shape='cascade', val_size=128, norm_size=128, base_seed=1,
    )
    torch.manual_seed(10)
    model = train_npe(**kwargs)
    return model, folder, kwargs


def test_train_npe_returns_fitted_model_in_eval_mode(trained_toy):
    """The trained model is an NPENetwork in eval mode whose normalisation was fitted on simulated data."""
    model, _, _ = trained_toy
    assert isinstance(model, NPENetwork)
    assert not model.training
    assert model.norm.fitted
    assert model.norm.n_features == N_DATA
    # summaries of a uniform prior on [-1, 1]: means 0, std 1/sqrt(3) and sqrt(2/3) for the sum
    assert_allclose(model.norm.mean_t.numpy(), [0.0, 0.0, 0.0], atol=0.15)
    assert_allclose(model.norm.std_t.numpy(), [0.58, 0.58, 0.82], rtol=0.15)


def test_train_npe_improves_heldout_loss(trained_toy):
    """On fresh simulations the trained model beats both the uniform prior (NLL 0) and an untrained network."""
    model, _, kwargs = trained_toy
    rng = np.random.default_rng(11)
    u = rng.uniform(size=(256, N_PARAMS))
    x = np.array([rng.normal(toy_mean(toy_prior_transform(ui)), NOISE_STD) for ui in u])
    u_t = torch.tensor(u, dtype=torch.float32)
    x_t = torch.tensor(x, dtype=torch.float32)

    torch.manual_seed(12)
    untrained = NPENetwork(n_data=N_DATA, n_params=N_PARAMS, depth=kwargs['npe_depth'], width=kwargs['npe_width'],
                           activation_name='ReLU', layer_shape='cascade')
    untrained.fit_norm(x)
    with torch.no_grad():
        nll_trained = nll_kuma_logistic_product(u_t, *model(x_t)).item()
        nll_untrained = nll_kuma_logistic_product(u_t, *untrained(x_t)).item()
    # the exact posterior scores about -3.8 on these samples; no normalised density can do much better
    assert -4.5 < nll_trained < -2.5
    assert nll_trained < nll_untrained - 1.0


@pytest.mark.parametrize("u_true", [[0.3, 0.75], [0.7, 0.2]])
def test_train_npe_posterior_moves_toward_truth(trained_toy, u_true):
    """For noiseless data at a known truth, the posterior centres on it with a spread close to the exact posterior's."""
    model, _, _ = trained_toy
    u_true = np.array(u_true)

    u, _ = sample_posterior(model, toy_mean(toy_prior_transform(u_true)), toy_prior_transform, N_PARAMS, 4000)
    assert np.all(np.abs(np.median(u, axis=0) - u_true) < 0.05)
    # exact marginal posterior sd in u is 0.5 * NOISE_STD * sqrt(2/3) ~ 0.04 (prior: 0.29);
    # the small network may be somewhat wider, but neither prior-like nor over-confident
    exact_sd = 0.5 * NOISE_STD * np.sqrt(2.0 / 3.0)
    assert np.all((u.std(axis=0) > 0.7 * exact_sd) & (u.std(axis=0) < 2.5 * exact_sd))


def test_train_npe_posterior_follows_noisy_data(trained_toy):
    """For noisy data the posterior median follows the least-squares estimate, which is the exact posterior centre here."""
    model, _, _ = trained_toy
    rng = np.random.default_rng(13)
    observed = rng.normal(toy_mean(toy_prior_transform(U_TRUE)), NOISE_STD)
    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    u_hat = (np.linalg.lstsq(design, observed, rcond=None)[0] + 1.0) / 2.0
    assert np.all(np.abs(u_hat - U_TRUE) > 0.05)

    u, _ = sample_posterior(model, observed, toy_prior_transform, N_PARAMS, 4000)
    assert np.all(np.abs(np.median(u, axis=0) - u_hat) < 0.05)


def test_train_npe_saves_and_reloads_checkpoint(trained_toy):
    """A second call with the same folder and architecture loads the saved weights instead of training again."""
    model, folder, kwargs = trained_toy
    assert (folder / "minisbi_npe_3d_32_32_32_ReLU_KLP_cascade.pt").is_file()

    calls = {'batches': [], 'noise': 0}
    _, inject = _make_counting_simulator(calls)
    reloaded = train_npe(**dict(kwargs, inject_noise=inject))
    assert isinstance(reloaded, NPENetwork)
    assert not reloaded.training
    # only the probe batch gets noise; no normalisation, validation or training data is made
    assert calls['noise'] == kwargs['fresh_sim_batch_size']
    expected_state = model.state_dict()
    state = reloaded.state_dict()
    assert state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        assert torch.equal(state[key], value), key


def test_train_npe_checkpoint_depends_on_architecture(trained_toy):
    """A different architecture trains and saves its own checkpoint instead of loading a mismatched one."""
    _, folder, kwargs = trained_toy
    torch.manual_seed(14)
    model = train_npe(**dict(kwargs, npe_width=16, max_model_evals=64 * 4))
    assert [lin.out_features for lin in model.backbone[0].hidden_layers] == [16, 8]
    assert sorted(p.name for p in folder.glob("*.pt")) == [
        "minisbi_npe_3d_16_16_16_ReLU_KLP_cascade.pt",
        "minisbi_npe_3d_32_32_32_ReLU_KLP_cascade.pt",
    ]
