"""Tests for the z-score input normalisation layer of minisbi."""
import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from ultranest.simbase.minisbi.norm import ZScoreNorm  # noqa: E402


def _calibration_data(n=200, seed=1):
    """Return a small calibration set with distinct per-feature location and scale."""
    rng = np.random.default_rng(seed)
    loc = np.array([-3.0, 0.5, 10.0])
    scale = np.array([0.1, 2.0, 7.0])
    return loc + scale * rng.normal(size=(n, 3))


def test_zscorenorm_initial_state():
    """A fresh layer is unfitted, holds identity statistics as float32 buffers and has no trainable parameters."""
    norm = ZScoreNorm(4)
    assert norm.n_features == 4
    assert norm.eps == 1e-8
    assert not norm.fitted
    assert norm.mean_t.shape == (4,) and norm.std_t.shape == (4,)
    assert norm.mean_t.dtype == torch.float32 and norm.std_t.dtype == torch.float32
    assert_array_equal(norm.mean_t.numpy(), np.zeros(4))
    assert_array_equal(norm.std_t.numpy(), np.ones(4))
    assert list(norm.parameters()) == []
    assert {"mean_t", "std_t"} <= set(norm.state_dict().keys())
    assert ZScoreNorm(2, eps=0.25).eps == 0.25


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zscorenorm_unfitted_forward_is_identity(dtype):
    """Before fit, forward returns its input unchanged, as documented."""
    norm = ZScoreNorm(3)
    x = torch.arange(12, dtype=dtype).reshape(4, 3) * 1.5 - 5
    out = norm(x)
    assert out.dtype == dtype
    assert torch.equal(out, x)


def test_zscorenorm_fit_matches_numpy_statistics():
    """Fit stores the per-feature mean and population std (ddof=0) of the calibration set."""
    x = _calibration_data()
    x_before = x.copy()
    norm = ZScoreNorm(3)
    norm.fit(x)
    assert norm.fitted
    assert_array_equal(x, x_before)
    assert norm.mean_t.dtype == torch.float32 and norm.std_t.dtype == torch.float32
    assert_allclose(norm.mean_t.numpy(), x.mean(axis=0), rtol=1e-6, atol=1e-6)
    assert_allclose(norm.std_t.numpy(), x.std(axis=0, ddof=0), rtol=1e-6)


def test_zscorenorm_whitens_calibration_set():
    """Normalising the calibration set itself yields zero mean and unit (ddof=0) std in every feature."""
    x = _calibration_data()
    norm = ZScoreNorm(3)
    norm.fit(x)
    z = norm(torch.tensor(x, dtype=torch.float64)).numpy()
    assert z.shape == x.shape
    assert_allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert_allclose(z.std(axis=0), 1.0, rtol=1e-5)


def test_zscorenorm_constant_feature():
    """A constant feature gets std 1, so it is only shifted and never produces inf or nan."""
    x = _calibration_data(n=50)
    x[:, 1] = 4.25
    norm = ZScoreNorm(3)
    norm.fit(x)
    assert norm.std_t[1].item() == 1.0
    assert norm.mean_t[1].item() == 4.25
    assert_allclose(norm.std_t.numpy()[[0, 2]], x[:, [0, 2]].std(axis=0), rtol=1e-6)

    z = norm(torch.tensor(x, dtype=torch.float32)).numpy()
    assert np.all(np.isfinite(z))
    assert_array_equal(z[:, 1], 0.0)

    x_new = torch.tensor([[0.0, 5.25, 0.0], [0.0, 2.25, 0.0]])
    assert_allclose(norm(x_new)[:, 1].numpy(), [1.0, -2.0])


def test_zscorenorm_eps_threshold():
    """Features whose std is not above eps are left unscaled, while features above eps are scaled."""
    rng = np.random.default_rng(3)
    x = np.empty((100, 3))
    x[:, 0] = 2.0 + 0.25 * rng.normal(size=100)
    x[:, 1] = -1.0 + 4.0 * rng.normal(size=100)
    x[:, 2] = 7.0 + 0.6 * rng.normal(size=100)
    norm = ZScoreNorm(3, eps=0.5)
    norm.fit(x)
    std = x.std(axis=0)
    assert std[0] < 0.5 < std[2] < std[1]
    assert norm.std_t[0].item() == 1.0
    assert_allclose(norm.std_t.numpy()[1:], std[1:], rtol=1e-6)
    assert_allclose(norm.mean_t.numpy(), x.mean(axis=0), rtol=1e-6)


def test_zscorenorm_default_eps_tiny_spread():
    """With the default eps, a spread far below eps counts as constant, but a small spread above eps is kept."""
    rng = np.random.default_rng(4)
    x = np.empty((100, 2))
    x[:, 0] = 1.0 + 1e-12 * rng.normal(size=100)
    x[:, 1] = 3.0 + 1e-6 * rng.normal(size=100)
    norm = ZScoreNorm(2)
    norm.fit(x)
    assert norm.std_t[0].item() == 1.0
    assert_allclose(norm.std_t[1].item(), x[:, 1].std(), rtol=1e-5)
    z = norm(torch.tensor(x)).numpy()
    assert np.abs(z[:, 0]).max() < 1e-9
    assert_allclose(z[:, 1].std(), 1.0, rtol=1e-5)


def test_zscorenorm_single_sample():
    """Fitting on a single sample gives zero spread everywhere, so that sample maps to the origin."""
    x = np.array([[1.5, -2.0, 30.0]])
    norm = ZScoreNorm(3)
    norm.fit(x)
    assert_array_equal(norm.std_t.numpy(), np.ones(3))
    assert_allclose(norm.mean_t.numpy(), x[0])
    assert_array_equal(norm(torch.tensor(x, dtype=torch.float32)).numpy(), np.zeros((1, 3)))


def test_zscorenorm_fit_integer_array():
    """Integer calibration arrays are accepted and give the same statistics as their float version."""
    x = np.array([[0, 10], [2, 10], [4, 13], [6, 15]])
    norm = ZScoreNorm(2)
    norm.fit(x)
    assert_allclose(norm.mean_t.numpy(), x.astype(float).mean(axis=0))
    assert_allclose(norm.std_t.numpy(), x.astype(float).std(axis=0), rtol=1e-6)


def test_zscorenorm_fit_wrong_feature_count():
    """Fit rejects calibration data with the wrong number of features and keeps the layer unfitted."""
    norm = ZScoreNorm(3)
    with pytest.raises(AssertionError, match="Expected 3 features, got 2"):
        norm.fit(np.ones((5, 2)))
    assert not norm.fitted
    assert_array_equal(norm.mean_t.numpy(), np.zeros(3))
    assert_array_equal(norm.std_t.numpy(), np.ones(3))


def test_zscorenorm_refit_overwrites_statistics():
    """A second fit replaces the statistics of the first one."""
    norm = ZScoreNorm(3)
    norm.fit(_calibration_data(seed=5))
    x2 = 100.0 + 0.5 * _calibration_data(seed=6)
    norm.fit(x2)
    assert_allclose(norm.mean_t.numpy(), x2.mean(axis=0), rtol=1e-6)
    assert_allclose(norm.std_t.numpy(), x2.std(axis=0), rtol=1e-6)


@pytest.mark.parametrize("shape", [(3,), (1, 3), (5, 3), (2, 4, 3)])
def test_zscorenorm_forward_shapes(shape):
    """Forward maps any (..., n_features) input to (x - mean) / std along the last axis, keeping its shape."""
    x_cal = _calibration_data()
    norm = ZScoreNorm(3)
    norm.fit(x_cal)
    x = _calibration_data(n=int(np.prod(shape[:-1])), seed=7).reshape(shape).astype(np.float32)
    z = norm(torch.from_numpy(x))
    assert z.shape == x.shape and z.dtype == torch.float32
    expected = (x - x_cal.mean(axis=0)) / x_cal.std(axis=0)
    assert_allclose(z.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_zscorenorm_forward_keeps_statistics_fixed():
    """Unlike batch norm, forward never updates the statistics, and train and eval mode give the same output."""
    x_cal = _calibration_data()
    norm = ZScoreNorm(3)
    norm.fit(x_cal)
    mean, std = norm.mean_t.clone(), norm.std_t.clone()
    x = torch.tensor(50.0 + 3.0 * _calibration_data(n=20, seed=8), dtype=torch.float32)
    z_train = norm.train()(x)
    z_eval = norm.eval()(x)
    assert torch.equal(norm.mean_t, mean) and torch.equal(norm.std_t, std)
    assert torch.equal(z_train, z_eval)


@pytest.mark.parametrize("in_dtype,out_dtype", [
    (torch.float32, torch.float32),
    (torch.float64, torch.float64),
    (torch.int64, torch.float32),
])
def test_zscorenorm_forward_dtype_promotion(in_dtype, out_dtype):
    """Float32 buffers follow torch type promotion: float inputs keep their precision, integer inputs become float32."""
    x = _calibration_data()
    norm = ZScoreNorm(3)
    norm.fit(x)
    x_in = torch.tensor([[-3, 1, 10], [-2, 0, 17]]).to(in_dtype)
    z = norm(x_in)
    assert z.dtype == out_dtype
    expected = (x_in.double().numpy() - x.mean(axis=0)) / x.std(axis=0)
    assert_allclose(z.numpy(), expected, rtol=1e-5)


def test_zscorenorm_double_module():
    """Casting the module to float64 converts the buffers, and a later fit stores the statistics in float64."""
    x = _calibration_data()
    norm = ZScoreNorm(3).double()
    assert norm.mean_t.dtype == torch.float64 and norm.std_t.dtype == torch.float64
    norm.fit(x)
    assert norm.mean_t.dtype == torch.float64 and norm.std_t.dtype == torch.float64
    assert_allclose(norm.mean_t.numpy(), x.mean(axis=0), rtol=1e-6)
    assert_allclose(norm.std_t.numpy(), x.std(axis=0), rtol=1e-6)
    z = norm(torch.tensor(x))
    assert z.dtype == torch.float64
    assert_allclose(z.numpy().std(axis=0), 1.0, rtol=1e-5)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
])
def test_zscorenorm_device(device):
    """Buffers move with the module, fit works on the moved module, and the output stays on the input device."""
    x = _calibration_data()
    norm = ZScoreNorm(3).to(device)
    assert norm.mean_t.device.type == device and norm.std_t.device.type == device
    norm.fit(x)
    assert norm.mean_t.device.type == device and norm.std_t.device.type == device
    z = norm(torch.tensor(x, dtype=torch.float32, device=device))
    assert z.device.type == device
    assert_allclose(z.cpu().numpy().mean(axis=0), 0.0, atol=1e-5)


def test_zscorenorm_gradient():
    """Forward is differentiable in its input with Jacobian diag(1 / std), and the buffers collect no gradient."""
    norm = ZScoreNorm(3)
    norm.fit(_calibration_data())
    x = torch.zeros(2, 3, requires_grad=True)
    norm(x).sum().backward()
    assert_allclose(x.grad.numpy(), np.tile(1.0 / norm.std_t.numpy(), (2, 1)), rtol=1e-6)
    assert not norm.mean_t.requires_grad and norm.mean_t.grad is None
    assert not norm.std_t.requires_grad and norm.std_t.grad is None


def test_zscorenorm_buffers_saved_in_state_dict(tmp_path):
    """The fitted statistics are stored in the state_dict and survive a torch.save / torch.load round trip."""
    x = _calibration_data()
    norm = ZScoreNorm(3)
    norm.fit(x)
    state = norm.state_dict()
    assert_array_equal(state["mean_t"].numpy(), norm.mean_t.numpy())
    assert_array_equal(state["std_t"].numpy(), norm.std_t.numpy())

    path = tmp_path / "norm.pt"
    torch.save({"state_dict": state}, path)
    loaded = ZScoreNorm(3)
    loaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    assert loaded.mean_t.dtype == torch.float32 and loaded.std_t.dtype == torch.float32
    assert_array_equal(loaded.mean_t.numpy(), norm.mean_t.numpy())
    assert_array_equal(loaded.std_t.numpy(), norm.std_t.numpy())

    with pytest.raises(RuntimeError):
        ZScoreNorm(4).load_state_dict(state)
