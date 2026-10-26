"""Tests for the sampling, caching and permutation helpers of minisbi."""
import pytest

torch = pytest.importorskip("torch")
joblib = pytest.importorskip("joblib")

import itertools  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from ultranest.simbase.minisbi.utils import (  # noqa: E402
    inject_noise_batch, make_cached_generate, make_memory, random_derangement, sample_prior_u)


def _ks_statistic_uniform(x):
    """Return the Kolmogorov-Smirnov distance between the samples x and U(0, 1)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    upper = np.arange(1, n + 1) / n - x
    lower = x - np.arange(n) / n
    return max(upper.max(), lower.max())


def _prior_transform(u):
    """Map the unit cube to a simple box, like the prior transforms used with minisbi."""
    return 10.0 * u - 5.0


class _Simulator:
    """Noiseless simulator that records every call made to it."""

    def __init__(self, n_data=3):
        self.n_data = n_data
        self.calls = []

    def __call__(self, idx, seed, theta):
        """Return the noiseless mean and noise level for one simulation."""
        self.calls.append((idx, seed, np.array(theta)))
        x = np.linspace(0.0, 1.0, self.n_data)
        return {'mean': theta[0] + theta[1] * x, 'noise_std': 0.1 + 0 * x}


def _gaussian_noise(props, rng):
    """Draw Gaussian noise around the noiseless mean, as in the minisbi example notebook."""
    return rng.normal(props['mean'], props['noise_std'])


# ---------------------------------------------------------------- sample_prior_u


@pytest.mark.parametrize("n_params", [1, 2, 7])
def test_sample_prior_u_shape_dtype_range(n_params):
    """A single draw is a float32 vector of length n_params inside the unit cube."""
    u = sample_prior_u(np.random.default_rng(1), n_params)
    assert isinstance(u, np.ndarray)
    assert u.shape == (n_params,)
    assert u.dtype == np.float32
    assert np.all(u >= 0.0)
    assert np.all(u < 1.0)


def test_sample_prior_u_zero_params():
    """Zero parameters give an empty float32 vector."""
    u = sample_prior_u(np.random.default_rng(1), 0)
    assert u.shape == (0,)
    assert u.dtype == np.float32


def test_sample_prior_u_reproducible_and_uses_rng():
    """The draw is fully determined by the generator state, and advances that state."""
    a = sample_prior_u(np.random.default_rng(42), 5)
    b = sample_prior_u(np.random.default_rng(42), 5)
    assert_array_equal(a, b)

    rng = np.random.default_rng(42)
    first = sample_prior_u(rng, 5)
    second = sample_prior_u(rng, 5)
    assert_array_equal(first, a)
    assert not np.any(first == second)

    c = sample_prior_u(np.random.default_rng(43), 5)
    assert not np.any(a == c)


def test_sample_prior_u_is_uniform():
    """Pooled draws follow U(0, 1) (KS test) and the coordinates are uncorrelated."""
    rng = np.random.default_rng(2024)
    u = np.stack([sample_prior_u(rng, 4) for _ in range(500)])
    n = u.size
    # critical value of the one-sample KS test at the 1% level
    assert _ks_statistic_uniform(u.ravel()) < 1.63 / np.sqrt(n)
    for j in range(u.shape[1]):
        assert _ks_statistic_uniform(u[:, j]) < 1.63 / np.sqrt(len(u))
    corr = np.corrcoef(u, rowvar=False)
    off_diagonal = corr[~np.eye(4, dtype=bool)]
    assert np.all(np.abs(off_diagonal) < 4 / np.sqrt(len(u)))
    assert_allclose(u.mean(axis=0), 0.5, atol=0.05)
    assert_allclose(u.var(axis=0), 1.0 / 12, atol=0.015)


# ---------------------------------------------------------------- make_memory


def test_make_memory_returns_joblib_memory(tmp_path):
    """make_memory returns a joblib.Memory that stores its cache in the given folder."""
    memory = make_memory(tmp_path)
    assert isinstance(memory, joblib.Memory)
    assert os.fspath(memory.location) == os.fspath(tmp_path)

    memory_str = make_memory(str(tmp_path))
    assert os.fspath(memory_str.location) == os.fspath(tmp_path)


def test_make_memory_caches_to_disk_quietly(tmp_path, capsys):
    """A function cached with the memory runs once, writes into the folder and prints nothing."""
    folder = tmp_path / "cache"
    memory = make_memory(folder)
    calls = []

    def square(x):
        calls.append(x)
        return x * x

    cached_square = memory.cache(square)
    assert cached_square(3) == 9
    assert cached_square(3) == 9
    assert cached_square(4) == 16
    assert calls == [3, 4]
    assert folder.is_dir()
    assert any(p.is_file() for p in folder.rglob("*"))
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""


# ---------------------------------------------------------------- make_cached_generate


@pytest.mark.parametrize("n_sim", [1, 4])
def test_cached_generate_output_contract(tmp_path, n_sim):
    """The batch holds float32 unit-cube draws and one simulator output per draw."""
    n_params = 3
    simulator = _Simulator()
    generate = make_cached_generate(make_memory(tmp_path), _prior_transform, simulator)
    batch = generate(batch_idx=0, n_sim=n_sim, seed=7, n_params=n_params)

    assert set(batch) == {'u_samples', 'mean_props'}
    u = batch['u_samples']
    assert isinstance(u, np.ndarray)
    assert u.shape == (n_sim, n_params)
    assert u.dtype == np.float32
    assert np.all((u >= 0.0) & (u < 1.0))
    assert isinstance(batch['mean_props'], list)
    assert len(batch['mean_props']) == n_sim

    # each simulation sees theta = prior_transform(u) of its own row, and the base seed
    assert len(simulator.calls) == n_sim
    reference = _Simulator()
    for i, (idx, seed, theta) in enumerate(simulator.calls):
        assert seed == 7
        assert_array_equal(theta, _prior_transform(u[i]))
        assert_array_equal(batch['mean_props'][i]['mean'], reference(idx, seed, theta)['mean'])


def test_cached_generate_simulation_indices_are_global(tmp_path):
    """Simulation indices are unique and contiguous across batches; the test batch (batch_idx=-1) gets negative ones."""
    n_sim = 3
    simulator = _Simulator()
    generate = make_cached_generate(make_memory(tmp_path), _prior_transform, simulator)
    for batch_idx in range(3):
        generate(batch_idx=batch_idx, n_sim=n_sim, seed=11, n_params=2)
    indices = [idx for idx, _, _ in simulator.calls]
    assert indices == list(range(3 * n_sim))

    # minisbi.plot draws its held-out test set with batch_idx=-1
    test_batch = generate(batch_idx=-1, n_sim=n_sim, seed=11, n_params=2)
    assert [idx for idx, _, _ in simulator.calls[3 * n_sim:]] == [-3, -2, -1]
    assert test_batch['u_samples'].shape == (n_sim, 2)


def test_cached_generate_serves_repeated_calls_from_cache(tmp_path):
    """Repeating a call does not rerun the simulator and returns identical results."""
    simulator = _Simulator()
    generate = make_cached_generate(make_memory(tmp_path), _prior_transform, simulator)
    first = generate(batch_idx=2, n_sim=4, seed=5, n_params=2)
    assert len(simulator.calls) == 4

    second = generate(batch_idx=2, n_sim=4, seed=5, n_params=2)
    third = generate(2, 4, 5, 2)
    assert len(simulator.calls) == 4
    for again in (second, third):
        assert again['u_samples'].dtype == np.float32
        assert_array_equal(again['u_samples'], first['u_samples'])
        assert len(again['mean_props']) == len(first['mean_props'])
        for p, q in zip(again['mean_props'], first['mean_props']):
            assert_array_equal(p['mean'], q['mean'])
            assert_array_equal(p['noise_std'], q['noise_std'])

    # any changed argument is a cache miss and runs the simulator again
    generate(batch_idx=3, n_sim=4, seed=5, n_params=2)
    generate(batch_idx=2, n_sim=4, seed=6, n_params=2)
    generate(batch_idx=2, n_sim=2, seed=5, n_params=2)
    generate(batch_idx=2, n_sim=4, seed=5, n_params=3)
    assert len(simulator.calls) == 4 + 4 + 4 + 2 + 4


def test_cached_generate_persists_across_memory_instances(tmp_path):
    """A new Memory on the same folder (e.g. a new session) reuses batches stored on disk."""
    simulator = _Simulator()
    generate = make_cached_generate(make_memory(tmp_path), _prior_transform, simulator)
    first = generate(batch_idx=0, n_sim=3, seed=1, n_params=2)
    assert len(simulator.calls) == 3

    generate_again = make_cached_generate(make_memory(tmp_path), _prior_transform, simulator)
    second = generate_again(batch_idx=0, n_sim=3, seed=1, n_params=2)
    assert len(simulator.calls) == 3
    assert_array_equal(second['u_samples'], first['u_samples'])
    for p, q in zip(second['mean_props'], first['mean_props']):
        assert_array_equal(p['mean'], q['mean'])


def test_cached_generate_reproducible_across_caches(tmp_path):
    """Separate caches give the same draws for the same batch and seed, and new draws otherwise."""
    gen_a = make_cached_generate(make_memory(tmp_path / "a"), _prior_transform, _Simulator())
    gen_b = make_cached_generate(make_memory(tmp_path / "b"), _prior_transform, _Simulator())
    a = gen_a(batch_idx=1, n_sim=5, seed=123, n_params=3)
    b = gen_b(batch_idx=1, n_sim=5, seed=123, n_params=3)
    assert_array_equal(a['u_samples'], b['u_samples'])
    for p, q in zip(a['mean_props'], b['mean_props']):
        assert_array_equal(p['mean'], q['mean'])

    other_batch = gen_a(batch_idx=2, n_sim=5, seed=123, n_params=3)['u_samples']
    other_seed = gen_a(batch_idx=1, n_sim=5, seed=999, n_params=3)['u_samples']
    for other in (other_batch, other_seed):
        assert not np.any(other == a['u_samples'])


def test_cached_generate_batches_are_uniform(tmp_path):
    """Unit-cube draws pooled over several batches follow U(0, 1) (KS test)."""
    generate = make_cached_generate(make_memory(tmp_path), lambda u: u, lambda idx, seed, theta: theta)
    u = np.concatenate([generate(batch_idx=b, n_sim=50, seed=3, n_params=2)['u_samples'] for b in range(6)])
    assert u.shape == (300, 2)
    assert len(np.unique(u)) == u.size
    assert _ks_statistic_uniform(u.ravel()) < 1.63 / np.sqrt(u.size)


# ---------------------------------------------------------------- inject_noise_batch


@pytest.mark.parametrize("n_sim", [1, 6])
def test_inject_noise_batch_shapes_and_dtype(tmp_path, n_sim):
    """Noisy data are float32 with one row per simulation; u_samples are passed through."""
    generate = make_cached_generate(make_memory(tmp_path), _prior_transform, _Simulator(n_data=5))
    batch = generate(batch_idx=0, n_sim=n_sim, seed=4, n_params=2)
    u, raw = inject_noise_batch(batch, np.random.default_rng(0), _gaussian_noise)
    assert u is batch['u_samples']
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (n_sim, 5)
    assert raw.dtype == np.float32
    assert np.all(np.isfinite(raw))


def test_inject_noise_batch_matches_sequential_noise_draws():
    """Row i is inject_noise(mean_props[i], rng), drawn in order from the given generator."""
    means = [np.array([0.0, 1.0, 2.0]), np.array([10.0, -3.0, 0.5]), np.array([1e3, 1e-3, -7.0])]
    batch = {
        'u_samples': np.zeros((3, 2), dtype=np.float32),
        'mean_props': [{'mean': m, 'noise_std': np.full(3, 0.5)} for m in means],
    }
    _, raw = inject_noise_batch(batch, np.random.default_rng(99), _gaussian_noise)

    rng = np.random.default_rng(99)
    expected = np.array([_gaussian_noise(p, rng) for p in batch['mean_props']], dtype=np.float32)
    assert_array_equal(raw, expected)


def test_inject_noise_batch_passes_props_and_rng_through():
    """The noise function receives each props object unchanged, in order, with the caller's rng."""
    props_list = [{'mean': np.array([float(i)])} for i in range(4)]
    batch = {'u_samples': np.zeros((4, 1), dtype=np.float32), 'mean_props': props_list}
    rng = np.random.default_rng(0)
    seen = []

    def no_noise(props, r):
        seen.append((props, r))
        return props['mean']

    _, raw = inject_noise_batch(batch, rng, no_noise)
    assert len(seen) == 4
    for (props, r), expected_props in zip(seen, props_list):
        assert props is expected_props
        assert r is rng
    assert_array_equal(raw, np.arange(4, dtype=np.float32)[:, None])


def test_inject_noise_batch_casts_to_float32():
    """Float64 noise realisations are rounded to float32 like np.float32 does."""
    values = np.array([[np.pi, 1e-8, 1.0 + 1e-12], [-np.e, 3e38, 123456.789]])
    batch = {'u_samples': np.zeros((2, 1), dtype=np.float32), 'mean_props': list(values)}
    _, raw = inject_noise_batch(batch, np.random.default_rng(0), lambda props, rng: props)
    assert raw.dtype == np.float32
    assert_array_equal(raw, values.astype(np.float32))


def test_inject_noise_batch_reproducible_given_rng(tmp_path):
    """The same seed gives identical noisy data; a different seed gives different noise."""
    generate = make_cached_generate(make_memory(tmp_path), _prior_transform, _Simulator(n_data=4))
    batch = generate(batch_idx=0, n_sim=3, seed=8, n_params=2)
    _, raw1 = inject_noise_batch(batch, np.random.default_rng(5), _gaussian_noise)
    _, raw2 = inject_noise_batch(batch, np.random.default_rng(5), _gaussian_noise)
    _, raw3 = inject_noise_batch(batch, np.random.default_rng(6), _gaussian_noise)
    assert_array_equal(raw1, raw2)
    assert not np.any(raw1 == raw3)

    # the noise is centred on the cached noiseless means at the requested level
    means = np.array([p['mean'] for p in batch['mean_props']])
    assert np.all(np.abs(raw1 - means) < 5 * 0.1)


# ---------------------------------------------------------------- random_derangement


def _check_derangement(perm, n):
    """Assert that perm is a 1-D integer permutation of range(n) without fixed points."""
    assert isinstance(perm, torch.Tensor)
    assert perm.shape == (n,)
    assert perm.dtype == torch.int64
    assert_array_equal(np.sort(perm.numpy()), np.arange(n))
    assert not np.any(perm.numpy() == np.arange(n))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 10, 257])
def test_random_derangement_is_derangement(n):
    """Every draw is a permutation of range(n) with no element left in place."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(n)
        for _ in range(20):
            _check_derangement(random_derangement(n), n)


def test_random_derangement_empty():
    """The empty permutation is the (vacuous) derangement of zero elements."""
    _check_derangement(random_derangement(0), 0)


def test_random_derangement_two_elements():
    """The only derangement of two elements is the swap."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for _ in range(5):
            assert random_derangement(2).tolist() == [1, 0]


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_random_derangement_device(device):
    """The permutation is created on the requested device."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        perm = random_derangement(6, device=device)
    assert perm.device.type == "cpu"
    _check_derangement(perm, 6)


def test_random_derangement_reproducible_with_torch_seed():
    """The draw follows the torch global random state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        a = random_derangement(12)
        torch.manual_seed(1234)
        b = random_derangement(12)
    assert torch.equal(a, b)


@pytest.mark.parametrize("n", [3, 4])
def test_random_derangement_is_uniform(n):
    """All derangements of a small set appear, with equal frequency (chi-square test)."""
    all_derangements = {
        p for p in itertools.permutations(range(n))
        if all(p[i] != i for i in range(n))
    }
    # number of derangements: !3 = 2, !4 = 9
    assert len(all_derangements) == {3: 2, 4: 9}[n]

    n_draws = 100 * len(all_derangements)
    counts = dict.fromkeys(all_derangements, 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2025)
        for _ in range(n_draws):
            counts[tuple(random_derangement(n).tolist())] += 1

    observed = np.array(list(counts.values()))
    assert observed.sum() == n_draws
    expected = n_draws / len(all_derangements)
    chi2 = ((observed - expected) ** 2 / expected).sum()
    # 0.1% upper quantiles of the chi-square distribution with 1 and 8 degrees of freedom
    assert chi2 < {3: 10.83, 4: 26.12}[n]
