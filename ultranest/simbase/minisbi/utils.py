import numpy as np
import torch
import joblib


def sample_prior_u(rng, n_params):
    """
    Sample a single draw from a uniform prior over the unit hypercube.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used to draw samples.
    n_params : int
        Number of parameters (dimensionality of the hypercube).

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(n_params,)`` with dtype ``float32``,
        containing values sampled uniformly from ``[0.0, 1.0)``.
    """
    return rng.uniform(0.0, 1.0, size=n_params).astype(np.float32)


def make_memory(folder):
    """
    Create a joblib ``Memory`` object for caching results to disk.

    Parameters
    ----------
    folder : str or os.PathLike
        Path to the directory where cached results will be stored.

    Returns
    -------
    joblib.Memory
        A ``Memory`` instance configured to use ``folder`` as its cache
        location, with verbosity set to 0.
    """
    return joblib.Memory(folder, verbose=0)


def make_cached_generate(memory, prior_transform, generate_mean_and_noise):
    """
    Create a cached function that generates noiseless simulation batches.

    The returned function is decorated with ``memory.cache`` so that repeated
    calls with identical arguments are served from disk rather than recomputed.

    Parameters
    ----------
    memory : joblib.Memory
        Joblib memory object used to cache the inner function.
    prior_transform : callable
        Function that maps a unit-hypercube sample ``u`` (1-D array of shape
        ``(n_params,)``) to a parameter vector ``theta`` in the model space.
    generate_mean_and_noise : callable
        Function with signature ``(idx, seed, theta)`` that returns the
        noiseless (mean) summary properties for a single simulation.

    Returns
    -------
    callable
        A cached function ``generate_noiseless_batch(batch_idx, n_sim, seed,
        n_params)`` that returns a dict with keys:

        ``'u_samples'``
            2-D ``float32`` array of shape ``(n_sim, n_params)`` containing
            unit-hypercube draws.
        ``'mean_props'``
            List of length ``n_sim`` containing the noiseless summary
            properties for each simulation.
    """
    @memory.cache
    def generate_noiseless_batch(batch_idx, n_sim, seed, n_params):
        """
        Generate a batch of noiseless simulations and cache the result.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch, used to offset the random seed so
            that different batches produce independent draws.
        n_sim : int
            Number of simulations to generate in this batch.
        seed : int
            Base random seed; combined with ``batch_idx`` to seed the RNG.
        n_params : int
            Dimensionality of the parameter space (length of each ``u``
            sample).

        Returns
        -------
        dict
            Dictionary with keys:

            ``'u_samples'``
                2-D ``float32`` array of shape ``(n_sim, n_params)``.
            ``'mean_props'``
                List of length ``n_sim`` with noiseless summary properties.
        """
        rng = np.random.default_rng(seed + batch_idx)
        u_samples = np.stack([sample_prior_u(rng, n_params) for _ in range(n_sim)])

        mean_props_list = []
        for i in range(n_sim):
            theta = prior_transform(u_samples[i])
            props = generate_mean_and_noise(
                idx=batch_idx * n_sim + i,
                seed=seed,
                theta=theta,
            )
            mean_props_list.append(props)

        return {
            'u_samples':  u_samples,
            'mean_props': mean_props_list,
        }
    return generate_noiseless_batch


def inject_noise_batch(batch_dict, rng, inject_noise):
    """
    Inject noise into a pre-generated batch of noiseless simulations.

    Parameters
    ----------
    batch_dict : dict
        Dictionary returned by ``generate_noiseless_batch``, containing:

        ``'u_samples'``
            2-D ``float32`` array of shape ``(n_sim, n_params)``.
        ``'mean_props'``
            List of length ``n_sim`` with noiseless summary properties.
    rng : numpy.random.Generator
        Random number generator used by ``inject_noise`` to draw noise
        realisations.
    inject_noise : callable
        Function with signature ``(props, rng)`` that takes noiseless
        summary properties and returns a 1-D array of noisy data values.

    Returns
    -------
    u_samples : numpy.ndarray
        2-D ``float32`` array of shape ``(n_sim, n_params)`` containing the
        unit-hypercube parameter draws.
    raw_data : numpy.ndarray
        2-D ``float32`` array of shape ``(n_sim, n_data)`` containing the
        noisy simulation outputs, where ``n_data`` is the length of the
        array returned by ``inject_noise``.
    """
    u_samples = batch_dict['u_samples']
    mean_props_list = batch_dict['mean_props']
    n = len(mean_props_list)

    raw_data = None
    for i, props in enumerate(mean_props_list):
        noisy = inject_noise(props, rng)
        n_data = len(noisy)
        if raw_data is None:
            raw_data = np.empty((n, n_data), dtype=np.float32)
        raw_data[i] = noisy

    return u_samples, raw_data

def random_derangement(n, device=None):
    """
    Generate a uniformly random derangement (permutation without fixed points).

    The function resamples until a valid derangement is found.

    Parameters
    ----------
    n : int
        Number of elements to permute.
    device : torch.device or None, optional
        Device on which to create the permutation tensor.

    Returns
    -------
    torch.Tensor
        1-D integer tensor of length ``n`` with no element equal to its index.
    """
    while True:
        perm = torch.randperm(n, device=device)
        if not torch.any(perm == torch.arange(n, device=device)):
            return perm
