"""Validation / diagnostic utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .utils import inject_noise_batch
from .logistic import kuma_logistic_cdf


def rank_histogram(
    *,
    model,
    generate_noiseless_batch,
    inject_noise,
    n_params,
    folder,
    n_test=500,
    n_posterior_samples=200,
    seed=98765,
    param_names=None,
    prior_transform=None,
    n_bins=20,
):
    """
    Rank-histogram (Tallagrand / PIT) test.

    For each of *n_test* test simulations, the rank of the true parameter
    value is computed analytically from the Kumaraswamy-Logistic CDF, then
    converted to a discrete rank in [0, n_posterior_samples].  A
    well-calibrated posterior yields a flat histogram.

    Results are saved to <folder>/rank_histograms.pdf.

    Parameters
    ----------
    model : NPENetwork
        The trained neural posterior estimation network.
    generate_noiseless_batch : callable
        Cached generator that produces noiseless simulation batches.
    inject_noise : callable
        Noise injector applied to each noiseless simulation.
    n_params : int
        Number of model parameters.
    folder : str
        Output directory in which to save the rank histogram PDF.
    n_test : int, optional
        Number of test simulations. Default is 500.
    n_posterior_samples : int, optional
        Posterior draws per simulation, used to discretise the CDF rank.
        Default is 200.
    seed : int, optional
        RNG seed. Default is 98765.
    param_names : list of str or None, optional
        Names for each parameter. If None, defaults to
        ['param_0', 'param_1', ...].
    n_bins : int, optional
        Number of histogram bins. Default is 20.

    Returns
    -------
    ranks : np.ndarray
        Rank of the true value in [0, n_posterior_samples].
        Shape: (n_test, n_params).
    """
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    rng = np.random.default_rng(seed)

    # generate test set
    test_batch = generate_noiseless_batch(
        batch_idx=-1,
        n_sim=n_test,
        seed=seed,
        n_params=n_params,
    )
    _, test_raw = inject_noise_batch(test_batch, rng, inject_noise)
    test_u = test_batch['u_samples']          # shape (n_test, n_params), in [0,1]

    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(test_raw, dtype=torch.float32)
        loc_t, scale_t, a_t, b_t = model(x_t)

    loc_np   = loc_t.numpy()    # (n_test, n_params)
    scale_np = scale_t.numpy()  # (n_test, n_params)
    a_np     = a_t.numpy()      # (n_test, n_params)
    b_np     = b_t.numpy()      # (n_test, n_params)

    # Compute rank analytically via Kumaraswamy-Logistic CDF
    ranks = np.zeros((n_test, n_params), dtype=np.int32)
    for i in tqdm(range(n_test), desc="Rank histogram", unit="sim"):
        for p in range(n_params):
            cdf_val = kuma_logistic_cdf(
                test_u[i, p],
                loc_np[i, p],
                scale_np[i, p],
                a_np[i, p],
                b_np[i, p],
            )
            ranks[i, p] = int(np.floor(cdf_val * n_posterior_samples))

    # plot
    n_cols = min(n_params, 4)
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)
    for p in range(n_params):
        ax = axes[p // n_cols][p % n_cols]
        ax.hist(ranks[:, p], bins=n_bins, range=(0, n_posterior_samples),
                color='k', density=True, histtype='step')
        ax.axhline(1.0 / n_posterior_samples, color='gray', linestyle='--',
                   alpha=0.5)
        ax.set_title(param_names[p])
        ax.set_xlabel("rank")
        ax.set_ylabel("density")

    for p in range(n_params, n_rows * n_cols):
        axes[p // n_cols][p % n_cols].set_visible(False)

    return ranks, fig, axes


def parameter_coverage_test(
    *,
    model,
    generate_noiseless_batch,
    inject_noise,
    n_params,
    folder,
    n_test=500,
    credible_levels=None,
    seed=11223,
    param_names=None,
    prior_transform=None,
):
    """
    Expected-coverage (parameter coverage) test.

    For each test simulation, the rank of the true parameter under the
    approximate posterior is computed analytically from the
    Kumaraswamy-Logistic CDF.  The rank (a value in [0, 1]) is then compared
    against the nominal credible levels to determine coverage.

    Results are saved to <folder>/coverage_test.pdf.

    Parameters
    ----------
    model : NPENetwork
    generate_noiseless_batch : callable
    inject_noise : callable
    n_params : int
    folder : str
    n_test : int, optional
    credible_levels : list of float or None, optional
    seed : int, optional
    param_names : list of str or None, optional

    Returns
    -------
    coverage : np.ndarray, shape (len(credible_levels), n_params)
        Empirical coverage fraction at each credible level.
    """
    if credible_levels is None:
        credible_levels = np.linspace(0.05, 0.99, 20).tolist()
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    levels = np.asarray(credible_levels)
    rng = np.random.default_rng(seed)

    test_batch = generate_noiseless_batch(
        batch_idx=-1,
        n_sim=n_test,
        seed=seed,
        n_params=n_params,
    )
    _, test_raw = inject_noise_batch(test_batch, rng, inject_noise)
    test_u = test_batch['u_samples']          # (n_test, n_params) in [0,1]

    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(test_raw, dtype=torch.float32)
        loc_t, scale_t, a_t, b_t = model(x_t)

    loc_np   = loc_t.numpy()    # (n_test, n_params)
    scale_np = scale_t.numpy()  # (n_test, n_params)
    a_np     = a_t.numpy()      # (n_test, n_params)
    b_np     = b_t.numpy()      # (n_test, n_params)

    # cdf_vals[i, p] = P(X <= true_u[i, p]) under the Kuma-Logistic posterior
    cdf_vals = np.empty((n_test, n_params), dtype=np.float64)
    for i in tqdm(range(n_test), desc="Coverage test", unit="sim"):
        for p in range(n_params):
            cdf_vals[i, p] = kuma_logistic_cdf(
                test_u[i, p],
                loc_np[i, p],
                scale_np[i, p],
                a_np[i, p],
                b_np[i, p],
            )

    # coverage[l, p] = fraction of test cases where the true parameter falls
    # inside the symmetric credible interval at level levels[l].
    coverage = np.zeros((len(levels), n_params))
    for li, level in enumerate(levels):
        lo_cdf = (1.0 - level) / 2.0
        hi_cdf = (1.0 + level) / 2.0
        inside = (cdf_vals >= lo_cdf) & (cdf_vals <= hi_cdf)  # (n_test, n_params)
        coverage[li] = inside.mean(axis=0)

    # plot
    n_cols = min(n_params, 4)
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)
    for p in range(n_params):
        ax = axes[p // n_cols][p % n_cols]
        ax.plot(levels, coverage[:, p], 'o-', color='steelblue',
                markersize=4, label='empirical')
        ax.plot([0, 1], [0, 1], 'r--', label='ideal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(param_names[p])
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical coverage")
        ax.legend(fontsize=7)

    for p in range(n_params, n_rows * n_cols):
        axes[p // n_cols][p % n_cols].set_visible(False)

    fig.suptitle("Parameter coverage test (diagonal = well-calibrated)", fontsize=13)
    return coverage, fig, axes


def posterior_predictive_check(
    *,
    posterior_samples_theta,
    observed_data,
    generate_mean_and_noise,
    inject_noise,
    folder,
    n_mean_curves=200,
    n_realisation_curves=50,
    seed=77777,
    x_coords=None,
):
    """
    Posterior predictive check.

    Draws parameter samples from the posterior and generates:
      * posterior mean curves  (noiseless signal for each sample)
      * posterior data realisations  (noisy draws)
    then plots them together with the true observed data and saves the result
    to <folder>/posterior.pdf.

    Parameters
    ----------
    posterior_samples_theta : np.ndarray
        Physical parameter samples from the posterior.
        Shape: (n_posterior, n_params).
    observed_data : np.ndarray
        The actual observed dataset. Shape: (n_data,).
    generate_mean_and_noise : callable
        Same function used during simulation; generates noiseless signal
        properties given a parameter vector.
    inject_noise : callable
        Noise injector that takes simulation properties and an RNG instance
        and returns a noisy realisation.
    folder : str
        Output directory in which to save the posterior predictive PDF.
    n_mean_curves : int, optional
        How many posterior mean curves to overlay. Default is 200.
    n_realisation_curves : int, optional
        How many noisy realisations to overlay. Default is 50.
    seed : int, optional
        RNG seed for noise injection. Default is 77777.
    x_coords : np.ndarray or None, optional
        x-axis coordinates for the data. If None, defaults to
        np.linspace(-5, 5, n_data).

    Returns
    -------
    None
        Saves the figure to <folder>/posterior.pdf and prints the path.
    """
    rng = np.random.default_rng(seed)
    n_data = len(observed_data)

    if x_coords is None:
        x_coords = np.linspace(-5, 5, n_data)

    n_post = len(posterior_samples_theta)
    n_mean_curves        = min(n_mean_curves,        n_post)
    n_realisation_curves = min(n_realisation_curves, n_post)

    idx_mean = rng.choice(n_post, size=n_mean_curves, replace=False)
    idx_real = rng.choice(n_post, size=n_realisation_curves, replace=False)

    # --- collect posterior mean curves ---
    mean_curves = None
    for k, i in enumerate(idx_mean):
        props = generate_mean_and_noise(idx=10000 + k, seed=seed, theta=posterior_samples_theta[i])
        if 'mean' not in props:
            break
        if mean_curves is None:
            mean_curves = np.empty((n_mean_curves, n_data), dtype=np.float64)
        mean_curves[k] = props['mean']

    # --- collect posterior noisy realisations ---
    realisation_curves = np.empty((n_realisation_curves, n_data), dtype=np.float64)
    for k, i in enumerate(idx_real):
        props = generate_mean_and_noise(idx=20000 + k, seed=seed, theta=posterior_samples_theta[i])
        realisation_curves[k] = inject_noise(props, rng)

    # --- plot ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # top panel: mean curves
    ax = axes[0]
    if mean_curves is not None:
        for k in range(n_mean_curves):
            ax.plot(x_coords, mean_curves[k], color='steelblue', alpha=0.15,
                    linewidth=0.8, zorder=1)
        ax.plot(x_coords, mean_curves.mean(axis=0), color='navy',
                linewidth=2.0, label='posterior mean (avg)', zorder=3)
    ax.scatter(x_coords, observed_data, color='red', s=12, zorder=4,
               label='observed data', linewidths=0)
    ax.set_ylabel("signal")
    ax.legend(fontsize=9)

    # bottom panel: noisy realisations
    ax = axes[1]
    for k in range(n_realisation_curves):
        ax.plot(x_coords, realisation_curves[k], color='darkorange', alpha=0.20,
                linewidth=0.8, zorder=1)
    ax.plot(x_coords, realisation_curves.mean(axis=0), color='saddlebrown',
            linewidth=2.0, label='posterior realisation (avg)', zorder=3)
    ax.scatter(x_coords, observed_data, color='red', s=12, zorder=4,
               label='observed data', linewidths=0)
    ax.set_xlabel("x")
    ax.set_ylabel("signal + noise")
    ax.set_title(f"Posterior data realisations  (n={n_realisation_curves})")
    ax.legend(fontsize=9)

    fig.suptitle("Posterior predictive check", fontsize=13)
    return realisation_curves, fig, axes
