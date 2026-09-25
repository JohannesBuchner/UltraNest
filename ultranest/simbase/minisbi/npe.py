"""Neural Posterior Estimation (NPE) training."""

import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

from .logistic import nll_kuma_logistic_product, sample_kuma_logistic_product
from .norm import ZScoreNorm
from .utils import inject_noise_batch


def _arch_suffix(depth, width, activation_cls):
    """
    Build a string suffix that encodes the architecture hyper-parameters.

    Parameters
    ----------
    depth : int
        Number of hidden layers.
    width : int
        Width of each hidden layer.
    activation_cls : str or type
        Activation class or its name.

    Returns
    -------
    str
        A ``'_'``-separated string such as ``'128_128_ReLU'``.
    """
    if isinstance(activation_cls, str):
        act = activation_cls
    else:
        act = activation_cls.__name__
    parts = [f"{width}"] * depth + [act]
    return "_".join(parts)


class CascadeNet(nn.Module):
    """Cascade MLP architecture.

    Each hidden layer forwards half of its output neurons directly to the
    final layer (skip connection) and the other half to the next hidden layer.

    Concretely, layer i produces ``hidden_widths[i]`` neurons. We split them
    evenly: the first half (``skip_size = hidden_widths[i] // 2``) accumulates
    in a "cascade buffer" that is concatenated to the input of the final
    linear layer; the second half (``pass_size``) is forwarded to the next
    hidden layer.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the network input.
    output_dim : int
        Dimensionality of the network output.
    hidden_widths : list of int
        Number of neurons in each hidden layer.
    activation_cls : type
        Activation class (e.g. ``nn.ReLU``); instantiated per layer.
    """

    def __init__(self, input_dim, output_dim, hidden_widths, activation_cls):
        """Initialise."""
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        in_dim = input_dim
        for hw in hidden_widths:
            self.hidden_layers.append(nn.Linear(in_dim, hw))
            self.activations.append(activation_cls())
            pass_size = hw - hw // 2
            in_dim = pass_size

        skip_total = sum(hw // 2 for hw in hidden_widths)
        final_in = (hidden_widths[-1] - hidden_widths[-1] // 2) + skip_total
        self.final_layer = nn.Linear(final_in, output_dim)

    def forward(self, x):
        """
        Forward pass through the cascade network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., input_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., output_dim)``.
        """
        skip_accumulator = []
        current = x
        for linear, act in zip(self.hidden_layers, self.activations):
            out = act(linear(current))
            skip_size = out.shape[-1] // 2
            skip = out[..., :skip_size]
            passed = out[..., skip_size:]
            skip_accumulator.append(skip)
            current = passed

        final_input = torch.cat([current] + skip_accumulator, dim=-1)
        return self.final_layer(final_input)


def _build_layers(input_dim, output_dim, depth, width, activation_cls, shape='rectangular'):
    """
    Build a list of ``nn.Module`` layers for an MLP.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the network input.
    output_dim : int
        Dimensionality of the network output.
    depth : int
        Number of hidden layers.
    width : int
        Base width; exact meaning depends on ``shape``.
    activation_cls : type
        Activation class (already resolved), e.g. ``nn.ReLU``.
    shape : str, optional
        One of ``'rectangular'``, ``'triangular'``, or ``'cascade'``.
        Default is ``'rectangular'``.

    Returns
    -------
    list of nn.Module
        Ordered list of layers ready to be passed to ``nn.Sequential``,
        or a single-element list containing a :class:`CascadeNet` when
        ``shape='cascade'``.
    """
    if shape == 'rectangular':
        hidden_widths = [width] * depth

    elif shape == 'triangular':
        # Linearly decrease from width down to output_dim across depth layers.
        if depth == 1:
            hidden_widths = [width]
        else:
            hidden_widths = [
                max(output_dim, int(round(width - (width - output_dim) * i / (depth - 1))))
                for i in range(depth)
            ]

    elif shape == 'cascade':
        # Each layer is half as wide as the previous one.
        # The first layer has `width` neurons; each subsequent layer has half.
        hidden_widths = []
        w = width
        # no use to go very deep if we do not have neurons left
        min_width = 4
        for _ in range(depth):
            if not w >= min_width * 2:
                break
            hidden_widths.append(max(2, w))
            w = max(2, w // 2)

    else:
        raise ValueError(f"Unknown layer shape '{shape}'. "
                         "Choose 'rectangular', 'triangular', or 'cascade'.")

    print("hidden layer widths:", hidden_widths)

    if shape == 'cascade':
        return [CascadeNet(input_dim, output_dim, hidden_widths, activation_cls)]

    else:
        layers = [nn.Linear(input_dim, hidden_widths[0]), activation_cls()]
        for i in range(1, depth):
            layers += [nn.Linear(hidden_widths[i - 1], hidden_widths[i]), activation_cls()]
        layers.append(nn.Linear(hidden_widths[-1], output_dim))

    return layers


class NPENetwork(nn.Module):
    """Network that outputs a product of Kumaraswamy-Logistic chained distributions living on the unit cube [0, 1]^d.

    For each parameter dimension the network predicts 4 values:
      - loc       (via sigmoid, in (0,1))
      - log_scale (unconstrained; exponentiated -> scale > 0)
      - log_a     (unconstrained; exponentiated -> a > 0, Kumaraswamy shape)
      - log_b     (unconstrained; exponentiated -> b > 0, Kumaraswamy shape)

    Parameters
    ----------
    n_data : int
        Raw input dimension.
    n_params : int
        Number of model parameters.
    depth : int
        Number of hidden layers.
    width : int
        Width of each hidden layer.
    activation_name : str
        Activation name (e.g. ``'ReLU'``).
    layer_shape : str
        Architecture shape: ``'rectangular'``, ``'triangular'``, or
        ``'cascade'``.
    """

    def __init__(
        self, n_data, n_params, depth, width,
        activation_name, layer_shape: str
    ):
        """Initialise."""
        super().__init__()
        activation_cls = getattr(nn, activation_name)
        self.n_params = n_params

        self.norm = ZScoreNorm(n_data)

        # 4 outputs per parameter: loc_raw, log_scale, log_a, log_b
        backbone_layers = _build_layers(
            input_dim=n_data,
            output_dim=4 * n_params,
            depth=depth,
            width=width,
            activation_cls=activation_cls,
            shape=layer_shape,
        )
        self.backbone = nn.Sequential(*backbone_layers)
        self.loc_activation = nn.Sigmoid()

    def fit_norm(self, x_np: np.ndarray) -> None:
        """
        Fit the moment-matching normalisation layer on a representative sample.

        Parameters
        ----------
        x_np : numpy.ndarray
            Array of shape ``(n_samples, n_data)`` used to compute
            the normalisation statistics.

        Returns
        -------
        None
        """
        self.norm.fit(x_np)

    def forward(self, x_raw):
        """
        Map raw data to per-parameter distribution parameters.

        Parameters
        ----------
        x_raw : torch.Tensor  (batch, n_data)

        Returns
        -------
        loc   : torch.Tensor  (batch, n_params)  in (0,1)
        scale : torch.Tensor  (batch, n_params)  > 0
        a     : torch.Tensor  (batch, n_params)  > 0
        b     : torch.Tensor  (batch, n_params)  > 0
        """
        x_norm = self.norm(x_raw)
        out = self.backbone(x_norm)          # (batch, 4 * n_params)

        n = self.n_params
        loc_raw = out[:, 0 * n:1 * n]
        log_scale = out[:, 1 * n:2 * n]
        log_a = out[:, 2 * n:3 * n]
        log_b = out[:, 3 * n:4 * n]

        loc = self.loc_activation(loc_raw)
        scale = torch.exp(log_scale.clamp(-8, 4))
        a = torch.exp(log_a.clamp(-4, 4))
        b = torch.exp(log_b.clamp(-4, 4))

        return loc, scale, a, b


def sample_posterior(
    model,
    observed_data,
    prior_transform,
    n_params,
    n_posterior_samples,
):
    """
    Draw posterior samples for a single observed dataset.

    Parameters
    ----------
    model : NPENetwork
        Trained NPE model.
    observed_data : numpy.ndarray
        1-D array of observed data values, shape ``(n_data,)``.
    prior_transform : callable
        Function ``(u) -> theta`` mapping unit-cube samples to the
        physical parameter space.
    n_params : int
        Number of model parameters.
    n_posterior_samples : int
        Number of posterior samples to return.

    Returns
    -------
    posterior_samples_u : numpy.ndarray
        Samples in unit-cube space, shape ``(n_posterior_samples, n_params)``.
    posterior_samples_theta : numpy.ndarray
        Samples in physical parameter space, shape
        ``(n_posterior_samples, n_params)``.
    """
    x_obs_t = torch.tensor(observed_data, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        loc, scale, a, b = model(x_obs_t)
        loc = loc.squeeze(0)
        scale = scale.squeeze(0)
        a = a.squeeze(0)
        b = b.squeeze(0)

    loc_np = loc.numpy()
    scale_np = scale.numpy()
    a_np = a.numpy()
    b_np = b.numpy()

    print("\nPredicted posterior (Kumaraswamy-Logistic product, unit-cube space):")
    for d in range(n_params):
        print(f"  Param {d}: loc={loc_np[d]:.4f}  scale={scale_np[d]:.4f}"
              f"  a={a_np[d]:.4f}  b={b_np[d]:.4f}")

    rng = np.random.default_rng(0)
    posterior_samples_u = sample_kuma_logistic_product(
        loc_np, scale_np, a_np, b_np, n_posterior_samples, rng
    )

    posterior_samples_theta = np.array([
        prior_transform(u) for u in posterior_samples_u
    ])

    return posterior_samples_u, posterior_samples_theta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_npe(
    *,
    folder,
    generate_noiseless_batch,
    inject_noise,
    n_params,
    fresh_sim_batch_size,
    npe_lr,
    base_seed,
    max_model_evals,
    num_processes=1,
    patience=30,
    patience_min_delta=1e-4,
    npe_batches_epoch=8,
    npe_width=1024,
    npe_activation_cls='ReLU',
    npe_depth=6,
    val_size=1024,
    norm_size=2000,
    layer_shape='cascade',
    fresh_example_fraction=0.5,
):
    """Train a Neural Posterior Estimator (NPE) with a product-of-Kumaraswamy-Logistic output.

    Parameters
    ----------
    folder : str
        Directory in which to save or load the trained model.
    generate_noiseless_batch : callable
        Simulation function with signature
        ``(batch_idx, n_sim, seed, n_params) -> dict``.
    inject_noise : callable
        Function ``(props, rng) -> noisy_vector``.
    n_params : int
        Number of model parameters.
    fresh_sim_batch_size : int
        Number of fresh simulator draws requested per sub-batch. This is the
        number of new simulations run each time the simulator is called; it is
        *not* the number of examples seen by the neural network in one gradient
        step (see ``effective_train_batch_size``).
    npe_lr : float
        Learning rate for the Adam optimiser.
    base_seed : int
        Base random seed used throughout training.
    max_model_evals : int
        Maximum total number of simulator evaluations to generate across all
        training batches.
    num_processes : int, optional
        Number of parallel worker processes. Default is ``1``.
    patience : int, optional
        Number of epochs without improvement before early stopping.
        Default is ``30``.
    patience_min_delta : float, optional
        Minimum validation-loss improvement to reset the patience counter.
        Default is ``1e-4``.
    npe_batches_epoch : int, optional
        Number of simulation batches drawn per epoch. Default is ``8``.
    npe_width : int, optional
        Width of each hidden layer. Default is ``1024``.
    npe_activation_cls : str
        Activation function name. Default is ``'ReLU'``.
    npe_depth : int, optional
        Number of hidden layers. Default is ``6``.
    val_size : int, optional
        Number of samples in the fixed validation set. Default is ``1024``.
    norm_size : int, optional
        Number of samples for determining the input normalisation validation set. Default is ``2000``.
    layer_shape : str, optional
        Architecture shape: ``'rectangular'`` (default), ``'triangular'``, or
        ``'cascade'``.
    fresh_example_fraction : float, optional
        Fraction of each training mini-batch that consists of freshly
        simulated examples. The remainder
        (``1 - fresh_example_fraction``) is filled by replaying examples
        from the history buffer, so the effective batch size seen by the
        network (``effective_train_batch_size``) is larger than
        ``fresh_sim_batch_size``.  Must be in ``(0, 1]``.  When set to
        ``1.0`` no replay is used and
        ``effective_train_batch_size == fresh_sim_batch_size``.
        Default is ``0.5``.

    Notes
    -----
    The relationship between the key batch-size quantities is::

        replay_example_count    = fresh_sim_batch_size
                                  * (1 - fresh_example_fraction)
                                  / fresh_example_fraction
        effective_train_batch_size = fresh_sim_batch_size
                                     + replay_example_count

    ``fresh_simulator_evals`` counts the cumulative number of simulator
    calls made so far (i.e. the total number of freshly generated
    parameter–data pairs, excluding replayed examples).

    Returns
    -------
    model : NPENetwork
        Trained NPE model in eval mode.
    """
    if not (0.0 < fresh_example_fraction <= 1.0):
        raise ValueError(f"fresh_example_fraction must be in (0, 1], got {fresh_example_fraction}")

    total_npe_batches = max(1, max_model_evals // fresh_sim_batch_size)
    npe_epochs = max(1, total_npe_batches // npe_batches_epoch)

    norm_batches_needed = max(1, (norm_size + fresh_sim_batch_size - 1) // fresh_sim_batch_size)
    val_batches_needed = max(1, (val_size + fresh_sim_batch_size - 1) // fresh_sim_batch_size)
    prefix_batches = norm_batches_needed + val_batches_needed
    total_batches = prefix_batches + total_npe_batches

    # Derive replay_example_count from fresh_example_fraction:
    # fresh_example_fraction = n_fresh / (n_fresh + replay_example_count)
    # => replay_example_count = n_fresh * (1 - fresh_example_fraction) / fresh_example_fraction
    n_fresh = fresh_sim_batch_size
    replay_example_count = int(round(n_fresh * (1.0 - fresh_example_fraction) / fresh_example_fraction))

    print(f"fresh_example_fraction={fresh_example_fraction:.3f} =>  "
          f"n_fresh={n_fresh}, replay_example_count={replay_example_count} per sub-batch")

    print(f"Starting parallel simulation generator ({total_batches} batches total, "
          f"{num_processes} workers) ...")

    parallel_gen = joblib.Parallel(
        n_jobs=num_processes,
        return_as='generator',
    )(
        joblib.delayed(generate_noiseless_batch)(
            batch_idx=b,
            n_sim=fresh_sim_batch_size,
            seed=base_seed,
            n_params=n_params,
        )
        for b in range(total_batches)
    )

    # Probe n_data from the very first batch.
    probe_batch = next(parallel_gen)
    probe_rng = np.random.default_rng(base_seed ^ 0xDEAD)
    _, probe_raw = inject_noise_batch(probe_batch, probe_rng, inject_noise)
    n_data = probe_raw.shape[1]
    print(f"Detected n_data={n_data} from probe simulation.")

    norm_batches_collected = [probe_batch]
    for _ in range(norm_batches_needed - 1):
        norm_batches_collected.append(next(parallel_gen))

    # ------------------------------------------------------------------ #
    # Build model and optionally load from disk.                           #
    # ------------------------------------------------------------------ #
    npe_suffix = _arch_suffix(npe_depth, npe_width, npe_activation_cls)
    model_path = os.path.join(
        folder,
        f"minisbi_npe_{n_data}d_{npe_suffix}_KLP_{layer_shape}.pt"
    )

    model = NPENetwork(
        n_data=n_data,
        n_params=n_params,
        depth=npe_depth,
        width=npe_width,
        activation_name=npe_activation_cls,
        layer_shape=layer_shape,
    )

    if os.path.exists(model_path):
        print(f"Loading trained NPE model from {model_path}")
        ckpt = torch.load(model_path, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    torchinfo.summary(model)
    print("Training NPE (product of Kumaraswamy-Logistic distributions) "
          "with on-the-fly batch generation ...")

    print("Computing normalisation statistics ...")
    norm_rng = np.random.default_rng(base_seed ^ 0xC0FFEE)
    norm_raw_parts = []
    for nb in norm_batches_collected:
        _, raw_part = inject_noise_batch(nb, norm_rng, inject_noise)
        norm_raw_parts.append(raw_part)
    norm_raw = np.concatenate(norm_raw_parts, axis=0)[:norm_size]
    model.fit_norm(norm_raw)
    print("Input normalisation fitted.")

    # ------------------------------------------------------------------ #
    # Build fixed validation set.                                          #
    # ------------------------------------------------------------------ #
    print(f"Generating fixed NPE validation set (val_size={val_size}) ...")
    val_rng = np.random.default_rng(base_seed ^ 0xBEEF)
    val_u_parts = []
    val_raw_parts = []
    for _ in range(val_batches_needed):
        vb = next(parallel_gen)
        vu, vr = inject_noise_batch(vb, val_rng, inject_noise)
        val_u_parts.append(vu)
        val_raw_parts.append(vr)
    val_u_np = np.concatenate(val_u_parts, axis=0)[:val_size]
    val_raw_np = np.concatenate(val_raw_parts, axis=0)[:val_size]
    val_uv_unit = torch.tensor(val_u_np, dtype=torch.float32)
    val_xv_raw = torch.tensor(val_raw_np, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    print("Training NPE with on-the-fly batch generation ...")

    optimizer = optim.Adam(model.parameters(), lr=npe_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=npe_epochs)

    best_val_loss = np.inf
    best_state = None
    patience_count = 0

    noiseless_history = []

    rng_reuse = np.random.default_rng(base_seed ^ 0x1234)
    batch_counter = 0
    fresh_simulator_evals = 0

    model.train()

    for epoch in range(1, npe_epochs + 1):
        epoch_loss = 0.0
        epoch_n = 0

        # each epoch has multiple sub-batches
        for _ in range(npe_batches_epoch):
            # we load a fresh dataset batch from the simulator
            fresh_batch = next(parallel_gen)
            batch_counter += 1
            fresh_simulator_evals += fresh_sim_batch_size

            combined_u = [fresh_batch['u_samples']]
            combined_props = [fresh_batch['mean_props']]

            # Determine how many historical samples to reuse based on
            # fresh_example_fraction:
            # replay_example_count = n_fresh
            #                        * (1 - fresh_example_fraction)
            #                        / fresh_example_fraction
            n_reuse = 0
            if len(noiseless_history) > 0 and replay_example_count > 0:
                all_u = np.concatenate(
                    [h['u_samples'] for h in noiseless_history], axis=0
                )
                all_props = [p for h in noiseless_history for p in h['mean_props']]
                history_total = len(all_props)

                n_reuse = min(replay_example_count, history_total)
                idx = rng_reuse.choice(
                    history_total, size=n_reuse,
                    replace=(n_reuse > history_total)
                )
                combined_u.append(all_u[idx])
                combined_props.append([all_props[i] for i in idx])

            effective_train_batch_size = n_fresh + n_reuse
            u_batch_np = np.concatenate(combined_u, axis=0)
            x_batch_np = None

            noise_seed = (base_seed ^ (batch_counter * 131071)) % (2**31)
            rng_inject = np.random.default_rng(noise_seed)

            flat_props = combined_props[0] + (
                combined_props[1] if len(combined_props) > 1 else []
            )
            # inject fresh simulation noise (this is assumed to be cheap)
            for i, props in enumerate(flat_props):
                noisy = inject_noise(props, rng_inject)
                if x_batch_np is None:
                    x_batch_np = np.empty(
                        (effective_train_batch_size, len(noisy)), dtype=np.float32
                    )
                x_batch_np[i] = noisy

            noiseless_history.append(fresh_batch)

            x_batch = torch.tensor(x_batch_np, dtype=torch.float32)
            u_batch_unit = torch.tensor(u_batch_np, dtype=torch.float32)

            optimizer.zero_grad()
            loc, scale, a, b_param = model(x_batch)
            loss = nll_kuma_logistic_product(u_batch_unit, loc, scale, a, b_param)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(x_batch)
            epoch_n += len(x_batch)

        avg_loss = epoch_loss / epoch_n

        # Validation
        model.eval()
        with torch.no_grad():
            vloc, vscale, va, vb = model(val_xv_raw)
            vloss = nll_kuma_logistic_product(
                val_uv_unit, vloc, vscale, va, vb
            ).item()
        model.train()

        improved = vloss < best_val_loss - patience_min_delta
        if improved:
            best_val_loss = vloss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
            marker = "*"
        else:
            patience_count += 1
            marker = f"[{patience_count}/{patience}]"

        scheduler.step()
        print(f"  Epoch {epoch:3d}/{npe_epochs}  train_loss={avg_loss:.4f}"
              f"  val_loss={vloss:.4f}"
              f"  fresh_simulator_evals={fresh_simulator_evals}"
              f"  {marker}")

        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience},"
                  f" fresh_simulator_evals={fresh_simulator_evals})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    torch.save(
        {"state_dict": model.state_dict()},
        model_path,
    )
    print(f"Model saved to {model_path}")
    model.eval()
    return model
