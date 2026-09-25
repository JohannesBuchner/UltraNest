"""Kumaraswamy-Logistic chained distribution defined on a unit hypercube."""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Kumaraswamy-Logistic chained distribution helpers
#
# Construction:
#   1. Draw v ~ TruncatedLogistic(loc, scale) on (0,1)   [the "base" draw]
#   2. Apply the Kumaraswamy CDF  u = 1 - (1 - v^a)^b   to map v -> u in (0,1)
#
# The joint density factors as:
#   p(u) = p_L(v(u)) * |dv/du|
# where v(u) is the inverse Kumaraswamy CDF (= Kumaraswamy quantile function):
#   v = (1 - (1 - u)^{1/b})^{1/a}
# and the log |dv/du| term comes from the Kumaraswamy log-density evaluated
# at v with the extra -log(a) -log(b) factored in correctly.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NumPy utility functions
# ---------------------------------------------------------------------------

def sigmoid(x):
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    x : numpy.ndarray or float

    Returns
    -------
    numpy.ndarray or float
        Sigmoid of x, clipped to avoid overflow.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


# Internal alias used within numpy helpers
_sigmoid = sigmoid


def logit(p):
    """
    Logit (inverse sigmoid) function.

    Parameters
    ----------
    p : numpy.ndarray or float
        Probability value(s) in (0, 1).

    Returns
    -------
    numpy.ndarray or float
        Log-odds of p.
    """
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))


# Internal alias used within numpy helpers
_logit = logit


# ---------------------------------------------------------------------------
# Kumaraswamy distribution helpers
# ---------------------------------------------------------------------------

def kuma_log_prob(u, a, b):
    """
    Log-density of Kumaraswamy(a, b) at u in (0, 1).

    log p(u) = log(a) + log(b) + (a-1)*log(u) + (b-1)*log(1 - u^a)

    Parameters
    ----------
    u : torch.Tensor
        Values in (0, 1).
    a : torch.Tensor
        Shape parameter, > 0.
    b : torch.Tensor
        Shape parameter, > 0.

    Returns
    -------
    torch.Tensor
        Log-density at u.
    """
    u = u.clamp(1e-7, 1.0 - 1e-7)
    ua = u.pow(a)
    log_p = (torch.log(a) + torch.log(b)
             + (a - 1.0) * torch.log(u)
             + (b - 1.0) * torch.log((1.0 - ua).clamp(min=1e-30)))
    return log_p


def kuma_icdf(u, a, b):
    """
    Inverse CDF (quantile function) of Kumaraswamy(a, b).

    Maps u in (0, 1) to v in (0, 1) via:
      v = (1 - (1 - u)^{1/b})^{1/a}

    Parameters
    ----------
    u : torch.Tensor
        Values in (0, 1).
    a : torch.Tensor
        Shape parameter, > 0.
    b : torch.Tensor
        Shape parameter, > 0.

    Returns
    -------
    torch.Tensor
        Quantile values in (0, 1).
    """
    u = u.clamp(1e-7, 1.0 - 1e-7)
    return (1.0 - (1.0 - u).pow(1.0 / b)).pow(1.0 / a)


def kuma_icdf_np(u, a, b):
    """
    Inverse CDF (quantile function) of Kumaraswamy(a, b) — NumPy version.

    Maps u in (0, 1) to v in (0, 1) via:
      v = (1 - (1 - u)^{1/b})^{1/a}

    Parameters
    ----------
    u : numpy.ndarray
        Values in (0, 1).
    a : numpy.ndarray
        Shape parameter, > 0.
    b : numpy.ndarray
        Shape parameter, > 0.

    Returns
    -------
    numpy.ndarray
        Quantile values in (0, 1).
    """
    u = np.clip(u, 1e-7, 1.0 - 1e-7)
    return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)


# ---------------------------------------------------------------------------
# Logistic distribution helpers (PyTorch)
# ---------------------------------------------------------------------------

def _logistic_log_prob(u, loc, scale):
    """
    Log-probability of u under Logistic(loc, scale).

    Parameters
    ----------
    u : torch.Tensor
        Shape (batch, n_params).
    loc : torch.Tensor
        Shape (batch, n_params).
    scale : torch.Tensor
        Shape (batch, n_params).

    Returns
    -------
    torch.Tensor
        Shape (batch, n_params), per-dimension log-probability.
    """
    z = (u - loc) / scale
    log_p = -z - torch.log(scale) - 2.0 * torch.nn.functional.softplus(-z)
    return log_p


def logistic_log_cdf(x, loc, scale):
    """
    Log CDF of Logistic(loc, scale).

    Computes log sigmoid((x - loc) / scale).

    Parameters
    ----------
    x : torch.Tensor
    loc : torch.Tensor
    scale : torch.Tensor

    Returns
    -------
    torch.Tensor
        Log CDF evaluated at x.
    """
    return torch.nn.functional.logsigmoid((x - loc) / scale)


def _logistic_log_sf(x, loc, scale):
    """
    Log survival function of Logistic(loc, scale).

    Computes log(1 - sigmoid((x - loc) / scale)) = log sigmoid(-(x - loc) / scale).

    Parameters
    ----------
    x : torch.Tensor
    loc : torch.Tensor
    scale : torch.Tensor

    Returns
    -------
    torch.Tensor
        Log survival function evaluated at x.
    """
    return torch.nn.functional.logsigmoid(-(x - loc) / scale)


def _logistic_log_normaliser(loc, scale):
    """
    Log probability mass of Logistic(loc, scale) inside (0, 1).

    Computes log(CDF(1) - CDF(0)) per dimension.

    Parameters
    ----------
    loc : torch.Tensor
        Shape (batch, n_params).
    scale : torch.Tensor
        Shape (batch, n_params).

    Returns
    -------
    torch.Tensor
        Shape (batch, n_params).
    """
    log_cdf1 = logistic_log_cdf(torch.ones_like(loc), loc, scale)
    log_cdf0 = logistic_log_cdf(torch.zeros_like(loc), loc, scale)
    log_mass = torch.log(
        torch.clamp(log_cdf1.exp() - log_cdf0.exp(), min=1e-30)
    )
    return log_mass


# ---------------------------------------------------------------------------
# Chained (Kumaraswamy o TruncatedLogistic) distribution — PyTorch
# ---------------------------------------------------------------------------

def nll_kuma_logistic_product(u, loc, scale, a, b):
    """
    Negative log-likelihood under a product of Kumaraswamy-Logistic chained
    distributions on (0, 1).

    The generative model per dimension is:
      v ~ TruncatedLogistic(loc, scale)  on (0, 1)
      u = Kuma_CDF(v; a, b)  = 1 - (1 - v^a)^b

    The density of u is obtained by the change of variables v = Kuma_ICDF(u):
      log p_U(u) = log p_L(v; loc, scale) - log Z(loc, scale) + log|dv/du|

    where log|dv/du| = -log p_Kuma(v; a, b) (the Kumaraswamy log-density
    evaluated at v gives the magnitude of the Jacobian of the inverse map).

    Parameters
    ----------
    u : torch.Tensor
        Shape (batch, n_params), values in (0, 1).
    loc : torch.Tensor
        Shape (batch, n_params).
    scale : torch.Tensor
        Shape (batch, n_params), > 0.
    a : torch.Tensor
        Shape (batch, n_params), > 0. Kumaraswamy shape parameter.
    b : torch.Tensor
        Shape (batch, n_params), > 0. Kumaraswamy shape parameter.

    Returns
    -------
    torch.Tensor
        Scalar mean NLL.
    """
    # v = Kumaraswamy inverse CDF applied to u
    v = kuma_icdf(u, a, b)                           # (batch, n_params), in (0,1)

    # log p_L(v) per dimension (untruncated logistic)
    log_pL_v = _logistic_log_prob(v, loc, scale)     # (batch, n_params)

    # log normaliser per dimension
    log_Z = _logistic_log_normaliser(loc, scale)     # (batch, n_params)

    # log|dv/du| = -log(Kuma density at v w.r.t. Kuma(a, b))
    # Kuma density: p_K(v) = a*b*v^{a-1}*(1-v^a)^{b-1}
    v_c = v.clamp(1e-7, 1.0 - 1e-7)
    va = v_c.pow(a)
    log_kuma_at_v = (torch.log(a) + torch.log(b)
                     + (a - 1.0) * torch.log(v_c)
                     + (b - 1.0) * torch.log((1.0 - va).clamp(min=1e-30)))
    log_abs_dv_du = -log_kuma_at_v                  # (batch, n_params)

    # log p_U(u) per dimension
    log_p_u = log_pL_v - log_Z + log_abs_dv_du      # (batch, n_params)

    # sum over parameter dimensions, mean over batch
    nll = -log_p_u.sum(dim=-1).mean()
    return nll


def sample_kuma_logistic_product(loc, scale, a, b, n_samples, rng):
    """
    Sample from a product of per-dimension Kumaraswamy-Logistic chained
    distributions on (0, 1) using analytic inverse CDF.

    The inverse CDF of U = Kuma_CDF(V) where V ~ TruncLogistic is:
      1. Draw uniform w in (0, 1).
      2. v = TruncLogistic_ICDF(w; loc, scale) via standard inversion.
      3. u = Kuma_CDF(v; a, b) = 1 - (1 - v^a)^b.

    Parameters
    ----------
    loc : numpy.ndarray
        Shape (n_params,).
    scale : numpy.ndarray
        Shape (n_params,).
    a : numpy.ndarray
        Shape (n_params,), Kumaraswamy shape parameter, > 0.
    b : numpy.ndarray
        Shape (n_params,), Kumaraswamy shape parameter, > 0.
    n_samples : int
        Number of samples to draw.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Shape (n_samples, n_params), values in (0, 1).
    """
    n_params = loc.shape[0]

    # Truncated-logistic CDF bounds
    p0 = sigmoid((0.0 - loc) / scale)   # (n_params,)
    p1 = sigmoid((1.0 - loc) / scale)   # (n_params,)

    # Step 1: uniform draws -> truncated-logistic samples v
    w = rng.uniform(0.0, 1.0, size=(n_samples, n_params))
    p_draw = p0[None, :] + w * (p1 - p0)[None, :]
    p_draw = np.clip(p_draw, 1e-15, 1.0 - 1e-15)
    v = loc[None, :] + scale[None, :] * logit(p_draw)   # (n_samples, n_params)
    v = np.clip(v, 1e-7, 1.0 - 1e-7)

    # Step 2: apply Kumaraswamy CDF  u = 1 - (1 - v^a)^b
    u = 1.0 - (1.0 - v ** a[None, :]) ** b[None, :]
    u = np.clip(u, 0.0, 1.0)
    return u.astype(np.float32)


# ---------------------------------------------------------------------------
# Chained (Kumaraswamy o TruncatedLogistic) distribution — NumPy
# ---------------------------------------------------------------------------

def kuma_logistic_cdf(x_val, loc, scale, a, b):
    """
    CDF of the Kumaraswamy-Logistic chained distribution at a scalar point.

    The generative model is:
      v ~ TruncatedLogistic(loc, scale) on (0, 1)
      u = 1 - (1 - v^a)^b

    The CDF of u at x is P(u <= x) = P(v <= Kuma_ICDF(x; a, b)) under the
    truncated logistic.

    Parameters
    ----------
    x_val : float
        Point in [0, 1].
    loc : float
    scale : float
        Must be > 0.
    a : float
        Kumaraswamy shape parameter, > 0.
    b : float
        Kumaraswamy shape parameter, > 0.

    Returns
    -------
    float
        CDF value in [0, 1].
    """
    x_val = float(np.clip(x_val, 1e-7, 1.0 - 1e-7))

    # v = Kuma_ICDF(x_val; a, b)
    v = (1.0 - (1.0 - x_val) ** (1.0 / b)) ** (1.0 / a)
    v = float(np.clip(v, 1e-7, 1.0 - 1e-7))

    # truncated-logistic CDF normaliser
    p0 = sigmoid((0.0 - loc) / scale)
    p1 = sigmoid((1.0 - loc) / scale)
    mass = p1 - p0
    if mass < 1e-30:
        mass = 1e-30

    # P(V <= v) under the truncated logistic
    pv = sigmoid((v - loc) / scale)
    cdf_val = (pv - p0) / mass
    return float(np.clip(cdf_val, 0.0, 1.0))


def kuma_logistic_cdf_vec(u_vec, loc, scale, a, b):
    """
    CDF of the Kumaraswamy-Logistic chained distribution, evaluated
    element-wise over a vector of points.

    Parameters
    ----------
    u_vec : numpy.ndarray
        Shape (d,), points at which to evaluate the CDF, each in (0, 1).
    loc : numpy.ndarray
        Shape (d,).
    scale : numpy.ndarray
        Shape (d,), must be positive.
    a : numpy.ndarray
        Shape (d,), Kumaraswamy shape parameter, must be positive.
    b : numpy.ndarray
        Shape (d,), Kumaraswamy shape parameter, must be positive.

    Returns
    -------
    numpy.ndarray
        Shape (d,), CDF values in [0, 1].
    """
    u_vec = np.clip(u_vec, 1e-7, 1.0 - 1e-7)

    # Step 1: map u -> v via Kumaraswamy inverse CDF
    #   v = (1 - (1 - u)^{1/b})^{1/a}
    v = (1.0 - (1.0 - u_vec) ** (1.0 / b)) ** (1.0 / a)
    v = np.clip(v, 1e-7, 1.0 - 1e-7)

    # Step 2: evaluate truncated-logistic CDF at v
    p0 = _sigmoid((0.0 - loc) / scale)
    p1 = _sigmoid((1.0 - loc) / scale)
    mass = np.clip(p1 - p0, 1e-30, None)

    pv = _sigmoid((v - loc) / scale)
    cdf = np.clip((pv - p0) / mass, 0.0, 1.0)
    return cdf


def kuma_logistic_icdf_vec(t_vec, loc, scale, a, b):
    """
    Inverse CDF (quantile function) of the Kumaraswamy-Logistic chained
    distribution, evaluated element-wise.

    Parameters
    ----------
    t_vec : numpy.ndarray
        Shape (d,), quantile levels in (0, 1).
    loc : numpy.ndarray
        Shape (d,).
    scale : numpy.ndarray
        Shape (d,), must be positive.
    a : numpy.ndarray
        Shape (d,), Kumaraswamy shape parameter, must be positive.
    b : numpy.ndarray
        Shape (d,), Kumaraswamy shape parameter, must be positive.

    Returns
    -------
    numpy.ndarray
        Shape (d,), quantile values in (0, 1).
    """
    t_vec = np.clip(t_vec, 1e-7, 1.0 - 1e-7)

    # Step 1: invert the truncated logistic
    #   t = (sigmoid((v - loc)/scale) - p0) / (p1 - p0)
    #   => sigmoid((v - loc)/scale) = p0 + t*(p1-p0)
    #   => v = loc + scale * logit(p0 + t*(p1-p0))
    p0 = _sigmoid((0.0 - loc) / scale)
    p1 = _sigmoid((1.0 - loc) / scale)
    mass = np.clip(p1 - p0, 1e-30, None)

    p_draw = np.clip(p0 + t_vec * mass, 1e-15, 1.0 - 1e-15)
    v = loc + scale * _logit(p_draw)
    v = np.clip(v, 1e-7, 1.0 - 1e-7)

    # Step 2: apply Kumaraswamy CDF  u = 1 - (1 - v^a)^b
    u = 1.0 - (1.0 - v ** a) ** b
    u = np.clip(u, 1e-7, 1.0 - 1e-7)
    return u


def kuma_logistic_logpdf_vec(u_vec, loc, scale, a, b):
    """
    Log-density of a single point under the product of Kumaraswamy-Logistic
    chained distributions, summed over dimensions.

    This equals the log Jacobian log|du/dt| needed to correct the
    nested-sampling likelihood when the prior is this distribution.

    Parameters
    ----------
    u_vec : numpy.ndarray
        Shape (d,), values in (0, 1).
    loc : numpy.ndarray
        Shape (d,).
    scale : numpy.ndarray
        Shape (d,).
    a : numpy.ndarray
        Shape (d,), Kumaraswamy shape parameter.
    b : numpy.ndarray
        Shape (d,), Kumaraswamy shape parameter.

    Returns
    -------
    float
        Sum of per-dimension log-densities.
    """
    u_vec = np.clip(u_vec, 1e-7, 1.0 - 1e-7)

    # v = Kuma_ICDF(u)
    v = (1.0 - (1.0 - u_vec) ** (1.0 / b)) ** (1.0 / a)
    v = np.clip(v, 1e-7, 1.0 - 1e-7)

    # log p_L(v) -- untruncated logistic
    z = (v - loc) / scale
    log_pL_v = -z - np.log(scale) - 2.0 * np.log1p(np.exp(-np.clip(z, -500, 500)))

    # log normaliser  log(sigmoid((1-loc)/scale) - sigmoid((0-loc)/scale))
    p0 = _sigmoid((0.0 - loc) / scale)
    p1 = _sigmoid((1.0 - loc) / scale)
    log_Z = np.log(np.clip(p1 - p0, 1e-30, None))

    # log|dv/du| = -log(Kuma density at v)
    #   log p_Kuma(v; a, b) = log(a)+log(b)+(a-1)log(v)+(b-1)log(1-v^a)
    va = v ** a
    log_kuma_at_v = (np.log(a) + np.log(b)
                     + (a - 1.0) * np.log(v)
                     + (b - 1.0) * np.log(np.clip(1.0 - va, 1e-30, None)))
    log_abs_dv_du = -log_kuma_at_v

    log_p_per_dim = log_pL_v - log_Z + log_abs_dv_du   # shape (d,)
    return float(np.sum(log_p_per_dim))
