"""Helpers for using nested sampling on top of an auxiliary distribution"""

import numpy as np
import torch
from .logistic import kuma_logistic_cdf_vec, kuma_logistic_logpdf_vec, kuma_logistic_icdf_vec

def get_distribution_parameters(model, observed_data):
    """
    Query a neural network model to obtain Kumaraswamy-logistic distribution
    parameters for the given observed data.

    Parameters
    ----------
    model : torch.nn.Module
        A trained neural posterior estimator that returns the tuple
        ``(loc, scale, a, b)`` when called with an input tensor.
    observed_data : array_like
        The observed data to condition on. Will be converted to a
        ``torch.float32`` tensor and given a batch dimension of 1.

    Returns
    -------
    dict
        A dictionary with keys ``'loc'``, ``'scale'``, ``'a'``, and ``'b'``,
        each mapping to a Python list of floats representing the corresponding
        distribution parameter for each dimension.
    """
    # Query the network once to get the distribution parameters
    x_t = torch.tensor(observed_data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        loc, scale, a, b = model(x_t)

    # Store as numpy arrays, shape (n_params,)
    return dict(
        loc   = loc.squeeze(0).numpy().tolist(),
        scale = scale.squeeze(0).numpy().tolist(),
        a     = a.squeeze(0).numpy().tolist(),
        b     = b.squeeze(0).numpy().tolist()
    )

class KLPTransform:
    """
    A coordinate transform based on a Kumaraswamy-logistic posterior (KLP)
    approximation from a neural posterior estimator (NPE).

    Provides mappings between nested-sampler unit-cube coordinates ``t`` and
    prior unit-cube coordinates ``u``, along with the associated log Jacobian
    and CDF evaluation.

    Parameters
    ----------
    loc : array_like
        Location parameters of the Kumaraswamy-logistic distribution,
        shape ``(n_params,)``.
    scale : array_like
        Scale parameters of the Kumaraswamy-logistic distribution,
        shape ``(n_params,)``.
    a : array_like
        First shape parameters of the Kumaraswamy distribution,
        shape ``(n_params,)``.
    b : array_like
        Second shape parameters of the Kumaraswamy distribution,
        shape ``(n_params,)``.
    """
    
    def __init__(self, loc, scale, a, b):
        self.loc = np.array(loc)
        self.scale = np.array(scale)
        self.a = np.array(a)
        self.b = np.array(b)
    
    def transform(self, t):
        """
        Map nested-sampler unit-cube coordinates ``t`` to prior unit-cube
        coordinates ``u`` via the analytic inverse CDF of the NPE posterior.

        Parameters
        ----------
        t : np.ndarray
            Uniformly distributed sample from the nested sampler,
            shape ``(n_params,)``.

        Returns
        -------
        u : np.ndarray
            Corresponding unit-cube coordinates of shape ``(n_params,)``
            and dtype ``float32``; pass to ``prior_transform`` to get
            physical parameters.
        """
        u = kuma_logistic_icdf_vec(t, self.loc, self.scale, self.a, self.b)
        return u.astype(np.float32)

    def log_jacobian(self, t):
        """
        Log Jacobian of the transform ``t -> u``, equal to the log-density
        of the NPE posterior at ``u = transform(t)``.

        Add this value to the true log-likelihood when passing to the
        nested sampler so that the sampler correctly targets the posterior.

        Parameters
        ----------
        t : np.ndarray
            Nested-sampler unit-cube coordinates, shape ``(n_params,)``.

        Returns
        -------
        log_q : float
            ``log q(u(t) | x_obs)`` -- always finite.
        """
        u = self.transform(t)
        return kuma_logistic_logpdf_vec(u, self.loc, self.scale, self.a, self.b)

    def cdf(self, u):
        """
        Evaluate the analytic CDF of the NPE posterior at a unit-cube point
        ``u`` (i.e. compute the ``t`` that maps to ``u``).

        Parameters
        ----------
        u : np.ndarray
            Prior unit-cube coordinates, shape ``(n_params,)``.

        Returns
        -------
        t : np.ndarray
            Nested-sampler unit-cube coordinates corresponding to ``u``,
            shape ``(n_params,)``.
        """
        return kuma_logistic_cdf_vec(u, self.loc, self.scale, self.a, self.b)

    def logpdf(self, u):
        """
        Evaluate the log-density of the NPE posterior at a unit-cube point
        ``u``.

        This is used by importance sampling to compute the log proposal
        density ``log q(u | x_obs)`` for each sample drawn from the NPE
        approximate posterior.

        Parameters
        ----------
        u : np.ndarray
            Prior unit-cube coordinates, shape ``(n_params,)``.

        Returns
        -------
        log_q : float
            Sum of log-densities across all dimensions,
            ``sum_i log q(u_i | x_obs)``.
        """
        return kuma_logistic_logpdf_vec(u, self.loc, self.scale, self.a, self.b)
