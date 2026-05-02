# noqa: D400 D205
"""
Snowballing Nested Sampling
---------------------------
"""

import os
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from ultranest.integrator import ReactiveNestedSampler
from ultranest.stepsampler import (SliceSampler,
                                   generate_mixture_random_direction)


def _make_plot(
    plot_dir: Optional[str],
    K_arr: List[int],
    logz_arr: List[float],
    logzerr_arr: List[float],
) -> None:
    """Store an updated plot of ln(Z) as a function of K.

    Parameters
    ----------
    plot_dir : str or None
        Directory where the plot will be saved.
        If None, no plot is produced.
    K_arr : list of int
        Values of K (number of live points) used in each run.
    logz_arr : list of float
        Corresponding log-evidence estimates.
    logzerr_arr : list of float
        Corresponding log-evidence uncertainties.

    Returns
    -------
    None
    """
    if plot_dir is None:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(K_arr, logz_arr, yerr=logzerr_arr, fmt='o-', capsize=4, color='k')
    ax.set_xlabel('K (Number of live points)')
    ax.set_ylabel('ln(Z)')
    ax.grid(True, alpha=0.3)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'snowballing.pdf')
    fig.savefig(plot_path, bbox_inches='tight')
    print("Updated ln(Z) vs K plot saved to: %s" % plot_path)
    plt.close(fig)


def snowball(
    sampler: ReactiveNestedSampler,
    Kfactor: float = 1.5,
    Kmin: int = 20,
    Kmax: int = 10000,
    frac_remain: float = 0.5,
    max_num_improvement_loops: int = 0,
    **kwargs: Any,
) -> Dict[str, List]:
    """Run nested sampling with a snowballing schedule of live points.

    Repeatedly calls ``sampler.run`` with an increasing number of live
    points *K*, multiplying by *Kfactor* after each run until *Kmax* is
    reached or exceeded.

    Parameters
    ----------
    sampler : ReactiveNestedSampler
        An initialised UltraNest sampler instance.
    Kfactor : float, optional
        Multiplicative growth factor applied to K after each run.
        Must be greater than 1. Default is 1.5.
    Kmin : int, optional
        Initial number of live points. Default is 20.
    Kmax : int, optional
        Maximum number of live points. The loop stops when K reaches
        or exceeds this value. Default is 10000.
    frac_remain : float, optional
        Passed to ``sampler.run`` as the ``frac_remain`` argument.
        Default is 0.5.
    max_num_improvement_loops : int, optional
        Passed to ``sampler.run`` as the ``max_num_improvement_loops``
        argument. Default is 0.
    **kwargs : dict
        Additional keyword arguments forwarded to ``sampler.run``.

    Returns
    -------
    dict
        A dictionary with the following keys:

        K : list of int
            Number of live points used in each run.
        logz : list of float
            Log-evidence estimate from each run.
        logzerr : list of float
            Log-evidence uncertainty from each run.
    """
    if sampler.stepsampler is None:
        nsteps = sampler.num_params
        sampler.stepsampler = SliceSampler(
            nsteps=nsteps,
            generate_direction=generate_mixture_random_direction,
        )
        print(
            "No step sampler was set. Automatically configured a SliceSampler "
            "with generate_mixture_random_direction and nsteps=%d (1x the number of parameters)." % nsteps
        )

    K = Kmin
    K_values: List[int] = []
    lnZs: List[float] = []
    lnZerrs: List[float] = []

    plot_dir = getattr(sampler, 'log_dir', None)

    while K < Kmax:
        K_int = int(K)
        result = sampler.run(
            min_num_live_points=K_int,
            frac_remain=frac_remain,
            max_num_improvement_loops=max_num_improvement_loops,
            **kwargs
        )

        print("K=%d  ln(Z)=%.2f +- %.2f" % (K_int, result['logz'], result['logzerr']))

        K_values.append(K_int)
        lnZs.append(result['logz'])
        lnZerrs.append(result['logzerr'])

        _make_plot(plot_dir, K_values, lnZs, lnZerrs)

        K = max(K * Kfactor, K + 1)

    return dict(K=K_values, logz=lnZs, logzerr=lnZerrs)
