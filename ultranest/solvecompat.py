"""Drop-in replacement for pymultinest.solve.

Example::

    from ultranest.solvecompat import pymultinest_solve_compat as solve

    # is a drop-in replacement for

    from pymultinest.solve import solve

"""


import numpy as np
import string

from .integrator import ReactiveNestedSampler
from .stepsampler import SliceSampler, generate_mixture_random_direction


def pymultinest_solve_compat(
    LogLikelihood, Prior, n_dims, paramnames=None,
    outputfiles_basename=None, resume=False,
    n_live_points=400, evidence_tolerance=0.5,
    seed=-1, max_iter=0, wrapped_params=None, verbose=True,
    speed="safe",
    **kwargs
):
    """Run nested sampling analysis.

    Disadvantages compared to using ReactiveNestedSampler directly:
    cannot resume easily, cannot plot interactively.
    Limited results.

    It is recommended that you directly use::

        sampler = ReactiveNestedSampler(paramnames, LogLikelihood, transform=Prior)
        sampler.run()

    following the UltraNest documentation and manuals,
    as this gives you more control on resuming and sampler options.
    """
    if paramnames is None:
        paramnames = list(string.ascii_lowercase)[:n_dims]
    if seed >= 0:
        np.random.seed(seed)
    assert len(paramnames) == n_dims
    min_ess = kwargs.pop('min_ess', 0)
    frac_remain = kwargs.pop('frac_remain', 0.01)
    Lepsilon = kwargs.pop('Lepsilon', 0.001)
    outputkwargs = {}
    if not verbose:
        outputkwargs = dict(viz_callback=False, show_status=False)

    sampler = ReactiveNestedSampler(
        paramnames, LogLikelihood, transform=Prior,
        log_dir=outputfiles_basename, resume='resume' if resume else 'overwrite',
        wrapped_params=wrapped_params, draw_multiple=False, vectorized=False,
        **outputkwargs)

    if speed == "safe":
        pass
    elif speed == "auto":
        sampler.run(
            dlogz=evidence_tolerance,
            max_iters=max_iter if max_iter > 0 else None,
            min_num_live_points=n_live_points,
            min_ess=min_ess, frac_remain=frac_remain,
            Lepsilon=Lepsilon, max_ncalls=40000)

        sampler.stepsampler = SliceSampler(
            nsteps=1000,
            generate_direction=generate_mixture_random_direction,
            adaptive_nsteps='move-distance',
            region_filter=kwargs.get('region_filter', True)
            )
    else:
        sampler.stepsampler = SliceSampler(
            generate_direction=generate_mixture_random_direction,
            nsteps=speed,
            adaptive_nsteps=False,
            region_filter=False)

    sampler.run(dlogz=evidence_tolerance,
                max_iters=max_iter if max_iter > 0 else None,
                min_num_live_points=n_live_points,
                min_ess=min_ess, frac_remain=frac_remain,
                Lepsilon=Lepsilon)

    if verbose:
        sampler.print_results()
    results = sampler.results
    sampler.plot()

    return dict(logZ=results['logz'],
                logZerr=results['logzerr'],
                samples=results['samples'],
                weighted_samples=results['weighted_samples'])
