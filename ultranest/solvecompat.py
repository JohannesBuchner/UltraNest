"""Drop-in replacement for pymultinest.solve.

Example::

    from ultranest.solvecompat import pymultinest_solve_compat as solve

    # is a drop-in replacement for

    from pymultinest.solve import solve

"""


import numpy as np
import string

from .integrator import ReactiveNestedSampler


def pymultinest_solve_compat(LogLikelihood, Prior, n_dims,
        paramnames=None,
        outputfiles_basename=None, resume=False,
        n_live_points=400, evidence_tolerance=0.5,
        seed=-1, max_iter=0, wrapped_params=None, verbose=True,
        **kwargs
):
    """Run nested sampling analysis.

    Disadvantages compared to using ReactiveNestedSampler directly:
    cannot resume easily, cannot plot interactively.
    Limited results.
    """
    if paramnames is None:
        paramnames = list(string.ascii_lowercase)[:n_dims]
    assert len(paramnames) == n_dims
    min_ess = kwargs.pop('min_ess', 0)
    frac_remain = kwargs.pop('frac_remain', 0.01)
    outputkwargs = {}
    if not verbose:
        outputkwargs = dict(viz_callback=False, show_status=False)

    sampler = ReactiveNestedSampler(paramnames,
        LogLikelihood,
        transform=Prior,
        log_dir=outputfiles_basename,
        resume='resume' if resume else 'overwrite',
        wrapped_params=wrapped_params,
        draw_multiple=False,
        vectorized=False,
        **outputkwargs
    )
    sampler.run(dlogz=evidence_tolerance,
        max_iters=max_iter if max_iter > 0 else None,
        min_num_live_points=n_live_points,
        min_ess=min_ess, frac_remain=frac_remain)
    if verbose:
        sampler.print_results()
    results = sampler.results
    sampler.plot()

    return dict(logZ=results['logz'],
        logZerr=results['logzerr'],
        samples=results['samples'],
        weighted_samples=results['weighted_samples'])
