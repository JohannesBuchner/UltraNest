"""Warm start and hot start helper functions."""

import numpy as np
import scipy.stats
from .integrator import ReactiveNestedSampler
from .hotstart import get_auxiliary_contbox_parameterization
from .stepsampler import SliceSampler
import os
import ultranest.stepsampler
import logging
from .mlfriends import SimpleRegion


class ProgressiveNestedSampler():
    """
    Progressive nested sampling method.

    Iteratively runs nested sampling with more and more live points,
    with a constant number of slice sampling steps, until the
    evidence estimate converges.
    """

    def __init__(
        self,
        param_names,
        loglike,
        transform=None,
        vectorized=False,
        log_dir=None,
        **sampler_kwargs
    ):
        """See :py:class:`ReactiveNestedSampler` for documentation of the parameters."""
        self.paramnames = param_names
        self.transform = transform
        self.loglike = loglike
        self.vectorized = vectorized
        self.log_dir = sampler_kwargs['log_dir'] = log_dir
        if vectorized:
            # make simple versions of the provided functions
            def flat_transform(x):
                return transform(np.asarray(x).reshape((1, -1)))[0]

            def flat_loglike(x):
                return loglike(np.asarray(x).reshape((1, -1)))[0]

            self.flat_transform = flat_transform
            self.flat_loglike = flat_loglike
        else:
            self.flat_transform = transform
            self.flat_loglike = loglike
        self.sampler_kwargs = sampler_kwargs
        self.sampler = None
        self.warmstarted = False

    def _get_laplace_approximation(self):
        if self.sampler is not None and self.sampler.log:
            self.logger.info("warming up: finding laplace approximation with snowline...")
        import snowline  # snowline is required, install it with pip
        # make laplace approximation
        fitsampler = snowline.ReactiveImportanceSampler(self.paramnames, self.flat_loglike, transform=self.flat_transform)
        fitsampler.laplace_approximate()
        return fitsampler.optu, fitsampler.cov

    def warmup(self, df=2, num_global_samples=400, verbose=True, num_quantile_samples=10000):
        """Prepare a efficient approximation of the posterior to speed up nested sampling.

        This uses snowline to find a laplace approximation.
        Snowline uses minuit to find the maximum likelihood and its local error (hessian) approximation.
        The marginals of this are used to deform the nested sampling parameter space,
        see :py:func:`ultranest.hotstart.get_auxiliary_contbox_parameterization`.
        For obtaining approximate residuals, a student-t with the laplace covariance with *df*
        degrees of freedom (low numbers promote heavy tails), is sampled
        *num_quantile_samples* times.

        If *log_dir* has been specified, the results of this function are
        stored in:

        * `chains/max_likelihood.txt` (maximum likelihood coordinates)
        * `chains/laplaceapprox.txt` (covariance matrix)
        * `chains/laplaceapprox_samples.txt` (samples drawn)

        Parameters
        ----------
        df: float
            Student-t's degrees of freedom for generating samples. Use 2 (heavy-tailed) or 100 (gaussian-like)
        num_global_samples: int
            number of samples to draw initially from the prior to
            identify a good starting point for the minuit optimization.
        num_quantile_samples: int
            number of samples to draw from laplace approximation to
            estimate marginal quantiles.
        verbose: bool
            whether snowline/minuit should show intermediate progress & status
        """
        store = self.sampler_kwargs.get('log_dir', None) is not None

        if store:
            upoints_path = os.path.join(self.sampler_kwargs['log_dir'], 'chains', 'laplaceapprox_samples.txt')
            try:
                upoints = np.loadtxt(upoints_path)
            except IOError:
                max_like_path = os.path.join(self.sampler_kwargs['log_dir'], 'chains', 'max_likelihood.txt')
                cov_path = os.path.join(self.sampler_kwargs['log_dir'], 'chains', 'laplaceapprox.txt')
                try:
                    optu = np.loadtxt(max_like_path)
                    cov = np.loadtxt(cov_path)
                except IOError:
                    optu, cov = self._get_laplace_approximation()
                    os.makedirs(os.path.join(self.sampler_kwargs['log_dir'], 'chains'), exist_ok=True)
                    np.savetxt(max_like_path, optu)
                    np.savetxt(cov_path, cov)

                upoints_all = scipy.stats.t.rvs(loc=optu, scale=np.diag(cov)**0.5, df=df, size=(num_quantile_samples * 10, len(optu)))
                mask = np.logical_and(upoints_all > 0, upoints_all < 1).all(axis=1)
                upoints = upoints_all[mask,:][:num_quantile_samples]
                np.savetxt(upoints_path, upoints)
        else:
            optu, cov = self._get_laplace_approximation()
            upoints_all = scipy.stats.t.rvs(loc=optu, scale=np.diag(cov)**0.5, df=df, size=(num_quantile_samples * 10, len(optu)))
            mask = np.logical_and(upoints_all > 0, upoints_all < 1).all(axis=1)
            upoints = upoints_all[mask,:][:num_quantile_samples]

        uweights = np.ones(len(upoints)) / len(upoints)

        aux_parameters, aux_loglike, aux_transform, vectorized = get_auxiliary_contbox_parameterization(
            self.paramnames, self.loglike, transform=self.transform,
            upoints=upoints, uweights=uweights, vectorized=self.vectorized)

        self.sampler = ReactiveNestedSampler(
            aux_parameters, aux_loglike, aux_transform, vectorized=vectorized,
            **self.sampler_kwargs
        )
        self.warmstarted = True

    def init(self):
        """Initialize underlying sampler."""
        if not self.sampler:
            self.sampler = ReactiveNestedSampler(
                self.paramnames, self.loglike, transform=self.transform,
                vectorized=self.vectorized, **self.sampler_kwargs)
            self.warmstarted = False
        self.mpi_rank = self.sampler.mpi_rank
        self.mpi_size = self.sampler.mpi_size
        self.log = self.sampler.log
        if self.log:
            self.logger = logging.getLogger('ultranest.progressive')

    def run_iter(
        self,
        slice_sampler_args={'nsteps': 10, 'generate_direction': ultranest.stepsampler.generate_mixture_random_direction},
        max_num_improvement_loops=0,
        update_interval_volume_fraction=0.1,
        region_class=SimpleRegion,
        viz_callback=None,
        min_num_live_points=10,
        max_iter=50,
        **run_kwargs,
    ):
        """Run nested sampling iteratively with increasing number of live points.

        see :py:class:`ReactiveNestedSampler.run_iter` for parameter documentation.
        """
        self.init()
        if 'dlogz' in run_kwargs:
            raise ValueError('dlogz was set: use run() instead of run_iter() to target a ln(Z) accuracy.')

        for i in range(max_iter):
            # need at least 2*d live points to compute variance
            # each iteration increases the number of live points
            nlive = min_num_live_points + 2 * max(20, len(self.paramnames)) * (i + 1)
            if self.sampler.log:
                self.logger.info("progressive NS iteration %d: %d live points, %d steps" % (i + 1, nlive, slice_sampler_args['nsteps']))
            self.sampler.stepsampler = SliceSampler(**slice_sampler_args)
            yield self.sampler.run(
                min_num_live_points=nlive,
                region_class=region_class, viz_callback=viz_callback,
                update_interval_volume_fraction=update_interval_volume_fraction,
                max_num_improvement_loops=max_num_improvement_loops,
                dlogz=1000, cluster_num_live_points=0,
                **run_kwargs
            )

    def _estimate_logz(self):
        """Estimate evidence from sequence of estimates."""
        # ignore first 20%
        iend = len(self.results_sequence)
        istart = max(0, iend // 5)
        results_sequence_here = self.results_sequence[istart:]
        # least square fit of number of live points vs lnZ estimate

        # x is logarithmically related to the number of live points
        x = np.log(np.arange(istart, iend) + 1)
        # ln(Z) values:
        y = np.array([res['logz'] for res in results_sequence_here])
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        m, c = result[0]
        mean_residual = float(result[1]) / len(y) if result[1] > 0 else np.nan

        # estimate for twice the number of samples
        xfuture = np.log(iend * 2 + 1)
        logz_predict = m * xfuture + c
        return logz_predict, mean_residual

    def _get_results(self):
        """Combine sequence of estimates into one result."""
        logz_predict, logz_prediction_error = self._estimate_logz()
        logz_current = self.results_sequence[-1]['logz']
        logz_convergence_error = ((logz_predict - logz_current)**2 + logz_prediction_error**2)**0.5

        results = dict(self.sampler.results)
        results['logz'] = logz_predict
        results['logzerr_convergence'] = logz_convergence_error
        results['logzerr'] = (results['logzerr']**2 + logz_convergence_error**2)**0.5

        if self.warmstarted:
            # hide warmstart weight variable
            mask = [p != 'aux_logweight' for p in results['paramnames']]
            paramnames = [p for p in results['paramnames'] if p != 'aux_logweight']
            results['paramnames'] = paramnames
            results['samples'] = results['samples'][:,mask]
            results['weighted_samples']['points'] = results['weighted_samples']['points'][:,mask]
            results['weighted_samples']['upoints'] = results['weighted_samples']['upoints'][:,mask]
        return results

    def run(
        self,
        slice_sampler_args={'nsteps':10, 'generate_direction': ultranest.stepsampler.generate_mixture_random_direction},
        max_num_improvement_loops=0,
        update_interval_volume_fraction=0.1,
        region_class=SimpleRegion,
        viz_callback=None,
        max_iter=50,
        dlogz=0.1,
        dlogz_extra_per_parameter=0.05,
        **run_kwargs,
    ):
        """Run nested sampling to convergence.

        see :py:class:`ReactiveNestedSampler.run_iter` for parameter documentation.
        """
        self.results_sequence = []
        dlogztot = dlogz + dlogz_extra_per_parameter * len(self.paramnames)

        for i, res in enumerate(self.run_iter(
            slice_sampler_args=slice_sampler_args,
            max_num_improvement_loops=max_num_improvement_loops,
            update_interval_volume_fraction=update_interval_volume_fraction,
            region_class=region_class, viz_callback=viz_callback,
            max_iter=max_iter,
            **run_kwargs)
        ):
            self.results_sequence.append(res)
            self.results = self._get_results()

            if len(self.results_sequence) > 2 and self.results['logzerr'] < dlogz + dlogz_extra_per_parameter * len(self.paramnames):
                if self.sampler.log:
                    self.logger.info('current logz = %.3f +- %.3f, target accuracy (<%.3f) satisfied', self.results['logz'], self.results['logzerr'], dlogztot)
                break
            elif self.sampler.log:
                self.logger.info('current logz = %.3f +- %.3f, target accuracy <%.3f', self.results['logz'], self.results['logzerr'], dlogztot)
        return self.results

    def plot_corner(self, **kwargs):
        """Make corner plot.

        Writes corner plot to plots/ directory if log directory was
        specified, otherwise show interactively.

        This does essentially::

            from ultranest.plot import cornerplot
            cornerplot(results)

        """
        from .plot import cornerplot
        import matplotlib.pyplot as plt
        if self.sampler.log:
            self.sampler.logger.debug('Making corner plot ...')
        cornerplot(self.results, logger=self.sampler.logger if self.sampler.log else None, **kwargs)
        if self.sampler.log_to_disk:
            plt.savefig(os.path.join(self.sampler.logs['plots'], 'corner.pdf'), bbox_inches='tight')
            plt.close()
            self.sampler.logger.debug('Making corner plot ... done')

    def plot(self, *args, **kwargs):
        """Make corner, run and trace plots.

        calls:

        * plot_corner()
        * plot_run()
        * plot_trace()
        """
        self.plot_progress()
        self.plot_corner(plot_datapoints=False, plot_density=False)
        self.plot_run()
        self.plot_trace()

    def plot_progress(self):
        import matplotlib.pyplot as plt
        if self.log:
            self.logger.debug('Making progress plot ...')
        y = np.array([res['logz'] for res in self.results_sequence])
        plt.plot(y)
        if self.sampler.log_to_disk:
            plt.savefig(os.path.join(self.sampler.logs['plots'], 'progress.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.debug('Making progress plot ... done')


    def print_results(self, *args, **kwargs):
        """See :py:func:`ReactiveNestedSampler.print_results`."""
        return self.sampler.print_results(*args, **kwargs)

    def plot_trace(self, *args, **kwargs):
        """See :py:func:`ReactiveNestedSampler.plot_trace`."""
        return self.sampler.plot_trace(*args, **kwargs)

    def plot_run(self, *args, **kwargs):
        """See :py:func:`ReactiveNestedSampler.plot_run`."""
        return self.sampler.plot_run(*args, **kwargs)
