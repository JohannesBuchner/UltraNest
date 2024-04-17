"""
Calibration of step sampler
"""

import numpy as np
from ultranest.integrator import ReactiveNestedSampler
import os


def _substitute_log_dir(init_args, nsteps):
    """Append `nsteps` to `log_dir` argument, if set.

    Parameters
    -----------
    init_args: dict
        arguments passed :py:class:`ReactiveNestedSampler`,
        may contain the key `'log_dir'`.
    nsteps: int
        number of steps

    Returns
    -------
    new_init_args: dict
        same as init_args, but if `'log_dir'` was set,
        it now has `'-nsteps'+str(nsteps)` appended.
    """
    if 'log_dir' in init_args:
        args = dict(init_args)
        args['log_dir'] = init_args['log_dir'] + '-nsteps%d' % nsteps
        return args
    return init_args


class ReactiveNestedCalibrator():
    """Calibrator for the number of steps in step samplers.

    The number of steps in a step sampler needs to be chosen.
    A calibration recommended (e.g. https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.2044H)
    is to run a sequence of nested sampling runs with increasing number of steps,
    and stop when log(Z) converges.

    This class automates this. See the :py:meth:`ReactiveNestedCalibrator.run`
    for details.

    Usage
    -----

    Usage is designed to be a drop-in replacement for ReactiveNestedSampler.

    If your code was::
        sampler = ReactiveNestedSampler(my_param_names, my_loglike, my_transform)
        sampler.stepsampler = SliceSampler(nsteps=10, generate_direction=region_oriented_direction)
        sampler.run(min_num_livepoints=400)

    You would change it to::
        sampler = ReactiveNestedCalibrator(my_param_names, my_loglike, my_transform)
        sampler.stepsampler = SliceSampler(nsteps=10, generate_direction=region_oriented_direction)
        sampler.run(min_num_livepoints=400)

    The run() command will print the number of slice sampler steps
    that appear safe for the inference task.

    The initial value for nsteps (e.g. in `SliceSampler(nsteps=...)`)
    is overwritten by this class.
    """

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 **kwargs
                 ):
        """Initialise nested sampler calibrator.

        Parameters
        -----------
        param_names: list of str
            Names of the parameters.
            Length gives dimensionality of the sampling problem.
        loglike: function
            log-likelihood function.
        transform: function
            parameter transform from unit cube to physical parameters.
        kwargs: dict
            further arguments passed to ReactiveNestedSampler

        if `log_dir` is set, then the suffix `-nsteps%d` is added for each
        run where %d is replaced with the number of steps (2, 4, 8 etc).
        """
        self.init_args = dict(param_names=param_names, loglike=loglike, transform=transform, **kwargs)
        self.stepsampler = None

    def run(self, **kwargs):
        """Run a sequence of ReactiveNestedSampler runs until convergence.

        The first run is made with the number of steps set to the number of parameters.
        Each subsequent run doubles the number of steps.
        Runs are made until convergence is reached.
        Then this generator stops yielding results.

        Convergence is defined as three consecutive runs which
        1) are not ordered in their log(Z) results,
        and 2) the consecutive log(Z) error bars must overlap.

        Parameters
        -----------
        **kwargs: dict
            All arguments are passed to :py:meth:`ReactiveNestedSampler.run`.

        Yields
        -------
        nsteps: int
            number of steps for the current run
        result: dict
            return value of :py:meth:`ReactiveNestedSampler.run` for the current run
        """
        assert self.stepsampler is not None
        self.run_args = kwargs

        # start with nsteps=d
        nsteps = len(self.init_args['param_names'])
        self.results = []
        self.nsteps = []
        self.relsteps = []

        while True:
            print("running with %d steps ..." % nsteps)
            init_args = _substitute_log_dir(self.init_args, nsteps)
            sampler = ReactiveNestedSampler(**init_args)
            sampler.stepsampler = self.stepsampler.__class__(
                nsteps=nsteps, generate_direction=self.stepsampler.generate_direction,
                check_nsteps=self.stepsampler.check_nsteps,
                adaptive_nsteps=self.stepsampler.adaptive_nsteps,
                log=open(init_args['log_dir'] + '/stepsampler.log', 'w') if 'log_dir' in self.init_args else None)
            self.sampler = sampler
            result = sampler.run(**self.run_args)
            print("Z=%(logz).2f +- %(logzerr).2f" % result)
            if self.sampler.log_to_disk:
                sampler.stepsampler.plot(os.path.join(self.sampler.logs['plots'], 'stepsampler.pdf'))
                sampler.stepsampler.plot_jump_diagnostic_histogram(
                    os.path.join(self.sampler.logs['plots'], 'stepsampler-jumphist.pdf'),
                    histtype='step', bins='auto')
            sampler.stepsampler.print_diagnostic()
            if 'jump-distance' in sampler.stepsampler.logstat_labels and 'reference-distance' in sampler.stepsampler.logstat_labels:
                i = sampler.stepsampler.logstat_labels.index('jump-distance')
                j = sampler.stepsampler.logstat_labels.index('reference-distance')
                jump_distances = np.array([entry[i] for entry in sampler.stepsampler.logstat])
                reference_distances = np.array([entry[j] for entry in sampler.stepsampler.logstat])
                self.relsteps.append(jump_distances / reference_distances)
            # TODO: handle population step samplers

            self.results.append(result)
            self.nsteps.append(nsteps)
            yield nsteps, result
            if len(self.results) > 2:
                last_result = self.results[-2]
                last_result2 = self.results[-3]
                # check if they agree within the error bars
                last_significant = abs(result['logz'] - last_result['logz']) > (result['logzerr']**2 + last_result['logzerr']**2)**0.5
                last2_significant = abs(last_result2['logz'] - last_result['logz']) > (last_result2['logzerr']**2 + last_result['logzerr']**2)**0.5
                # check if there is order
                monotonic_increase = result['logz'] > last_result['logz'] > last_result2['logz']
                monotonic_decrease = result['logz'] < last_result['logz'] < last_result2['logz']
                if last_significant:
                    print("not converged: last two Z were significantly different")
                elif last2_significant:
                    print("not yet converged: previous two Z were significantly different")
                elif monotonic_increase:
                    print("not converged: monotonic increase in the last three Z results")
                elif monotonic_decrease:
                    print("not converged: monotonic decrease in the last three Z results")
                else:
                    print("converged! nsteps=%d appears safe" % nsteps)
                    break

            nsteps *= 2

    def plot(self):
        """Visualise the convergence diagnostics.

        Stores into `<log_dir>/plots/` folder:
        * stepsampler.pdf: diagnostic of stepsampler, see :py:meth:`StepSampler.plot`
        * nsteps-calibration-jumps.pdf: distribution of relative jump distance
        * nsteps-calibration.pdf: evolution of ln(Z) with nsteps
        """
        self.sampler.stepsampler.plot(os.path.join(self.sampler.logs['plots'], 'stepsampler.pdf'))

        # plot U-test convergence run length (at 4 sigma) (or niter) vs nsteps
        # plot step > reference fraction vs nsteps
        calibration_results = []

        import matplotlib.pyplot as plt
        plt.figure("jump-distance")
        print("jump distance diagnostic:")
        for nsteps, relsteps, result in zip(self.nsteps, self.relsteps, self.results):
            calibration_results.append([
                nsteps, result['logz'], result['logzerr'],
                min(result['niter'], result['insertion_order_MWW_test']['independent_iterations']),
                result['insertion_order_MWW_test']['converged'] * 1,
                np.nanmean(relsteps > 1)])
            plt.hist(np.log10(relsteps + 1e-10), histtype='step', bins='auto', label=nsteps)
            print('  %-4d: %.2f%%  avg:%.2f' % (nsteps, np.nanmean(relsteps > 1) * 100.0, np.exp(np.nanmean(np.log(relsteps)))))
        if 'log_dir' in self.init_args:
            np.savetxt(
                self.init_args['log_dir'] + 'calibration.csv',
                calibration_results, delimiter=',', comments='',
                header='nsteps,logz,logzerr,maxUrun,Uconverged,stepfrac',
                fmt='%d,%.3f,%.3f,%d,%d,%.5f')
        plt.xlabel('$log_{10}$(relative step distance)')
        plt.ylabel('Frequency')
        plt.legend(title='nsteps', loc='best')
        if self.sampler.log_to_disk:
            plt.savefig(os.path.join(self.sampler.logs['plots'], 'nsteps-calibration-jumps.pdf'), bbox_inches='tight')
            plt.close()

        plt.figure("logz")
        plt.errorbar(
            x=self.nsteps,
            y=[result['logz'] for result in self.results],
            yerr=[result['logzerr'] for result in self.results],
        )
        plt.title('Step sampler calibration')
        plt.xlabel('Number of steps')
        plt.ylabel('ln(Z)')
        if self.sampler.log_to_disk:
            plt.savefig(os.path.join(self.sampler.logs['plots'], 'nsteps-calibration.pdf'), bbox_inches='tight')
            plt.close()
            self.sampler.logger.debug('Making nsteps calibration plot ... done')
