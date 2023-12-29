from ultranest.integrator import ReactiveNestedSampler

def substitute_log_dir(init_args, nsteps):
    if 'log_dir' in init_args:
        args = dict(init_args)
        args['log_dir'] = init_args['log_dir'] + '-nsteps%d' % nsteps
        return args
    return init_args


class ReactiveNestedCalibrator():
    """Calibrator for the number of steps in step samplers.

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

    The initial value for nsteps is ignored, and set to len(param_names)
    instead.
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
        """Run a sequence of ReactiveNestedSampler with nsteps doubling.

        All arguments are passed to :py:meth:`ReactiveNestedSampler.run`.

        """
        assert self.stepsampler is not None
        self.run_args = kwargs
        
        # start with nsteps=d
        nsteps = len(self.init_args['param_names'])
        self.results = []
        self.nsteps = []

        while True:
            print("running with %d steps ..." % nsteps)
            sampler = ReactiveNestedSampler(**substitute_log_dir(self.init_args, nsteps))
            sampler.stepsampler = self.stepsampler.__class__(nsteps, generate_direction=self.stepsampler.generate_direction)
            result = sampler.run(**self.run_args)
            self.results.append(result)
            self.nsteps.append(nsteps)
            print("lnZ=%(logz).2f +- %(logzerr).2f" % result)
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
