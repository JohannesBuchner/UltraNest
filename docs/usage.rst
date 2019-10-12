=====
Usage
=====

To use UltraNest in a project::

    import ultranest

    sampler = ultranest.ReactiveNestedSampler(
        param_names, 
        loglike=my_likelihood, 
        transform=my_prior_transform,
        ...
        )

`ReactiveNestedSampler <ultranest.html#ultranest.integrator.ReactiveNestedSampler>`_ 
has several options to specify what logging and file output it should produce::

    sampler = ultranest.ReactiveNestedSampler(
        param_names, 
        loglike=my_likelihood, 
        transform=my_prior_transform,
        ## additional parameter properties: 
        # identify circular parameters
        wrapped_params=None,
        # add derived calculations
        derived_param_names=[], 
        # where to store the output
        log_dir=None,
        # whether to continue from existing output
        resume=False,
        # make a new folder for each run?
        append_run_num=True,
        run_num=None,
        num_test_samples=2,
        draw_multiple=True,
        num_bootstraps=30,
        show_status=True
        )


Next, we start running the sampler::

    result = sampler.run()
    sampler.print_results()
    

Both `ReactiveNestedSampler <ultranest.html#ultranest.integrator.ReactiveNestedSampler>`_ 
and the `run function <ultranest.html#ultranest.integrator.ReactiveNestedSampler.run>`_ 
have several options to specify what logging and file output they should produce,
and how they should explore the parameter space.

You can find several examples in the tutorials.




