import argparse
import numpy as np

def main(args):
    ndim = args.x_dim
    sigma = args.sigma
    sigma = np.logspace(-1, np.log10(args.sigma), ndim)
    width = 1 - 5 * sigma
    width[width < 1e-20] = 1e-20
    centers = (np.sin(np.arange(ndim)/2.) * width + 1.) / 2.
    #centers = np.ones(ndim) * 0.5

    adaptive_nsteps = args.adapt_steps
    if adaptive_nsteps is None:
        adaptive_nsteps = False

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2).sum()
        return like

    def transform(x):
        return x
    
    def transform_loglike_gradient(u):
        theta = u
        like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2).sum()
        grad = (theta - centers)/sigma
        return u, like, grad

    def gradient(theta):
        return (theta - centers) / sigma
            
    paramnames = ['param%d' % (i+1) for i in range(ndim)]
    
    if args.pymultinest:
        from pymultinest.solve import solve
        
        def flat_loglike(theta):
            return loglike(theta.reshape((1, -1)))
        
        result = solve(LogLikelihood=flat_loglike, Prior=transform, 
            n_dims=ndim, outputfiles_basename=args.log_dir + 'MN-%dd' % ndim,
            verbose=True, resume=True, importance_nested_sampling=False)
        
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(paramnames, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    
    elif args.reactive:
        if args.slice:
            log_dir = args.log_dir + 'RNS-%dd-slice%d' % (ndim, args.slice_steps)
        elif args.harm:
            log_dir = args.log_dir + 'RNS-%dd-harm%d' % (ndim, args.slice_steps)
        elif args.dyhmc:
            log_dir = args.log_dir + 'RNS-%dd-dyhmc%d' % (ndim, args.slice_steps)
        elif args.dychmc:
            log_dir = args.log_dir + 'RNS-%dd-dychmc%d' % (ndim, args.slice_steps)
        else:
            log_dir = args.log_dir + 'RNS-%dd' % (ndim)
        if adaptive_nsteps:
            log_dir = log_dir + '-adapt%s' % (adaptive_nsteps)
        
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=log_dir, resume=True,
            vectorized=True)
        if args.slice:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps,
                log=open(log_dir + '/stepsampler.log', 'w'))
        if args.harm:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps,
                log=open(log_dir + '/stepsampler.log', 'w'))
        if args.dyhmc:
            import ultranest.dyhmc
            from ultranest.utils import verify_gradient
            verify_gradient(ndim, transform, loglike, transform_loglike_gradient, combination=True)
            sampler.stepsampler = ultranest.dyhmc.DynamicHMCSampler(ndim=ndim, nsteps=args.slice_steps, 
                transform_loglike_gradient=transform_loglike_gradient, adaptive_nsteps=adaptive_nsteps)
        if args.dychmc:
            import ultranest.dychmc
            from ultranest.utils import verify_gradient
            verify_gradient(ndim, transform, loglike, gradient)
            sampler.stepsampler = ultranest.dychmc.DynamicCHMCSampler(ndim=ndim, nsteps=args.slice_steps, 
                transform=transform, loglike=loglike, gradient=gradient, adaptive_nsteps=adaptive_nsteps)
        sampler.run(frac_remain=0.5, min_num_live_points=args.num_live_points, max_num_improvement_loops=1)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename = log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 20:
            sampler.plot()
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            num_live_points=args.num_live_points, vectorized=True,
            log_dir=args.log_dir + '-%dd' % ndim, resume=True)
        sampler.run()
        sampler.print_results()
        sampler.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='logs/asymgauss')
    parser.add_argument('--pymultinest', action='store_true')
    parser.add_argument('--reactive', action='store_true')
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--harm', action='store_true')
    parser.add_argument('--dyhmc', action='store_true')
    parser.add_argument('--dychmc', action='store_true')
    parser.add_argument('--slice_steps', type=int, default=100)
    parser.add_argument('--adapt_steps', type=str)

    args = parser.parse_args()
    main(args)
