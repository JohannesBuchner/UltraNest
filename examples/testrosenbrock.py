import argparse
import numpy as np

def main(args):
    ndim = args.x_dim
    adaptive_nsteps = args.adapt_steps
    if adaptive_nsteps is None:
        adaptive_nsteps = False

    def loglike(theta):
        a = theta[:,:-1]
        b = theta[:,1:]
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum(axis=1)

    def transform(u):
        return u * 20 - 10
    
    
    def transform_loglike_gradient(u):
        theta = u * 20 - 10
        
        a = theta[:-1]
        b = theta[1:]
        grad = theta.copy()
        L = -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum()
        for i in range(ndim):
            a = theta[i]
            if i < ndim-1:
                b = theta[i+1]
                grad[i] = -2*(-400 * a * (b - a**2) - 2 * (1 - a))
            if i > 0:
                c = theta[i-1]
                grad[i] += - 400 * (a - c**2)
        
        prior_factor = 20
        
        return theta, L, grad * prior_factor

    def gradient(u):
        theta = u * 20 - 10
        
        grad = theta.copy()
        for i in range(ndim):
            a = theta[i]
            if i < ndim-1:
                b = theta[i+1]
                grad[i] = -2*(-400 * a * (b - a**2) - 2 * (1 - a))
            if i > 0:
                c = theta[i-1]
                grad[i] += - 400 * (a - c**2)
        
        prior_factor = 20
        
        return grad * prior_factor
            
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
        elif args.aharm:
            log_dir = args.log_dir + 'RNS-%dd-aharm%d' % (ndim, args.slice_steps)
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
                log=open(log_dir + '/stepsampler.log', 'w'), max_nsteps=ndim)
        if args.harm:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps,
                log=open(log_dir + '/stepsampler.log', 'w'), max_nsteps=ndim)
        if args.aharm:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.AHARMSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps,
                log=open(log_dir + '/stepsampler.log', 'w'), max_nsteps=ndim)
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
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument('--log_dir', type=str, default='logs/rosen')
    parser.add_argument('--pymultinest', action='store_true')
    parser.add_argument('--reactive', action='store_true')
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--harm', action='store_true')
    parser.add_argument('--aharm', action='store_true')
    parser.add_argument('--dyhmc', action='store_true')
    parser.add_argument('--dychmc', action='store_true')
    parser.add_argument('--slice_steps', type=int, default=100)
    parser.add_argument('--adapt_steps', type=str)

    args = parser.parse_args()
    main(args)
