import argparse
import numpy as np
from numpy import log
import scipy.stats
from ultranest.utils import verify_gradient

def main(args):
    ndim = args.x_dim
    scale = args.scale
    adaptive_nsteps = args.adapt_steps
    if adaptive_nsteps is None:
        adaptive_nsteps = False

    rv1a = scipy.stats.loggamma(1, loc=2./3, scale=scale)
    rv1b = scipy.stats.loggamma(1, loc=1./3, scale=scale)
    rv2a = scipy.stats.norm(2./3, scale)
    rv2b = scipy.stats.norm(1./3, scale)
    rv_rest = []
    for i in range(2, ndim):
	    if i <= (ndim+2)/2:
		    rv = scipy.stats.loggamma(1, loc=2./3., scale=scale)
	    else:
		    rv = scipy.stats.norm(2./3, scale)
	    rv_rest.append(rv)
	    del rv

    def loglike(theta):
        L1 = log(0.5 * rv1a.pdf(theta[:,0]) + 0.5 * rv1b.pdf(theta[:,0]) + 1e-300)
        L2 = log(0.5 * rv2a.pdf(theta[:,1]) + 0.5 * rv2b.pdf(theta[:,1]) + 1e-300)
        Lrest = np.sum([rv.logpdf(t) for rv, t in zip(rv_rest, theta[:,2:].transpose())], axis=0)
        #assert L1.shape == (len(theta),)
        #assert L2.shape == (len(theta),)
        #assert Lrest.shape == (len(theta),), Lrest.shape
        like = L1 + L2 + Lrest
        like = np.where(like < -1e300, -1e300 - ((np.asarray(theta) - 0.5)**2).sum(), like)
        assert like.shape == (len(theta),), (like.shape, theta.shape)
        return like

    def transform(x):
        return x
    
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
                log=open(log_dir + '/stepsampler.log', 'w') if sampler.mpi_rank == 0 else False)
        if args.harm:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps,
                log=open(log_dir + '/stepsampler.log', 'w') if sampler.mpi_rank == 0 else False)
        #if args.dyhmc:
        #    import ultranest.dyhmc
        #    verify_gradient(ndim, transform, loglike, transform_loglike_gradient, combination=True)
        #    sampler.stepsampler = ultranest.dyhmc.DynamicHMCSampler(ndim=ndim, nsteps=args.slice_steps, 
        #        transform_loglike_gradient=transform_loglike_gradient, adaptive_nsteps=adaptive_nsteps)
        #if args.dychmc:
        #    import ultranest.dychmc
        #    verify_gradient(ndim, transform, loglike, gradient)
        #    sampler.stepsampler = ultranest.dychmc.DynamicCHMCSampler(ndim=ndim, nsteps=args.slice_steps, 
        #        transform=transform, loglike=loglike, gradient=gradient, adaptive_nsteps=adaptive_nsteps)
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
    parser.add_argument('--scale', type=float, default=1/30., help="Peak widths")
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument('--log_dir', type=str, default='logs/loggamma')
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
