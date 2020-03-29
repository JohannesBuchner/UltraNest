import argparse
import numpy as np
from numpy import exp, log, pi
import scipy.stats

# analytic solution:
def shell_vol(ndim, r, w):
    # integral along the radius
    mom = scipy.stats.norm.moment(ndim - 1, loc=r, scale=w)
    # integral along the angles is surface of hyper-ball
    # which is volume of one higher dimension x (ndim + 1)
    vol = pi**((ndim)/2.) / scipy.special.gamma((ndim)/2. + 1)
    surf = vol * ndim
    return mom * surf

"""
for ndim in [2, 4, 8, 16, 32, 64, 128, 256]:
    r = 0.2
    C = 0.01
    #r = (C * scipy.special.gamma((ndim+3)/2)*ndim*pi**(-(ndim+1)/2) / (
    #        scipy.special.gamma((ndim+2)/2) * pi**(-ndim/2)))**(1 / (ndim+1))
    w = (r**(ndim+1) + C * scipy.special.gamma((ndim+3)/2)*ndim*pi**(-(ndim+1)/2) / (
            scipy.special.gamma((ndim+2)/2) * pi**(-ndim/2)))**(1 / (ndim+1)) - r
    
    vol_sphere = pi**((ndim)/2.) / scipy.special.gamma((ndim)/2. + 1)
    surf_shell = pi**((ndim+1)/2.) / scipy.special.gamma((ndim+1)/2. + 1)
    vol_shell = surf_shell * ((w+r)**(ndim+1) - r**(ndim+1)) / ndim
    #vol_shell = surf_shell * (r**(ndim+1)) / ndim
    
    print('%4d %.3f %.4e %.4e %.4e' % (ndim, w, vol_sphere, vol_shell, vol_shell / vol_sphere))

#import sys; sys.exit()
"""

def main(args):
    ndim = args.x_dim
    adaptive_nsteps = args.adapt_steps
    if adaptive_nsteps is None:
        adaptive_nsteps = False

    
    #C = 0.01
    r = 0.2
    # the shell thickness is 
    #w = (r**(ndim+1) + C * scipy.special.gamma((ndim+3)/2)*ndim*pi**(-(ndim+1)/2) / (
    #        scipy.special.gamma((ndim+2)/2) * pi**(-ndim/2)))**(1 / (ndim+1)) - r
    w = 0.001 / ndim
    
    r1, r2 = r, r
    w1, w2 = w, w
    c1, c2 = np.zeros(ndim) + 0.5, np.zeros(ndim) + 0.5
    c1[0] -= r1 / 2
    c2[0] += r2 / 2
    N1 = -0.5 * log(2 * pi * w1**2)
    N2 = -0.5 * log(2 * pi * w2**2)
    Z_analytic = log(shell_vol(ndim, r1, w1) + shell_vol(ndim, r2, w2))
    
    def loglike(theta):
        d1 = ((theta - c1)**2).sum(axis=1)**0.5
        d2 = ((theta - c2)**2).sum(axis=1)**0.5
        L1 = -0.5 * ((d1 - r1)**2) / w1**2 + N1
        L2 = -0.5 * ((d2 - r2)**2) / w2**2 + N2
        return np.logaddexp(L1, L2)

    def transform(x):
        return x
    
    def gradient(theta):
        delta1 = theta - c1
        delta2 = theta - c1
        d1 = (delta1**2).sum()**0.5
        d2 = (delta2**2).sum()**0.5
        g1 = -delta1 * (1 - r1 / d1) / w1**2
        g2 = -delta2 * (1 - r2 / d2) / w2**2
        return np.logaddexp(g1, g2)
    
    
    """
    N = 10000
    x = np.random.normal(size=(N, ndim))
    x *= (np.random.uniform(size=N)**(1./ndim) / (x**2).sum(axis=1)**0.5).reshape((-1, 1))
    x = x * r1 + c1
    print(loglike(x) - N1)
    print('%.3f%%' % ((loglike(x) - N1 > -ndim*2).mean() * 100))
    
    import sys; sys.exit()
    """
    
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
        #if args.dyhmc:
        #    import ultranest.dyhmc
        #    from ultranest.utils import verify_gradient
        #    verify_gradient(ndim, transform, loglike, transform_loglike_gradient, combination=True)
        #    sampler.stepsampler = ultranest.dyhmc.DynamicHMCSampler(ndim=ndim, nsteps=args.slice_steps, 
        #        transform_loglike_gradient=transform_loglike_gradient)
        if args.dychmc:
            import ultranest.dychmc
            from ultranest.utils import verify_gradient
            verify_gradient(ndim, transform, loglike, gradient, verbose=True)
            sampler.stepsampler = ultranest.dychmc.DynamicCHMCSampler(ndim=ndim, nsteps=args.slice_steps, 
                transform=transform, loglike=loglike, gradient=gradient)
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
    print("expected Z=%.3f (analytic solution)" % Z_analytic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument('--log_dir', type=str, default='logs/multishell')
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
