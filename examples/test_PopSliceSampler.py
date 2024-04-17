import argparse
import numpy as np

def main(args):

    ndim = args.x_dim
    paramnames = ['param%d' % (i+1) for i in range(ndim)]
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.problem == 'rosenbrock':
        def loglike(theta):
            a = theta[:,:-1]
            b = theta[:,1:]
            return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum(axis=1)

        def transform(u):
            return u * 20 - 10
    if args.problem == 'multishell':
        from numpy import exp, log, pi
        import scipy
        def shell_vol(ndim, r, w):
            # integral along the radius
            mom = scipy.stats.norm.moment(ndim - 1, loc=r, scale=w)
            # integral along the angles is surface of hyper-ball
            # which is volume of one higher dimension x (ndim + 1)
            vol = pi**((ndim)/2.) / scipy.special.gamma((ndim)/2. + 1)
            surf = vol * ndim
            return mom * surf

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

    if args.problem == 'gaussian':
        sigma = args.sigma
        width = max(0, 1 - 5 * sigma)    
        centers = (np.sin(np.arange(ndim)/2.) * width + 1.) / 2.
        sigma = np.random.uniform(0.01, 1., ndim)*sigma
        centers=centers.reshape((1,ndim))
        sigma=np.array(sigma.reshape((1,ndim)))

        norm = -0.5 * np.log(2 * np.pi * sigma**2).sum()
        def loglike(theta):
            return -0.5 * (((theta - centers) / sigma)**2).sum(axis=1) + norm

        def transform(x):
            return x
    if args.problem == 'eggbox':
        def loglike(theta):
            return np.cos(theta).prod(axis=1)**2

        def transform(x):
            return x * 10 * np.pi

    if args.problem == 'funnel':
        sigma = args.sigma
        centers = np.sin(np.arange(ndim) / 2.)
        data = np.random.normal(centers, sigma).reshape((1, -1))

        def loglike(theta):
            sigma = 10**theta[:,0]
            
            like = -0.5 * (((theta[:,1:] - data)/sigma.reshape((-1, 1)))**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
            return like

        def transform(x):
            z = x * 20 - 10
            z[:,0] = x[:,0] * 6 - 3
            return z
        import string
        #print(ndim, len(list(string.ascii_lowercase)))
        paramnames = ['sigma'] + ['param%d' % (i+1) for i in range(ndim)][:ndim]

    
    from ultranest import ReactiveNestedSampler
    from ultranest.calibrator import ReactiveNestedCalibrator
    
    if args.run_type=='Calibration':
        sampler = ReactiveNestedCalibrator(paramnames, loglike,\
                  transform=transform, log_dir=args.log_dir, resume='overwrite',\
                  draw_multiple=False, vectorized=True,)
    if args.run_type=='Normal':
        sampler = ReactiveNestedSampler(paramnames, loglike,\
                  transform=transform, log_dir=args.log_dir, resume='overwrite',\
                  draw_multiple=False, vectorized=True,)
    if args.Sampler=='SimSlice':
        import ultranest.popstepsampler as ultrapop
        direction=[ultrapop.generate_cube_oriented_direction,ultrapop.generate_mixture_random_direction,ultrapop.generate_differential_direction,ultrapop.generate_region_random_direction,ultrapop.generate_region_oriented_direction,ultrapop.generate_random_direction]
        sampler.stepsampler = ultrapop.PopulationSimpleSliceSampler(popsize=args.popsize,nsteps=args.nstep,generate_direction=direction[args.direction])
    if args.Sampler=='PopSlice':
        import ultranest.popstepsampler as ultrapop
        direction=[ultrapop.generate_cube_oriented_direction,ultrapop.generate_mixture_random_direction,ultrapop.generate_differential_direction,ultrapop.generate_region_random_direction,ultrapop.generate_region_oriented_direction,ultrapop.generate_random_direction]
        sampler.stepsampler = ultrapop.PopulationSliceSampler(popsize=args.popsize,nsteps=args.nstep,generate_direction=direction[args.direction],scale=1.0)
    if args.Sampler=='Slice':
        import ultranest.stepsampler as stepsampler
        sampler.stepsampler = stepsampler.SliceSampler(nsteps=args.nstep,generate_direction=stepsampler.generate_mixture_random_direction,)
    if args.Sampler=='PopGaussWalk':
        import ultranest.popstepsampler as ultrapop
        direction=[ultrapop.generate_cube_oriented_direction,ultrapop.generate_random_direction, ultrapop.generate_region_oriented_direction, ultrapop.generate_region_random_direction]
        sampler.stepsampler = ultrapop.PopulationRandomWalkSampler(popsize=args.popsize, nsteps=args.nstep, generate_direction=direction[args.direction],scale=1.0,)

    result=sampler.run(frac_remain=0.5, min_num_live_points=args.num_live_points, max_num_improvement_loops=3)
    
    if args.run_type=='Normal':
        sampler.print_results()
        if ndim <= 20:
            sampler.plot()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--problem', type=str,required=True,choices=['rosenbrock', 'multishell', 'gaussian', 'eggbox', 'funnel'])
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--run_type', type=str, default='Normal', choices=['Normal', 'Calibration'])
    parser.add_argument('--Sampler',type=str,required=True,choices=['SimSlice','PopSlice','Slice','PopGaussWalk'])
    parser.add_argument('--popsize', type=int)
    parser.add_argument('--nstep', type=int)
    parser.add_argument('--direction', type=int)
    main(parser.parse_args())
