import argparse
import numpy as np
from numpy import pi, sin, log
import matplotlib.pyplot as plt

def main(args):
    adaptive_nsteps = args.adapt_steps
    if adaptive_nsteps is None:
        adaptive_nsteps = False

    np.random.seed(2)
    Ndata = args.ndata
    jitter_true = 0.1
    phase_true = 0.
    period_true = 180
    amplitude_true = args.contrast / Ndata * jitter_true
    paramnames = ['amplitude', 'jitter', 'phase', 'period']
    ndim = 4
    derivednames = [] #'frequency']
    wrapped_params = [False, False, True, False]
    #wrapped_params = None
    
    x = np.linspace(0, 360, 1000)
    y = amplitude_true * sin(x / period_true * 2 * pi + phase_true)
    
    if True:
        plt.plot(x, y)
        x = np.random.uniform(0, 360, Ndata)
        y = np.random.normal(amplitude_true * sin(x / period_true * 2 * pi + phase_true), jitter_true)
        plt.errorbar(x, y, yerr=jitter_true, marker='x', ls=' ')
        plt.savefig('testsine.pdf', bbox_inches='tight')
        plt.close()
    
    
    def loglike(params):
        amplitude, jitter, phase, period = params.transpose()[:4]
        predicty = amplitude * sin(x.reshape((-1,1)) / period * 2 * pi + phase)
        logl = (-0.5 * log(2 * pi * jitter**2) - 0.5 * ((predicty - y.reshape((-1,1))) / jitter)**2).sum(axis=0)
        assert logl.shape == jitter.shape
        return logl
    
    def transform(x):
        z = np.empty((len(x), 4))
        z[:,0] = 10**(x[:,0] * 4 - 2)
        z[:,1] = 10**(x[:,1] * 1 - 1.5)
        z[:,2] = 2 * pi * x[:,2]
        z[:,3] = 10**(x[:,3] * 4 - 1)
        #z[:,4] = 2 * pi / x[:,3]
        return z

    loglike(transform(np.ones((2, ndim))*0.5))
    if args.pymultinest:
        from pymultinest.solve import solve
        global Lmax
        Lmax = -np.inf
        
        def flat_loglike(theta):
            L = loglike(theta.reshape((1, -1)))[0]
            global Lmax
            if L > Lmax:
                print("Like: %.2f" % L)
                Lmax = L
            return L
        
        def flat_transform(cube):
            return transform(cube.reshape((1, -1)))[0]
        
        result = solve(LogLikelihood=flat_loglike, Prior=flat_transform, 
            n_dims=ndim, outputfiles_basename=args.log_dir + 'MN-%dd' % ndim,
            n_live_points=args.num_live_points,
            verbose=True, resume=False, importance_nested_sampling=False)
        
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(paramnames, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
        return
    
    elif args.reactive:
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir, vectorized=True,
            derived_param_names=derivednames, wrapped_params=wrapped_params,
            resume='overwrite')
        if args.harm:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps)
        if args.slice:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps)
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir, vectorized=True,
            derived_param_names=derivednames, wrapped_params=wrapped_params,
            resume='overwrite')
    
    sampler.run(min_num_live_points=args.num_live_points)
        
    print()
    sampler.print_results()
    sampler.plot()
    
    for i, p in enumerate(paramnames + derivednames):
        v = sampler.results['samples'][:,i]
        print('%20s: %5.3f +- %5.3f' % (p, v.mean(), v.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--contrast', type=int, default=100,
                        help="Signal-to-Noise level")
    parser.add_argument('--ndata', type=int, default=40,
                        help="Number of simulated data points")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='logs/testsine')
    parser.add_argument('--reactive', action='store_true', default=False)
    parser.add_argument('--pymultinest', action='store_true')
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--harm', action='store_true')
    parser.add_argument('--dyhmc', action='store_true')
    parser.add_argument('--dychmc', action='store_true')
    parser.add_argument('--slice_steps', type=int, default=100)
    parser.add_argument('--adapt_steps', type=str)

    args = parser.parse_args()
    main(args)
