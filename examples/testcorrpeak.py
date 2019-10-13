# combination of:
#   - varying sigmas
#   - varying priors (uniform, log-uniform (every third))
#   - tight parameter correlation

import argparse
import numpy as np
#from numpy import log

def main(args):
    ndim = args.x_dim
    sigmas = 10**(-2.0 + 2.0 * np.cos(np.arange(ndim)-2)) / (np.arange(ndim)-2)
    sigmas[:2] = 1.0

    def transform(x):
        y = x #.copy()
        #y[:,1::3] = 10**-y[:,1::3]
        #y[:,::3] *= x[:,2::3]
        return y
    
    centers = transform(np.ones((1, ndim)) * 0.2).flatten()
    degsigmas = 0.01
    crosssigmas = args.sigma
    # * sigmas[3:-1:] * sigmas[4::]
    
    def loglike(theta):
        # gaussian
        like = -0.5 * (np.abs((theta[:,1:] - centers[1:])/sigmas[1:])**2).sum(axis=1)
        # non-linear degeneracy correlation
        like2 = -0.5 * (np.abs((theta[:,1] * theta[:,0] - centers[1] * centers[0])/degsigmas)**2) #.sum(axis=1)
        # pair-wise correlation
        a = (theta[:,3:-1:] - centers[3:-1:]) / sigmas[3:-1:]
        b = (theta[:,4::] - centers[4::]) / sigmas[4::]
        like3 = -0.5 * (np.abs((a - b) / crosssigmas)**2).sum(axis=1)
        return like + like2 + like3

    print(centers, crosssigmas, sigmas)
    import string
    paramnames = list(string.ascii_lowercase)[:ndim]
    
    if args.pymultinest:
        from pymultinest.solve import solve
        import json
        
        def flat_loglike(theta):
            return loglike(theta.reshape((1, -1))).flatten()
        
        def flat_transform(cube):
            return transform(cube.reshape((1, -1))).flatten()
        
        result = solve(LogLikelihood=flat_loglike, Prior=flat_transform, 
            n_dims=ndim, outputfiles_basename=args.log_dir + 'MN-%dd' % ndim,
            verbose=True, resume=True, n_live_points=args.num_live_points,
            importance_nested_sampling=False)
        json.dump(paramnames, open(args.log_dir + 'MN-%ddparams.json' % ndim, 'w'))
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(paramnames, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    
    elif args.reactive:
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir + 'RNS-%dd' % ndim,
            vectorized=True)
        sampler.run(frac_remain=0.5, min_ess=400, min_num_live_points=args.num_live_points)
        sampler.print_results()
        sampler.plot()
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            num_live_points=args.num_live_points, vectorized=True,
            log_dir=args.log_dir + '-%dd' % ndim)
        sampler.run()
        sampler.print_results()
        sampler.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/corrpeak')
    parser.add_argument('--reactive', action='store_true')
    parser.add_argument('--pymultinest', action='store_true')

    args = parser.parse_args()
    main(args)
