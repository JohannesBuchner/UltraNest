import argparse
import numpy as np
from numpy import log
import scipy.stats

def main(args):
    ndim = args.x_dim

    rv1a = scipy.stats.loggamma(1, loc=2./3, scale=1./30)
    rv1b = scipy.stats.loggamma(1, loc=1./3, scale=1./30)
    rv2a = scipy.stats.norm(2./3, 1./30)
    rv2b = scipy.stats.norm(1./3, 1./30)
    rv_rest = []
    for i in range(2, ndim):
	    if i <= (ndim+2)/2:
		    rv = scipy.stats.loggamma(1, loc=2./3., scale=1./30)
	    else:
		    rv = scipy.stats.norm(2./3, 1./30)
	    rv_rest.append(rv)
	    del rv

    def loglike(theta):
        L1 = log(0.5 * rv1a.pdf(theta[:,0]) + 0.5 * rv1b.pdf(theta[:,0]))
        L2 = log(0.5 * rv2a.pdf(theta[:,1]) + 0.5 * rv2b.pdf(theta[:,1]))
        Lrest = np.sum([rv.logpdf(t) for rv, t in zip(rv_rest, theta[:,2:].transpose())], axis=0)
        #assert L1.shape == (len(theta),)
        #assert L2.shape == (len(theta),)
        #assert Lrest.shape == (len(theta),), Lrest.shape
        like = L1 + L2 + Lrest
        like = np.where(like < -300, -300 - ((np.asarray(theta) - 0.5)**2).sum(), like)
        assert like.shape == (len(theta),), (like.shape, theta.shape)
        return like

    def transform(x):
        return x
    
    import string
    paramnames = list(string.ascii_lowercase)[:ndim]
    
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
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir + 'RNS-%dd' % ndim, resume=True,
	    vectorized=True)
        sampler.run(frac_remain=0.5, min_ess=400, min_num_live_points=args.num_live_points)
        sampler.print_results()
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
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/loggamma')
    parser.add_argument('--reactive', action='store_true')
    parser.add_argument('--pymultinest', action='store_true')

    args = parser.parse_args()
    main(args)
