import argparse
import numpy as np
from numpy import log
import scipy.stats

def main(args):
    ndim = args.x_dim

    def flat_loglike(theta):
        delta = np.max(np.abs(theta - 0.5))
        volume_enclosed = ndim * np.log(delta + 1e-15)
        if volume_enclosed > -100:
            return -volume_enclosed
        else:
            return +100
        #like = min(-volume_enclosed, +100)
        #return like

    def flat_transform(x):
        return x
    
    import string
    paramnames = ['param%d' % (i+1) for i in range(ndim)]
    
    if args.pymultinest:
        from pymultinest.solve import solve
        
        result = solve(LogLikelihood=flat_loglike, Prior=flat_transform, 
            n_dims=ndim, outputfiles_basename=args.log_dir + 'MN-%dd' % ndim,
            verbose=True, resume=True, importance_nested_sampling=False)
        
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(paramnames, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    
    elif args.reactive:
        from ultranest.solvecompat import pymultinest_solve_compat as solve
        result = solve(LogLikelihood=flat_loglike, Prior=flat_transform, 
            n_dims=ndim, outputfiles_basename=args.log_dir + 'MN-%dd' % ndim,
            verbose=True, resume=True)
        
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(paramnames, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='logs/hyperrect')
    parser.add_argument('--reactive', action='store_true')
    parser.add_argument('--pymultinest', action='store_true')

    args = parser.parse_args()
    main(args)
