import argparse
import numpy as np
from numpy import log
import scipy.stats

def main(args):
    ndim = args.x_dim
    sigma = args.sigma
    centers = (np.sin(np.arange(ndim)/2.) + 1.) / 2.

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1)
        return like

    def transform(x):
        return x
    
    import string
    paramnames = list(string.ascii_lowercase)[:ndim]
    
    if args.reactive:
        from mininest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            min_num_live_points=args.num_live_points,
            log_dir=args.log_dir + '-%dd' % ndim, append_run_num=True)
        sampler.run()
        sampler.plot()
    else:
        from mininest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            num_live_points=args.num_live_points,
            log_dir=args.log_dir, append_run_num=True)
        sampler.run()
        sampler.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/loggauss')
    parser.add_argument('--reactive', action='store_true')

    args = parser.parse_args()
    main(args)
