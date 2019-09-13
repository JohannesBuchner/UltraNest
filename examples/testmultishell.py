import os
import sys
import argparse
import numpy as np
from numpy import sin, pi

def main(args):

    def loglike(z):
        x = z[:,0] - 2
        y = (1 + z[:,1])*10
        L1 = - (np.abs(x**2 + y**2 - 4) / 0.2) + 3*sin(x + y)
        x = z[:,0] + 2
        y = (1 - z[:,1])*10
        L2 = - (np.abs(x**2 + y**2 - 4) / 0.2) + 3*sin(x + y)
        L = np.logaddexp(L1, L2)
        return L

    def transform(x):
        return x * 10 - 5
    
    import string
    paramnames = list(string.ascii_lowercase)[:args.x_dim]
    
    if args.reactive:
        from mininest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir + 'RNS-%dd' % args.x_dim, append_run_num=False)
            #log_dir=None)
        sampler.run(log_interval=20, min_num_live_points=args.num_live_points)
        sampler.plot()
    else:
        from mininest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            num_live_points=args.num_live_points,
            log_dir=args.log_dir + '%dd' % args.x_dim, append_run_num=True)
            #log_dir=None)
        sampler.run(log_interval=20)
        sampler.plot()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument('--log_dir', type=str, default='logs/multishell')
    parser.add_argument('--reactive', action='store_true')

    args = parser.parse_args()
    main(args)
