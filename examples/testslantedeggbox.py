import os
import sys
import argparse
import numpy as np
from numpy import cos, pi

def main(args):

    def loglike(z):
        chi = (2. + (cos(z[:,:2] / 2.)).prod(axis=1))**5
        chi2 = -np.abs((z - 5 * pi) / 0.5).sum(axis=1)
        return chi + chi2

    def transform(x):
        return x * 100
    
    import string
    paramnames = list(string.ascii_lowercase)[:args.x_dim]
    
    if args.reactive:
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir + 'RNS-%dd' % args.x_dim, resume=True,
            vectorized=True)
            #log_dir=None)
        sampler.run(log_interval=20, min_num_live_points=args.num_live_points)
        sampler.plot()
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            num_live_points=args.num_live_points, vectorized=True,
            log_dir=args.log_dir + '%dd' % args.x_dim, resume=True)
            #log_dir=None)
        sampler.run(log_interval=20)
        sampler.plot()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/slantedeggbox')
    parser.add_argument('--reactive', action='store_true')

    args = parser.parse_args()
    main(args)
