import argparse
import numpy as np
from numpy import log

def main(args):
    np.random.seed(2)
    ndim = args.x_dim
    sigma = args.sigma
    centers = np.sin(np.arange(ndim) / 2.)
    data = np.random.normal(centers, sigma).reshape((1, -1))

    def loglike(theta):
        sigma = 10**theta[:,0]
        like = -0.5 * (((theta[:,1:] - data)/sigma.reshape((-1, 1)))**2).sum(axis=1) - 0.5 * log(2 * np.pi * sigma**2) * ndim
        return like

    def transform(x):
        z = x * 20 - 10
        z[:,0] = x[:,0] * 6 - 3
        return z
    
    import string
    paramnames = ['sigma'] + list(string.ascii_lowercase)[:ndim]
    
    if args.reactive:
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir + 'RNS-%dd' % ndim, vectorized=True,
            resume=True)
        sampler.run(log_interval=20, min_num_live_points=args.num_live_points)
        sampler.plot()
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            num_live_points=args.num_live_points, vectorized=True,
            log_dir=args.log_dir + '-%dd' % ndim, resume=True)
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
    parser.add_argument('--log_dir', type=str, default='logs/funnel')
    parser.add_argument('--reactive', action='store_true')

    args = parser.parse_args()
    main(args)
