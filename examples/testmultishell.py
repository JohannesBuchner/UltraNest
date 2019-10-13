import argparse
import numpy as np
from numpy import sin

def main(args):
    ndim = args.x_dim

    def loglike(z):
        x = z[:,0] - 2
        y = (1 + z[:,1])*10
        L1 = - (np.abs(x**2 + y**2 - 4) / 0.2) + 3*sin(x + y)
        x = z[:,0] + 2
        y = (1 - z[:,1])*10
        L2 = - (np.abs(x**2 + y**2 - 4) / 0.2) + 3*sin(x + y)
        L = np.logaddexp(L1, L2)
        L += -(z[:,2:]**2/0.01).sum(axis=1)
        return L

    def transform(x):
        return x * 10 - 5
    
    import string
    paramnames = list(string.ascii_lowercase)[:ndim]
    
    if args.reactive:
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
            log_dir=args.log_dir + 'RNS-%dd' % args.x_dim, resume='overwrite',
            vectorized=True)
            #log_dir=None)
        if args.slice:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=max(10, 4 * ndim))
        sampler.run(
            #frac_remain=0.5, 
            #min_ess=20000,
            #dKL=np.inf, dlogz=0.5 + 0.1 * ndim,
            #update_interval_iter_fraction=0.4 if ndim > 20 else 0.2,
            #cluster_num_live_points=40,
            #max_num_improvement_loops=3,
            min_num_live_points=args.num_live_points)
        sampler.print_results()
        if args.slice:
            sampler.stepsampler.plot(filename = args.log_dir + 'RNS-%dd/stepsampler_stats_regionslice.pdf' % ndim)
        #sampler.run(log_interval=20, min_num_live_points=args.num_live_points)
        sampler.plot()
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform, 
            num_live_points=args.num_live_points,
            log_dir=args.log_dir + '%dd' % args.x_dim, resume='overwrite')
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
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--reactive', action='store_true')

    args = parser.parse_args()
    main(args)
