import os
import sys
import argparse
import numpy as np

def main(args):
    from ultranest import NestedSampler

    #def loglike(z):
    #    return np.array([-sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) for x in z])
    def loglike_(z):
        return np.array([-sum(100.0 * (x[1::2] - x[::2] ** 2.0) ** 2.0 + (1 - x[::2]) ** 2.0) for x in z])

    def loglike(z):
        a = np.array([-0.5 * sum([((xi - 0.83456 + i*0.1)/0.01)**2 for i, xi in enumerate(x)]) for x in z])
        b = np.array([-0.5 * sum([((xi - 0.43456 - i*0.1)/0.01)**2 for i, xi in enumerate(x)]) for x in z])
        return np.logaddexp(a, b)

    def transform(x):
        return 10. * x - 5.
    
    import string
    paramnames = list(string.ascii_lowercase)[:args.x_dim]

    sampler = NestedSampler(paramnames, loglike, transform=transform, 
        vectorized=True, log_dir=args.log_dir)
    sampler.run(log_interval=20, num_live_points=args.num_live_points)
    sampler.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=50,
                        help="number of train iters")
    parser.add_argument("--mcmc_steps", type=int, default=0)
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--test_mcmc_steps", type=int, default=1000)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/rosenbrock')

    args = parser.parse_args()
    main(args)
