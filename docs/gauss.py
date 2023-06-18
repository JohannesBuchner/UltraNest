import argparse
import numpy as np
from numpy import log

# define command line arguments:
parser = argparse.ArgumentParser()

parser.add_argument('--x_dim', type=int, default=2,
                    help="Dimensionality")
parser.add_argument("--num_live_points", type=int, default=400)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--slice', action='store_true')
parser.add_argument('--slice_steps', type=int, default=100)
parser.add_argument('--log_dir', type=str, default='logs/loggauss')

args = parser.parse_args()

ndim = args.x_dim
sigma = args.sigma
width = max(0, 1 - 5 * sigma)
centers = (np.sin(np.arange(ndim)/2.) * width + 1.) / 2.

# Here, we implement a vectorized loglikelihood, which can
# process many points at the same time. This reduces function calls.
def loglike(theta):
    like = -0.5 * (((theta - centers)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
    return like

def transform(x):
    return x

paramnames = ['param%d' % (i+1) for i in range(ndim)]

# set up nested sampler:

from ultranest import ReactiveNestedSampler

sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
    log_dir=args.log_dir + 'RNS-%dd' % ndim, resume=True,
    vectorized=True)

if args.slice:
    # set up step sampler. Here, we use a differential evolution slice sampler:
    import ultranest.stepsampler
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=args.slice_steps,
        generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
    )

# run sampler, with a few custom arguments:
sampler.run(dlogz=0.5 + 0.1 * ndim,
    update_interval_volume_fraction=0.4 if ndim > 20 else 0.2,
    max_num_improvement_loops=3,
    min_num_live_points=args.num_live_points)

sampler.print_results()

if args.slice:
    sampler.stepsampler.plot(filename = args.log_dir + 'RNS-%dd/stepsampler_stats_regionslice.pdf' % ndim)

sampler.plot()
