import argparse
import numpy as np
from numpy import pi, sin, log
import matplotlib.pyplot as plt

def main(args):
    from mininest import NestedSampler

    np.random.seed(2)
    Ndata = args.ndata
    jitter_true = 0.1
    phase_true = 0.
    period_true = 180
    amplitude_true = args.contrast / Ndata * jitter_true
    paramnames = ['amplitude', 'jitter', 'phase', 'period']
    derivednames = ['frequency']
    wrapped_params = [False, False, True, False]
    #wrapped_params = None
    
    x = np.linspace(0, 360, 1000)
    y = amplitude_true * sin(x / period_true * 2 * pi + phase_true)
    
    if True:
        plt.plot(x, y)
        x = np.random.uniform(0, 360, Ndata)
        y = np.random.normal(amplitude_true * sin(x / period_true * 2 * pi + phase_true), jitter_true)
        plt.errorbar(x, y, yerr=jitter_true, marker='x', ls=' ')
        plt.savefig('testsine.pdf', bbox_inches='tight')
        plt.close()
    
    
    def loglike(params):
        amplitude, jitter, phase, period, freq = params.transpose()
        predicty = amplitude * sin(x.reshape((-1,1)) / period * 2 * pi + phase)
        logl = (-0.5 * log(2 * pi * jitter**2) - 0.5 * ((predicty - y.reshape((-1,1))) / jitter)**2).sum(axis=0)
        assert logl.shape == jitter.shape
        return logl
    
    def transform(x):
        z = np.empty((len(x), 5))
        z[:,0] = 10**(x[:,0] * 4 - 2)
        z[:,1] = 10**(x[:,1] * 1 - 1.5)
        z[:,2] = 2 * pi * x[:,2]
        z[:,3] = 10**(x[:,3] * 4 - 1)
        z[:,4] = 2 * pi / x[:,3]
        return z

    
    loglike(transform(0.5 * np.ones((1, len(paramnames)))))
    sampler = NestedSampler(paramnames, loglike, transform=transform, 
        log_dir=args.log_dir, num_live_points=args.num_live_points,
        derived_param_names=derivednames, wrapped_params=wrapped_params,
        append_run_num=False)
    sampler.run(log_interval=100)
    #sampler.plot()
    
    for i, p in enumerate(paramnames + derivednames):
        v = sampler.results['samples'][:,i]
        print('%20s: %5.3f +- %5.3f' % (p, v.mean(), v.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--contrast', type=int, default=100,
                        help="Signal-to-Noise level")
    parser.add_argument('--ndata', type=int, default=40,
                        help="Number of simulated data points")
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
    parser.add_argument('--log_dir', type=str, default='logs/testsine')

    args = parser.parse_args()
    main(args)
