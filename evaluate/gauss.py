import argparse
import numpy as np
from numpy import log, pi
import scipy.stats
#import sys
import time
import numpy
#import scipy.stats
import tqdm
import matplotlib.pyplot as plt
import joblib

mem = joblib.Memory('.', verbose=False)


def logpdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape == x.shape), ('mu and x must have the same dimensions', mu.shape, x.shape)
    assert(len(mu.shape) == 1), 'mu must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    delta = x - mu
    ndim = len(mu)
    part1 = -0.5 * (log(2 * pi) * ndim + log(np.linalg.det(cov)) )
    part2 = -0.5 * (delta.T.dot(np.linalg.inv(cov))).dot(delta)
    return part1 + part2

parser = argparse.ArgumentParser()

parser.add_argument('--x_dim', type=int, default=2,
                    help="Dimensionality")
parser.add_argument('--gamma', type=float, default=0.9)

args = parser.parse_args()

ndim = args.x_dim
gamma = args.gamma
mean = np.zeros(ndim)
M = np.ones((ndim, ndim)) * gamma
np.fill_diagonal(M, 1)
Minv = np.linalg.inv(M)
logMdet = log(np.linalg.det(M))

multiM = M.reshape((1, ndim, ndim))

prefactor = log(2 * pi) * ndim

def loglike(theta):
    var = np.exp(theta[0])
    cov = M * var
    np.fill_diagonal(cov, var)
    delta = theta[1:]
    like = -0.5 * (prefactor + log(np.linalg.det(cov)) + delta.T.dot(np.linalg.inv(cov)).dot(delta))
    return like

def loglike_vectorized(theta):
    var = np.exp(theta[:,0])
    r = np.einsum('ij,jk,ik->i', theta[:,1:], Minv, theta[:,1:]) / var
    like = -0.5 * (prefactor + logMdet + theta[:,0] * ndim + r)
    return like

def transform(x):
    z = x * 200 - 100
    z[0] = scipy.stats.norm.ppf(x[0])
    return z
def transform_vectorized(x):
    z = x * 200 - 100
    z[:,0] = scipy.stats.norm.ppf(x[:,0])
    return z
def untransform_vectorized(z):
    x = (z + 100) / 200
    x[:,0] = scipy.stats.norm.cdf(z[:,0])
    return x

paramnames = ['lnvar'] + ['p%d' % i for i in range(ndim)]
nparams = len(paramnames)
l1 = loglike(transform(np.ones(ndim+1)*0.5))
l2 = loglike_vectorized(transform_vectorized(np.ones((2,ndim+1))*0.5))[0]
assert l1 == l2, (l1, l2)

gsamples = np.random.multivariate_normal(mean, M, size=1000)
#gsamples = np.random.normal(size=(1000, ndim))
#sigmas = np.random.normal(size=1000).reshape((-1, 1))
sigmas = scipy.stats.norm.cdf(np.random.uniform(0, 0.5, size=1000).reshape((-1, 1)))
samples = np.hstack((gsamples, sigmas))
usamples = untransform_vectorized(samples)
print(samples.shape, samples[:3])
L = loglike_vectorized(samples)
print(L.mean(), L.max(), L.min(), np.median(L))
Lmin = np.median(L)
i = np.argsort(L)[600:]
#us = samples[i,:]
#Ls = L[i]
startpoint = usamples[i[0],:]
borders = list(zip(usamples.min(axis=0), usamples.max(axis=0)))
print(startpoint.shape, startpoint)
print(borders)

def flat_indicator(params):
    if (params > 0).all() and (params < 1).all():
        like = loglike(transform(params))
        if like >= Lmin:
            return 0.
    return -np.inf

def frac_filled(samples):
    key = 0
    for i, (xlo, xhi) in enumerate(borders):
        # divide x and y in a 3x3 grid:
        a = (samples[:,i] - xlo) / (xhi - xlo) * 5
        a[a > 4] = 4
        a[a < 0] = 0
        # assign each grid a index, compute global index by shifting
        key = key * 5 + a.astype(int)
        print(np.unique(a.astype(int)))

    # count how many of the grid parts are filled:
    keys, discovery_indices = np.unique(key, return_index=True)
    discovery_indices.sort()
    return discovery_indices 

@mem.cache
def mcmc(logfunction, x0, nsteps, sigma_p):
    samples = np.empty((nsteps, len(x0)))
    logL0 = logfunction(x0)
    naccepts = 0
    for i in tqdm.trange(nsteps):
        x1 = np.random.normal(x0, sigma_p)
        logL1 = logfunction(x1)
        if logL1 - logL0 > np.log(np.random.uniform()):
            x0, logL0 = x1, logL1
            naccepts += 1
        samples[i,:] = x0
    return samples, naccepts

from ultranest.mlfriends import MLFriends, AffineLayer

def setup_region(startpoint):
    print("finding some initial live points...")
    samples, naccepts = mcmc(flat_indicator, startpoint, 40000, 1e-5)
    us = samples[::100,:]
    Ls = loglike_vectorized(transform_vectorized(us))
    Nuniq = len(np.unique(Ls))
    print('unique live points:', Nuniq, len(us), naccepts)
    assert Nuniq > 1
    transformLayer = AffineLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    return us, Ls, region

import ultranest.stepsampler

def fetch_samples(sampler, startpoint, nsteps):
    us, Ls, region = setup_region(startpoint)
    Nlive = len(us)
    samples = np.empty((nsteps, nparams))
    ncalls = np.zeros(nsteps, dtype=int)
    nc_cum = 0
    for i in tqdm.trange(nsteps):
        u = None
        while u is None:
            u, p, L, nc = sampler.__next__(region, Lmin, us, Ls, transform_vectorized, loglike_vectorized)
            nc_cum += nc
        samples[i,:] = u
        ncalls[i] = nc_cum
        assert L > Lmin
        # replace a live point
        k = np.random.randint(Nlive)
        if k == 0:
            print("scale:", sampler.scale, "span:", us.max(axis=0) - us.min(axis=0))
        us[k,:] = u
        Ls[k] = L
        if i % Nlive == Nlive - 1:
            # update region:
            transformLayer = AffineLayer()
            transformLayer.optimize(us, us)
            region = MLFriends(us, transformLayer)
            region.maxradiussq, region.enlarge = region.compute_enlargement()
            region.create_ellipsoid()
        
    return samples, ncalls

from ultranest.stepsampler import generate_cube_oriented_direction, generate_region_oriented_direction, generate_region_random_direction, generate_random_direction
from ultranest.stepsampler import SliceSampler, MHSampler, OrthogonalProposalGenerator
step_matrix=np.arange(nparams).reshape((-1, 1))
K = max(10, nparams)

samplers = [
    ('mh', 100000, MHSampler(nsteps=K, generate_direction=generate_random_direction)),
    #('regionmh', 100000,  MHSampler(nsteps=K, generate_direction= generate_region_random_direction)),
    ('cubeslice', 100000,  SliceSampler(nsteps=K, generate_direction=generate_cube_oriented_direction)),
    ('regionslice', 100000,  SliceSampler(nsteps=K, generate_direction=generate_region_oriented_direction)),
    ('regionball', 100000,  SliceSampler(nsteps=K, generate_direction=generate_region_random_direction)),
    ('cubeslice-orth', 100000,  SliceSampler(nsteps=K, generate_direction=OrthogonalProposalGenerator(generate_cube_oriented_direction))),
    ('regionslice-orth', 100000,  SliceSampler(nsteps=K, generate_direction=OrthogonalProposalGenerator(generate_region_oriented_direction))),
    ('regionball-orth', 100000,  SliceSampler(nsteps=K, generate_direction=OrthogonalProposalGenerator(generate_region_random_direction))),
    #('seqregionslice', 100000, ultranest.stepsampler.SliceSampler(nsteps=K,
    #    generate_direction=ultranest.stepsampler.SpeedVariableGenerator(step_matrix=step_matrix, generate_direction=ultranest.stepsampler.generate_region_oriented_direction))
    #),
    #('seqcubeslice', 10000, ultranest.stepsampler.SliceSampler(nsteps=K,
    #    generate_direction=ultranest.stepsampler.SpeedVariableGenerator(step_matrix=step_matrix, generate_direction=ultranest.stepsampler.generate_cube_oriented_direction))
    #)
]

@mem.cache
def get_samples(samplername, nsteps, nparams, seed=1):
    np.random.seed(seed)
    for samplernamei, _, sampler in samplers:
        if samplername == samplernamei:
            tstart = time.time()
            samples, ncalls = fetch_samples(
                startpoint=startpoint,
                sampler=sampler,
                nsteps=nsteps
            )
            return samples, ncalls, time.time() - tstart
    assert False

def main():
    print('start point:', loglike(transform(startpoint)), startpoint, Lmin)
    print('start point:', flat_indicator(startpoint), startpoint)
    assert flat_indicator(startpoint) == 0.0
    # generate live points

    for samplername, nsteps, sampler in samplers:
        print("checking sampler: %s" % samplername, sampler)
        l = None
        samples, ncalls, T = get_samples(samplername, nsteps, nparams)
        discovery_indices = frac_filled(samples) + 1
        ndiscovered = np.arange(len(discovery_indices)) + 1
        cost = discovery_indices * ncalls[-1] / nsteps
        print("discovery indices:", discovery_indices, "average cost per step:", ncalls[-1] / nsteps)
        if l is None:
            l, = plt.plot(cost, ndiscovered, label=samplername)
        else:
            plt.plot(cost, ndiscovered, color=l.get_color())
        print(cost.max())

        plt.legend(loc='best', title='%d+1 dim' % (ndim), prop=dict(size=8))
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(1, None)
        plt.xlabel('Number of model evaluations')
        plt.ylabel('Number of regions discovered')
        plt.savefig('gauss_discovery_%d.pdf' % ndim, bbox_inches='tight')
        
        import corner
        corner.corner(samples, truths=startpoint)
        plt.savefig('gauss_sampled_%s.pdf' % ndim, bbox_inches='tight')
        plt.close()

        plt.figure()
        r, pvalue = scipy.stats.spearmanr(samples)
        r[np.arange(nparams),np.arange(nparams)] = 0
        plt.imshow(r)
        plt.colorbar()
        plt.savefig('gauss_corr.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
