import argparse
import numpy as np
from numpy import log, pi
import scipy.stats

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

def main(args):
    ndim = args.x_dim
    adaptive_nsteps = args.adapt_steps
    if adaptive_nsteps is None:
        adaptive_nsteps = False
    gamma = args.gamma
    mean = np.zeros(ndim)
    M = np.ones((ndim, ndim)) * gamma
    np.fill_diagonal(M, 1)
    Minv = np.linalg.inv(M)
    logMdet = log(np.linalg.det(M))

    #multiD = np.zeros((1, ndim, ndim))
    multiM = M.reshape((1, ndim, ndim))

    prefactor = log(2 * pi) * ndim
    #diagonal_indices = np.arange(ndim)

    if gamma == 0:
        def loglike(theta):
            sigma = np.exp(theta[0] * 0.5)
            like = -0.5 * (((theta[1:])/sigma)**2).sum() - 0.5 * log(2 * pi * sigma**2) * ndim
            return like
    else:
        def loglike(theta):
            var = np.exp(theta[0])
            cov = M * var
            np.fill_diagonal(cov, var)
            #assert cov.shape == (ndim, ndim), cov.shape
            #assert theta[1:].shape == (ndim,), theta[1:].shape
            #like = scipy.stats.multivariate_normal.logpdf(theta[1:], mean=mean, cov=cov).sum()
            #like = logpdf_multivariate_gauss(theta[1:], mean, cov)
            delta = theta[1:]
            like = -0.5 * (prefactor + log(np.linalg.det(cov)) + delta.T.dot(np.linalg.inv(cov)).dot(delta))
            return like
        def loglike_vectorized1(theta):
            var = np.exp(theta[:,0])
            covs = multiM * var.reshape((-1, 1, 1))
            r = np.einsum('ij,ijk,ik->i', theta[:,1:], np.linalg.inv(covs), theta[:,1:])
            like = -0.5 * (prefactor + log(np.linalg.det(covs)) + r)
            return like
        def loglike_vectorized(theta):
            var = np.exp(theta[:,0])
            r = np.einsum('ij,jk,ik->i', theta[:,1:], Minv, theta[:,1:]) / var
            like = -0.5 * (prefactor + logMdet + theta[:,0] * ndim + r)
            return like
        def loglike_orig(theta):
            var = np.exp(theta[0])
            cov = M * var
            np.fill_diagonal(cov, var)
            assert cov.shape == (ndim, ndim), cov.shape
            assert theta[1:].shape == (ndim,), theta[1:].shape
            like = scipy.stats.multivariate_normal.logpdf(theta[1:], mean=mean, cov=cov).sum()
            return like
    def transform(x):
        z = x * 200 - 100
        z[0] = scipy.stats.norm.ppf(x[0])
        return z
    def transform_vectorized(x):
        z = x * 200 - 100
        z[:,0] = scipy.stats.norm.ppf(x[:,0])
        return z

    paramnames = ['lnvar'] + ['p%d' % i for i in range(ndim)]
    l1 = loglike(transform(np.ones(ndim+1)*0.5))
    l2 = loglike_vectorized(transform_vectorized(np.ones((2,ndim+1))*0.5))[0]
    assert l1 == l2, (l1, l2)
    sigma_dims = [0] if args.with_cones else []


    if args.pymultinest:
        from pymultinest.solve import solve

        result = solve(LogLikelihood=loglike, Prior=transform,
            n_dims=len(paramnames), outputfiles_basename=args.log_dir + 'MN-%dd' % ndim,
            verbose=True, resume=True, importance_nested_sampling=False,
            sampling_efficiency=0.3,
            )

        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(paramnames, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    elif args.reactive:
        if args.slice:
            log_dir = args.log_dir + 'RNS-%dd-slice%d' % (ndim, args.slice_steps)
        elif args.harm:
            log_dir = args.log_dir + 'RNS-%dd-harm%d' % (ndim, args.slice_steps)
        else:
            log_dir = args.log_dir + 'RNS-%dd' % (ndim)
        if adaptive_nsteps:
            log_dir = log_dir + '-adapt%s' % (adaptive_nsteps)
        if sigma_dims != []:
            log_dir = log_dir + '-cones'

        from ultranest import ReactiveNestedSampler
        #sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform,
        #    log_dir=log_dir, resume=True)
        sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform_vectorized,
            log_dir=log_dir, resume=True, vectorized=True, sigma_dims=sigma_dims)
        if args.slice:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
                nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps, region_filter=True)
        if args.harm:
            import ultranest.stepsampler
            sampler.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(
                nsteps=args.slice_steps, adaptive_nsteps=adaptive_nsteps, region_filter=True)
        sampler.run(
            frac_remain=0.5,
            min_num_live_points=args.num_live_points,
            cluster_num_live_points=0,
            max_num_improvement_loops=0,
            dlogz=(0.5 + ndim)**0.5)
        sampler.print_results()
        if sampler.stepsampler is not None:
            sampler.stepsampler.plot(filename = log_dir + '/stepsampler_stats_region.pdf')
        if ndim <= 30:
            sampler.plot()
    else:
        from ultranest import NestedSampler
        sampler = NestedSampler(paramnames, loglike, transform=transform,
            num_live_points=args.num_live_points,
            log_dir=args.log_dir + '-gamma%.2f-%dd' % (gamma, ndim), resume=True)
        sampler.run()
        sampler.print_results()
        sampler.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/stdfunnel')
    parser.add_argument('--pymultinest', action='store_true')
    parser.add_argument('--reactive', action='store_true')
    parser.add_argument('--slice', action='store_true')
    parser.add_argument('--harm', action='store_true')
    parser.add_argument('--slice_steps', type=int, default=100)
    parser.add_argument('--adapt_steps', type=str)
    parser.add_argument('--with-cones', action='store_true')

    args = parser.parse_args()
    main(args)
