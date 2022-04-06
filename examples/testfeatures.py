import numpy as np
import shutil
import scipy.stats
import traceback
import json
import random
import sys
import os
import signal
import string
import hashlib


def get_arg_hash(runargs):
    return hashlib.md5(str(runargs).encode()).hexdigest()[:10]


def main(args):
    ndim = args.x_dim
    paramnames = list(string.ascii_lowercase)[:ndim]

    np.random.seed(args.seed)
    if args.wrapped_dims:
        wrapped_params = [True] * ndim
    else:
        wrapped_params = None

    true_Z = None

    if args.log_dir is None:
        if args.delete_dir:
            return
        log_dir = None
    else:
        log_dir = args.log_dir + '-%s' % args.problem
        log_dir += '-%dd' % ndim
        if args.wrapped_dims:
            log_dir += '-wrapped'

        if args.delete_dir:
            shutil.rmtree(log_dir, ignore_errors=True)

    if ndim >= 20 and args.num_live_points >= 1000:
        print("skipping, probably too slow to run")
        return

    if args.problem == 'gauss':
        sigma = 0.01
        if args.wrapped_dims:
            centers = (np.sin(np.arange(ndim) / 2.) + 1.) / 2.
        else:
            centers = (np.sin(np.arange(ndim) / 2.) / 2. + 1.) / 2.
        true_Z = 0

        def loglike(theta):
            like = -0.5 * (((theta - centers) / sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
            return like

        def transform(x):
            return x
    elif args.problem == 'slantedeggbox':
        if not args.pass_transform:
            return

        def loglike(z):
            chi = (2. + (np.cos(z[:,:2] / 2.)).prod(axis=1))**5
            chi2 = -np.abs((z - 5 * np.pi) / 0.5).sum(axis=1)
            return chi + chi2

        def transform(x):
            return x * 100
    elif args.problem == 'funnel':
        if args.wrapped_dims:
            return
        if not args.pass_transform:
            return

        sigma = 0.01
        centers = np.sin(np.arange(ndim) / 2.)
        data = np.random.normal(centers, sigma).reshape((1, -1))

        def loglike(theta):
            sigma = 10**theta[:,0]
            like = -0.5 * (((theta[:,1:] - data) / sigma.reshape((-1, 1)))**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
            return like

        def transform(x):
            z = x * 20 - 10
            z[:,0] = x[:,0] * 6 - 3
            return z

        paramnames.insert(0, 'sigma')
    elif args.problem == 'loggamma':
        true_Z = 0.0
        if args.wrapped_dims:
            return
        rv1a = scipy.stats.loggamma(1, loc=2. / 3, scale=1. / 30)
        rv1b = scipy.stats.loggamma(1, loc=1. / 3, scale=1. / 30)
        rv2a = scipy.stats.norm(2. / 3, 1. / 30)
        rv2b = scipy.stats.norm(1. / 3, 1. / 30)
        rv_rest = []
        for i in range(2, ndim):
            if i <= (ndim + 2) / 2:
                rv = scipy.stats.loggamma(1, loc=2. / 3., scale=1. / 30)
            else:
                rv = scipy.stats.norm(2. / 3, 1. / 30)
            rv_rest.append(rv)
            del rv

        def loglike(theta):
            L1 = np.log(0.5 * rv1a.pdf(theta[:,0]) + 0.5 * rv1b.pdf(theta[:,0]))
            L2 = np.log(0.5 * rv2a.pdf(theta[:,1]) + 0.5 * rv2b.pdf(theta[:,1]))
            Lrest = np.sum([rv.logpdf(t) for rv, t in zip(rv_rest, theta[:,2:].transpose())], axis=0)
            like = L1 + L2 + Lrest
            like = np.where(like < -1e100, -1e100 - ((np.asarray(theta) - 0.5)**2).sum(), like)
            assert like.shape == (len(theta),), (like.shape, theta.shape)
            return like

        def transform(x):
            return x

    from ultranest import ReactiveNestedSampler
    from ultranest.mlfriends import MLFriends, RobustEllipsoidRegion, SimpleRegion, ScalingLayer
    sampler = ReactiveNestedSampler(
        paramnames, loglike,
        transform=transform if args.pass_transform else None,
        log_dir=log_dir, vectorized=True,
        resume='resume' if args.resume else 'overwrite',
        wrapped_params=wrapped_params,
    )
    if hasattr(args, 'axis_aligned') and args.axis_aligned:
        sampler.transform_layer_class = ScalingLayer
        region_class = SimpleRegion
    else:
        region_class = RobustEllipsoidRegion if hasattr(args, 'ellipsoidal') and args.ellipsoidal else MLFriends
    print("MPI:", sampler.mpi_size, sampler.mpi_rank)
    for result in sampler.run_iter(
        update_interval_volume_fraction=args.update_interval_iter_fraction,
        dlogz=args.dlogz,
        dKL=args.dKL,
        frac_remain=args.frac_remain,
        min_ess=args.min_ess,
        max_iters=args.max_iters,
        cluster_num_live_points=args.cluster_num_live_points,
        min_num_live_points=args.num_live_points,
        max_ncalls=int(args.max_ncalls),
        region_class=region_class,
    ):
        sampler.print_results()
        print(
            " (remember, we are trying to achive: %s ) " % (
                dict(
                    dlogz=args.dlogz,
                    dKL=args.dKL,
                    frac_remain=args.frac_remain,
                    min_ess=args.min_ess,
                )))

    results = sampler.results
    try:
        sampler.plot()
    except AssertionError as e:
        if "I don't believe that you want more dimensions than samples" in str(e) and results['ess'] <= ndim + 1:
            pass
        else:
            raise e
    sampler.pointstore.close()
    if results['logzerr_tail'] < 0.5 and results['logzerr'] < 1.0 and true_Z is not None and args.num_live_points > 50:
        assert results['logz'] - results['logzerr'] * 3 < true_Z < results['logz'] + results['logzerr'] * 3
    return results


def run_safely(runargs):
    id = get_arg_hash(runargs)
    if os.path.exists('testfeatures/%s.done' % id):
        print("not rerunning %s" % id)
        return

    print("Running %s with options:" % id, runargs)

    def timeout_handler(signum, frame):
        raise Exception("Timeout")

    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(60 * (1 + runargs['x_dim']))  # give a few minutes
    try:
        main(AttrDict(runargs))
    except Exception:
        traceback.print_exc()
        filename = 'testfeatures/runsettings-%s-error.json' % id
        print("Storing configuration as '%s'. Options were:" % filename, runargs)
        with open(filename, 'w') as f:
            json.dump(runargs, f, indent=2)
        sys.exit(1)
    signal.alarm(0)
    with open('testfeatures/%s.done' % id, 'w'):
        pass


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--random', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timeout', action='store_true')
    parser.add_argument('--nrounds', type=int, default=1,
                        help="Number of random configurations to generate")
    parser.add_argument('conf', nargs='*', help='config files')

    progargs = parser.parse_args()

    if len(progargs.conf) > 0:
        for filename in progargs.conf:
            print("loading configuration from file '%s'..." % filename)
            runargs = json.load(open(filename))
            print("Running with options:", runargs)
            main(AttrDict(runargs))
            if progargs.timeout:
                run_safely(runargs)
        sys.exit(0)

    if progargs.random:
        random.seed(progargs.seed)
        def choose(myargs):
            # pick first (default) option most of the time
            if random.random() < 0.25:
                return myargs[0]
            else:
                return random.choice(myargs)
    else:
        def choose(myargs):
            return myargs

    Nrounds = progargs.nrounds
    i = 0
    while True:
        print("generating a random configuration...")

        runargs = dict(
            problem = choose(['gauss', 'slantedeggbox', 'funnel', 'loggamma']),
            x_dim = choose([2, 1, 6, 20]),
            seed = choose([1, 2, 3]),
            wrapped_dims = choose([False, True]),
            log_dir = choose(['logs/features', None]),
            delete_dir = choose([False, False, False, True]),
            pass_transform = choose([True, False]),
            num_live_points = choose([100, 50, 400, 1000]),
            resume = choose([False, True]),
            cluster_num_live_points = choose([50, 0]),
            update_interval_iter_fraction=choose([0.2, 1.0]),
            dlogz = choose([2.0, 0.5]),
            dKL = choose([1.0, 0.1]),
            frac_remain = choose([0.5, 0.001]),
            min_ess = choose([0, 4000]),
            max_iters = choose([None, 10000]),
            max_ncalls = choose([10000000., 10000., 100000.]),
            axis_aligned = choose([False, True]),
            ellipsoidal = choose([False, True]),
        )
        if not progargs.random:
            key = i
            nkeys = len(runargs.keys())
            for k, v in runargs.items():
                if 0 <= key <= len(v):
                    j = key % len(v)
                    runargs[k] = v[j]
                    key -= len(v)
                else:
                    runargs[k] = v[0]
                    key -= len(v)
            filename = 'testfeatures/runsettings-%s-iterated.json' % get_arg_hash(runargs)
            print("Storing configuration as '%s'. Options were:" % filename, runargs)
            with open(filename, 'w') as f:
                json.dump(runargs, f, indent=2)
            # run_safely(runargs)
            if key > 0:
                break
        else:
            run_safely(runargs)
            if i + 1 >= progargs.nrounds:
                break
        i = i + 1
