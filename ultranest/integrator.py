"""Ultranest calculates the Bayesian evidence and posterior samples of arbitrary models."""

# Some parts are from the Nestle library by Kyle Barbary (https://github.com/kbarbary/nestle)
# Some parts are from the nnest library by Adam Moss (https://github.com/adammoss/nnest)

from __future__ import print_function
from __future__ import division

import os
import sys
import csv
import json
import operator
import time
import warnings

from numpy import log, exp, logaddexp
import numpy as np

from .utils import create_logger, make_run_dir, resample_equal, vol_prefactor, vectorize
from ultranest.mlfriends import MLFriends, AffineLayer, ScalingLayer, find_nearby
from .store import HDF5PointStore, NullPointStore
from .viz import get_default_viz_callback, nicelogger
from .netiter import PointPile, MultiCounter, BreadthFirstIterator, TreeNode, count_tree_between, find_nodes_before, logz_sequence
from .netiter import dump_tree

__all__ = ['ReactiveNestedSampler', 'NestedSampler']


def _get_cumsum_range(pi, dp):
    """From probabilities pi compute the quantile indices of `dp` and ``1-dp``."""
    ci = pi.cumsum()
    ilo = np.where(ci > dp)[0]
    ilo = ilo[0] if len(ilo) > 0 else 0
    ihi = np.where(ci < 1. - dp)[0]
    ihi = ihi[-1] if len(ihi) > 0 else -1
    return ilo, ihi


def _sequentialize_width_sequence(minimal_widths, min_width):
    """Turn a list of required tree width into an ordered sequence.

    Parameters
    ------------
    minimal_widths: list of (Llo, Lhi, width)
        Defines the required width between Llo and Lhi.
    min_width: int
        Minimum width everywhere.

    Returns:
    ---------
    Lsequence: list of (L, width)
        A sequence of L points and the expected tree width at and above it.

    """
    Lpoints = np.unique([-np.inf] + [L for L, _, _ in minimal_widths] +
                        [L for _, L, _ in minimal_widths] + [np.inf])
    widths = np.ones(len(Lpoints)) * min_width

    for Llo, Lhi, width in minimal_widths:
        # all Lpoints within that range should be maximized to width
        # mask = np.logical_and(Lpoints >= Llo, Lpoints <= Lhi)
        # the following allows segments to specify -inf..L ranges
        mask = ~np.logical_or(Lpoints < Llo, Lpoints > Lhi)
        widths[mask] = np.where(widths[mask] < width, width, widths[mask])

    # the width has to monotonically increase to the maximum from both sides
    # so we fill up any intermediate dips
    max_width = widths.max()
    mid = np.where(widths == max_width)[0][0]
    widest = 0
    for i in range(mid):
        widths[i] = max(widest, widths[i])
        widest = widths[i]
    widest = 0
    for i in range(len(widths) - 1, mid, -1):
        widths[i] = max(widest, widths[i])
        widest = widths[i]

    return list(zip(Lpoints, widths))


class NestedSampler(object):
    """Simple Nested sampler for reference."""

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 derived_param_names=[],
                 resume='subfolder',
                 run_num=None,
                 log_dir='logs/test',
                 num_live_points=1000,
                 vectorized=False,
                 wrapped_params=[],
                 ):
        """Set up nested sampler.

        Parameters
        -----------
        param_names: list of str, names of the parameters.
            Length gives dimensionality of the sampling problem.
        loglike: function
            log-likelihood function.
            Receives multiple parameter vectors, returns vector of likelihood.
        transform: function
            parameter transform from unit cube to physical parameters.
            Receives multiple cube vectors, returns multiple parameter vectors.
        derived_param_names: list of str
            Additional derived parameters created by transform. (empty by default)
        log_dir: str
            where to store output files
        resume: 'resume', 'overwrite' or 'subfolder'
            if 'overwrite', overwrite previous data.
            if 'subfolder', create a fresh subdirectory in log_dir
            if 'resume' or True, continue previous run if available.
        wrapped_params: list of bools
            indicating whether this parameter wraps around (circular parameter).
        num_live_points: int
            Number of live points
        vectorized: bool
            If true, loglike and transform function can receive arrays
            of points.

        """
        self.paramnames = param_names
        x_dim = len(self.paramnames)
        self.num_live_points = num_live_points
        self.sampler = 'nested'
        self.x_dim = x_dim
        self.derivedparamnames = derived_param_names
        num_derived = len(self.derivedparamnames)
        self.num_params = x_dim + num_derived
        self.volfactor = vol_prefactor(self.x_dim)
        if wrapped_params is None:
            self.wrapped_axes = []
        else:
            self.wrapped_axes = np.where(wrapped_params)[0]

        assert resume in (True, 'overwrite', 'subfolder', 'resume'), "resume should be one of 'overwrite' 'subfolder' or 'resume'"
        append_run_num = resume == 'subfolder'
        resume = resume == 'resume' or resume == True

        if not vectorized:
            transform = vectorize(transform)
            loglike = vectorize(loglike)

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        u = np.random.uniform(size=(2, self.x_dim))
        p = self.transform(u)
        assert p.shape == (2, self.num_params), ("Error in transform function: returned shape is %s, expected %s" % (p.shape, (2, self.num_params)))
        logl = loglike(p)
        assert np.logical_and(u > 0, u < 1).all(), ("Error in transform function: u was modified!")
        assert logl.shape == (2,), ("Error in loglikelihood function: returned shape is %s, expected %s" % (p.shape, (2, self.num_params)))
        assert np.isfinite(logl).all(), ("Error in loglikelihood function: returned non-finite number: %s for input u=%s p=%s" % (logl, u, p))

        def safe_loglike(x):
            x = np.asarray(x)
            logl = loglike(x)
            assert np.isfinite(logl).all(), ('loglikelihood returned non-finite value:',
                   logl[~np.isfinite(logl)], x[~np.isfinite(logl),:])
            return logl

        self.loglike = safe_loglike

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
        except Exception:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = self.mpi_rank == 0
        self.log_to_disk = self.log and log_dir is not None

        if self.log and log_dir is not None:
            self.logs = make_run_dir(log_dir, run_num, append_run_num=append_run_num)
            log_dir = self.logs['run_dir']
        else:
            log_dir = None

        self.logger = create_logger(__name__ + '.' + type(self).__name__, log_dir=log_dir)

        if self.log:
            self.logger.info('Num live points [%d]', self.num_live_points)

        if self.log_to_disk:
            # self.pointstore = TextPointStore(os.path.join(self.logs['results'], 'points.tsv'), 2 + self.x_dim + self.num_params)
            self.pointstore = HDF5PointStore(os.path.join(self.logs['results'], 'points.hdf5'),
                3 + self.x_dim + self.num_params, mode='a' if resume else 'w')
        else:
            self.pointstore = NullPointStore(3 + self.x_dim + self.num_params)

    def run(
            self,
            update_interval_iter=None,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=0.001,
            max_iters=None):
        """Explore parameter space.

        Parameters
        ----------
        update_interval_iter:
            Update region after this many iterations.
        update_interval_ncall:
            Update region after update_interval_ncall likelihood calls.
        log_interval:
            Update stdout status line every log_interval iterations
        dlogz:
            Target evidence uncertainty.
        max_iters:
            maximum number of integration iterations.

        """
        if update_interval_ncall is None:
            update_interval_ncall = max(1, round(self.num_live_points))

        if update_interval_iter is None:
            if update_interval_ncall == 0:
                update_interval_iter = max(1, round(self.num_live_points))
            else:
                update_interval_iter = max(1, round(0.2 * self.num_live_points))

        if log_interval is None:
            log_interval = max(1, round(0.2 * self.num_live_points))
        else:
            log_interval = round(log_interval)
            if log_interval < 1:
                raise ValueError("log_interval must be >= 1")

        viz_callback = get_default_viz_callback()

        prev_u = []
        prev_v = []
        prev_logl = []
        if self.log:
            # try to resume:
            self.logger.info('Resuming...')
            for i in range(self.num_live_points):
                _, row = self.pointstore.pop(-np.inf)
                if row is not None:
                    prev_logl.append(row[1])
                    prev_u.append(row[3:3 + self.x_dim])
                    prev_v.append(row[3 + self.x_dim:3 + self.x_dim + self.num_params])
                else:
                    break

            prev_u = np.array(prev_u)
            prev_v = np.array(prev_v)
            prev_logl = np.array(prev_logl)

            num_live_points_missing = self.num_live_points - len(prev_logl)
        else:
            num_live_points_missing = -1

        if self.use_mpi:
            num_live_points_missing = self.comm.bcast(num_live_points_missing, root=0)
            prev_u = self.comm.bcast(prev_u, root=0)
            prev_v = self.comm.bcast(prev_v, root=0)
            prev_logl = self.comm.bcast(prev_logl, root=0)

        use_point_stack = True

        assert num_live_points_missing >= 0
        if num_live_points_missing > 0:
            if self.use_mpi:
                self.logger.info('Using MPI with rank [%d]', self.mpi_rank)
                if self.mpi_rank == 0:
                    active_u = np.random.uniform(size=(num_live_points_missing, self.x_dim))
                else:
                    active_u = np.empty((num_live_points_missing, self.x_dim), dtype=np.float64)
                active_u = self.comm.bcast(active_u, root=0)
            else:
                active_u = np.random.uniform(size=(num_live_points_missing, self.x_dim))
            active_v = self.transform(active_u)

            if self.use_mpi:
                if self.mpi_rank == 0:
                    chunks = [[] for _ in range(self.mpi_size)]
                    for i, chunk in enumerate(active_v):
                        chunks[i % self.mpi_size].append(chunk)
                else:
                    chunks = None
                data = self.comm.scatter(chunks, root=0)
                active_logl = self.loglike(data)
                recv_active_logl = self.comm.gather(active_logl, root=0)
                recv_active_logl = self.comm.bcast(recv_active_logl, root=0)
                active_logl = np.concatenate(recv_active_logl, axis=0)
            else:
                active_logl = self.loglike(active_v)

            if self.log_to_disk:
                for i in range(num_live_points_missing):
                    self.pointstore.add([-np.inf, active_logl[i], 0.] + active_u[i,:].tolist() + active_v[i,:].tolist())

            if len(prev_u) > 0:
                active_u = np.concatenate((prev_u, active_u))
                active_v = np.concatenate((prev_v, active_v))
                active_logl = np.concatenate((prev_logl, active_logl))
            assert active_u.shape == (self.num_live_points, self.x_dim)
            assert active_v.shape == (self.num_live_points, self.num_params)
            assert active_logl.shape == (self.num_live_points,)
        else:
            active_u = prev_u
            active_v = prev_v
            active_logl = prev_logl

        saved_u = []
        saved_v = []  # Stored points for posterior results
        saved_logl = []
        saved_logwt = []
        h = 0.0  # Information, initially 0.
        logz = -1e300  # ln(Evidence Z), initially Z=0
        logvol = log(1.0 - exp(-1.0 / self.num_live_points))
        logz_remain = np.max(active_logl)
        fraction_remain = 1.0
        ncall = num_live_points_missing  # number of calls we already made
        first_time = True
        if self.x_dim > 1:
            transformLayer = AffineLayer(wrapped_dims=self.wrapped_axes)
        else:
            transformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
        transformLayer.optimize(active_u, active_u)
        region = MLFriends(active_u, transformLayer)

        if self.log:
            self.logger.info('Starting sampling ...')
        ib = 0
        samples = []
        ndraw = 100
        it = 0
        next_update_interval_ncall = -1
        next_update_interval_iter = -1

        while max_iters is None or it < max_iters:

            # Worst object in collection and its weight (= volume * likelihood)
            worst = np.argmin(active_logl)
            logwt = logvol + active_logl[worst]

            # Update evidence Z and information h.
            logz_new = np.logaddexp(logz, logwt)
            h = (exp(logwt - logz_new) * active_logl[worst] + exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new

            # Add worst object to samples.
            saved_u.append(np.array(active_u[worst]))
            saved_v.append(np.array(active_v[worst]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[worst])

            # expected_vol = np.exp(-it / self.num_live_points)

            # The new likelihood constraint is that of the worst object.
            loglstar = active_logl[worst]

            if ncall > next_update_interval_ncall and it > next_update_interval_iter:

                if first_time:
                    nextregion = region
                else:
                    # rebuild space
                    # print()
                    # print("rebuilding space...", active_u.shape, active_u)
                    nextTransformLayer = transformLayer.create_new(active_u, region.maxradiussq)
                    nextregion = MLFriends(active_u, nextTransformLayer)

                # print("computing maxradius...")
                r, f = nextregion.compute_enlargement(nbootstraps=max(1, 30 // self.mpi_size))
                # print("MLFriends built. r=%f" % r**0.5)
                if self.use_mpi:
                    recv_maxradii = self.comm.gather(r, root=0)
                    recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                    r = np.max(recv_maxradii)
                    recv_enlarge = self.comm.gather(f, root=0)
                    recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
                    f = np.max(recv_enlarge)

                nextregion.maxradiussq = r
                nextregion.enlarge = f
                # force shrinkage of volume
                # this is to avoid re-connection of dying out nodes
                if nextregion.estimate_volume() < region.estimate_volume():
                    region = nextregion
                    transformLayer = region.transformLayer
                region.create_ellipsoid(minvol=exp(-it / self.num_live_points) * self.volfactor)

                if self.log:
                    viz_callback(points=dict(u=active_u, p=active_v, logl=active_logl),
                        info=dict(it=it, ncall=ncall, logz=logz, logz_remain=logz_remain,
                        paramnames=self.paramnames + self.derivedparamnames, logvol=logvol),
                        region=region, transformLayer=transformLayer)
                    self.pointstore.flush()

                next_update_interval_ncall = ncall + update_interval_ncall
                next_update_interval_iter = it + update_interval_iter
                first_time = False

            while True:
                if ib >= len(samples) and use_point_stack:
                    # root checks the point store
                    next_point = np.zeros((1, 3 + self.x_dim + self.num_params))

                    if self.log_to_disk:
                        _, stored_point = self.pointstore.pop(loglstar)
                        if stored_point is not None:
                            next_point[0,:] = stored_point
                        else:
                            next_point[0,:] = -np.inf
                        use_point_stack = not self.pointstore.stack_empty

                    if self.use_mpi:  # and informs everyone
                        use_point_stack = self.comm.bcast(use_point_stack, root=0)
                        next_point = self.comm.bcast(next_point, root=0)

                    # assert not use_point_stack

                    # unpack
                    likes = next_point[:,1]
                    samples = next_point[:,3:3 + self.x_dim]
                    samplesv = next_point[:,3 + self.x_dim:3 + self.x_dim + self.num_params]
                    # skip if we already know it is not useful
                    ib = 0 if np.isfinite(likes[0]) else 1

                while ib >= len(samples):
                    # get new samples
                    ib = 0

                    nc = 0
                    u, father = region.sample(nsamples=ndraw)
                    nu = u.shape[0]
                    if nu == 0:
                        v = np.empty((0, self.x_dim))
                        logl = np.empty((0,))
                    else:
                        v = self.transform(u)
                        logl = self.loglike(v)
                        nc += nu
                        accepted = logl > loglstar
                        u = u[accepted,:]
                        v = v[accepted,:]
                        logl = logl[accepted]
                        father = father[accepted]

                    if self.use_mpi:
                        recv_father = self.comm.gather(father, root=0)
                        recv_samples = self.comm.gather(u, root=0)
                        recv_samplesv = self.comm.gather(v, root=0)
                        recv_likes = self.comm.gather(logl, root=0)
                        recv_nc = self.comm.gather(nc, root=0)
                        recv_father = self.comm.bcast(recv_father, root=0)
                        recv_samples = self.comm.bcast(recv_samples, root=0)
                        recv_samplesv = self.comm.bcast(recv_samplesv, root=0)
                        recv_likes = self.comm.bcast(recv_likes, root=0)
                        recv_nc = self.comm.bcast(recv_nc, root=0)
                        samples = np.concatenate(recv_samples, axis=0)
                        samplesv = np.concatenate(recv_samplesv, axis=0)
                        father = np.concatenate(recv_father, axis=0)
                        likes = np.concatenate(recv_likes, axis=0)
                        ncall += sum(recv_nc)
                    else:
                        samples = np.array(u)
                        samplesv = np.array(v)
                        likes = np.array(logl)
                        ncall += nc

                    if self.log:
                        for ui, vi, logli in zip(samples, samplesv, likes):
                            self.pointstore.add([loglstar, logli, 0.0] + ui.tolist() + vi.tolist())

                if likes[ib] > loglstar:
                    active_u[worst] = samples[ib, :]
                    active_v[worst] = samplesv[ib,:]
                    active_logl[worst] = likes[ib]

                    # if we keep the region informed about the new live points
                    # then the region follows the live points even if maxradius is not updated
                    region.u[worst,:] = active_u[worst]
                    region.unormed[worst,:] = region.transformLayer.transform(region.u[worst,:])

                    # if we track the cluster assignment, then in the next round
                    # the ids with the same members are likely to have the same id
                    # this is imperfect
                    # transformLayer.clusterids[worst] = transformLayer.clusterids[father[ib]]
                    # so we just mark the replaced ones as "unassigned"
                    transformLayer.clusterids[worst] = 0
                    ib = ib + 1
                    break
                else:
                    ib = ib + 1

            # Shrink interval
            logvol -= 1.0 / self.num_live_points
            logz_remain = np.max(active_logl) - it / self.num_live_points
            fraction_remain = np.logaddexp(logz, logz_remain) - logz

            if it % log_interval == 0 and self.log:
                # nicelogger(self.paramnames, active_u, active_v, active_logl, it, ncall, logz, logz_remain, region=region)
                sys.stdout.write('Z=%.1g+%.1g | Like=%.1g..%.1g | it/evals=%d/%d eff=%.4f%%  \r' % (
                    logz, logz_remain, loglstar, np.max(active_logl), it,
                    ncall, np.inf if ncall == 0 else it * 100 / ncall))
                sys.stdout.flush()

                # if efficiency becomes low, bulk-process larger arrays
                ndraw = max(128, min(16384, round((ncall + 1) / (it + 1) / self.mpi_size)))

            # Stopping criterion
            if fraction_remain < dlogz:
                break
            it = it + 1

        logvol = -len(saved_v) / self.num_live_points - log(self.num_live_points)
        for i in range(self.num_live_points):
            logwt = logvol + active_logl[i]
            logz_new = np.logaddexp(logz, logwt)
            h = (exp(logwt - logz_new) * active_logl[i] + exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new
            saved_u.append(np.array(active_u[i]))
            saved_v.append(np.array(active_v[i]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[i])

        saved_u = np.array(saved_u)
        saved_v = np.array(saved_v)
        saved_wt = exp(np.array(saved_logwt) - logz)
        saved_logl = np.array(saved_logl)
        logzerr = np.sqrt(h / self.num_live_points)

        if self.log_to_disk:
            with open(os.path.join(self.logs['results'], 'final.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['niter', 'ncall', 'logz', 'logzerr', 'h'])
                writer.writerow([it + 1, ncall, logz, logzerr, h])
            self.pointstore.close()

        if not self.use_mpi or self.mpi_rank == 0:
            print()
            print("niter: {:d}\n ncall: {:d}\n nsamples: {:d}\n logz: {:6.3f} +/- {:6.3f}\n h: {:6.3f}"
                  .format(it + 1, ncall, len(saved_v), logz, logzerr, h))

        self.results = dict(samples=resample_equal(saved_v, saved_wt / saved_wt.sum()),
            ncall=ncall, niter=it, logz=logz, logzerr=logzerr,
            weighted_samples=dict(u=saved_u, v=saved_v, w=saved_wt, logw=saved_logwt, L=saved_logl),
        )

        return self.results

    def print_results(self, logZ=True, posterior=True):
        """Give summary of marginal likelihood and parameters."""
        print()
        print('logZ = %(logz).3f +- %(logzerr).3f' % self.results)

        print()
        for i, p in enumerate(self.paramnames + self.derivedparamnames):
            v = self.results['samples'][:,i]
            sigma = v.std()
            med = v.mean()
            if sigma == 0:
                i = 3
            else:
                i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join(['    %-20s' + fmt + " +- " + fmt])
            print(fmts % (p, med, sigma))

    def plot(self):
        """Make corner plot."""
        if self.log_to_disk:
            import matplotlib.pyplot as plt
            import corner
            data = np.array(self.results['weighted_samples']['v'])
            weights = np.array(self.results['weighted_samples']['w'])
            weights /= weights.sum()
            cumsumweights = np.cumsum(weights)

            mask = cumsumweights > 1e-4

            corner.corner(data[mask,:], weights=weights[mask],
                    labels=self.paramnames + self.derivedparamnames, show_titles=True)
            plt.savefig(os.path.join(self.logs['plots'], 'corner.pdf'), bbox_inches='tight')
            plt.close()


class ReactiveNestedSampler(object):
    """Nested sampler with reactive exploration strategy.

    Storage & resume capable, optionally MPI parallelised.
    """

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 derived_param_names=[],
                 wrapped_params=None,
                 resume='subfolder',
                 run_num=None,
                 log_dir=None,
                 num_test_samples=2,
                 draw_multiple=True,
                 num_bootstraps=30,
                 vectorized=False,
                 ):
        """Initialise nested sampler.

        Parameters
        -----------
        param_names: list of str, names of the parameters.
            Length gives dimensionality of the sampling problem.

        loglike: function
            log-likelihood function.
            Receives multiple parameter vectors, returns vector of likelihood.
        transform: function
            parameter transform from unit cube to physical parameters.
            Receives multiple cube vectors, returns multiple parameter vectors.

        derived_param_names: list of str
            Additional derived parameters created by transform. (empty by default)

        log_dir: str
            where to store output files
        resume: 'resume', 'overwrite' or 'subfolder'
            if 'overwrite', overwrite previous data.
            if 'subfolder', create a fresh subdirectory in log_dir
            if 'resume' or True, continue previous run if available.

        wrapped_params: list of bools
            indicating whether this parameter wraps around (circular parameter).

        num_test_samples: int
            test transform and likelihood with this number of
            random points for errors first. Useful to catch bugs.

        draw_multiple: bool
            draw more points if efficiency goes down.
            If set to False, few points are sampled at once.

        num_bootstraps: int
            number of logZ estimators and MLFriends region
            bootstrap rounds.

        vectorized: bool
            If true, loglike and transform function can receive arrays
            of points.

        """
        self.paramnames = param_names
        x_dim = len(self.paramnames)

        self.sampler = 'reactive-nested'
        self.x_dim = x_dim
        self.derivedparamnames = derived_param_names
        self.num_bootstraps = int(num_bootstraps)
        num_derived = len(self.derivedparamnames)
        self.num_params = x_dim + num_derived
        if wrapped_params is None:
            self.wrapped_axes = []
        else:
            assert len(wrapped_params) == self.x_dim, ("wrapped_params has the number of entries:", wrapped_params, ", expected", self.x_dim)
            self.wrapped_axes = np.where(wrapped_params)[0]

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
                self._setup_distributed_seeds()
        except Exception:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = self.mpi_rank == 0
        self.log_to_disk = self.log and log_dir is not None

        assert resume in (True, 'overwrite', 'subfolder', 'resume'), "resume should be one of 'overwrite' 'subfolder' or 'resume'"
        append_run_num = resume == 'subfolder'
        resume = resume == 'resume' or resume == True
        
        if self.log and log_dir is not None:
            self.logs = make_run_dir(log_dir, run_num, append_run_num=append_run_num)
            log_dir = self.logs['run_dir']
        else:
            log_dir = None

        if self.log:
            self.logger = create_logger('ultranest', log_dir=log_dir)
        self.root = TreeNode(id=-1, value=-np.inf)

        self.pointpile = PointPile(self.x_dim, self.num_params)
        self.ncall = 0
        self.ncall_region = 0
        if self.log_to_disk:
            # self.pointstore = TextPointStore(os.path.join(self.logs['results'], 'points.tsv'), 2 + self.x_dim + self.num_params)
            self.pointstore = HDF5PointStore(os.path.join(self.logs['results'], 'points.hdf5'),
                3 + self.x_dim + self.num_params, mode='a' if resume else 'w')
            self.ncall = len(self.pointstore.stack)
        else:
            self.pointstore = NullPointStore(3 + self.x_dim + self.num_params)

        if not vectorized:
            if transform is not None:
                transform = vectorize(transform)
            loglike = vectorize(loglike)
            draw_multiple = False

        self.draw_multiple = draw_multiple
        self._set_likelihood_function(transform, loglike, num_test_samples)
        self.stepsampler = None

    def _setup_distributed_seeds(self):
        if not self.use_mpi:
            return
        seed = 0
        if self.mpi_rank == 0:
            seed = np.random.randint(0, 1000000)

        seed = self.comm.bcast(seed, root=0)
        if self.mpi_rank > 0:
            # from http://arxiv.org/abs/1005.4117
            seed = int(abs(((seed * 181) * ((self.mpi_rank - 83) * 359)) % 104729))
            # print('setting seed:', self.mpi_rank, seed)
            np.random.seed(seed)

    def _set_likelihood_function(self, transform, loglike, num_test_samples, make_safe=False):
        """Store the transform and log-likelihood functions.

        Tests with `num_test_samples` whether they work and give the correct output.

        if make_safe is set, make functions safer by accepting misformed
        return shapes and non-finite likelihood values.
        """
        # do some checks on the likelihood function
        # this makes debugging easier by failing early with meaningful errors

        # if we are resuming, check that last sample still gives same result
        num_resume_test_samples = 0
        if num_test_samples and not self.pointstore.stack_empty:
            num_resume_test_samples = 1
            num_test_samples -= 1

        if num_test_samples > 0:
            # test with num_test_samples random points
            u = np.random.uniform(size=(num_test_samples, self.x_dim))
            p = transform(u) if transform is not None else u
            assert p.shape == (num_test_samples, self.num_params), ("Error in transform function: returned shape is %s, expected %s" % (p.shape, (num_test_samples, self.num_params)))
            logl = loglike(p)
            assert np.logical_and(u > 0, u < 1).all(), ("Error in transform function: u was modified!")
            assert logl.shape == (num_test_samples,), ("Error in loglikelihood function: returned shape is %s, expected %s" % (p.shape, (num_test_samples, self.num_params)))
            assert np.isfinite(logl).all(), ("Error in loglikelihood function: returned non-finite number: %s for input u=%s p=%s" % (logl, u, p))

        if not self.pointstore.stack_empty and num_resume_test_samples > 0:
            # test that last sample gives the same likelihood value
            _, lastrow = self.pointstore.stack[-1]
            assert len(lastrow) == 3 + self.x_dim + self.num_params, ("Cannot resume: problem has different dimensionality", len(lastrow), (2, self.x_dim, self.num_params))
            lastL = lastrow[1]
            lastu = lastrow[3:3 + self.x_dim]
            u = lastu.reshape((1, -1))
            lastp = lastrow[3 + self.x_dim:3 + self.x_dim + self.num_params]
            if self.log:
                self.logger.debug("Testing resume consistency: %s: u=%s -> p=%s -> L=%s ", lastrow, lastu, lastp, lastL)
            p = transform(u) if transform is not None else u
            if not np.allclose(p.flatten(), lastp) and self.log:
                self.logger.warning("Trying to resume from previous run, but transform function gives different result: %s gave %s, now %s", lastu, lastp, p.flatten())
            assert np.allclose(p.flatten(), lastp), "Cannot resume because transform function changed. To start from scratch, delete '%s'." % (self.logs['run_dir'])
            logl = loglike(p).flatten()[0]
            if not np.isclose(logl, lastL) and self.log:
                self.logger.warning("Trying to resume from previous run, but likelihood function gives different result: %s gave %s, now %s", lastu.flatten(), lastL, logl)
            assert np.isclose(logl, lastL), "Cannot resume because loglikelihood function changed. To start from scratch, delete '%s'." % (self.logs['run_dir'])

        def safe_loglike(x):
            x = np.asarray(x)
            if len(x.shape) == 1:
                assert x.shape[0] == self.x_dim
                x = np.expand_dims(x, 0)
            logl = loglike(x)
            if len(logl.shape) == 0:
                logl = np.expand_dims(logl, 0)
            logl[np.logical_not(np.isfinite(logl))] = -1e100
            return logl

        if make_safe:
            self.loglike = safe_loglike
        else:
            self.loglike = loglike

        if transform is None:
            self.transform = lambda x: x
        elif make_safe:
            def safe_transform(x):
                x = np.asarray(x)
                if len(x.shape) == 1:
                    assert x.shape[0] == self.x_dim
                    x = np.expand_dims(x, 0)
                return transform(x)
            self.transform = safe_transform
        else:
            self.transform = transform

        lims = np.ones((2, self.x_dim))
        lims[0,:] = 1e-6
        lims[1,:] = 1 - 1e-6
        self.transform_limits = self.transform(lims).transpose()

        self.volfactor = vol_prefactor(self.x_dim)

    def _widen_nodes(self, weighted_parents, weights, nnodes_needed, update_interval_ncall):
        """Ensure that at parents have `nnodes_needed` live points (parallel arcs).

        If not, fill up by sampling.
        """
        ndone = len(weighted_parents)
        if ndone == 0:
            if self.log:
                self.logger.info('No parents, so widening roots')
            self._widen_roots(nnodes_needed)
            return {}

        # select parents with weight 1/parent_weights
        p = 1. / np.array(weights)
        if (p == p[0]).all():
            parents = weighted_parents
        else:
            # preferentially select nodes with few parents, as those
            # have most weight
            i = np.random.choice(len(weighted_parents), size=nnodes_needed, p=p / p.sum())
            parents = [weighted_parents[ii] for ii in i]

        del weighted_parents, weights

        # sort from low to high
        parents.sort(key=operator.attrgetter('value'))
        Lmin = parents[0].value
        if np.isinf(Lmin):
            # some of the parents were born by sampling from the entire
            # prior volume. So we can efficiently apply a solution:
            # expand the roots
            if self.log:
                self.logger.info('parent value is -inf, so widening roots')
            self._widen_roots(nnodes_needed)
            return {}

        # double until we reach the necessary points
        # this is probably 1, from (2K - K) / K
        nsamples = int(np.ceil((nnodes_needed - ndone) / len(parents)))

        if self.log:
            self.logger.info('Will add %d live points (x%d) at L=%.1g ...', nnodes_needed - ndone, nsamples, Lmin)

        # add points where necessary (parents can have multiple entries)
        target_min_num_children = {}
        for n in parents:
            orign = target_min_num_children.get(n.id, len(n.children))
            target_min_num_children[n.id] = orign + nsamples

        return target_min_num_children

    def _widen_roots(self, nroots):
        """Ensure root has `nroots` children.

        Sample from prior to fill up (if needed).
        """
        if self.log and len(self.root.children) > 0:
            self.logger.info('Widening roots to %d live points (have %d already) ...', nroots, len(self.root.children))
        nnewroots = nroots - len(self.root.children)
        if nnewroots <= 0:
            # nothing to do
            return

        prev_u = []
        prev_v = []
        prev_logl = []
        prev_rowid = []

        if self.log and self.use_point_stack:
            # try to resume:
            self.logger.info('Resuming...')
            for i in range(nnewroots):
                rowid, row = self.pointstore.pop(-np.inf)
                if row is None:
                    break
                prev_logl.append(row[1])
                prev_u.append(row[3:3 + self.x_dim])
                prev_v.append(row[3 + self.x_dim:3 + self.x_dim + self.num_params])
                prev_rowid.append(rowid)

        if self.log:
            prev_u = np.array(prev_u)
            prev_v = np.array(prev_v)
            prev_logl = np.array(prev_logl)

            num_live_points_missing = nnewroots - len(prev_logl)
        else:
            num_live_points_missing = -1

        if self.use_mpi:
            num_live_points_missing = self.comm.bcast(num_live_points_missing, root=0)
            prev_u = self.comm.bcast(prev_u, root=0)
            prev_v = self.comm.bcast(prev_v, root=0)
            prev_logl = self.comm.bcast(prev_logl, root=0)

        assert num_live_points_missing >= 0
        if self.log and num_live_points_missing > 0:
            self.logger.info('Sampling %d live points from prior ...', num_live_points_missing)
        if num_live_points_missing > 0:
            if self.use_mpi:
                if self.mpi_rank == 0:
                    active_u = np.random.uniform(size=(num_live_points_missing, self.x_dim))
                else:
                    active_u = np.empty((num_live_points_missing, self.x_dim), dtype=np.float64)
                active_u = self.comm.bcast(active_u, root=0)
            else:
                active_u = np.random.uniform(size=(num_live_points_missing, self.x_dim))
            active_v = self.transform(active_u)

            if self.use_mpi:
                if self.mpi_rank == 0:
                    chunks = [[] for _ in range(self.mpi_size)]
                    for i, chunk in enumerate(active_v):
                        chunks[i % self.mpi_size].append(chunk)
                    self.ncall += num_live_points_missing
                else:
                    chunks = None
                data = self.comm.scatter(chunks, root=0)
                active_logl = self.loglike(data)
                assert active_logl.shape == (len(data),), (active_logl.shape, len(data))
                recv_active_logl = self.comm.gather(active_logl, root=0)
                recv_active_logl = self.comm.bcast(recv_active_logl, root=0)
                self.ncall = self.comm.bcast(self.ncall, root=0)
                active_logl = np.concatenate(recv_active_logl, axis=0)
                assert active_logl.shape == (num_live_points_missing,), (active_logl.shape, num_live_points_missing, chunks)
            else:
                self.ncall += num_live_points_missing
                active_logl = self.loglike(active_v)

            assert active_logl.shape == (num_live_points_missing,), (active_logl.shape, num_live_points_missing)

            if self.log_to_disk:
                for i in range(num_live_points_missing):
                    rowid = self.pointstore.add([-np.inf, active_logl[i], 0.0] + active_u[i,:].tolist() + active_v[i,:].tolist())

            if len(prev_u) > 0:
                active_u = np.concatenate((prev_u, active_u))
                active_v = np.concatenate((prev_v, active_v))
                active_logl = np.concatenate((prev_logl, active_logl))
            assert active_u.shape == (nnewroots, self.x_dim), (active_u.shape, nnewroots, self.x_dim, num_live_points_missing, len(prev_u))
            assert active_v.shape == (nnewroots, self.num_params), (active_v.shape, nnewroots, self.num_params, num_live_points_missing, len(prev_u))
            assert active_logl.shape == (nnewroots,), (active_logl.shape, nnewroots)
        else:
            active_u = prev_u
            active_v = prev_v
            active_logl = prev_logl

        roots = [self.pointpile.make_node(logl, u, p) for u, p, logl in zip(active_u, active_v, active_logl)]
        self.root.children += roots

    def _adaptive_strategy_advice(self, Lmin, parallel_values, main_iterator, minimal_widths, frac_remain):
        """Check if integration is done.

        Returns
        --------
        Llo, Lhi: floats
            range where more sampling is needed
            if done, both are nan

        Parameters
        -----------
        Lmin: float
            current loglikelihood threshold
        parallel_values: array of floats
            loglikelihoods of live points
        main_iterator: BreadthFirstIterator
            current tree exploration iterator
        minimal_widths: list
            current width required
        frac_remain: float
            maximum fraction of integral in remainder for termination

        """
        Ls = parallel_values.copy()
        Ls.sort()
        # Ls = [node.value] + [n.value for rootid2, n in parallel_nodes]
        Lmax = Ls[-1]
        Lmin = Ls[0]

        # all points the same, stop
        if Lmax - Lmin < 0.001:
            return np.nan, np.nan

        # max remainder contribution is Lmax + weight, to be added to main_iterator.logZ
        # the likelihood that would add an equal amount as main_iterator.logZ is:
        logZmax = main_iterator.logZremain
        Lnext = logZmax - (main_iterator.logVolremaining + log(frac_remain)) - log(len(Ls))
        L1 = Ls[1] if len(Ls) > 1 else Ls[0]
        Lmax1 = Ls[-2] if len(Ls) > 1 else Ls[-1]
        Lnext = max(min(Lnext, Lmax1), L1)

        # if the remainder dominates, return that range
        if main_iterator.logZremain > main_iterator.logZ:
            return Lmin, Lnext

        if main_iterator.remainder_fraction > frac_remain:
            return Lmin, Lnext

        return np.nan, np.nan

    def _find_strategy(self, saved_logl, main_iterator, dlogz, dKL, min_ess):
        """Ask each strategy which log-likelihood interval needs more exploration.

        Returns
        -------
        (Llo_Z, Lhi_Z): floats
            interval where dlogz strategy requires more samples.
        (Llo_KL, Lhi_KL): floats
            interval where posterior uncertainty strategy requires more samples.
        (Llo_ess, Lhi_ess): floats
            interval where effective sample strategy requires more samples.

        Parameters
        ----------
        saved_logl: array of float
            loglikelihood values in integration
        main_iterator: BreadthFirstIterator
            current tree exploration iterator
        dlogz: float
            required logZ accuracy (smaller is stricter)
        dKL: float
            required Kulback-Leibler information gain between bootstrapped
            nested sampling incarnations (smaller is stricter).
        min_ess: float
            required number of effective samples (higher is stricter).

        """
        saved_logl = np.asarray(saved_logl)
        logw = np.asarray(main_iterator.logweights) + saved_logl.reshape((-1,1)) - main_iterator.all_logZ
        ref_logw = logw[:,0].reshape((-1,1))
        other_logw = logw[:,1:]

        Llo_ess = np.inf
        Lhi_ess = -np.inf
        w = exp(ref_logw.flatten())
        w /= w.sum()
        ess = len(w) / (1.0 + ((len(w) * w - 1)**2).sum() / len(w))
        if ess < min_ess:
            samples = np.random.choice(len(w), p=w, size=min_ess)
            Llo_ess = saved_logl[samples].min()
            Lhi_ess = saved_logl[samples].max()
        if self.log and Lhi_ess > Llo_ess:
            self.logger.info("Effective samples strategy wants to improve: %.2f..%.2f (ESS = %.1f, need >%d)",
                             Llo_ess, Lhi_ess, ess, min_ess)
        elif self.log and min_ess > 0:
            self.logger.info("Effective samples strategy satisfied (ESS = %.1f, need >%d)",
                             ess, min_ess)

        # compute KL divergence
        with np.errstate(invalid='ignore'):
            KL = np.where(np.isfinite(other_logw), exp(other_logw) * (other_logw - ref_logw), 0)
        KLtot = KL.sum(axis=0)
        dKLtot = np.abs(KLtot - KLtot.mean())
        p = np.where(KL > 0, KL, 0)
        p /= p.sum(axis=0).reshape((1, -1))

        Llo_KL = np.inf
        Lhi_KL = -np.inf
        for i, (pi, dKLi, logwi) in enumerate(zip(p.transpose(), dKLtot, other_logw)):
            if dKLi > dKL:
                ilo, ihi = _get_cumsum_range(pi, 1. / 400)
                # ilo and ihi are most likely missing in this iterator
                # --> select the one before/after in this iterator
                ilos = np.where(np.isfinite(logwi[:ilo]))[0]
                ihis = np.where(np.isfinite(logwi[ihi:]))[0]
                ilo2 = ilos[-1] if len(ilos) > 0 else 0
                ihi2 = (ihi + ihis[0]) if len(ihis) > 0 else -1
                # self.logger.info('   - KL[%d] = %.2f: need to improve near %.2f..%.2f --> %.2f..%.2f' % (i, dKLi, saved_logl[ilo], saved_logl[ihi], saved_logl[ilo2], saved_logl[ihi2]))
                Llo_KL = min(Llo_KL, saved_logl[ilo2])
                Lhi_KL = max(Lhi_KL, saved_logl[ihi2])

        if self.log and Lhi_KL > Llo_KL:
            self.logger.info("Posterior uncertainty strategy wants to improve: %.2f..%.2f (KL: %.2f+-%.2f nat, need <%.2f nat)",
                             Llo_KL, Lhi_KL, KLtot.mean(), dKLtot.max(), dKL)
        elif self.log:
            self.logger.info("Posterior uncertainty strategy is satisfied (KL: %.2f+-%.2f nat, need <%.2f nat)",
                             KLtot.mean(), dKLtot.max(), dKL)

        Llo_Z = np.inf
        Lhi_Z = -np.inf
        # compute difference between lnZ cumsum
        p = exp(logw)
        p /= p.sum(axis=0).reshape((1, -1))
        deltalogZ = np.abs(main_iterator.all_logZ[1:] - main_iterator.logZ)

        tail_fraction = w[np.asarray(main_iterator.istail)].sum() / w.sum()
        logzerr_tail = logaddexp(log(tail_fraction) + main_iterator.logZ, main_iterator.logZ) - main_iterator.logZ
        if (deltalogZ > dlogz).any() and (main_iterator.logZerr_bs**2 + logzerr_tail**2)**0.5 > dlogz:
            for i, (pi, deltalogZi) in enumerate(zip(p.transpose(), deltalogZ)):
                if deltalogZi > dlogz:
                    # break up samples with too much weight
                    samples = np.random.choice(len(ref_logw), p=pi, size=400)
                    # if self.log:
                    #     self.logger.info('   - deltalogZi[%d] = %.2f: need to improve near %.2f..%.2f' % (
                    #         i, deltalogZi, saved_logl[samples].min(), saved_logl[samples].max()))
                    Llo_Z = min(Llo_Z, saved_logl[samples].min())
                    Lhi_Z = max(Lhi_Z, saved_logl[samples].max())

        if self.log and Lhi_Z > Llo_Z:
            self.logger.info("Evidency uncertainty strategy wants to improve: %.2f..%.2f (dlogz from %.2f to %.2f, need <%s)",
                             Llo_Z, Lhi_Z, deltalogZ.mean(), deltalogZ.max(), dlogz)
        elif self.log:
            self.logger.info("Evidency uncertainty strategy is satisfied (dlogz=%.2f, need <%s)",
                             deltalogZ.max(), dlogz)
        self.logger.info('  logZ error budget: single: %.2f bs:%.2f tail:%.2f total:%.2f required:<%.2f',
                         main_iterator.logZerr, main_iterator.logZerr_bs, logzerr_tail,
                         (main_iterator.logZerr_bs**2 + logzerr_tail**2)**0.5, dlogz)

        return (Llo_Z, Lhi_Z), (Llo_KL, Lhi_KL), (Llo_ess, Lhi_ess)

    def _pump_region(self, Lmin, ndraw, nit, active_u, active_values):
        """Get new samples at `Lmin` with attribute `.stepsampler`."""
        # tlive = self.region.transformLayer.transform(ulive)
        u, v, logl, nc = self.stepsampler.__next__(self.region,
            transform=self.transform, loglike=self.loglike,
            Lmin=Lmin, us=active_u, Ls=active_values,
            ndraw=min(10, ndraw))
        if logl is None:
            u = np.empty((0, self.x_dim))
            v = np.empty((0, self.num_params))
            logl = np.empty((0,))
        else:
            assert np.logical_and(u > 0, u < 1).all(), (u)
            u = u.reshape((1, self.x_dim))
            v = v.reshape((1, self.num_params))
            logl = logl.reshape((1,))

        if self.use_mpi:
            recv_samples = self.comm.gather(u, root=0)
            recv_samplesv = self.comm.gather(v, root=0)
            recv_likes = self.comm.gather(logl, root=0)
            recv_nc = self.comm.gather(nc, root=0)
            recv_samples = self.comm.bcast(recv_samples, root=0)
            recv_samplesv = self.comm.bcast(recv_samplesv, root=0)
            recv_likes = self.comm.bcast(recv_likes, root=0)
            recv_nc = self.comm.bcast(recv_nc, root=0)
            self.samples = np.concatenate(recv_samples, axis=0)
            self.samplesv = np.concatenate(recv_samplesv, axis=0)
            self.likes = np.concatenate(recv_likes, axis=0)
            self.ncall += sum(recv_nc)
        else:
            self.samples = u
            self.samplesv = v
            self.likes = logl
            self.ncall += nc
        self.ncall_region += ndraw

        if self.log:
            quality = self.stepsampler.nsteps
            for ui, vi, logli in zip(self.samples, self.samplesv, self.likes):
                self.pointstore.add([Lmin, logli, quality] + ui.tolist() + vi.tolist())

    def _refill_samples(self, Lmin, ndraw, nit):
        """Get new samples from region."""
        nc = 0
        u, father = self.region.sample(nsamples=ndraw)
        assert np.logical_and(u > 0, u < 1).all(), (u)
        nu = u.shape[0]
        if nu == 0:
            v = np.empty((0, self.num_params))
            logl = np.empty((0,))
        else:
            if nu > 1 and not self.draw_multiple:
                nu = 1
                u = u[:1,:]
                father = father[:1]

            v = self.transform(u)
            logl = self.loglike(v)
        nc += nu
        accepted = logl > Lmin
        if nit >= 100000 / ndraw and nit % (100000 // ndraw) == 0 and not self.sampling_slow_warned:
            np.savez('sampling-stuck-it%d.npz' % nit, u=self.region.u, unormed=self.region.unormed, maxradiussq=self.region.maxradiussq,
                sample_u=u, sample_v=v, sample_logl=logl)
            warnings.warn("Sampling from region seems inefficient. You can try increasing nlive, frac_remain, dlogz, dKL, decrease min_ess). [%d/%d accepted, it=%d]" % (accepted.sum(), ndraw, nit))
            logl_region = self.loglike(self.transform(self.region.u))
            if (logl_region == Lmin).all():
                raise ValueError("Region cannot sample a higher point. All remaining live points have the same value.")
            if not (logl_region > Lmin).any():
                raise ValueError("Region cannot sample a higher point. Perhaps you are resuming from a different problem? Delete the output files and start again.")
            self.sampling_slow_warned = True

        u = u[accepted,:]
        v = v[accepted,:]
        logl = logl[accepted]

        if self.use_mpi:
            recv_samples = self.comm.gather(u, root=0)
            recv_samplesv = self.comm.gather(v, root=0)
            recv_likes = self.comm.gather(logl, root=0)
            recv_nc = self.comm.gather(nc, root=0)
            recv_samples = self.comm.bcast(recv_samples, root=0)
            recv_samplesv = self.comm.bcast(recv_samplesv, root=0)
            recv_likes = self.comm.bcast(recv_likes, root=0)
            recv_nc = self.comm.bcast(recv_nc, root=0)
            self.samples = np.concatenate(recv_samples, axis=0)
            self.samplesv = np.concatenate(recv_samplesv, axis=0)
            self.likes = np.concatenate(recv_likes, axis=0)
            self.ncall += sum(recv_nc)
        else:
            self.samples = np.array(u)
            self.samplesv = np.array(v)
            self.likes = np.array(logl)
            self.ncall += nc
        self.ncall_region += ndraw

        if self.log:
            for ui, vi, logli in zip(self.samples, self.samplesv, self.likes):
                self.pointstore.add([Lmin, logli, 0.0] + ui.tolist() + vi.tolist())

    def _create_point(self, Lmin, ndraw, active_u, active_values):
        """Draw a new point above likelihood threshold `Lmin`.

        Parameters
        -----------
        Lmin: float
            loglikelihood threshold to draw above
        ndraw: float
            number of points to try to sample at once
        active_u: array of floats
            current live points
        active_values
            loglikelihoods of current live points

        """
        nit = 0
        while True:
            ib = self.ib
            if ib >= len(self.samples) and self.use_point_stack:
                # root checks the point store
                next_point = np.zeros((1, 3 + self.x_dim + self.num_params)) * np.nan

                if self.log_to_disk:
                    _, stored_point = self.pointstore.pop(Lmin)
                    if stored_point is not None:
                        next_point[0,:] = stored_point
                    else:
                        next_point[0,:] = -np.inf
                    self.use_point_stack = not self.pointstore.stack_empty

                if self.use_mpi:  # and informs everyone
                    self.use_point_stack = self.comm.bcast(self.use_point_stack, root=0)
                    next_point = self.comm.bcast(next_point, root=0)

                # unpack
                self.likes = next_point[:,1]
                self.samples = next_point[:,3:3 + self.x_dim]
                self.samplesv = next_point[:,3 + self.x_dim:3 + self.x_dim + self.num_params]
                # skip if we already know it is not useful
                ib = 0 if np.isfinite(self.likes[0]) else 1

            assert self.region.inside(active_u).any(), ("None of the live points satisfies the current region!",
                self.region.maxradiussq, self.region.u, self.region.unormed, active_u)
            if self.stepsampler is None:
                while ib >= len(self.samples):
                    ib = 0
                    self._refill_samples(Lmin, ndraw, nit)
                    nit += 1
            else:
                while ib >= len(self.samples):
                    ib = 0
                    self._pump_region(Lmin, ndraw, nit, active_u=active_u, active_values=active_values)
                    nit += 1

            if self.likes[ib] > Lmin:
                u = self.samples[ib, :]
                assert np.logical_and(u > 0, u < 1).all(), (u)
                p = self.samplesv[ib, :]
                logl = self.likes[ib]

                self.ib = ib + 1
                return u, p, logl
            else:
                self.ib = ib + 1

    def _update_region(self, active_u, active_node_ids,
        bootstrap_rootids=None, active_rootids=None,
        nbootstraps=30, minvol=0.
    ):
        """Build a new MLFriends region from `active_u`.

        Parameters
        -----------
        active_u: array of floats
            current live points
        active_node_ids: 2d array of ints
            which bootstrap initialisation the points belong to.
        active_rootids: 2d array of ints
            roots active in each bootstrap initialisation
        bootstrap_rootids: array of ints
            bootstrap samples. if None, they are drawn fresh.
        nbootstraps: int
            number of bootstrap rounds
        minvol: float
            expected current minimum volume of region.

        Returns
        --------
        updated: bool
            True if update was made, False if previous region remained.

        """
        assert nbootstraps > 0
        updated = False
        if self.region is None:
            # if self.log:
            #    self.logger.debug("building first region ...")
            if self.x_dim > 1:
                self.transformLayer = AffineLayer(wrapped_dims=self.wrapped_axes)
            else:
                self.transformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
            self.transformLayer.optimize(active_u, active_u, minvol=minvol)
            self.region = MLFriends(active_u, self.transformLayer)
            self.region_nodes = active_node_ids.copy()
            assert self.region.maxradiussq is None

            r, f = self.region.compute_enlargement(minvol=minvol,
                nbootstraps=max(1, nbootstraps // self.mpi_size))
            # rng=np.random.RandomState(self.mpi_rank))
            # print("MLFriends built. r=%f" % r**0.5)
            if self.use_mpi:
                recv_maxradii = self.comm.gather(r, root=0)
                recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                r = np.max(recv_maxradii)
                recv_enlarge = self.comm.gather(f, root=0)
                recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
                f = np.max(recv_enlarge)

            self.region.maxradiussq = r
            self.region.enlarge = f
            # if self.log:
            #     self.logger.debug("building first region ... r=%e, f=%e" % (r, f))

            # verify correctness:
            # self.region.create_ellipsoid(minvol=minvol)
            # assert self.region.inside(active_u).all(), self.region.inside(active_u).mean()

        assert self.transformLayer is not None
        need_accept = False

        if self.region.maxradiussq is None:
            # we have been told that radius is currently invalid
            # we need to bootstrap back to a valid state

            # compute radius given current transformLayer
            oldu = self.region.u
            self.region.u = active_u
            self.region.set_transformLayer(self.transformLayer)
            r, f = self.region.compute_enlargement(minvol=minvol,
                nbootstraps=max(1, nbootstraps // self.mpi_size))
            # rng=np.random.RandomState(self.mpi_rank))
            # print("MLFriends built. r=%f" % r**0.5)
            if self.use_mpi:
                recv_maxradii = self.comm.gather(r, root=0)
                recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                r = np.max(recv_maxradii)
                recv_enlarge = self.comm.gather(f, root=0)
                recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
                f = np.max(recv_enlarge)

            self.region.maxradiussq = r
            self.region.enlarge = f

            # print("made first region, r=%e" % (r))

            # now that we have r, can do clustering
            # self.transformLayer.nclusters, self.transformLayer.clusterids, _ = update_clusters(
            #    self.region.u, self.region.unormed, self.region.maxradiussq)
            # but such reclustering would forget the cluster ids

            # instead, track the clusters from before by matching manually
            oldt = self.transformLayer.transform(oldu)
            clusterids = np.zeros(len(active_u), dtype=int)
            nnearby = np.empty(len(self.region.unormed), dtype=int)
            for ci in np.unique(self.transformLayer.clusterids):
                if ci == 0:
                    continue

                # find points from that cluster
                oldti = oldt[self.transformLayer.clusterids == ci]
                # identify which new points are near this cluster
                find_nearby(oldti, self.region.unormed, self.region.maxradiussq, nnearby)
                mask = nnearby != 0
                # assign the nearby ones to this cluster
                # if they have not been set yet
                # if they have, set them to -1
                clusterids[mask] = np.where(clusterids[mask] == 0, ci, -1)

            # print("following clusters, nc=%d" % r, self.transformLayer.nclusters,
            #    np.unique(clusterids, return_counts=True))

            # clusters we are unsure about (double assignments) go unassigned
            clusterids[clusterids == -1] = 0

            # tell scaling layer the correct cluster information
            self.transformLayer.clusterids = clusterids

            # we want the clustering to repeat to remove remaining zeros
            need_accept = (self.transformLayer.clusterids == 0).any()

            updated = True
            assert len(self.region.u) == len(self.transformLayer.clusterids)

            # verify correctness:
            # self.region.create_ellipsoid(minvol=minvol)
            # assert self.region.inside(active_u).all(), self.region.inside(active_u).mean()

        assert len(self.region.u) == len(self.transformLayer.clusterids)
        # rebuild space
        # print()
        # print("rebuilding space...", active_u.shape, active_u)
        with warnings.catch_warnings(), np.errstate(all='raise'):
            try:
                nextTransformLayer = self.transformLayer.create_new(active_u, self.region.maxradiussq, minvol=minvol)
                # nextTransformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
                # nextTransformLayer.optimize(active_u, active_u)
                assert not (nextTransformLayer.clusterids == 0).any()
                _, cluster_sizes = np.unique(nextTransformLayer.clusterids, return_counts=True)
                smallest_cluster = cluster_sizes.min()
                if self.log and smallest_cluster == 1:
                    self.logger.debug("clustering found some stray points [need_accept=%s] %s",
                        need_accept, np.unique(nextTransformLayer.clusterids, return_counts=True))

                nextregion = MLFriends(active_u, nextTransformLayer)
                if not np.isfinite(nextregion.unormed).all():
                    assert False
                    # self.logger.warn("not updating region because new transform gave inf/nans")
                    # self.region.create_ellipsoid(minvol=minvol)
                    # return updated

                if not nextTransformLayer.nclusters < 20:
                    filename = 'overclustered_%d.npz' % nextTransformLayer.nclusters
                    if self.log:
                        self.logger.info("Found a lot of clusters: %d (%d with >1 members)",
                                         nextTransformLayer.nclusters, (cluster_sizes > 1).sum())

                    if not os.path.exists(filename):
                        self.logger.info("A lot of clusters! writing debug output file '%s'", filename)
                        np.savez(filename,
                            u=nextregion.u, unormed=nextregion.unormed,
                            maxradiussq=nextregion.maxradiussq,
                            u0=self.region.u, unormed0=self.region.unormed,
                            maxradiussq0=self.region.maxradiussq)
                        np.savetxt('overclustered_u_%d.txt' % nextTransformLayer.nclusters, nextregion.u)
                    # assert nextTransformLayer.nclusters < 20, nextTransformLayer.nclusters

                # if self.log:
                #     self.logger.info("computing maxradius...")
                r, f = nextregion.compute_enlargement(minvol=minvol,
                    nbootstraps=max(1, nbootstraps // self.mpi_size))
                # rng=np.random.RandomState(self.mpi_rank))
                # print("MLFriends built. r=%f" % r**0.5)
                if self.use_mpi:
                    recv_maxradii = self.comm.gather(r, root=0)
                    recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                    r = np.max(recv_maxradii)
                    recv_enlarge = self.comm.gather(f, root=0)
                    recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
                    f = np.max(recv_enlarge)

                nextregion.maxradiussq = r
                nextregion.enlarge = f
                # verify correctness:
                nextregion.create_ellipsoid(minvol=minvol)
                assert (nextregion.u == active_u).all()
                assert np.allclose(nextregion.unormed, nextregion.transformLayer.transform(active_u))
                # assert nextregion.inside(active_u).all(), ("live points should live in new region, but only %.3f%% do." % (100 * nextregion.inside(active_u).mean()), active_u)
                good_region = nextregion.inside(active_u).all()
                # assert good_region
                if not good_region:
                    self.logger.warning("constructed region is inconsistent (maxr=%f,enlarge=%f)", r, f)
                    np.savez('inconsistent_region.npz',
                        u=nextregion.u, unormed=nextregion.unormed,
                        maxradiussq=nextregion.maxradiussq,
                        u0=self.region.u, unormed0=self.region.unormed,
                        maxradiussq0=self.region.maxradiussq)
                    np.savetxt('inconsistent_region.txt', nextregion.u)

                # good_region = good_region and region.transformLayer.nclusters * 5 < len(active_u)

                # if self.log:
                #     self.logger.debug("building new region ... r=%e, f=%e" % (r, f))
                # print("MLFriends computed: r=%e nc=%d" % (r, nextTransformLayer.nclusters))
                # force shrinkage of volume
                # this is to avoid re-connection of dying out nodes
                if good_region and (need_accept or nextregion.estimate_volume() <= self.region.estimate_volume()):
                    self.region = nextregion
                    self.transformLayer = self.region.transformLayer
                    self.region_nodes = active_node_ids.copy()
                    # if self.log:
                    #     self.logger.debug("region updated: V=%e R=%e" % (self.region.estimate_volume(), r))
                    updated = True

                    assert not (self.transformLayer.clusterids == 0).any(), (self.transformLayer.clusterids, need_accept, updated)

            except Warning as w:
                if self.log:
                    self.logger.warning("not updating region", exc_info=True)
            except FloatingPointError as e:
                if self.log:
                    self.logger.warning("not updating region", exc_info=True)
            except np.linalg.LinAlgError as e:
                if self.log:
                    self.logger.warning("not updating region", exc_info=True)

        self.region.create_ellipsoid(minvol=minvol)
        assert len(self.region.u) == len(self.transformLayer.clusterids)
        return updated

    def _expand_nodes_before(self, Lmin, nnodes_needed, update_interval_ncall):
        """Expand nodes before `Lmin` to have `nnodes_needed`.

        Returns
        --------
        Llo: float
            lowest parent sampled (-np.inf if sampling from root)
        Lhi: float
            Lmin
        target_min_num_children: int
            number of children that need to be maintained between Llo, Lhi

        """
        self.pointstore.reset()
        parents, weights = find_nodes_before(self.root, Lmin)
        target_min_num_children = self._widen_nodes(parents, weights, nnodes_needed, update_interval_ncall)
        if len(parents) == 0:
            Llo = -np.inf
        else:
            Llo = min(n.value for n in parents)
        Lhi = Lmin
        return Llo, Lhi, target_min_num_children

    def _should_node_be_expanded(self, it,
        Llo, Lhi, minimal_widths_sequence, target_min_num_children,
        node, parallel_values, max_ncalls, max_iters
    ):
        """Check if node needs new children.

        Returns
        -------
        expand_node: bool
            True if should sample a new point
            based on this node (above its likelihood value Lmin).

        Parameters
        ----------
        it: int
            current iteration
        node: node
            The node to consider
        parallel_values: array of floats
            loglikelihoods of live points
        max_ncalls: int
            maximum number of likelihood function calls allowed
        max_iters: int
            maximum number of nested sampling iteration allowed
        Llo, Lhi, minimal_widths_sequence, target_min_num_children:
            Current strategy parameters

        """
        Lmin = node.value
        nlive = len(parallel_values)
        expand_node = False
        if Lmin <= Lhi and Llo <= Lhi:
            # we should continue to progress towards Lhi
            while Lmin > minimal_widths_sequence[0][0]:
                minimal_widths_sequence.pop(0)

            # get currently desired width
            if self.region is None:
                minimal_width_clusters = 0
            else:
                # compute number of clusters with more than 1 element
                _, cluster_sizes = np.unique(self.region.transformLayer.clusterids, return_counts=True)
                nclusters = (cluster_sizes > 1).sum()
                minimal_width_clusters = self.cluster_num_live_points * nclusters

            minimal_width = max(minimal_widths_sequence[0][1], minimal_width_clusters)

            # if already has children, no need to expand
            # if we are wider than the width required
            # we do not need to expand this one
            # expand_node = len(node.children) == 0
            # prefer 1 child, or the number required, if specified
            expand_node = len(node.children) < target_min_num_children.get(node.id, 1)

            # some exceptions:
            if it > 0:
                too_wide = nlive > minimal_width
                # we have to expand the first iteration,
                # otherwise the integrator never sets H

                if too_wide:
                    # print("not expanding, because we are quite wide", nlive, minimal_width, minimal_widths_sequence)
                    expand_node = False

                if max_ncalls is not None and self.ncall >= max_ncalls:
                    # print("not expanding, because above max_ncall")
                    expand_node = False

                if max_iters is not None and it >= max_iters:
                    # print("not expanding, because above max_iters")
                    expand_node = False

        return expand_node

    def run(
            self,
            update_interval_iter_fraction=0.2,
            update_interval_ncall=None,
            log_interval=None,
            show_status=True,
            viz_callback='auto',
            dlogz=0.5,
            dKL=0.5,
            frac_remain=0.01,
            min_ess=400,
            max_iters=None,
            max_ncalls=None,
            max_num_improvement_loops=-1,
            min_num_live_points=400,
            cluster_num_live_points=40,
    ):
        """Run until target convergence criteria are fulfilled.

        Parameters
        ----------
        update_interval_iter_fraction: float
            Update region after (update_interval_iter_fraction*nlive) iterations.

        update_interval_ncall: int
            Update region after update_interval_ncall likelihood calls (not used).

        log_interval: int
            Update stdout status line every log_interval iterations

        show_status: bool
            show integration progress as a status line.
            If no output desired, set to False.

        viz_callback: function
            callback function when region was rebuilt. Allows to
            show current state of the live points. 
            See :func:`nicelogger` or :class:`LivePointsWidget`.
            If no output desired, set to False.

        dlogz: float
            Target evidence uncertainty. This is the std
            between bootstrapped logz integrators.

        dKL: float
            Target posterior uncertainty. This is the
            Kullback-Leibler divergence in nat between bootstrapped integrators.

        frac_remain: float
            Integrate until this fraction of the integral is left in the remainder.
            Set to a low number (1e-2 ... 1e-5) to make sure peaks are discovered.
            Set to a higher number (0.5) if you know the posterior is simple.

        min_ess: int
            Target number of effective posterior samples.

        max_iters: int
            maximum number of integration iterations.

        max_ncalls: int
            stop after this many likelihood evaluations.

        max_num_improvement_loops: int
            run() tries to assess iteratively where more samples are needed.
            This number limits the number of improvement loops.

        min_num_live_points: int
            minimum number of live points throughout the run

        cluster_num_live_points: int
            require at least this many live points per detected cluster

        """
        # if viz_callback == 'auto':
        #    viz_callback = get_default_viz_callback()

        for result in self.run_iter(
            update_interval_iter_fraction=update_interval_iter_fraction,
            update_interval_ncall=update_interval_ncall,
            log_interval=log_interval,
            dlogz=dlogz, dKL=dKL, frac_remain=frac_remain,
            min_ess=min_ess, max_iters=max_iters,
            max_ncalls=max_ncalls, max_num_improvement_loops=max_num_improvement_loops,
            min_num_live_points=min_num_live_points,
            cluster_num_live_points=cluster_num_live_points,
            show_status=show_status,
            viz_callback=viz_callback,
        ):
            if self.log:
                self.logger.debug("did a run_iter pass!")
            pass
        if self.log:
            self.logger.info("done iterating.")

        return self.results

    def run_iter(
            self,
            update_interval_iter_fraction=0.2,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=0.5,
            dKL=0.5,
            frac_remain=0.01,
            min_ess=400,
            max_iters=None,
            max_ncalls=None,
            max_num_improvement_loops=-1,
            min_num_live_points=400,
            cluster_num_live_points=40,
            show_status=True,
            viz_callback='auto',
    ):
        """Iterate towards convergence.

        Use as an iterator like so::

            for result in sampler.run_iter(...):
                print('lnZ = %(logz).2f +- %(logzerr).2f' % result)

        Parameters as described in run() function.
        """
        # frac_remain=1  means 1:1 -> dlogz=log(0.5)
        # frac_remain=0.1 means 1:10 -> dlogz=log(0.1)
        # dlogz_min = log(1./(1 + frac_remain))
        # dlogz_min = -log1p(frac_remain)
        if -np.log1p(frac_remain) > dlogz:
            raise ValueError("To achieve the desired logz accuracy, set frac_remain to a value much smaller than %s (currently: %s)" % (
                exp(-dlogz) - 1, frac_remain))

        # if self.log:
        #     self.logger.info('Using MPI with rank [%d]' % (self.mpi_rank))
        if self.log_to_disk:
            self.logger.info("PointStore: have %d items", len(self.pointstore.stack))
            self.use_point_stack = not self.pointstore.stack_empty
        else:
            self.use_point_stack = False

        assert min_num_live_points >= cluster_num_live_points, ('min_num_live_points(%d) cannot be less than cluster_num_live_points(%d)' % (min_num_live_points, cluster_num_live_points))
        self.min_num_live_points = min_num_live_points
        self.cluster_num_live_points = cluster_num_live_points
        self.sampling_slow_warned = False

        if viz_callback == 'auto':
            viz_callback = get_default_viz_callback()

        self._widen_roots(min_num_live_points)

        Llo, Lhi = -np.inf, np.inf
        Lmax = -np.inf
        strategy_stale = True
        minimal_widths = []
        target_min_num_children = {}
        improvement_it = 0

        assert max_iters is None or max_iters > 0, ("Invalid value for max_iters: %s. Set to None or positive number" % max_iters)
        assert max_ncalls is None or max_ncalls > 0, ("Invalid value for max_ncalls: %s. Set to None or positive number" % max_ncalls)

        self.results = None

        while True:
            roots = self.root.children

            nroots = len(roots)

            if update_interval_ncall is None:
                update_interval_ncall = nroots

            if log_interval is None:
                log_interval = max(1, round(0.1 * nroots))
            else:
                log_interval = round(log_interval)
                if log_interval < 1:
                    raise ValueError("log_interval must be >= 1")

            explorer = BreadthFirstIterator(roots)
            # Integrating thing
            main_iterator = MultiCounter(nroots=len(roots),
                nbootstraps=max(1, self.num_bootstraps // self.mpi_size),
                random=False)
            main_iterator.Lmax = max(Lmax, max(n.value for n in roots))

            self.transformLayer = None
            self.region = None
            it_at_first_region = 0
            self.ib = 0
            self.samples = []
            if self.draw_multiple:
                ndraw = 100
            else:
                ndraw = 40
            self.pointstore.reset()
            if self.log_to_disk:
                self.use_point_stack = not self.pointstore.stack_empty
            else:
                self.use_point_stack = False
            if self.use_mpi:
                self.use_point_stack = self.comm.bcast(self.use_point_stack, root=0)

            if self.log and np.isfinite(Llo) or np.isfinite(Lhi):
                self.logger.info("Exploring (in particular: L=%.2f..%.2f) ...", Llo, Lhi)
            # print_tree(roots[:5], title="Tree:")
            region_sequence = []
            minimal_widths_sequence = _sequentialize_width_sequence(minimal_widths, self.min_num_live_points)
            if self.log:
                self.logger.debug('minimal_widths_sequence: %s', minimal_widths_sequence)

            saved_nodeids = []
            saved_logl = []
            it = 0
            ncall_at_run_start = self.ncall
            ncall_region_at_run_start = self.ncall_region
            # next_update_interval_ncall = -1
            next_update_interval_iter = -1
            last_status = time.time()

            # we go through each live point (regardless of root) by likelihood value
            while True:
                next_node = explorer.next_node()
                if next_node is None:
                    break
                rootid, node, (active_nodes, active_rootids, active_values, active_node_ids) = next_node
                assert not isinstance(rootid, float)
                # this is the likelihood level we have to improve upon
                Lmin = node.value

                # if within suggested range, expand
                if strategy_stale or not (Lmin <= Lhi) or not np.isfinite(Lhi) or (active_values == Lmin).all():
                    # check with advisor if we want to expand this node
                    Llo_prev, Lhi_prev = Llo, Lhi
                    Llo, Lhi = self._adaptive_strategy_advice(Lmin, active_values, main_iterator, minimal_widths, frac_remain)
                    if np.isfinite(Lhi):
                        strategy_altered = Llo != Llo_prev or Lhi != Lhi_prev
                    else:
                        strategy_altered = np.isfinite(Llo_prev) != np.isfinite(Llo) or np.isfinite(Lhi_prev) != np.isfinite(Lhi)

                    if self.log and strategy_altered:
                        self.logger.debug("strategy update: L range to expand: %.3f-%.3f have: %.2f logZ=%.2f logZremain=%.2f",
                                          Llo, Lhi, Lmin, main_iterator.logZ, main_iterator.logZremain)

                    # when we are going to the peak, numerical accuracy
                    # can become an issue. We should try not to get stuck there
                    strategy_stale = Lhi - Llo < 0.01

                expand_node = self._should_node_be_expanded(it, Llo, Lhi, minimal_widths_sequence, target_min_num_children, node, active_values, max_ncalls, max_iters)

                region_fresh = False
                if expand_node:
                    # sample a new point above Lmin
                    active_u = self.pointpile.getu(active_node_ids)
                    nlive = len(active_u)
                    # first we check that the region is up-to-date
                    if it > next_update_interval_iter:
                        if self.region is None:
                            it_at_first_region = it
                        region_fresh = self._update_region(
                            active_u=active_u, active_node_ids=active_node_ids,
                            active_rootids=active_rootids,
                            bootstrap_rootids=main_iterator.rootids[1:,],
                            nbootstraps=self.num_bootstraps,
                            minvol=exp(main_iterator.logVolremaining))

                        _, cluster_sizes = np.unique(self.region.transformLayer.clusterids, return_counts=True)
                        nclusters = (cluster_sizes > 1).sum()
                        region_sequence.append((Lmin, nlive, nclusters))

                        # next_update_interval_ncall = self.ncall + (update_interval_ncall or nlive)
                        update_interval_iter = max(1, round(update_interval_iter_fraction * nlive))
                        next_update_interval_iter = it + update_interval_iter

                        # provide nice output to follow what is going on
                        # but skip if we are resuming
                        #  and (self.ncall != ncall_at_run_start and it_at_first_region == it)
                        if self.log and viz_callback:
                            active_p = self.pointpile.getp(active_node_ids)
                            viz_callback(points=dict(u=active_u, p=active_p, logl=active_values),
                                info=dict(it=it, ncall=self.ncall,
                                logz=main_iterator.logZ, logz_remain=main_iterator.logZremain,
                                logvol=main_iterator.logVolremaining,
                                paramnames=self.paramnames + self.derivedparamnames,
                                paramlims=self.transform_limits,
                                ),
                                region=self.region, transformLayer=self.transformLayer,
                                region_fresh=region_fresh)
                            self.pointstore.flush()

                    if nlive < cluster_num_live_points * nclusters and improvement_it < max_num_improvement_loops:
                        # make wider here
                        if self.log:
                            self.logger.info("Found %d clusters, but only have %d live points, want %d.",
                                             self.region.transformLayer.nclusters, nlive,
                                             cluster_num_live_points * nclusters)
                        break

                    # sample point
                    u, p, L = self._create_point(Lmin=Lmin, ndraw=ndraw, active_u=active_u, active_values=active_values)
                    child = self.pointpile.make_node(L, u, p)
                    main_iterator.Lmax = max(main_iterator.Lmax, L)

                    # identify which point is being replaced (from when we built the region)
                    worst = np.where(self.region_nodes == node.id)[0]
                    self.region_nodes[worst] = child.id
                    # if we keep the region informed about the new live points
                    # then the region follows the live points even if maxradius is not updated
                    self.region.u[worst] = u
                    self.region.unormed[worst] = self.region.transformLayer.transform(u)

                    # if we track the cluster assignment, then in the next round
                    # the ids with the same members are likely to have the same id
                    # this is imperfect
                    # transformLayer.clusterids[worst] = transformLayer.clusterids[father[ib]]
                    # so we just mark the replaced ones as "unassigned"
                    self.transformLayer.clusterids[worst] = 0

                    # if self.log:
                    #     self.logger.debug("replacing node", Lmin, "from", rootid, "with", L)
                    node.children.append(child)

                    if self.log and (region_fresh or it % log_interval == 0 or time.time() > last_status + 0.1):
                        # nicelogger(self.paramnames, active_u, active_v, active_logl, it, ncall, logz, logz_remain, region=region)
                        last_status = time.time()
                        ncall_here = self.ncall - ncall_at_run_start
                        it_here = it - it_at_first_region
                        if show_status:
                            if Lmin < -1e8:
                                txt = 'Z=%.1g(%.2f%%) | Like=%.2g..%.2g [%.4g..%.4g]%s| it/evals=%d/%d eff=%.4f%% N=%d \r'
                            else:
                                txt = 'Z=%.1f(%.2f%%) | Like=%.2f..%.2f [%.4f..%.4f]%s| it/evals=%d/%d eff=%.4f%% N=%d \r'
                            sys.stdout.write(txt % (
                                             main_iterator.logZ, 100 * (1 - main_iterator.remainder_fraction),
                                             Lmin, main_iterator.Lmax, Llo, Lhi, '*' if strategy_stale else ' ', it, self.ncall,
                                             np.inf if ncall_here == 0 else it_here * 100 / ncall_here,
                                             nlive))
                            sys.stdout.flush()

                        # if efficiency becomes low, bulk-process larger arrays
                        if self.draw_multiple:
                            # inefficiency is the number of (region) proposals per successful number of iterations
                            # but improves by parallelism (because we need only the per-process inefficiency)
                            # sampling_inefficiency = (self.ncall - ncall_at_run_start + 1) / (it + 1) / self.mpi_size
                            sampling_inefficiency = (self.ncall_region - ncall_region_at_run_start + 1) / (it + 1) / self.mpi_size
                            # (self.ncall - ncall_at_run_start + 1) (self.ncall_region - self.ncall_region_at_run_start) / self.
                            ndraw = max(128, min(16384, round(sampling_inefficiency)))

                else:
                    # we do not want to count iterations without work
                    # otherwise efficiency becomes > 1
                    it_at_first_region += 1

                saved_nodeids.append(node.id)
                saved_logl.append(Lmin)

                # inform iterators (if it is their business) about the arc
                main_iterator.passing_node(rootid, node, active_rootids, active_values)
                if len(node.children) == 0 and self.region is not None:
                    # the region radius needs to increase if nlive decreases
                    # radius is not reliable, so set to inf
                    # (heuristics do not work in practice)
                    self.region.maxradiussq = None
                    # ask for the region to be rebuilt
                    next_update_interval_iter = -1

                it += 1
                explorer.expand_children_of(rootid, node)

            if self.log:
                self.logger.info("Explored until L=%.1g  ", node.value)
            # print_tree(roots[::10])

            self._update_results(main_iterator, saved_logl, saved_nodeids)
            yield self.results

            if max_ncalls is not None and self.ncall >= max_ncalls:
                if self.log:
                    self.logger.info('Reached maximum number of likelihood calls (%d > %d)...',
                                     self.ncall, max_ncalls)
                break

            improvement_it += 1
            if max_num_improvement_loops >= 0 and improvement_it > max_num_improvement_loops:
                if self.log:
                    self.logger.info('Reached maximum number of improvement loops.')
                break

            if ncall_at_run_start == self.ncall and improvement_it > 1:
                if self.log:
                    self.logger.info('No changes made. Probably the strategy was to explore in the remainder, but it is irrelevant already; try decreasing frac_remain.')
                break

            Lmax = main_iterator.Lmax
            if len(region_sequence) > 0:
                Lmin, nlive, nclusters = region_sequence[-1]
                nnodes_needed = cluster_num_live_points * nclusters
                if nlive < nnodes_needed:
                    Llo, Lhi, target_min_num_children_new = self._expand_nodes_before(Lmin, nnodes_needed, update_interval_ncall or nlive)
                    target_min_num_children.update(target_min_num_children_new)
                    # if self.log:
                    #     print_tree(self.root.children[::10])
                    minimal_widths.append((Llo, Lmin, nnodes_needed))
                    Llo, Lhi = -np.inf, np.inf
                    continue

            if self.log:
                # self.logger.info('  logZ = %.4f +- %.4f (main)' % (main_iterator.logZ, main_iterator.logZerr))
                self.logger.info('  logZ = %.4g +- %.4g', main_iterator.logZ_bs, main_iterator.logZerr_bs)

            saved_logl = np.asarray(saved_logl)
            (Llo_Z, Lhi_Z), (Llo_KL, Lhi_KL), (Llo_ess, Lhi_ess) = self._find_strategy(saved_logl, main_iterator, dlogz=dlogz, dKL=dKL, min_ess=min_ess)
            Llo = min(Llo_ess, Llo_KL, Llo_Z)
            Lhi = max(Lhi_ess, Lhi_KL, Lhi_Z)
            # to avoid numerical issues when all likelihood values are the same
            Lhi = min(Lhi, saved_logl.max() - 0.001)

            if self.use_mpi:
                recv_Llo = self.comm.gather(Llo, root=0)
                recv_Llo = self.comm.bcast(recv_Llo, root=0)
                recv_Lhi = self.comm.gather(Lhi, root=0)
                recv_Lhi = self.comm.bcast(recv_Lhi, root=0)
                Llo = min(recv_Llo)
                Lhi = max(recv_Lhi)

            if Llo <= Lhi:
                # if self.log:
                #     print_tree(roots, title="Tree before forking:")
                parents, parent_weights = find_nodes_before(self.root, Llo)
                _, width = count_tree_between(self.root.children, Llo, Lhi)
                nnodes_needed = width * 2
                if self.log:
                    self.logger.info('Widening from %d to %d live points before L=%.1g...',
                                     len(parents), nnodes_needed, Llo)

                if len(parents) == 0:
                    Llo = -np.inf
                else:
                    Llo = min(n.value for n in parents)
                self.pointstore.reset()
                target_min_num_children.update(self._widen_nodes(parents, parent_weights, nnodes_needed, update_interval_ncall))
                minimal_widths.append((Llo, Lhi, nnodes_needed))
                # if self.log:
                #     print_tree(roots, title="Tree after forking:")
                # print('tree size:', count_tree(roots))
            else:
                break

    def _update_results(self, main_iterator, saved_logl, saved_nodeids):
        # print_tree(roots[0:5])
        if self.log:
            self.logger.info('Likelihood function evaluations: %d', self.ncall)
            # self.logger.info('Tree size: height=%d width=%d' % count_tree(self.root.children))

        # points with weights
        # saved_u = np.array([pp[nodeid].u for nodeid in saved_nodeids])
        assert np.shape(main_iterator.logweights) == (len(saved_logl), len(main_iterator.all_logZ)), (
            np.shape(main_iterator.logweights),
            np.shape(saved_logl),
            np.shape(main_iterator.all_logZ))

        saved_logl = np.array(saved_logl)
        saved_u = self.pointpile.getu(saved_nodeids)
        saved_v = self.pointpile.getp(saved_nodeids)
        saved_logwt = np.array(main_iterator.logweights)
        saved_logwt0 = saved_logwt[:,0]
        saved_logwt_bs = saved_logwt[:,1:]
        logZ_bs = main_iterator.all_logZ[1:]
        assert len(saved_logwt_bs) == len(saved_nodeids), (saved_logwt_bs.shape, len(saved_nodeids))

        if self.use_mpi:
            # spread logZ_bs, saved_logwt_bs
            recv_saved_logwt_bs = self.comm.gather(saved_logwt_bs, root=0)
            recv_saved_logwt_bs = self.comm.bcast(recv_saved_logwt_bs, root=0)
            saved_logwt_bs = np.concatenate(recv_saved_logwt_bs, axis=1)

            recv_logZ_bs = self.comm.gather(logZ_bs, root=0)
            recv_logZ_bs = self.comm.bcast(recv_logZ_bs, root=0)
            logZ_bs = np.concatenate(recv_logZ_bs)

        saved_wt_bs = exp(saved_logwt_bs + saved_logl.reshape((-1, 1)) - logZ_bs)
        saved_wt0 = exp(saved_logwt0 + saved_logl - main_iterator.all_logZ[0])

        # compute fraction in tail
        w = saved_wt0 / saved_wt0.sum()
        ess = len(w) / (1.0 + ((len(w) * w - 1)**2).sum() / len(w))
        tail_fraction = w[np.asarray(main_iterator.istail)].sum()
        if tail_fraction != 0:
            logzerr_tail = logaddexp(log(tail_fraction) + main_iterator.logZ, main_iterator.logZ) - main_iterator.logZ
        else:
            logzerr_tail = 0

        logzerr_bs = (logZ_bs - main_iterator.logZ).max()
        logzerr_total = (logzerr_tail**2 + logzerr_bs**2)**0.5
        samples = resample_equal(saved_v, w)
        
        results = dict(niter=len(saved_logl),
            logz=main_iterator.logZ, logzerr=logzerr_total,
            logz_bs=logZ_bs.mean(),
            logzerr_bs=logzerr_bs,
            logz_single=main_iterator.logZ,
            logzerr_tail=logzerr_tail,
            logzerr_single=(main_iterator.all_H[0] / self.min_num_live_points)**0.5,
            ess=ess,
            paramnames=self.paramnames + self.derivedparamnames,
            ncall=int(self.ncall),
            posterior=dict(
                means=samples.mean(axis=0).tolist(),
                stds=samples.std(axis=0).tolist(),
                median=np.percentile(samples, 0.5, axis=0).tolist(),
                errlo=np.percentile(samples, 0.158655, axis=0).tolist(),
                errup=np.percentile(samples, 0.841345, axis=0).tolist(),
            ),
        )
        
        if self.log_to_disk:
            if self.log:
                self.logger.info("Writing samples and results to disk ...")
            np.savetxt(os.path.join(self.logs['chains'], 'equal_weighted_post.txt'),
                samples,
                header=' '.join(self.paramnames + self.derivedparamnames),
                comments='')
            np.savetxt(os.path.join(self.logs['chains'], 'weighted_post.txt'),
                np.hstack((saved_wt0.reshape((-1, 1)), -saved_logl.reshape((-1, 1)), saved_v)),
                comments='#')
            np.savetxt(os.path.join(self.logs['chains'], 'weighted_post_untransformed.txt'),
                np.hstack((saved_wt0.reshape((-1, 1)), saved_logl.reshape((-1, 1)), saved_u)),
                comments='')
            with open(os.path.join(self.logs['chains'], 'weighted_post.paramnames'), 'w') as f:
                f.write('\n'.join(self.paramnames + self.derivedparamnames) + '\n')
            with open(os.path.join(self.logs['info'], 'results.json'), 'w') as f:
                json.dump(results, f)
            with open(os.path.join(self.logs['info'], 'post_summary.csv'), 'w') as f:
                np.savetxt(f, 
                    [np.hstack([results['posterior'][k] for k in ('means', 'stds', 'median', 'errlo', 'errup')])],
                    header=', '.join(['"{0}_mean", "{0}_std", "{0}_median", "{0}_errlo", "{0}_errup"'.format(k) 
                        for k in self.paramnames + self.derivedparamnames]),
                    delimiter=',', comments='',
                    )

            if self.log:
                self.logger.info("Writing samples and results to disk ... done")

        results.update(
            weighted_samples=dict(v=saved_v, w=saved_wt0, logw=saved_logwt0,
                bs_w=saved_wt_bs, L=saved_logl),
            samples=samples,
        )
        self.results = results

    def store_tree(self):
        """Store tree to disk (results/tree.hdf5)."""
        if self.log_to_disk:
            dump_tree(os.path.join(self.logs['results'], 'tree.hdf5'),
                self.root.children, self.pointpile)

    def print_results(self, logZ=True, posterior=True):
        """Give summary of marginal likelihood and parameters."""
        if self.log:
            print()
            print('logZ = %(logz).3f +- %(logzerr).3f' % self.results)
            print('  single instance: logZ = %(logz_single).3f +- %(logzerr_single).3f' % self.results)
            print('  bootstrapped   : logZ = %(logz_bs).3f +- %(logzerr_bs).3f' % self.results)
            print('  tail           : logZ = +- %(logzerr_tail).3f' % self.results)

            print()
            for i, p in enumerate(self.paramnames + self.derivedparamnames):
                v = self.results['samples'][:,i]
                sigma = v.std()
                med = v.mean()
                if sigma == 0:
                    i = 3
                else:
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-20s' + fmt + " +- " + fmt])
                print(fmts % (p, med, sigma))

    def plot(self):
        """Make corner, run and trace plots."""
        self.plot_corner()
        self.plot_run()
        self.plot_trace()

    def plot_corner(self):
        """Make corner plot.

        Writes corner plot to plots/ directory if log directory was
        specified, otherwise show interactively.

        This does essentially::

            from ultranest.plot import cornerplot
            cornerplot(results)

        """
        from .plot import cornerplot
        import matplotlib.pyplot as plt
        if self.log:
            self.logger.debug('Making corner plot ...')
        cornerplot(self.results, logger=self.logger if self.log else None)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.logs['plots'], 'corner.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.debug('Making corner plot ... done')

    def plot_trace(self):
        """Make trace plot.

        Write parameter trace diagnostic plots to plots/ directory
        if log directory specified, otherwise show interactively.

        This does essentially::

            from ultranest.plot import traceplot
            traceplot(results=results, labels=paramnames + derivedparamnames)

        """
        from .plot import traceplot
        import matplotlib.pyplot as plt
        if self.log:
            self.logger.debug('Making trace plot ... ')
        paramnames = self.paramnames + self.derivedparamnames
        # get dynesty-compatible sequences
        results = logz_sequence(self.root, self.pointpile)
        traceplot(results=results, labels=paramnames)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.logs['plots'], 'trace.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.debug('Making trace plot ... done')

    def plot_run(self):
        """Make run plot.

        Write run diagnostic plots to plots/ directory
        if log directory specified, otherwise show interactively.

        This does essentially::

            from ultranest.plot import runplot
            runplot(results=results)

        """
        from .plot import runplot
        import matplotlib.pyplot as plt
        if self.log:
            self.logger.debug('Making run plot ... ')
        # get dynesty-compatible sequences
        results = logz_sequence(self.root, self.pointpile)
        runplot(results=results, logplot=True)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.logs['plots'], 'run.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.debug('Making run plot ... done')
