"""
Performs nested sampling to calculate the Bayesian evidence and posterior samples
Some parts are from the Nestle library by Kyle Barbary (https://github.com/kbarbary/nestle)
Some parts are from the nnest library by Adam Moss (https://github.com/adammoss/nnest)
"""

from __future__ import print_function
from __future__ import division

import os
import sys
import csv
import json
import operator
import time
import logging
import warnings
from numpy import log, exp, logaddexp

from .utils import create_logger, make_run_dir, resample_equal, vol_prefactor
from mininest.mlfriends import MLFriends, AffineLayer, ScalingLayer, update_clusters, find_nearby
from .store import TextPointStore, HDF5PointStore, NullPointStore
from .viz import nicelogger
from .netiter import PointPile, MultiCounter, BreadthFirstIterator, TreeNode, print_tree, count_tree, count_tree_between, find_nodes_before, logz_sequence

import numpy as np

def get_cumsum_range(pi, dp):
    ci = pi.cumsum()
    ilo = np.where(ci > dp)[0]
    ilo = ilo[0] if len(ilo) > 0 else 0 
    ihi = np.where(ci < 1. - dp)[0]
    ihi = ihi[-1] if len(ihi) > 0 else -1
    return ilo, ihi

def sequentialize_width_sequence(minimal_widths, min_width):
    Lpoints = np.unique([-np.inf] + [L for L, _, _ in minimal_widths] + [L for _, L, _ in minimal_widths] + [np.inf])
    widths = np.ones(len(Lpoints)) * min_width
    
    for Llo, Lhi, width in minimal_widths:
        # all Lpoints within that range should be maximized to width
        #mask = np.logical_and(Lpoints >= Llo, Lpoints <= Lhi)
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

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 append_run_num=True,
                 wrapped_params=None,
                 derived_param_names=[],
                 run_num=None,
                 log_dir='logs/test',
                 num_live_points=1000
                 ):

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

        u = np.random.uniform(size=(2, self.x_dim))
        p = transform(u)
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

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
        except:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = self.mpi_rank == 0
        self.log_to_disk = self.log and log_dir is not None

        if self.log and log_dir is not None:
            self.logs = make_run_dir(log_dir, run_num, append_run_num= append_run_num)
            log_dir = self.logs['run_dir']
        else:
            log_dir = None
        
        self.logger = create_logger(__name__ + '.' + type(self).__name__, log_dir=log_dir)

        if self.log:
            self.logger.info('Num live points [%d]' % (self.num_live_points))

        if self.log_to_disk:
            #self.pointstore = TextPointStore(os.path.join(self.logs['results'], 'points.tsv'), 2 + self.x_dim + self.num_params)
            self.pointstore = HDF5PointStore(os.path.join(self.logs['results'], 'points.hdf5'), 2 + self.x_dim + self.num_params)
        else:
            self.pointstore = NullPointStore(2 + self.x_dim + self.num_params)

    def run(
            self,
            update_interval_iter=None,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=0.001,
            max_iters=None,
            volume_switch=0):

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
                    prev_u.append(row[2:2+self.x_dim])
                    prev_v.append(row[2+self.x_dim:2+self.x_dim+self.num_params])
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
                self.logger.info('Using MPI with rank [%d]' % (self.mpi_rank))
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
                    self.pointstore.add([-np.inf, active_logl[i]] + active_u[i,:].tolist() + active_v[i,:].tolist())
            
            if len(prev_u) > 0:
                active_u = np.concatenate((prev_u, active_u))
                active_v = np.concatenate((prev_v, active_v))
                active_logl = np.concatenate((prev_logl, active_logl))
            assert active_u.shape ==(self.num_live_points, self.x_dim)
            assert active_v.shape ==(self.num_live_points, self.num_params)
            assert active_logl.shape ==(self.num_live_points,)
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

            #expected_vol = np.exp(-it / self.num_live_points)

            # The new likelihood constraint is that of the worst object.
            loglstar = active_logl[worst]
            
            if ncall > next_update_interval_ncall and it > next_update_interval_iter:

                if first_time:
                    nextregion = region
                else:
                    # rebuild space
                    #print()
                    #print("rebuilding space...", active_u.shape, active_u)
                    nextTransformLayer = transformLayer.create_new(active_u, region.maxradiussq)
                    nextregion = MLFriends(active_u, nextTransformLayer)
                
                #print("computing maxradius...")
                r, f = nextregion.compute_enlargement(nbootstraps=max(1, 30 // self.mpi_size))
                #print("MLFriends built. r=%f" % r**0.5)
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
                    nicelogger(points=dict(u=active_u, p=active_v, logl=active_logl), 
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
                    next_point = np.zeros((1, 2 + self.x_dim + self.num_params))
                    
                    if self.log_to_disk:
                        _, stored_point = self.pointstore.pop(loglstar)
                        if stored_point is not None:
                            next_point[0,:] = stored_point
                        else:
                            next_point[0,:] = -np.inf
                        use_point_stack = not self.pointstore.stack_empty
                    
                    if self.use_mpi: # and informs everyone
                        use_point_stack = self.comm.bcast(use_point_stack, root=0)
                        next_point = self.comm.bcast(next_point, root=0)
                    
                    #assert not use_point_stack
                    
                    # unpack
                    likes = next_point[:,1]
                    samples = next_point[:,2:2+self.x_dim]
                    samplesv = next_point[:,2+self.x_dim:2+self.x_dim+self.num_params]
                    # skip if we already know it is not useful
                    ib = 0 if np.isfinite(likes[0]) else 1
                
                while ib >= len(samples):
                    # get new samples
                    ib = 0
                    
                    nc = 0
                    u, father = region.sample(nsamples=ndraw)
                    nu = u.shape[0]
                    
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
                            self.pointstore.add([loglstar, logli] + ui.tolist() + vi.tolist())
                
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
                    #transformLayer.clusterids[worst] = transformLayer.clusterids[father[ib]]
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
                #nicelogger(self.paramnames, active_u, active_v, active_logl, it, ncall, logz, logz_remain, region=region)
                sys.stdout.write('Z=%.1f+%.1f | Like=%.1f..%.1f | it/evals=%d/%d eff=%.4f%%  \r' % (
                      logz, logz_remain, loglstar, np.max(active_logl), it, ncall, np.inf if ncall == 0 else it * 100 / ncall))
                sys.stdout.flush()
                
                # if efficiency becomes low, bulk-process larger arrays
                ndraw = max(128, min(16384, round((ncall+1) / (it + 1) / self.mpi_size)))
            
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
            weighted_samples=dict(u=saved_u, v=saved_v, w = saved_wt, logw = saved_logwt, L=saved_logl),
        )
        
        return self.results

    def print_results(self, logZ=True, posterior=True):
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

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 append_run_num=True,
                 wrapped_params=None,
                 derived_param_names=[],
                 run_num=None,
                 log_dir='logs/test',
                 min_num_live_points=100,
                 cluster_num_live_points=40,
                 max_num_live_points_for_efficiency=400,
                 num_test_samples=2,
                 draw_multiple=True,
                 num_bootstraps=30,
                 viz_callback=nicelogger,
                 show_status=True,
                 ):
        """
        
        param_names: list of str, names of the parameters. 
            Length gives dimensionality of the sampling problem.
        
        loglike: log-likelihood function
        transform: parameter transform from unit cube to physical 
            parameters.
        
        derived_param_names: list of str
            Additional derived parameters created by transform. (empty by default)
        
        log_dir: where to store output files
        append_run_num: if true, create a fresh subdirectory in log_dir
        
        wrapped_params: list of bools, indicating whether this parameter 
            wraps around in a circular parameter space.
        
        min_num_live_points: number of live points
        cluster_num_live_points: require at least this many live points per cluster
        max_num_live_points_for_efficiency: 
            Increasing the number of live points can make the region
            more accurate, increasing performance. If efficiency is low,
            the number of live points is allowed to grow to this value.
        
        num_test_samples: test transform and likelihood with this number of
            random points for errors first. Useful to catch bugs.
        
        draw_multiple: draw more points if efficiency goes down. 
            If set to False, few points are sampled at once.
        
        num_bootstraps: number of logZ estimators and MLFriends region 
            bootstrap rounds.
            
        viz_callback: callback function when region was rebuilt. Allows to 
            show current state of the live points. See nicelogger() function.
            If no output desired, set to False.
        
        show_status: show integration progress as a status line. 
            If no output desired, set to False.
        
        """

        self.paramnames = param_names
        x_dim = len(self.paramnames)
        self.min_num_live_points = min_num_live_points
        self.cluster_num_live_points = cluster_num_live_points
        self.max_num_live_points_for_efficiency = max_num_live_points_for_efficiency
        assert min_num_live_points >= cluster_num_live_points, ('min_num_live_points(%d) cannot be less than cluster_num_live_points(%d)' % (min_num_live_points, cluster_num_live_points))
        self.draw_multiple = draw_multiple
        
        self.sampler = 'reactive-nested'
        self.x_dim = x_dim
        self.derivedparamnames = derived_param_names
        self.num_bootstraps=int(num_bootstraps)
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
        except:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = self.mpi_rank == 0
        self.log_to_disk = self.log and log_dir is not None

        if self.log and log_dir is not None:
            self.logs = make_run_dir(log_dir, run_num, append_run_num= append_run_num)
            log_dir = self.logs['run_dir']
        else:
            log_dir = None
        
        #self.logger = create_logger(__name__ + '.' + type(self).__name__)
        if self.log:
            self.logger = create_logger('mininest', log_dir=log_dir)
        self.root = TreeNode(id=-1, value=-np.inf)

        if self.log_to_disk:
            with open(os.path.join(self.logs['results'], 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'acceptance', 'min_ess',
                                 'max_ess', 'jump_distance', 'scale', 'loglstar', 'logz', 'fraction_remain', 'ncall'])
        
        self.pointpile = PointPile(self.x_dim, self.num_params)
        self.ncall = 0
        self.ncall_region = 0
        if self.log_to_disk:
            #self.pointstore = TextPointStore(os.path.join(self.logs['results'], 'points.tsv'), 2 + self.x_dim + self.num_params)
            self.pointstore = HDF5PointStore(os.path.join(self.logs['results'], 'points.hdf5'), 2 + self.x_dim + self.num_params)
            self.ncall = len(self.pointstore.stack)
        else:
            self.pointstore = NullPointStore(2 + self.x_dim + self.num_params)
        
        self.set_likelihood_function(transform, loglike, num_test_samples)
        self.viz_callback = viz_callback
        self.show_status = show_status
    
    def set_likelihood_function(self, transform, loglike, num_test_samples, make_safe=True):
        
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
            assert len(lastrow) == 2 + self.x_dim + self.num_params, ("Cannot resume: problem has different dimensionality", len(lastrow), (2, self.x_dim, self.num_params))
            lastL = lastrow[1]
            lastu = lastrow[2:2+self.x_dim]
            u = lastu.reshape((1, -1))
            lastp = lastrow[2+self.x_dim:2+self.x_dim+self.num_params]
            if self.log:
                self.logger.debug("Testing resume consistency: %s: u=%s -> p=%s -> L=%s " % (lastrow, lastu, lastp, lastL))
            p = transform(u) if transform is not None else u
            if not np.allclose(p.flatten(), lastp) and self.log:
                self.logger.warn("Trying to resume from previous run, but transform function gives different result: %s gave %s, now %s" % (lastu, lastp, p.flatten()))
            assert np.allclose(p.flatten(), lastp), "Cannot resume because transform function changed. To start from scratch, delete '%s'." % (self.logs['run_dir'])
            logl = loglike(p).flatten()[0]
            if not np.isclose(logl, lastL) and self.log:
                self.logger.warn("Trying to resume from previous run, but likelihood function gives different result: %s gave %s, now %s" % (lastu.flatten(), lastL, logl))
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

    def widen_nodes(self, parents, nnodes_needed, update_interval_ncall):
        """
        Make sure that at Labove, there are nnodes_needed live points
        (parallel arcs).
        Sample from an appropriate region
        """
        
        ndone = len(parents)
        if ndone == 0:
            if self.log:
                self.logger.info('No parents, so widening roots')
            self.widen_roots(nnodes_needed)
            return
        
        # sort from low to high
        parents.sort(key=operator.attrgetter('value'))
        Lmin = parents[0].value
        if np.isinf(Lmin):
            # some of the parents were born by sampling from the entire
            # prior volume. So we can efficiently apply a solution: 
            # expand the roots
            if self.log:
                self.logger.info('parent value is -inf, so widening roots')
            self.widen_roots(nnodes_needed)
            return
        
        if self.log:
            self.logger.info('Sampling %d live points at L=%.1f...' % (nnodes_needed - ndone, Lmin))
        
        initial_ncall = self.ncall
        next_update_interval_ncall = self.ncall
        # double until we reach the necessary points
        nsamples = int(np.ceil((nnodes_needed - ndone) / len(parents)))
        
        self.region = None
        for i, n in enumerate(parents):
            # create region if it does not exist
            # or update after some likelihood calls (unless only few nodes left)
            if self.region is None or (self.ncall > next_update_interval_ncall and len(parents) - i > 50):
                Lmin = n.value
                # sort so that the last ones go away first
                active_nodes = parents[i:][::-1]
                
                #if self.log:
                #    self.logger.info('Making region from %d parents at L=%.1f...' % (len(active_nodes), Lmin))
                active_node_ids = [n.id for n in active_nodes]
                active_u = self.pointpile.getu(active_node_ids)
                self.update_region(
                    active_u=active_u, active_node_ids=active_node_ids)
                active_u = self.pointpile.getu(active_node_ids)
            
                next_update_interval_ncall = self.ncall + update_interval_ncall
            
            for j in range(nsamples):
                u, p, L = self.create_point(Lmin=n.value, ndraw=100)
                child = self.pointpile.make_node(L, u, p)
                n.children.append(child)
                if self.log and self.show_status:
                    sys.stdout.write('%d/%d Region@%.1f Like=%.1f->%.1f | it/evals=%d/%d = %.4f%%  \r' % (
                          ndone, nnodes_needed, Lmin, n.value, L, 
                          i * nsamples + j + 1, self.ncall - initial_ncall,
                          np.inf if self.ncall == initial_ncall else (i * nsamples + j + 1) * 100. / (self.ncall - initial_ncall) ))
                    sys.stdout.flush()
                ndone += 1
    
    def widen_roots(self, nroots):
        """
        Make sure that the root has nroots children.
        Sample from prior to fill up.
        """

        if self.log:
            self.logger.info('Widening roots to %d live points (have %d already) ...' % (nroots, len(self.root.children)))
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
                prev_u.append(row[2:2+self.x_dim])
                prev_v.append(row[2+self.x_dim:2+self.x_dim+self.num_params])
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
            self.logger.info('Sampling %d live points from prior ...' % (num_live_points_missing))
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
                    rowid = self.pointstore.add([-np.inf, active_logl[i]] + active_u[i,:].tolist() + active_v[i,:].tolist())
            
            if len(prev_u) > 0:
                active_u = np.concatenate((prev_u, active_u))
                active_v = np.concatenate((prev_v, active_v))
                active_logl = np.concatenate((prev_logl, active_logl))
            assert active_u.shape == (nnewroots, self.x_dim), (active_u.shape, nnewroots, self.x_dim, num_live_points_missing, len(prev_u))
            assert active_v.shape == (nnewroots, self.num_params), (active_v.shape, nnewroots, self.num_params, num_live_points_missing, len(prev_u))
            assert active_logl.shape ==(nnewroots,), (active_logl.shape, nnewroots)
        else:
            active_u = prev_u
            active_v = prev_v
            active_logl = prev_logl
        
        roots = [self.pointpile.make_node(logl, u, p) for u, p, logl in zip(active_u, active_v, active_logl)]
        self.root.children += roots


    def adaptive_strategy_advice(self, Lmin, parallel_values, main_iterator, minimal_widths, frac_remain):
        Ls = parallel_values.copy()
        Ls.sort()
        #Ls = [node.value] + [n.value for rootid2, n in parallel_nodes]
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
    
    def find_strategy(self, saved_logl, main_iterator, dlogz, dKL, min_ess):
        saved_logl = np.asarray(saved_logl)
        logw = np.asarray(main_iterator.logweights) + saved_logl.reshape((-1,1)) - main_iterator.all_logZ
        ref_logw = logw[:,0].reshape((-1,1))
        other_logw = logw[:,1:]

        if self.log:
            self.logger.info("Effective samples strategy:")

        Llo_ess = np.inf
        Lhi_ess = -np.inf
        w = exp(ref_logw.flatten())
        w /= w.sum()
        ess = len(w) / (1.0 + ((len(w) * w - 1)**2).sum() / len(w))
        if self.log:
            self.logger.info('  ESS = %.1f' % (ess))
        if ess < min_ess:
            samples = np.random.choice(len(w), p=w, size=min_ess)
            if self.log:
                self.logger.info('  need to improve near %.2f..%.2f' % (
                    saved_logl[samples].min(), saved_logl[samples].max()))
            Llo_ess = saved_logl[samples].min()
            Lhi_ess = saved_logl[samples].max()
        if self.log and Lhi_ess > Llo_ess:
            self.logger.info("Effective samples strategy wants to improve: %.2f..%.2f" % (Llo_ess, Lhi_ess))
        elif self.log:
            self.logger.info("Effective samples strategy is happy")
        
        # compute KL divergence
        if self.log:
            self.logger.info("Posterior uncertainty strategy:")
        with np.errstate(invalid='ignore'):
            KL = np.where(np.isfinite(other_logw), exp(other_logw) * (other_logw - ref_logw), 0)
        #print(KL.cumsum(axis=0)[-1,:])
        KLtot = KL.sum(axis=0)
        dKLtot = np.abs(KLtot - KLtot.mean())
        if self.log:
            self.logger.info('  KL: %.2f +- %s (max:%.2f)' % (KLtot.mean(), np.percentile(dKLtot, [0.01, 0.1, 0.5, 0.9, 0.99]), dKLtot.max()))
        p = np.where(KL > 0, KL, 0)
        p /= p.sum(axis=0).reshape((1, -1))

        Llo_KL = np.inf
        Lhi_KL = -np.inf
        for i, (pi, dKLi, logwi) in enumerate(zip(p.transpose(), dKLtot, other_logw)):
            if dKLi > dKL:
                ilo, ihi = get_cumsum_range(pi, 1./400)
                # ilo and ihi are most likely missing in this iterator
                # --> select the one before/after in this iterator
                ilos = np.where(np.isfinite(logwi[:ilo]))[0]
                ihis = np.where(np.isfinite(logwi[ihi:]))[0]
                ilo2 = ilos[-1] if len(ilos) > 0 else 0
                ihi2 = (ihi + ihis[0]) if len(ihis) > 0 else -1
                self.logger.info('   - KL[%d] = %.2f: need to improve near %.2f..%.2f --> %.2f..%.2f' % (i, dKLi, saved_logl[ilo], saved_logl[ihi], saved_logl[ilo2], saved_logl[ihi2]))
                Llo_KL = min(Llo_KL, saved_logl[ilo2])
                Lhi_KL = max(Lhi_KL, saved_logl[ihi2])
        
        if self.log and Lhi_KL > Llo_KL:
            self.logger.info("Posterior uncertainty strategy wants to improve: %.2f..%.2f" % (Llo_KL, Lhi_KL))
        elif self.log:
            self.logger.info("Posterior uncertainty strategy is happy")
        
        if self.log:
            self.logger.info("Evidency uncertainty strategy:")
        Llo_Z = np.inf
        Lhi_Z = -np.inf
        # compute difference between lnZ cumsum
        p = exp(logw)
        p /= p.sum(axis=0).reshape((1, -1))
        deltalogZ = np.abs(main_iterator.all_logZ[1:] - main_iterator.logZ)
        if self.log:
            self.logger.info('  deltalogZ: %s (max:%.2f)' % (np.percentile(deltalogZ, [0.01, 0.1, 0.5, 0.9, 0.99]), deltalogZ.max()))
        
        tail_fraction = w[np.asarray(main_iterator.istail)].sum() / w.sum()
        logzerr_tail = logaddexp(log(tail_fraction) + main_iterator.logZ, main_iterator.logZ) - main_iterator.logZ
        if self.log:
            self.logger.info('  logZ error budget: single: %.2f bs:%.2f tail:%.2f total:%.2f required:<%.2f' % (
                main_iterator.logZerr, main_iterator.logZerr_bs, logzerr_tail, 
                (main_iterator.logZerr_bs**2 + logzerr_tail**2)**0.5, dlogz))
        if (deltalogZ > dlogz).any() and (main_iterator.logZerr_bs**2 + logzerr_tail**2)**0.5 > dlogz:
            for i, (pi, deltalogZi) in enumerate(zip(p.transpose(), deltalogZ)):
                if deltalogZi > dlogz:
                    # break up samples with too much weight
                    samples = np.random.choice(len(ref_logw), p=pi, size=400)
                    if self.log:
                        self.logger.info('   - deltalogZi[%d] = %.2f: need to improve near %.2f..%.2f' % (
                            i, deltalogZi, saved_logl[samples].min(), saved_logl[samples].max()))
                    Llo_Z = min(Llo_Z, saved_logl[samples].min())
                    Lhi_Z = max(Lhi_Z, saved_logl[samples].max())
        
        if self.log and Lhi_Z > Llo_Z:
            self.logger.info("Evidency uncertainty strategy wants to improve: %.2f..%.2f" % (Llo_Z, Lhi_Z))
        elif self.log:
            self.logger.info("Evidency uncertainty strategy is happy")
        return (Llo_Z, Lhi_Z), (Llo_KL, Lhi_KL), (Llo_ess, Lhi_ess)
    
    
    def create_point(self, Lmin, ndraw):
        """
        draw a new point above likelihood threshold Lmin
        
        :param Lmin: likelihood threshold to draw above
        :param ndraw: number of points to try to sample at once
        """
        nit = 0
        while True:
            ib = self.ib
            if ib >= len(self.samples) and self.use_point_stack:
                # root checks the point store
                next_point = np.zeros((1, 2 + self.x_dim + self.num_params)) * np.nan
                
                if self.log_to_disk:
                    _, stored_point = self.pointstore.pop(Lmin)
                    if stored_point is not None:
                        next_point[0,:] = stored_point
                    else:
                        next_point[0,:] = -np.inf
                    self.use_point_stack = not self.pointstore.stack_empty
                
                if self.use_mpi: # and informs everyone
                    self.use_point_stack = self.comm.bcast(self.use_point_stack, root=0)
                    next_point = self.comm.bcast(next_point, root=0)
                
                # unpack
                self.likes = next_point[:,1]
                self.samples = next_point[:,2:2+self.x_dim]
                self.samplesv = next_point[:,2+self.x_dim:2+self.x_dim+self.num_params]
                # skip if we already know it is not useful
                ib = 0 if np.isfinite(self.likes[0]) else 1
            
            while ib >= len(self.samples):
                # get new samples
                ib = 0
                
                nc = 0
                nit += 1
                u, father = self.region.sample(nsamples=ndraw)
                assert np.logical_and(u > 0, u < 1).all(), (u)
                nu = u.shape[0]
                if nu == 0:
                    v = np.empty((0, self.num_params))
                    logl = np.empty((0,))
                else:
                    v = self.transform(u)
                    logl = self.loglike(v)
                nc += nu
                accepted = logl > Lmin
                if nit >= 100000 / ndraw and nit % (100000 // ndraw) == 0:
                    #self.logger.warn("Sampling seems stuck. Writing debug output file 'sampling-stuck-it%d.npz'..." % nit)
                    np.savez('sampling-stuck-it%d.npz' % nit, u=self.region.u, unormed=self.region.unormed, maxradiussq=self.region.maxradiussq, 
                        sample_u=u, sample_v=v, sample_logl=logl)
                    warnings.warn("Sampling seems stuck, this could be numerical issue: You are probably trying to integrate to deep into the volume where all points become equal in logL; so cannot draw a higher point. Try loosening the quality constraints (increase frac_remain, dlogz, dKL, decrease min_ess). [%d/%d accepted, it=%d]" % (accepted.sum(), ndraw, nit))
                    logl_region = self.loglike(self.transform(self.region.u))
                    if not (logl_region > Lmin).any():
                        raise ValueError("Region cannot sample a point. Perhaps you are resuming from a different problem? Delete the output files and start again.")
                
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
                    self.samples = np.concatenate(recv_samples, axis=0)
                    self.samplesv = np.concatenate(recv_samplesv, axis=0)
                    self.father = np.concatenate(recv_father, axis=0)
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
                        self.pointstore.add([Lmin, logli] + ui.tolist() + vi.tolist())
            
            if self.likes[ib] > Lmin:
                u = self.samples[ib, :]
                assert np.logical_and(u > 0, u < 1).all(), (u)
                p = self.samplesv[ib, :]
                logl = self.likes[ib]

                self.ib = ib + 1
                return u, p, logl
            else:
                self.ib = ib + 1
    
    def update_region(self, active_u, active_node_ids, 
        bootstrap_rootids=None, active_rootids=None,
        nbootstraps=30, minvol=0.
    ):
        """
        Build a new MLFriends region from active_u
        """
        assert nbootstraps > 0
        updated = False
        if self.region is None:
            #print("building first region...")
            if self.x_dim > 1:
                self.transformLayer = AffineLayer(wrapped_dims=self.wrapped_axes)
            else:
                self.transformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
            self.transformLayer.optimize(active_u, active_u, minvol=minvol)
            self.region = MLFriends(active_u, self.transformLayer)
            self.region_nodes = active_node_ids.copy()
            assert self.region.maxradiussq is None
            r, f = self.region.compute_enlargement(nbootstraps=max(1, nbootstraps // self.mpi_size), minvol=minvol)
            #print("MLFriends built. r=%f" % r**0.5)
            if self.use_mpi:
                recv_maxradii = self.comm.gather(r, root=0)
                recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                r = np.max(recv_maxradii)
                recv_enlarge = self.comm.gather(f, root=0)
                recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
                f = np.max(recv_enlarge)
            
            self.region.maxradiussq = r
            self.region.enlarge = f
            #print("building first region... r=%e" % r)


        assert self.transformLayer is not None
        need_accept = False

        if self.region.maxradiussq is None:
            # we have been told that radius is currently invalid
            # we need to bootstrap back to a valid state
            
            # compute radius given current transformLayer
            oldu = self.region.u
            self.region.u = active_u
            self.region.set_transformLayer(self.transformLayer)
            r, f = self.region.compute_enlargement(nbootstraps=max(1, nbootstraps // self.mpi_size), minvol=minvol)
            #print("MLFriends built. r=%f" % r**0.5)
            if self.use_mpi:
                recv_maxradii = self.comm.gather(r, root=0)
                recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                r = np.max(recv_maxradii)
                recv_enlarge = self.comm.gather(f, root=0)
                recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
                f = np.max(recv_enlarge)
            
            self.region.maxradiussq = r
            self.region.enlarge = f
            
            #print("made first region, r=%e" % (r))
            
            # now that we have r, can do clustering 
            #self.transformLayer.nclusters, self.transformLayer.clusterids, _ = update_clusters(
            #    self.region.u, self.region.unormed, self.region.maxradiussq)
            # but such reclustering would forget the cluster ids
            
            # instead, track the clusters from before by matching manually
            oldt = self.transformLayer.transform(oldu)
            clusterids = np.zeros(len(active_u), dtype=int)
            nnearby = np.empty(len(self.region.unormed), dtype=int)
            for ci in np.unique(self.transformLayer.clusterids):
                if ci == 0: continue
                
                # find points from that cluster
                oldti = oldt[self.transformLayer.clusterids == ci]
                # identify which new points are near this cluster
                find_nearby(oldti, self.region.unormed, self.region.maxradiussq, nnearby)
                mask = nnearby != 0
                # assign the nearby ones to this cluster
                # if they have not been set yet
                # if they have, set them to -1
                clusterids[mask] = np.where(clusterids[mask] == 0, ci, -1)
        
            #print("following clusters, nc=%d" % r, self.transformLayer.nclusters, 
            #    np.unique(clusterids, return_counts=True))
            
            # clusters we are unsure about (double assignments) go unassigned
            clusterids[clusterids == -1] = 0
            
            # tell scaling layer the correct cluster information
            self.transformLayer.clusterids = clusterids
            
            # we want the clustering to repeat to remove remaining zeros
            need_accept = (self.transformLayer.clusterids == 0).any()
            
            updated = True
            assert len(self.region.u) == len(self.transformLayer.clusterids)


        assert len(self.region.u) == len(self.transformLayer.clusterids)
        # rebuild space
        #print()
        #print("rebuilding space...", active_u.shape, active_u)
        nextTransformLayer = self.transformLayer.create_new(active_u, self.region.maxradiussq, minvol=minvol)
        #nextTransformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
        #nextTransformLayer.optimize(active_u, active_u)
        assert not (nextTransformLayer.clusterids == 0).any()
        smallest_cluster = min((nextTransformLayer.clusterids == i).sum() for i in np.unique(nextTransformLayer.clusterids))
        if self.log and smallest_cluster == 1:
            self.logger.debug("clustering found some lonely points [need_accept=%s] %s" % (
                need_accept, np.unique(nextTransformLayer.clusterids, return_counts=True)))
        
        nextregion = MLFriends(active_u, nextTransformLayer)
        
        if not nextTransformLayer.nclusters < 20:
            filename = 'overclustered_%d.npz' % nextTransformLayer.nclusters
            if self.log:
                self.logger.info("Found a lot of clusters: %d" % nextTransformLayer.nclusters)
            
            if not os.path.exists(filename):
                self.logger.info("A lot of clusters! writing debug output file '%s'" % filename)
                np.savez(filename, 
                    u=nextregion.u, unormed=nextregion.unormed, 
                    maxradiussq=nextregion.maxradiussq,
                    u0=self.region.u, unormed0=self.region.unormed, 
                    maxradiussq0=self.region.maxradiussq)
                np.savetxt('overclustered_u_%d.txt' % nextTransformLayer.nclusters, nextregion.u)
            #assert nextTransformLayer.nclusters < 20, nextTransformLayer.nclusters
        
        #if self.log:
        #    self.logger.info("computing maxradius...")
        r, f = nextregion.compute_enlargement(nbootstraps=max(1, nbootstraps // self.mpi_size))
        #print("MLFriends built. r=%f" % r**0.5)
        if self.use_mpi:
            recv_maxradii = self.comm.gather(r, root=0)
            recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
            r = np.max(recv_maxradii)
            recv_enlarge = self.comm.gather(f, root=0)
            recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
            f = np.max(recv_enlarge)
        
        nextregion.maxradiussq = r
        nextregion.enlarge = f
        
        #print("MLFriends computed: r=%e nc=%d" % (r, nextTransformLayer.nclusters))
        # force shrinkage of volume
        # this is to avoid re-connection of dying out nodes
        if need_accept or nextregion.estimate_volume() <= self.region.estimate_volume():
            self.region = nextregion
            self.transformLayer = self.region.transformLayer
            self.region_nodes = active_node_ids.copy()
            #print("MLFriends updated: V=%e R=%e" % (self.region.estimate_volume(), r))
            updated = True
            
            assert not (self.transformLayer.clusterids == 0).any(), (self.transformLayer.clusterids, need_accept, updated)
        
        self.region.create_ellipsoid(minvol=minvol)
        assert len(self.region.u) == len(self.transformLayer.clusterids)
        return updated
    
    def expand_nodes_before(self, Lmin, nnodes_needed, update_interval_ncall):
        self.pointstore.reset()
        parents = find_nodes_before(self.root, Lmin)
        self.widen_nodes(parents, nnodes_needed, update_interval_ncall)
        if len(parents) == 0:
            Llo = -np.inf
        else:
            Llo = min(n.value for n in parents)
        Lhi = Lmin
        return Llo, Lhi
    
    def should_node_be_expanded(self, it, Llo, Lhi, minimal_widths_sequence, node, parallel_values, max_ncalls, max_iters):
        """
        Should we sample a new point based on this node (above its likelihood value Lmin)?
        
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
                minimal_width_clusters = self.cluster_num_live_points * self.region.transformLayer.nclusters
            
            minimal_width = max(minimal_widths_sequence[0][1], minimal_width_clusters)
            
            # if already has children, no need to expand
            # if we are wider than the width required
            # we do not need to expand this one
            expand_node = len(node.children) == 0 
            
            # some exceptions:
            if it > 0: 
                too_wide = nlive > minimal_width
                # exception for widening for efficiency
                if nlive <= self.max_num_live_points_for_efficiency:
                    too_wide = False
                
                # we have to expand the first iteration, 
                # otherwise the integrator never sets H
                
                if too_wide:
                    #print("not expanding, because we are quite wide", nlive, minimal_width, minimal_widths_sequence, self.max_num_live_points_for_efficiency)
                    expand_node = False
                
                if max_ncalls is not None and self.ncall >= max_ncalls:
                    #print("not expanding, because above max_ncall")
                    expand_node = False
                
                if max_iters is not None and it >= max_iters:
                    #print("not expanding, because above max_iters")
                    expand_node = False
        
        return expand_node

    def run(
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
            max_num_improvement_loops=-1):
        """
        Run until target convergence criteria are fulfilled:
        
        update_interval_iter_fraction: 
            Update region after (update_interval_iter_fraction*nlive) iterations.
        update_interval_ncall: (not actually used)
            Update region after update_interval_ncall likelihood calls.
        log_interval:
            Update stdout status line every log_interval iterations
        
        Termination criteria
        ---------------------
        
        dlogz: 
            Target evidence uncertainty. This is the std
            between bootstrapped logz integrators.
        
        dKL:
            Target posterior uncertainty. This is the 
            Kullback-Leibler divergence in nat between bootstrapped integrators.
        
        frac_remain:
            Integrate until this fraction of the integral is left in the remainder.
            Set to a low number (1e-2 ... 1e-5) to make sure peaks are discovered.
            Set to a higher number (0.5) if you know the posterior is simple.
        
        min_ess:
            Target number of effective posterior samples. 
        
        max_iters: maximum number of integration iterations.
        
        max_ncalls: stop after this many likelihood evaluations.
            
        max_num_improvement_loops: int
            run() tries to assess iteratively where more samples are needed.
            This number limits the number of improvement loops.
        
        """
        
        for result in self.run_iter(
            update_interval_iter_fraction=update_interval_iter_fraction,
            update_interval_ncall=update_interval_ncall,
            log_interval=log_interval,
            dlogz=dlogz, dKL=dKL, frac_remain=frac_remain,
            min_ess=min_ess, max_iters=max_iters,
            max_ncalls=max_ncalls, max_num_improvement_loops=max_num_improvement_loops):
            if self.log:
                self.logger.info("did a run_iter pass!")
            pass
        if self.log:
            self.logger.info("done running!")
        
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
            max_num_improvement_loops=-1):
        """
        Use as an iterator like so:
        
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
        
        #if self.log:
        #    self.logger.info('Using MPI with rank [%d]' % (self.mpi_rank))
        if self.log_to_disk:
            self.logger.info("PointStore: have %d items" % len(self.pointstore.stack))
            self.use_point_stack = not self.pointstore.stack_empty
        else:
            self.use_point_stack = False
        self.widen_roots(self.min_num_live_points)

        Llo, Lhi = -np.inf, np.inf
        Lmax = -np.inf
        strategy_stale = True
        minimal_widths = []
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

            if self.log:
                self.logger.info("Exploring (in particular: L=%.2f..%.2f) ..." % (Llo, Lhi))
            #print_tree(roots[:5], title="Tree:")
            region_sequence = []
            minimal_widths_sequence = sequentialize_width_sequence(minimal_widths, self.min_num_live_points)
            if self.log:
                self.logger.debug('minimal_widths_sequence: %s' % minimal_widths_sequence)
            
            saved_nodeids = []
            saved_logl = []
            it = 0
            ncall_at_run_start = self.ncall
            ncall_region_at_run_start = self.ncall_region
            #next_update_interval_ncall = -1
            next_update_interval_iter = -1
            last_status = time.time()
            
            # we go through each live point (regardless of root) by likelihood value
            while True:
                next = explorer.next_node()
                if next is None:
                    break
                rootid, node, (active_nodes, active_rootids, active_values, active_node_ids) = next
                assert not isinstance(rootid, float)
                # this is the likelihood level we have to improve upon
                Lmin = node.value
                
                # if within suggested range, expand
                if strategy_stale or not (Lmin <= Lhi) or not np.isfinite(Lhi):
                    # check with advisor if we want to expand this node
                    Llo_prev, Lhi_prev = Llo, Lhi
                    Llo, Lhi = self.adaptive_strategy_advice(Lmin, active_values, main_iterator, minimal_widths, frac_remain)
                    if np.isfinite(Lhi):
                        strategy_altered = Llo != Llo_prev or Lhi != Lhi_prev
                    else:
                        strategy_altered = np.isfinite(Llo_prev) != np.isfinite(Llo) or np.isfinite(Lhi_prev) != np.isfinite(Lhi)
                    
                    if self.log and strategy_altered:
                        self.logger.debug("strategy update: L range to expand: %.3f-%.3f have: %.2f logZ=%.2f logZremain=%.2f" % (
                            Llo, Lhi, Lmin, main_iterator.logZ, main_iterator.logZremain))
                    
                    # when we are going to the peak, numerical accuracy
                    # can become an issue. We should try not to get stuck there
                    strategy_stale = Lhi - Llo < 0.01
                
                expand_node = self.should_node_be_expanded(it, Llo, Lhi, minimal_widths_sequence, node, active_values, max_ncalls, max_iters)
                
                region_fresh = False
                if expand_node:
                    # sample a new point above Lmin
                    active_u = self.pointpile.getu(active_node_ids)
                    nlive = len(active_u)
                    # first we check that the region is up-to-date
                    if it > next_update_interval_iter:
                        if self.region is None:
                            it_at_first_region = it
                        region_fresh = self.update_region(
                            active_u=active_u, active_node_ids=active_node_ids, 
                            active_rootids=active_rootids, 
                            bootstrap_rootids=main_iterator.rootids[1:,],
                            nbootstraps=self.num_bootstraps, 
                            minvol=exp(-it / nlive) * self.volfactor)
                        
                        nclusters = self.transformLayer.nclusters
                        region_sequence.append((Lmin, nlive, nclusters))
                        
                        if nlive < self.cluster_num_live_points * nclusters:
                            # make wider here
                            if self.log:
                                self.logger.info("Found %d clusters, but only have %d live points, want %d." % (
                                    nclusters, nlive, self.cluster_num_live_points * nclusters))
                            break
                        
                        #next_update_interval_ncall = self.ncall + (update_interval_ncall or nlive)
                        update_interval_iter = max(1, round(update_interval_iter_fraction * nlive))
                        next_update_interval_iter = it + update_interval_iter
                        
                        # provide nice output to follow what is going on
                        # but skip if we are resuming
                        #  and (self.ncall != ncall_at_run_start and it_at_first_region == it)
                        if self.log and self.viz_callback:
                            active_p = self.pointpile.getp(active_node_ids)
                            self.viz_callback(points=dict(u=active_u, p=active_p, logl=active_values), 
                                info=dict(it=it, ncall=self.ncall, 
                                logz=main_iterator.logZ, logz_remain=main_iterator.logZremain, 
                                logvol=main_iterator.logVolremaining, 
                                paramnames=self.paramnames + self.derivedparamnames,
                                paramlims=self.transform_limits,
                                ),
                                region=self.region, transformLayer=self.transformLayer,
                                region_fresh=region_fresh)
                            self.pointstore.flush()
                    
                    for i in range(2):
                        # sample point
                        u, p, L = self.create_point(Lmin=Lmin, ndraw=ndraw)
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
                        #transformLayer.clusterids[worst] = transformLayer.clusterids[father[ib]]
                        # so we just mark the replaced ones as "unassigned"
                        self.transformLayer.clusterids[worst] = 0
                        
                        #if self.log: 
                        #    self.logger.debug("replacing node", Lmin, "from", rootid, "with", L)
                        node.children.append(child)
                        
                        if i == 0 and nlive < self.max_num_live_points_for_efficiency:
                            efficiency_here = (it - it_at_first_region) / (self.ncall - ncall_at_run_start + 1.)
                            if efficiency_here < 1. / (nlive + 1):
                                # running inefficiently; more live points could make sampling more efficient
                                if self.log:
                                    self.logger.debug("Running inefficiently with %d live points. Sampling another." % (nlive))
                                continue
                        break
                    
                    if self.log and (region_fresh or it % log_interval == 0 or time.time() > last_status + 0.1):
                        #nicelogger(self.paramnames, active_u, active_v, active_logl, it, ncall, logz, logz_remain, region=region)
                        last_status = time.time()
                        ncall_here = self.ncall - ncall_at_run_start
                        it_here = it - it_at_first_region
                        if self.show_status:
                            sys.stdout.write('Z=%.1f(%.2f%%) | Like=%.2f..%.2f | it/evals=%d/%d eff=%.4f%% N=%d ndraw=%d[%s]\r' % (
                                  main_iterator.logZ, 100 * (1 - main_iterator.remainder_fraction), 
                                  Lmin, main_iterator.Lmax, it, self.ncall, 
                                  np.inf if ncall_here == 0 else it_here * 100 / ncall_here, 
                                  nlive, ndraw, self.region.current_sampling_method.__name__[len('sample_from_')]))
                            sys.stdout.flush()
                        
                        # if efficiency becomes low, bulk-process larger arrays
                        if self.draw_multiple:
                            # inefficiency is the number of (region) proposals per successful number of iterations
                            # but improves by parallelism (because we need only the per-process inefficiency)
                            #sampling_inefficiency = (self.ncall - ncall_at_run_start + 1) / (it + 1) / self.mpi_size
                            sampling_inefficiency = (self.ncall_region - ncall_region_at_run_start + 1) / (it + 1) / self.mpi_size
                            #(self.ncall - ncall_at_run_start + 1) (self.ncall_region - self.ncall_region_at_run_start) / self.
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
                self.logger.info("Explored until L=%.1f  " % node.value)
            #print_tree(roots[::10])

            self._update_results(main_iterator, saved_logl, saved_nodeids)
            yield self.results

            if max_ncalls is not None and self.ncall >= max_ncalls:
                if self.log:
                    self.logger.info('Reached maximum number of likelihood calls (%d > %d)...' % (self.ncall, max_ncalls))
                break
            
            improvement_it += 1
            if max_num_improvement_loops >= 0 and improvement_it > max_num_improvement_loops:
                if self.log:
                    self.logger.info('Reached maximum number of improvement loops.')
                break
            
            if ncall_at_run_start == self.ncall:
                if self.log:
                    self.logger.info('No changes made. Probably the strategy was to explore in the remainder, but it is irrelevant already; try decreasing frac_remain.')
                break
            
            Lmax = main_iterator.Lmax
            if len(region_sequence) > 0:
                Lmin, nlive, nclusters = region_sequence[-1]
                nnodes_needed = self.cluster_num_live_points * nclusters
                if nlive < nnodes_needed:
                    #self.root.children = []
                    Llo, Lhi = self.expand_nodes_before(Lmin, nnodes_needed, update_interval_ncall or nlive)
                    #if self.log:
                    #    print_tree(self.root.children[::10])
                    minimal_widths.append((Llo, Lmin, nnodes_needed))
                    Llo, Lhi = -np.inf, np.inf
                    continue
            
            if self.log:
                #self.logger.info('  logZ = %.4f +- %.4f (main)' % (main_iterator.logZ, main_iterator.logZerr))
                self.logger.info('  logZ = %.4f +- %.4f' % (main_iterator.logZ_bs, main_iterator.logZerr_bs))
            
            saved_logl = np.asarray(saved_logl)
            (Llo_Z, Lhi_Z), (Llo_KL, Lhi_KL), (Llo_ess, Lhi_ess) = self.find_strategy(saved_logl, main_iterator, dlogz, dKL, min_ess)
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
                #if self.log: 
                #    print_tree(roots, title="Tree before forking:")
                parents = find_nodes_before(self.root, Llo)
                _, width = count_tree_between(self.root.children, Llo, Lhi)
                nnodes_needed = width * 2
                if self.log:
                    self.logger.info('Widening from %d to %d live points before L=%.1f...' % (len(parents), nnodes_needed, Llo))
                
                if len(parents) == 0:
                    Llo = -np.inf
                else:
                    Llo = min(n.value for n in parents)
                self.pointstore.reset()
                self.widen_nodes(parents, nnodes_needed, update_interval_ncall)
                minimal_widths.append((Llo, Lhi, nnodes_needed))
                #if self.log: 
                #    print_tree(roots, title="Tree after forking:")
                #print('tree size:', count_tree(roots))
            else:
                break
            
    def _update_results(self, main_iterator, saved_logl, saved_nodeids):
        #print_tree(roots[0:5])
        if self.log:
            self.logger.info('nevals: %d' % self.ncall)
            #self.logger.info('Tree size: height=%d width=%d' % count_tree(self.root.children))
        
        # points with weights
        #saved_u = np.array([pp[nodeid].u for nodeid in saved_nodeids])
        assert np.shape(main_iterator.logweights) == (len(saved_logl), len(main_iterator.all_logZ)), (
            np.shape(main_iterator.logweights), 
            np.shape(saved_logl), 
            np.shape(main_iterator.all_logZ))
        
        saved_logl = np.array(saved_logl)
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
        logzerr_tail = logaddexp(log(tail_fraction) + main_iterator.logZ, main_iterator.logZ) - main_iterator.logZ
        
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
        )
        if self.log_to_disk:
            if self.log:
                self.logger.info("Writing samples and results to disk ...")
            np.savetxt(os.path.join(self.logs['chains'], 'equal_weighted_post.txt'), 
                samples,
                header=' '.join(self.paramnames + self.derivedparamnames), 
                comments='')
            np.savetxt(os.path.join(self.logs['chains'], 'weighted_post.txt'), 
                np.hstack((saved_wt0.reshape((-1, 1)), saved_logl.reshape((-1, 1)), saved_v)), 
                header=' '.join(self.paramnames + self.derivedparamnames), 
                comments='')
            with open(os.path.join(self.logs['info'], 'parameters.txt'), 'w') as f:
                f.write('\n'.join(self.paramnames + self.derivedparamnames) + '\n')
            with open(os.path.join(self.logs['info'], 'results.json'), 'w') as f:
                json.dump(results, f)
            
            if self.log:
                self.logger.info("Writing samples and results to disk ... done")
        
        results.update(
            weighted_samples=dict(v=saved_v, w=saved_wt0, logw=saved_logwt0, 
                bs_w=saved_wt_bs, L=saved_logl),
            samples=samples,
        )
        self.results = results
                    
        
    
    def print_results(self, logZ=True, posterior=True):
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
        """
        Make corner, run and trace plots
        """
        self.plot_corner()
        self.plot_run()
    
    def plot_corner(self):
        """
        Write corner plot to plots/ directory.
        """
        if self.log_to_disk:
            self.logger.info('Making corner plot ...')
            from .plot import cornerplot
            import matplotlib.pyplot as plt
            cornerplot(self.results, logger=self.logger if self.log else None)
            plt.savefig(os.path.join(self.logs['plots'], 'corner.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.info('Making corner plot ... done')

    def plot_run(self):
        """
        Write run and parameter trace diagnostic plots to plots/ directory.
        """
        if self.log_to_disk:
            self.logger.info('Making run plot ... ')
            from .plot import runplot, traceplot
            import matplotlib.pyplot as plt
            # get dynesty-compatible sequences
            self.logger.debug("computing dynesty-compatible sequences")
            results = logz_sequence(self.root, self.pointpile)
            self.logger.debug("computing dynesty-compatible sequences done.")
            runplot(results = results, logplot=True)
            plt.savefig(os.path.join(self.logs['plots'], 'run.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.info('Making run plot ... done')

            self.logger.info('Making trace plot ... ')
            paramnames = self.paramnames + self.derivedparamnames
            traceplot(results = results, labels=paramnames)
            plt.savefig(os.path.join(self.logs['plots'], 'trace.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.info('Making trace plot ... done')
            self.logger.info('Plotting done')

