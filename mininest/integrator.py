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
from numpy import log, exp

from .utils import create_logger, make_run_dir
from .utils import acceptance_rate, effective_sample_size, mean_jump_distance, resample_equal
from mininest.mlfriends import MLFriends, AffineLayer, ScalingLayer, compute_maxradiussq
from .store import TextPointStore, HDF5PointStore, NullPointStore
from .viz import nicelogger
from .netiter import PointPile, MultiCounter, BreadthFirstIterator, TreeNode, print_tree, count_tree


import numpy as np

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
        if wrapped_params is None:
            self.wrapped_axes = []
        else:
            self.wrapped_axes = np.where(wrapped_params)[0]

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

        self.loglike = safe_loglike

        if transform is None:
            self.transform = lambda x: x
        else:
            def safe_transform(x):
                x = np.asarray(x)
                if len(x.shape) == 1:
                    assert x.shape[0] == self.x_dim
                    x = np.expand_dims(x, 0)
                return transform(x)
            self.transform = safe_transform

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
        
        self.logger = create_logger(__name__ + '.' + type(self).__name__)

        if self.log:
            self.logger.info('Num live points [%d]' % (self.num_live_points))

        if self.log_to_disk:
            with open(os.path.join(self.logs['results'], 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'acceptance', 'min_ess',
                                 'max_ess', 'jump_distance', 'scale', 'loglstar', 'logz', 'fraction_remain', 'ncall'])
        
        if self.log_to_disk:
            #self.pointstore = TextPointStore(os.path.join(self.logs['results'], 'points.tsv'), 2 + self.x_dim + self.num_params)
            self.pointstore = HDF5PointStore(os.path.join(self.logs['results'], 'points.hdf5'), 2 + self.x_dim + self.num_params)
        else:
            self.pointstore = NullPointStore(2 + self.x_dim + self.num_params)

    def _save_params(self, my_dict):
        my_dict = {k: str(v) for k, v in my_dict.items()}
        with open(os.path.join(self.logs['info'], 'params.txt'), 'w') as f:
            json.dump(my_dict, f, indent=4)

    def _chain_stats(self, samples, mean=None, std=None):
        acceptance = acceptance_rate(samples)
        if mean is None:
            mean = np.mean(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        if std is None:
            std = np.std(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        ess = effective_sample_size(samples, mean, std)
        jump_distance = mean_jump_distance(samples)
        self.logger.info(
            'Acceptance [%5.4f] min ESS [%5.4f] max ESS [%5.4f] jump distance [%5.4f]' %
            (acceptance, np.min(ess), np.max(ess), jump_distance))
        return acceptance, ess, jump_distance

    def _save_samples(self, samples, weights, logl, min_weight=1e-30, outfile='chain'):
        if len(samples.shape) == 2:
            with open(os.path.join(self.logs['chains'], outfile + '.txt'), 'w') as f:
                for i in range(samples.shape[0]):
                    f.write("%.5E " % max(weights[i], min_weight))
                    f.write("%.5E " % -logl[i])
                    f.write(" ".join(["%.5E" % v for v in samples[i, :]]))
                    f.write("\n")
        elif len(samples.shape) == 3:
            for ib in range(samples.shape[0]):
                with open(os.path.join(self.logs['chains'], outfile + '_%s.txt' % (ib+1)), 'w') as f:
                    for i in range(samples.shape[1]):
                        f.write("%.5E " % max(weights[ib, i], min_weight))
                        f.write("%.5E " % -logl[ib, i])
                        f.write(" ".join(["%.5E" % v for v in samples[ib, i, :]]))
                        f.write("\n")

    def run(
            self,
            update_interval_iter=None,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=0.5,
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
        logvol = np.log(1.0 - np.exp(-1.0 / self.num_live_points))
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
            h = (np.exp(logwt - logz_new) * active_logl[worst] + np.exp(logz - logz_new) * (h + logz) - logz_new)
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
                    #print("rebuilding space...")
                    nextTransformLayer = transformLayer.create_new(active_u, region.maxradiussq)
                    nextregion = MLFriends(active_u, nextTransformLayer)
                
                #print("computing maxradius...")
                r = nextregion.compute_maxradiussq(nbootstraps=30 // self.mpi_size)
                #print("MLFriends built. r=%f" % r**0.5)
                if self.use_mpi:
                    recv_maxradii = self.comm.gather(r, root=0)
                    recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                    r = np.max(recv_maxradii)
                
                nextregion.maxradiussq = r
                # force shrinkage of volume
                # this is to avoid re-connection of dying out nodes
                if nextregion.estimate_volume() < region.estimate_volume():
                    region = nextregion
                    transformLayer = region.transformLayer
                
                if self.log:
                    nicelogger(points=dict(u=active_u, p=active_v, logl=active_logl), 
                        info=dict(it=it, ncall=ncall, logz=logz, logz_remain=logz_remain, 
                        paramnames=self.paramnames + self.derivedparamnames), 
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
                sys.stdout.write('Z=%.1f+%.1f | Like=%.1f..%.1f | it/evals=%d/%d = %.4f%%  \r' % (
                      logz, logz_remain, loglstar,np.max(active_logl), it, ncall, np.inf if ncall == 0 else it * 100 / ncall))
                sys.stdout.flush()
                
                # if efficiency becomes low, bulk-process larger arrays
                ndraw = max(128, min(16384, round((ncall+1) / (it + 1) / self.mpi_size)))
            
            # Stopping criterion
            if fraction_remain < dlogz:
                break
            it = it + 1

        logvol = -len(saved_v) / self.num_live_points - np.log(self.num_live_points)
        for i in range(self.num_live_points):
            logwt = logvol + active_logl[i]
            logz_new = np.logaddexp(logz, logwt)
            h = (np.exp(logwt - logz_new) * active_logl[i] + np.exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new
            saved_u.append(np.array(active_u[i]))
            saved_v.append(np.array(active_v[i]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[i])
        
        saved_u = np.array(saved_u)
        saved_v = np.array(saved_v)
        saved_wt = np.exp(np.array(saved_logwt) - logz)
        saved_logl = np.array(saved_logl)
        logzerr = np.sqrt(h / self.num_live_points)

        if self.log_to_disk:
            with open(os.path.join(self.logs['results'], 'final.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['niter', 'ncall', 'logz', 'logzerr', 'h'])
                writer.writerow([it + 1, ncall, logz, logzerr, h])
            self._save_samples(saved_v, saved_wt, saved_logl)
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
                 cluster_num_live_points=100,
                 ):

        self.paramnames = param_names
        x_dim = len(self.paramnames)
        self.min_num_live_points = min_num_live_points
        self.sampler = 'nested'
        self.x_dim = x_dim
        self.derivedparamnames = derived_param_names
        num_derived = len(self.derivedparamnames)
        self.num_params = x_dim + num_derived
        if wrapped_params is None:
            self.wrapped_axes = []
        else:
            self.wrapped_axes = np.where(wrapped_params)[0]

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

        self.loglike = safe_loglike

        if transform is None:
            self.transform = lambda x: x
        else:
            def safe_transform(x):
                x = np.asarray(x)
                if len(x.shape) == 1:
                    assert x.shape[0] == self.x_dim
                    x = np.expand_dims(x, 0)
                return transform(x)
            self.transform = safe_transform

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
        
        self.logger = create_logger(__name__ + '.' + type(self).__name__)
        self.root = TreeNode()

        if self.log_to_disk:
            with open(os.path.join(self.logs['results'], 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'acceptance', 'min_ess',
                                 'max_ess', 'jump_distance', 'scale', 'loglstar', 'logz', 'fraction_remain', 'ncall'])
        
        self.pointpile = PointPile(self.x_dim, self.num_params)
        self.ncall = 0
        if self.log_to_disk:
            #self.pointstore = TextPointStore(os.path.join(self.logs['results'], 'points.tsv'), 2 + self.x_dim + self.num_params)
            self.pointstore = HDF5PointStore(os.path.join(self.logs['results'], 'points.hdf5'), 2 + self.x_dim + self.num_params)
            self.ncall = len(self.pointstore.stack)
        else:
            self.pointstore = NullPointStore(2 + self.x_dim + self.num_params)

    def _save_params(self, my_dict):
        my_dict = {k: str(v) for k, v in my_dict.items()}
        with open(os.path.join(self.logs['info'], 'params.txt'), 'w') as f:
            json.dump(my_dict, f, indent=4)

    def _chain_stats(self, samples, mean=None, std=None):
        acceptance = acceptance_rate(samples)
        if mean is None:
            mean = np.mean(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        if std is None:
            std = np.std(np.reshape(samples, (-1, samples.shape[2])), axis=0)
        ess = effective_sample_size(samples, mean, std)
        jump_distance = mean_jump_distance(samples)
        self.logger.info(
            'Acceptance [%5.4f] min ESS [%5.4f] max ESS [%5.4f] jump distance [%5.4f]' %
            (acceptance, np.min(ess), np.max(ess), jump_distance))
        return acceptance, ess, jump_distance

    def _save_samples(self, samples, weights, logl, min_weight=1e-30, outfile='chain'):
        if len(samples.shape) == 2:
            with open(os.path.join(self.logs['chains'], outfile + '.txt'), 'w') as f:
                for i in range(samples.shape[0]):
                    f.write("%.5E " % max(weights[i], min_weight))
                    f.write("%.5E " % -logl[i])
                    f.write(" ".join(["%.5E" % v for v in samples[i, :]]))
                    f.write("\n")
        elif len(samples.shape) == 3:
            for ib in range(samples.shape[0]):
                with open(os.path.join(self.logs['chains'], outfile + '_%s.txt' % (ib+1)), 'w') as f:
                    for i in range(samples.shape[1]):
                        f.write("%.5E " % max(weights[ib, i], min_weight))
                        f.write("%.5E " % -logl[ib, i])
                        f.write(" ".join(["%.5E" % v for v in samples[ib, i, :]]))
                        f.write("\n")

    def widen_roots(self, nroots):
        if self.log:
            self.logger.info('Sampling %d live points from prior ...' % (nroots))

        nnewroots = nroots - len(self.root.children)
        prev_u = []
        prev_v = []
        prev_logl = []
        prev_rowid = []

        if self.log:
            # try to resume:
            self.logger.info('Resuming...')
            for i in range(nnewroots):
                rowid, row = self.pointstore.pop(-np.inf)
                if row is not None:
                    prev_logl.append(row[1])
                    prev_u.append(row[2:2+self.x_dim])
                    prev_v.append(row[2+self.x_dim:2+self.x_dim+self.num_params])
                    prev_rowid.append(rowid)
                else:
                    break
            
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
                    self.ncall += num_live_points_missing
                else:
                    chunks = None
                data = self.comm.scatter(chunks, root=0)
                active_logl = self.loglike(data)
                recv_active_logl = self.comm.gather(active_logl, root=0)
                recv_active_logl = self.comm.bcast(recv_active_logl, root=0)
                self.ncall = self.comm.bcast(self.ncall, root=0)
                active_logl = np.concatenate(recv_active_logl, axis=0)
            else:
                self.ncall += num_live_points_missing
                active_logl = self.loglike(active_v)
        
            if self.log_to_disk:
                for i in range(num_live_points_missing):
                    rowid = self.pointstore.add([-np.inf, active_logl[i]] + active_u[i,:].tolist() + active_v[i,:].tolist())
            
            if len(prev_u) > 0:
                active_u = np.concatenate((prev_u, active_u))
                active_v = np.concatenate((prev_v, active_v))
                active_logl = np.concatenate((prev_logl, active_logl))
            assert active_u.shape ==(nroots, self.x_dim)
            assert active_v.shape ==(nroots, self.num_params)
            assert active_logl.shape ==(nroots,)
        else:
            active_u = prev_u
            active_v = prev_v
            active_logl = prev_logl
        
        roots = [self.pointpile.make_node(logl, p, u) for u, p, logl in zip(active_u, active_v, active_logl)]
        self.root.children = roots


    def adaptive_strategy_advice(self, node, parallel_values, main_iterator, counting_iterators, rootid):
        Ls = parallel_values
        #Ls = [node.value] + [n.value for rootid2, n in parallel_nodes]
        Lmax = Ls.max()
        Lmin = Ls.min() #main_iterator.logZ - weight
        weight = main_iterator.logVolremaining
        # max remainder contribution is Lmax + weight, to be added to main_iterator.logZ
        # the likelihood that would add an equal amount as main_iterator.logZ is:
        
        # if the remainder dominates, return that range
        if main_iterator.logZremain > main_iterator.logZ:
            return Lmin, Lmax
        
        #print("not expanding, remainder not dominant")
        return np.nan, np.nan
    
    def create_point(self, it, Lmin, ndraw):
        # draw a new point!
        
        while True:
            ib = self.ib
            if ib >= len(self.samples) and self.use_point_stack:
                # root checks the point store
                next_point = np.zeros((1, 2 + self.x_dim + self.num_params))
                
                if self.log_to_disk:
                    _, stored_point = self.pointstore.pop(Lmin)
                    if stored_point is not None:
                        next_point[0,:] = stored_point
                    else:
                        next_point[0,:] = -np.inf
                    use_point_stack = not self.pointstore.stack_empty
                
                if self.use_mpi: # and informs everyone
                    self.use_point_stack = self.comm.bcast(use_point_stack, root=0)
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
                u, father = self.region.sample(nsamples=ndraw)
                nu = u.shape[0]
                
                v = self.transform(u)
                logl = self.loglike(v)
                nc += nu
                accepted = logl > Lmin
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
                
                if self.log:
                    for ui, vi, logli in zip(self.samples, self.samplesv, self.likes):
                        self.pointstore.add([Lmin, logli] + ui.tolist() + vi.tolist())
            
            if self.likes[ib] > Lmin:
                u = self.samples[ib, :]
                p = self.samplesv[ib, :]
                logl = self.likes[ib]
                
                self.ib = ib + 1
                return u, p, logl
            else:
                self.ib = ib + 1
        
    def update_region(self, Lmin, active_u, active_nodes, active_rootids, active_node_ids, bootstrap_rootids):
        if self.region is None:
            self.transformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
            self.transformLayer.optimize(active_u, active_u)
            self.region = MLFriends(active_u, self.transformLayer)
            self.region_nodes = active_node_ids.copy()
            nextregion = self.region
        else:
            # rebuild space
            #print()
            #print("rebuilding space...")
            nextTransformLayer = self.transformLayer.create_new(active_u, self.region.maxradiussq)
            nextregion = MLFriends(active_u, nextTransformLayer)
        
        #if self.log:
        #    self.logger.info("computing maxradius...")
        #r = nextregion.compute_maxradiussq(nbootstraps=30 // self.mpi_size)
        r = 0.
        for selected in bootstrap_rootids[:,active_rootids]:
            a = nextregion.unormed[selected,:]
            b = nextregion.unormed[~selected,:]
            r = max(r, compute_maxradiussq(a, b))
        
        #print("MLFriends built. r=%f" % r**0.5)
        if self.use_mpi:
            recv_maxradii = self.comm.gather(r, root=0)
            recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
            r = np.max(recv_maxradii)
        
        nextregion.maxradiussq = r
        # force shrinkage of volume
        # this is to avoid re-connection of dying out nodes
        if nextregion.estimate_volume() < self.region.estimate_volume():
            self.region = nextregion
            self.transformLayer = self.region.transformLayer
            self.region_nodes = active_node_ids.copy()
        
    def run(
            self,
            update_interval_iter=None,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=0.5,
            dKL=0.5,
            max_iters=None,
            volume_switch=0):

        self.use_point_stack = True
        
        self.widen_roots(self.min_num_live_points)

        Llo, Lhi = -np.inf, np.inf
        strategy_stale = True
        
        while True:
            roots = self.root.children
            
            nroots = len(roots)

            if update_interval_ncall is None:
                update_interval_ncall = nroots
            
            if update_interval_iter is None:
                if update_interval_ncall == 0:
                    update_interval_iter = max(1, round(nroots))
                else:
                    update_interval_iter = max(1, round(0.2 * nroots))
            

            if log_interval is None:
                log_interval = max(1, round(0.2 * nroots))
            else:
                log_interval = round(log_interval)
                if log_interval < 1:
                    raise ValueError("log_interval must be >= 1")
            
            
            explorer = BreadthFirstIterator(roots)
            # Integrating thing
            main_iterator = MultiCounter(nroots=len(roots), nbootstraps=max(1, 30 // self.mpi_size), random=False)
            main_iterator.Lmax = max(n.value for n in roots)
            
            self.transformLayer = None
            self.region = None
            self.ib = 0
            self.samples = []
            
            saved_nodeids = []
            saved_logl = []
            it = 0
            next_update_interval_ncall = -1
            next_update_interval_iter = -1
            # we go through each live point (regardless of root) by likelihood value
            while True:
                next = explorer.next_node()
                if next is None:
                    break
                rootid, node, (active_nodes, active_rootids, active_values, active_node_ids) = next
                assert not isinstance(rootid, float)
                # this is the likelihood level we have to improve upon
                Lmin = node.value
                
                saved_nodeids.append(node.id)
                saved_logl.append(Lmin)
                
                expand_node = Lmin <= Lhi and Llo <= Lhi and len(node.children) == 0
                # if within suggested range, expand
                if strategy_stale or not (Lmin <= Lhi):
                    # check with advisor if we want to expand this node
                    Llo, Lhi = self.adaptive_strategy_advice(node, active_values, main_iterator, [], rootid)
                    #print("L range to expand: %.1f-%.1f" % (Llo, Lhi), "have:", Lmin, "=>", Lmin <= Lhi, Llo <= Lhi)
                    strategy_stale = False
                strategy_stale = True
                
                if expand_node:
                    # sample a new point above Lmin
                    active_u = self.pointpile.getu(active_node_ids)
                    if self.ncall > next_update_interval_ncall and it > next_update_interval_iter:
                        self.update_region(Lmin=Lmin, 
                            active_u=active_u, active_node_ids=active_node_ids, active_nodes=active_nodes, active_rootids=active_rootids, 
                            bootstrap_rootids=main_iterator.rootids[1:,])
                        
                        next_update_interval_ncall = self.ncall + update_interval_ncall
                        next_update_interval_iter = it + update_interval_iter
                        
                        if self.log:
                            active_p = self.pointpile.getp(active_node_ids)
                            nicelogger(points=dict(u=active_u, p=active_p, logl=active_values), 
                                info=dict(it=it, ncall=self.ncall, 
                                logz=main_iterator.logZ, logz_remain=main_iterator.logZremain, 
                                paramnames=self.paramnames + self.derivedparamnames), 
                                region=self.region, transformLayer=self.transformLayer)
                            self.pointstore.flush()
                    
                    u, p, L = self.create_point(it, Lmin, ndraw=100)
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
                    
                    if self.log: 
                        self.logger.debug("replacing node", Lmin, "from", rootid, "with", L)
                    node.children.append(child)
                else:
                    #print("ending node", Lmin)
                    pass
                
                # inform iterators (if it is their business) about the arc
                main_iterator.passing_node(rootid, node, active_rootids, active_values)
                it += 1
                explorer.expand_children_of(rootid, node)
            
            break # disable smart stuff for now
            
            
            print("Ranges according to posterior contribution")
            # compute KL divergence
            saved_logl = np.array(saved_logl)
            logw = np.asarray(main_iterator.logweights) + saved_logl.reshape((-1,1)) - main_iterator.all_logZ
            ref_logw = logw[:,0].reshape((-1,1))
            other_logw = logw[:,1:]
            KL = np.where(np.isfinite(other_logw), exp(other_logw) * (other_logw - ref_logw), 0)
            #print(KL.cumsum(axis=0)[-1,:])
            print('KL:', KL.sum(axis=0))
            p = np.where(KL > 0, KL, 0)
            p /= p.sum(axis=0).reshape((1, -1))
            Llo = np.inf
            Lhi = -np.inf

            for pi, KLi in zip(p.transpose(), KL.sum(axis=0)):
                if KLi > dKL:
                    samples = np.random.choice(len(ref_logw), p=pi, size=400)
                    Llo = min(Llo, saved_logl[samples].min())
                    Lhi = max(Lhi, saved_logl[samples].max())
            
            print("Ranges according to posterior contribution:", Llo, Lhi)
            
            print("Ranges according to lnZ contribution")
            # compute difference between lnZ cumsum
            p = exp(logw)
            p /= p.sum(axis=0).reshape((1, -1))
            deltalogZ = np.abs(main_iterator.all_logZ[1:] - main_iterator.logZ)
            for pi, deltalogZi in zip(p.transpose(), deltalogZ):
                if deltalogZi > dlogz:
                    samples = np.random.choice(len(ref_logw), p=pi, size=400)
                    Llo = min(Llo, saved_logl[samples].min())
                    Lhi = max(Lhi, saved_logl[samples].max())
            
            print("Ranges according to lnZ contribution", Llo, Lhi)
            
            # if still inf: measure lnZ contribution when number of live points decreases
            # if more than 0.001 of total lnZ, we want to integrate that away too
            
            if Llo <= Lhi:
                # fork off all roots at Llo
                #if self.log: 
                #    print_tree(roots, title="Tree before forking:")
                #fork_roots(create_point=create_point, pp=pp, pointstore=pointstore, roots=roots, Llo=Llo, Lhi=Lhi, verbose=verbose)
                #double_roots(create_point=create_point, pp=pp, pointstore=pointstore, roots=roots, Llo=Llo, Lhi=Lhi, verbose=verbose)
                
                # simply double roots
                self.widen_roots(len(self.root.children) * 2)
                #if self.log: 
                #    print_tree(roots, title="Tree after forking:")
                print('tree size:', count_tree(roots))
            else:
                break
            
        
        print('tree size:', count_tree(roots))
        # points with weights
        #saved_u = np.array([pp[nodeid].u for nodeid in saved_nodeids])
        saved_v = self.pointpile.getp(saved_nodeids)
        saved_logwt = np.array(main_iterator.logweights)
        saved_wt = np.exp(saved_logwt - main_iterator.logZ)
        saved_logl = np.array(saved_logl)
        print('%.4f +- %.4f (main)' % (main_iterator.logZ, main_iterator.logZerr))
        print('%.4f +- %.4f (bs)' % (main_iterator.all_logZ[1:].mean(), main_iterator.all_logZ[1:].std()))

        self.results = dict(niter=len(saved_logwt), 
            logz=main_iterator.logZ, logzerr=main_iterator.all_logZ[1:].std(),
            weighted_samples=dict(v=saved_v, w = saved_wt[:,0], logw = saved_logwt[:,0], bs_w = saved_wt, L=saved_logl),
            tree=TreeNode(-np.inf, children=roots),
        )
        return self.results


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























"""
        saved_u = []
        saved_v = []  # Stored points for posterior results
        saved_logl = []
        saved_logwt = []
        h = 0.0  # Information, initially 0.
        logz = -1e300  # ln(Evidence Z), initially Z=0
        logvol = np.log(1.0 - np.exp(-1.0 / self.num_live_points))
        logz_remain = np.max(active_logl)
        fraction_remain = 1.0
        ncall = num_live_points_missing  # number of calls we already made
        self.transformLayer = None
        self.retion = None
        
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
            h = (np.exp(logwt - logz_new) * active_logl[worst] + np.exp(logz - logz_new) * (h + logz) - logz_new)
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
                    #print("rebuilding space...")
                    nextTransformLayer = transformLayer.create_new(active_u, region.maxradiussq)
                    nextregion = MLFriends(active_u, nextTransformLayer)
                
                #print("computing maxradius...")
                r = nextregion.compute_maxradiussq(nbootstraps=30 // self.mpi_size)
                #print("MLFriends built. r=%f" % r**0.5)
                if self.use_mpi:
                    recv_maxradii = self.comm.gather(r, root=0)
                    recv_maxradii = self.comm.bcast(recv_maxradii, root=0)
                    r = np.max(recv_maxradii)
                
                nextregion.maxradiussq = r
                # force shrinkage of volume
                # this is to avoid re-connection of dying out nodes
                if nextregion.estimate_volume() < region.estimate_volume():
                    region = nextregion
                    transformLayer = region.transformLayer
                
                if self.log:
                    nicelogger(points=dict(u=active_u, p=active_v, logl=active_logl), 
                        info=dict(it=it, ncall=ncall, logz=logz, logz_remain=logz_remain, 
                        paramnames=self.paramnames + self.derivedparamnames), 
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
                sys.stdout.write('Z=%.1f+%.1f | Like=%.1f..%.1f | it/evals=%d/%d = %.4f%%  \r' % (
                      logz, logz_remain, loglstar,np.max(active_logl), it, ncall, np.inf if ncall == 0 else it * 100 / ncall))
                sys.stdout.flush()
                
                # if efficiency becomes low, bulk-process larger arrays
                ndraw = max(128, min(16384, round((ncall+1) / (it + 1) / self.mpi_size)))
            
            # Stopping criterion
            if fraction_remain < dlogz:
                break
            it = it + 1

        logvol = -len(saved_v) / self.num_live_points - np.log(self.num_live_points)
        for i in range(self.num_live_points):
            logwt = logvol + active_logl[i]
            logz_new = np.logaddexp(logz, logwt)
            h = (np.exp(logwt - logz_new) * active_logl[i] + np.exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new
            saved_u.append(np.array(active_u[i]))
            saved_v.append(np.array(active_v[i]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[i])
        
        saved_u = np.array(saved_u)
        saved_v = np.array(saved_v)
        saved_wt = np.exp(np.array(saved_logwt) - logz)
        saved_logl = np.array(saved_logl)
        logzerr = np.sqrt(h / self.num_live_points)

        if self.log_to_disk:
            with open(os.path.join(self.logs['results'], 'final.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['niter', 'ncall', 'logz', 'logzerr', 'h'])
                writer.writerow([it + 1, ncall, logz, logzerr, h])
            self._save_samples(saved_v, saved_wt, saved_logl)
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
    
"""
