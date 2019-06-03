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
import shutil

from .utils import create_logger, make_run_dir
from .utils import acceptance_rate, effective_sample_size, mean_jump_distance, resample_equal
from mininest.mlfriends import MLFriends, AffineLayer, ScalingLayer
from .store import TextPointStore, HDF5PointStore, NullPointStore

from numpy import log10
import numpy as np
import scipy.stats
import string
clusteridstrings = ['%d' % i for i in range(10)] + list(string.ascii_uppercase)

def nicelogger(points, info, region, transformLayer):
    #u, p, logl = points['u'], points['p'], points['logl']
    p = points['p']
    paramnames = info['paramnames']
    #print()
    #print('lnZ = %.1f, remainder = %.1f, lnLike = %.1f | Efficiency: %d/%d = %.4f%%\r' % (
    #      logz, logz_remain, np.max(logl), ncall, it, it * 100 / ncall))
    
    plo = p.min(axis=0)
    phi = p.max(axis=0)
    expos = log10(np.abs([plo, phi]))
    expolo = np.floor(np.min(expos, axis=0))
    expohi = np.ceil(np.max(expos, axis=0))
    is_negative = plo < 0
    plo = np.where(is_negative, -10**expohi, 10**expolo)
    phi = np.where(is_negative,  10**expohi, 10**expohi)

    if sys.stderr.isatty() and hasattr(shutil, 'get_terminal_size'):
        columns, _rows = shutil.get_terminal_size(fallback=(80, 25))
    else:
        columns, _rows = 80, 25

    width = columns - 22 - max([len(pname) for pname in paramnames])
    indices = ((p - plo) * width / (phi - plo).reshape((1, -1))).astype(int)
    indices[indices >= width] = width - 1
    indices[indices < 0] = 0
    ndim = len(plo)
    
    print()
    clusterids = transformLayer.clusterids
    nmodes = transformLayer.nclusters
    print("Volume: %.2e" % region.estimate_volume())
    if nmodes == 1:
        print("Mono-modal")
    else: 
        print("Have %d modes" % nmodes)
    
    for i, param in enumerate(paramnames):
        if nmodes == 1:
            line = [' ' for i in range(width)]
            for j in np.unique(indices[:,i]):
                line[j] = '*'
            linestr = ''.join(line)
        else:
            line = [' ' for i in range(width)]
            for clusterid, j in zip(clusterids, indices[:,i]):
                if clusterid == 0 and line[j] == ' ':
                    # empty, so set it although we don't know the cluster id
                    line[j] = clusteridstrings[clusterid]
                elif clusterid > 0 and line[j] in (' ', '*'):
                    # set it to correct cluster id
                    line[j] = clusteridstrings[clusterid]
                #else:
                #    line[j] = '*'
            linestr = ''.join(line)
        
        fmt = '%+.1e'
        if -1 <= expolo[i] <= 2 and -1 <= expohi[i] <= 2:
            if not is_negative[i]:
                plo[i] = 0
            fmt = '%+.1f'
        if -4 <= expolo[i] <= 0 and -4 <= expohi[i] <= 0:
            fmt = '%%+.%df' % (-min(expolo[i], expohi[i]))
        print('%09s|%s|%9s %s' % (fmt % plo[i], linestr, fmt % phi[i], param))
    
    print()
    if ndim == 1:
        pass
    elif ndim == 2:
        rho, pval = scipy.stats.spearmanr(p)
        if pval < 0.01 and abs(rho) > 0.75:
            print("   %s between %s and %s: rho=%.2f" % (
                'positive degeneracy' if rho > 0 else 'negative degeneracy',
                paramnames[0], paramnames[1], rho))
    else:
        rho, pval = scipy.stats.spearmanr(p)
        for i, param in enumerate(paramnames):
            for j, param2 in enumerate(paramnames[:i]):
                if pval[i,j] < 0.01 and abs(rho[i,j]) > 0.75:
                    print("   %s between %s and %s: rho=%.2f" % (
                        'positive degeneracy' if rho[i,j] > 0 else 'negative degeneracy',
                        param, param2, rho[i,j]))
    


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
        
        self.logger = create_logger(__name__)

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

        #if mcmc_steps <= 0:
        #    mcmc_steps = 5 * self.x_dim

        #if volume_switch <= 0:
        #    volume_switch = 1 / mcmc_steps

        #if alpha == 0.0:
        #    alpha = 2 / self.x_dim ** 0.5

        #if self.log:
        #    self.logger.info('MCMC steps [%d] alpha [%5.4f] volume switch [%5.4f]' % (mcmc_steps, alpha, volume_switch))

        prev_u = []
        prev_v = []
        prev_logl = []
        if self.log:
            # try to resume:
            self.logger.info('Resuming...')
            for i in range(self.num_live_points):
                row = self.pointstore.pop(-np.inf)
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
                        stored_point = self.pointstore.pop(loglstar)
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
