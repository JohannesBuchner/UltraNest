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

from .utils import create_logger, make_run_dir
from .utils import acceptance_rate, effective_sample_size, mean_jump_distance
from .mlfriends import MLFriends, AffineLayer, ScalingLayer

import numpy as np
from numpy import log10
import scipy.stats

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
    width = 60
    indices = ((p - plo) * width / (phi - plo).reshape((1, -1))).astype(int)
    indices[indices >= width] = width - 1
    indices[indices < 0] = 0
    
    print()
    print()
    clusterids = transformLayer.clusterids
    nmodes = transformLayer.nclusters
    print("Volume:", region.estimate_volume())
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
                if line[j] == ' ' or line[j] == '%d' % clusterid:
                    line[j] = '%d' % clusterid
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
    
    for i, param in enumerate(paramnames):
        for j, param2 in enumerate(paramnames[:i]):
            rho, pval = scipy.stats.spearmanr(p[:,i], p[:,j])
            if pval < 0.01:
                print("   %s between %s and %s: rho=%.2f" % (
                    'positive correlation' if rho > 0 else 'negative correlation',
                    param, param2, rho))
    


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
            self.wrapped_axes = np.zeros(0, dtype=int)
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

        if self.log:
            self.logs = make_run_dir(log_dir, run_num, append_run_num= append_run_num)
            log_dir = self.logs['run_dir']
        else:
            log_dir = None
        
        self.logger = create_logger(__name__)

        if self.log:
            self.logger.info('Num live points [%d]' % (self.num_live_points))

        if self.log:
            with open(os.path.join(self.logs['results'], 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'acceptance', 'min_ess',
                                 'max_ess', 'jump_distance', 'scale', 'loglstar', 'logz', 'fraction_remain', 'ncall'])

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
                update_interval_iter = 0
        

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

        if self.use_mpi:
            self.logger.info('Using MPI with rank [%d]' % (self.mpi_rank))
            if self.mpi_rank == 0:
                active_u = np.random.uniform(size=(self.num_live_points, self.x_dim))
            else:
                active_u = np.empty((self.num_live_points, self.x_dim), dtype=np.float64)
            self.comm.Bcast(active_u, root=0)
        else:
            active_u = np.random.uniform(size=(self.num_live_points, self.x_dim))
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

        saved_v = []  # Stored points for posterior results
        saved_logl = []
        saved_logwt = []
        h = 0.0  # Information, initially 0.
        logz = -1e300  # ln(Evidence Z), initially Z=0
        logvol = np.log(1.0 - np.exp(-1.0 / self.num_live_points))
        logz_remain = np.max(active_logl)
        fraction_remain = 1.0
        ncall = self.num_live_points  # number of calls we already made
        first_time = True
        #nb = self.mpi_size * mcmc_batch_size
        #direct_draw_efficient = True
        #mlfriends_efficient = True
        transformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
        region = MLFriends(active_u, transformLayer)
        
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
            saved_v.append(np.array(active_v[worst]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[worst])

            expected_vol = np.exp(-it / self.num_live_points)

            # The new likelihood constraint is that of the worst object.
            loglstar = active_logl[worst]
            
            if ncall > next_update_interval_ncall and it > next_update_interval_iter:
                if first_time:
                    nextregion = region
                else:
                    nextregion = MLFriends(active_u, transformLayer=transformLayer.create_new(active_u, region.maxradiussq))
                r = nextregion.compute_maxradiussq(nbootstraps=30 // self.mpi_size)
                #print("MLFriends built. r=%f" % r**0.5)
                if self.use_mpi:
                    recv_minradii = self.comm.gather(r, root=0)
                    recv_minradii = self.comm.bcast(recv_minradii, root=0)
                    r = np.max(recv_minradii)
                
                nextregion.maxradiussq = r
                print()
                print('proposed volume:', nextregion.estimate_volume())
                if nextregion.estimate_volume() < region.estimate_volume():
                    region = nextregion
                    transformLayer = region.transformLayer
                
                if self.log:
                    nicelogger(points=dict(u=active_u, p=active_v, logl=active_logl), 
                        info=dict(it=it, ncall=ncall, logz=logz, logz_remain=logz_remain, paramnames=self.paramnames), 
                        region=region, transformLayer=transformLayer)
                next_update_interval_ncall = ncall + update_interval_ncall
                next_update_interval_iter = it + update_interval_iter
                first_time = False
            
            while True:
                while ib >= len(samples):
                    # get new samples
                    ib = 0
                    
                    nc = 0
                    # Simple rejection sampling over prior
                    """
                    draw_efficiency = 1.0
                    if direct_draw_efficient:
                        u = np.random.uniform(size=(ndraw, self.x_dim))
                        mask = region.inside(u)
                        u = u[mask,:]
                        nu = len(u)
                        father = np.ones(nu, dtype=int)
                        draw_efficiency = mask.mean()
                        if draw_efficiency < 1.0 / self.num_live_points:
                            print()
                            print("switching to MLFriends draw...")
                            direct_draw_efficient = False
                    else:
                        u, father = region.sample(nsamples=ndraw)
                        nu = len(u)
                        draw_efficiency = nu / ndraw
                    """
                    u, father = region.sample(nsamples=ndraw)
                    nu = len(u)
                    
                    v = self.transform(u)
                    #assert v.shape == (nu, self.num_params)
                    logl = self.loglike(v)
                    #assert logl.shape == (nu,), (logl.shape, v.shape, u.shape)
                    nc += nu
                    accepted = logl > loglstar
                    #assert logl.shape == accepted.shape, (logl.shape, accepted.shape)
                    u = u[accepted,:]
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
                
                if likes[ib] > active_logl[worst]:
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
                sys.stdout.write('lnZ = %.1f, remainder = %.1f, lnLike = %.1f | Efficiency: %d/%d = %.4f%% (draw:%d) \r' % (
                      logz, logz_remain, np.max(active_logl), it, ncall, it * 100 / ncall, ndraw))
                sys.stdout.flush()
                
                # if low efficiency, bulk-process larger arrays
                #if it * 100 > ncall:
                #    ndraw = max(4000, self.num_live_points)
                ndraw = max(32, min(16384, round((ncall+1) / (it + 1))))
                #else:
                #    ndraw = max(100, self.num_live_points)
                
                #self.logger.info(
                #    '[it=%d,nevals=%d,eff=%f] Like=%5.1f..%5.1f lnZ=%5.1f' %
                #    (it, ncall, it * 100. / ncall, loglstar, np.max(active_logl), logz))
            
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
            saved_v.append(np.array(active_v[i]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[i])

        if self.log:
            with open(os.path.join(self.logs['results'], 'final.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['niter', 'ncall', 'logz', 'logzerr', 'h'])
                writer.writerow([it + 1, ncall, logz, np.sqrt(h / self.num_live_points), h])
            self._save_samples(np.array(saved_v), np.exp(np.array(saved_logwt) - logz), np.array(saved_logl))
        
        if not self.use_mpi or self.mpi_rank == 0:
            print("niter: {:d}\n ncall: {:d}\n nsamples: {:d}\n logz: {:6.3f} +/- {:6.3f}\n h: {:6.3f}"
                  .format(it + 1, ncall, len(np.array(saved_v)), logz, np.sqrt(h / self.num_live_points), h))
