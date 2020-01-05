"""MCMC-like step sampling within a region.

The classes implemented here are generators that, in each iteration,
only make one likelihood call. This allows keeping a population of
samplers that have the same execution time per call, even if they
do not terminate at the same number of iterations.
"""

import numpy as np

import matplotlib.pyplot as plt


def generate_random_direction(ui, region, scale=1):
    """Draw uniform direction vector in unit cube space of length `scale`.

    Region is not used.
    """
    v = np.random.normal(0, 1, size=len(ui))
    v *= scale / (v**2).sum()**0.5
    return v


def generate_cube_oriented_direction(ui, region):
    """Draw a unit direction vector in direction of a random unit cube axes.

    `region` is not used.
    """
    ndim = len(ui)
    # choose axis
    j = np.random.randint(ndim)
    # use doubling procedure to identify left and right maxima borders
    v = np.zeros(ndim)
    v[j] = 1.0
    return v


def generate_region_oriented_direction(ui, region, tscale=1, scale=None):
    """Draw a random direction vector in direction of one of the `region` axes.

    The vector length is `scale` (if given).
    If not, the vector length in transformed space is `tscale`.
    """
    ndim = len(ui)
    ti = region.transformLayer.transform(ui)

    # choose axis in transformed space:
    j = np.random.randint(ndim)
    tv = np.zeros(ndim)
    tv[j] = tscale
    # convert back to unit cube space:
    uj = region.transformLayer.untransform(ti + tv)
    v = uj - ui
    if scale is not None:
        v *= scale / (v**2).sum()**0.5
    return v


def generate_region_random_direction(ui, region, scale=1):
    """Draw a direction vector in a random direction of the region.

    The vector length is `scale` (in unit cube space).
    """
    ti = region.transformLayer.transform(ui)

    # choose axis in transformed space:
    ti = np.random.normal(ti, 1)
    # ti *= scale / (ti**2).sum()**0.5
    # convert back to unit cube space:
    uj = region.transformLayer.untransform(ti)
    v = uj - ui
    v *= scale / (v**2).sum()**0.5
    return v


def generate_mixture_random_direction(ui, region, scale=1, uniform_weight=1e-6):
    """Draw from a mix of a ball proposal and a region-shaped proposal.

    Parameters
    -----------
    region: MLFriends
        region
    uniform_weight: float
        sets the weight for the equal-axis ball contribution
    scale: float
        length of the vector.

    """
    v1 = generate_random_direction(ui, region)
    v1 /= (v1**2).sum()**0.5
    v2 = generate_region_random_direction(ui, region)
    v2 /= (v2**2).sum()**0.5
    v = (v1 * uniform_weight + v2 * (1 - uniform_weight))
    v *= scale * (v2**2).sum()**0.5
    return v


def inside_region(region, unew, uold):
    """Check if `unew` is inside region.

    always returns True at the moment.
    """
    #return np.ones(len(unew), dtype=bool)

    tnew = region.transformLayer.transform(unew)
    told = region.transformLayer.transform(uold)
    mask2 = ((told.reshape((1, -1)) - tnew)**2).sum(axis=1) < region.maxradiussq
    if mask2.all():
        return mask2

    mask = region.inside(unew)
    return np.logical_or(mask, mask2)


class StepSampler(object):
    """Base class for a simple step sampler, staggering around.

    Scales proposal towards a 50% acceptance rate.
    """

    def __init__(self, nsteps, scale=1.0, adaptive_nsteps=False, region_filter=False, log=False):
        """Initialise sampler.

        Parameters
        -----------
        scale: float
            initial proposal size
        
        nsteps: int
            number of accepted steps until the sample is considered independent.
        
        adaptive_nsteps: False, 'proposal-distance', 'move-distance'
            if not false, allow earlier termination than nsteps.
            The 'proposal-distance' strategy stops when the sum of 
            all proposed vectors exceeds the mean distance
            between pairs of live points.
            As distance, the Mahalanobis distance is used.
            The 'move-distance' strategy stops when the distance between
            start point and current position exceeds the mean distance
            between pairs of live points.
        
        region_filter: bool
            if True, use region to check if a proposed point can be inside
            before calling likelihood.

        """
        self.history = []
        self.nsteps = nsteps
        self.nrejects = 0
        self.scale = 1.0
        self.next_scale = self.scale
        self.last = None, None
        self.nudge = 1.1**(1. / self.nsteps)
        self.nsteps_nudge = 1.01
        adaptive_nsteps_options = (False, 'proposal-total-distances-NN', 'proposal-summed-distances-NN', 
            'proposal-total-distances', 'proposal-summed-distances', 
            'move-distance', 'move-distance-midway', 'proposal-summed-distances-min-NN',
            'proposal-variance-min', 'proposal-variance-min-NN')
        
        if adaptive_nsteps not in adaptive_nsteps_options:
            raise ValueError("adaptive_nsteps must be one of: %s, not '%s'" % (adaptive_nsteps_options, adaptive_nsteps))
        self.adaptive_nsteps = adaptive_nsteps
        self.mean_pair_distance = np.nan
        self.region_filter = region_filter
        self.log = log
        
        self.logstat = []
        self.logstat_labels = ['rejection_rate', 'scale', 'steps']
        if adaptive_nsteps:
            self.logstat_labels += ['jump-distance', 'reference-distance']

    def __str__(self):
        """Get string representation."""
        if not self.adaptive_nsteps:
            return type(self).__name__ + '(nsteps=%d)' % self.nsteps
        else:
            return type(self).__name__ + '(adaptive_nsteps=%s)' % self.adaptive_nsteps

    def plot(self, filename):
        """Plot sampler statistics."""
        if len(self.logstat) == 0:
            return

        plt.figure(figsize=(10, 1 + 3 * len(self.logstat_labels)))
        for i, label in enumerate(self.logstat_labels):
            part = [entry[i] for entry in self.logstat]
            plt.subplot(len(self.logstat_labels), 1, 1 + i)
            plt.ylabel(label)
            plt.plot(part)
            x = []
            y = []
            for j in range(0, len(part), 20):
                x.append(j)
                y.append(np.mean(part[j:j + 20]))
            plt.plot(x, y)
            if np.min(part) > 0:
                plt.yscale('log')
        plt.savefig(filename, bbox_inches='tight')
        np.savetxt(filename + '.txt.gz', self.logstat, 
            header=','.join(self.logstat_labels), delimiter=',')
        plt.close()

    def move(self, ui, region, ndraw=1, plot=False):
        """Move around ui. Stub to be implemented."""
        raise NotImplementedError()

    def adjust_outside_region(self):
        """Adjust proposal given that we landed outside region."""
        print("ineffective proposal scale (%g). shrinking..." % self.scale)
        
        # Usually the region is very generous.
        # Being here means that the scale is very wrong and we are probably stuck.
        # Adjust it and restart the chain
        self.scale /= self.nudge**10
        self.next_scale /= self.nudge**10
        assert self.scale > 0
        assert self.next_scale > 0
        # reset chain
        self.last = None, None
        if self.adaptive_nsteps:
            self.logstat.append([-1.0, self.scale, self.nsteps, np.nan, np.nan])
        else:
            self.logstat.append([-1.0, self.scale, self.nsteps])

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        """Adjust proposal given that we have been `accepted` at a new point after `nc` calls."""
        #print("accept: ", accepted)
        if accepted:
            self.next_scale *= self.nudge
            self.last = unew, Lnew
            self.history.append((unew.copy(), Lnew.copy()))
        else:
            self.next_scale /= self.nudge**10
            self.nrejects += 1
            self.history.append(self.history[-1])
        assert self.next_scale > 0, self.next_scale

    def adapt_nsteps(self, region):
        if not self.adaptive_nsteps:
            return
        elif len(self.history) < self.nsteps:
            # incomplete or aborted for some reason
            print("not adapting, incomplete history", len(self.history), self.nsteps)
            return
        
        #assert self.nrejects < len(self.history), (self.nsteps, self.nrejects, len(self.history))
        #assert self.nrejects <= self.nsteps, (self.nsteps, self.nrejects, len(self.history))
        assert np.isfinite(self.mean_pair_distance)
        nlive, ndim = region.u.shape
        if self.adaptive_nsteps == 'proposal-total-distances':
            # compute mean vector of each proposed jump
            # compute total distance of all jumps
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            assert len(tproposed.sum(axis=1)) == len(tproposed)
            d2 = ((((tproposed[0] - tproposed)**2).sum(axis=1))**0.5).sum()
            far_enough = d2 > self.mean_pair_distance / ndim
            
            self.logstat[-1] = self.logstat[-1] + [d2, self.mean_pair_distance]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, self.mean_pair_distance, d2)
        elif self.adaptive_nsteps == 'proposal-total-distances-NN':
            # compute mean vector of each proposed jump
            # compute total distance of all jumps
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            assert len(tproposed.sum(axis=1)) == len(tproposed)
            d2 = ((((tproposed[0] - tproposed)**2).sum(axis=1))**0.5).sum()
            far_enough = d2 > region.maxradiussq**0.5

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-summed-distances':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = (((tproposed[1:,:] - tproposed[:-1,:])**2).sum(axis=1)**0.5).sum()
            far_enough = d2 > self.mean_pair_distance / ndim
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, self.mean_pair_distance, d2)

            self.logstat[-1] = self.logstat[-1] + [d2, self.mean_pair_distance]
        elif self.adaptive_nsteps == 'proposal-summed-distances-NN':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = (((tproposed[1:,:] - tproposed[:-1,:])**2).sum(axis=1)**0.5).sum()
            far_enough = d2 > region.maxradiussq**0.5
            
            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-summed-distances-min-NN':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = (np.abs(tproposed[1:,:] - tproposed[:-1,:]).sum(axis=1)).min()
            far_enough = d2 > region.maxradiussq**0.5
            
            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-variance-min':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = tproposed.std(axis=0).min()
            far_enough = d2 > self.mean_pair_distance / ndim
            
            self.logstat[-1] = self.logstat[-1] + [d2, self.mean_pair_distance]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-variance-min-NN':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = tproposed.std(axis=0).min()
            far_enough = d2 > region.maxradiussq**0.5

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'move-distance':
            # compute distance from start to end
            ustart, _ = self.history[0]
            ufinal, _ = self.history[-1]
            tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
            d2 = ((tstart - tfinal)**2).sum()
            far_enough = d2 > region.maxradiussq
            
            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'move-distance-midway':
            # compute distance from start to end
            ustart, _ = self.history[0]
            middle = max(1, len(self.history) // 2)
            ufinal, _ = self.history[middle]
            tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
            d2 = ((tstart - tfinal)**2).sum()
            far_enough = d2 > region.maxradiussq
            
            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            #print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        else:
            assert False, self.adaptive_nsteps
        
        # adjust nsteps
        if far_enough:
            self.nsteps = min(self.nsteps - 1, int(self.nsteps / self.nsteps_nudge))
        else:
            self.nsteps = max(self.nsteps + 1, int(self.nsteps * self.nsteps_nudge))
        self.nsteps = max(1, min(1000, self.nsteps))

    def finalize_chain(self, region=None, Lmin=None, Ls=None):
        """Store chain statistics and adapt proposal."""
        self.logstat.append([self.nrejects / self.nsteps, self.scale, self.nsteps])
        if self.log:
            ustart, Lstart = self.history[0]
            ufinal, Lfinal = self.history[-1]
            #mean_pair_distance = region.compute_mean_pair_distance()
            mean_pair_distance = self.mean_pair_distance
            tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
            # L index of start and end
            #Ls_sorted = np.sort(Ls)
            iLstart = np.sum(Ls > Lstart)
            iLfinal = np.sum(Ls > Lfinal)
            # nearest neighbor index of start and end
            itstart = np.argmin((region.unormed - tstart.reshape((1, -1)))**2)
            itfinal = np.argmin((region.unormed - tfinal.reshape((1, -1)))**2)
            np.savetxt(self.log, [[Lmin] + list(ustart) + list(ufinal) + list(tstart) + list(tfinal) + \
                [self.nsteps, region.maxradiussq**0.5, mean_pair_distance, iLstart, iLfinal, itstart, itfinal]])
        
        if self.adaptive_nsteps:
            self.adapt_nsteps(region=region)
        
        if self.next_scale > self.scale * self.nudge**10:
            self.next_scale = self.scale * self.nudge**10
        elif self.next_scale < self.scale / self.nudge**10:
            self.next_scale = self.scale / self.nudge**10
        #print("updating scale: %g -> %g" % (self.scale, self.next_scale))
        self.scale = self.next_scale
        self.last = None, None
        self.history = []
        self.nrejects = 0

    def new_chain(self, region=None):
        """Starts a new path, reset statistics."""
        self.history = []
        self.nrejects = 0
    
    def region_changed(self, Ls, region):
        """React to change of region. """
        
        if self.adaptive_nsteps or True:
            self.mean_pair_distance = region.compute_mean_pair_distance()
            #print("region changed. new mean_pair_distance: %g" % self.mean_pair_distance)

    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=10, plot=False):
        """Get next point.

        Parameters
        ----------
        region: MLFriends
            region.
        Lmin: float
            loglikelihood threshold
        us: array of vectors
            current live points
        Ls: array of floats
            current live point likelihoods
        transform: function
            transform function
        loglike: function
            loglikelihood function
        ndraw: int
            number of draws to attempt simultaneously.
        plot: bool
            whether to produce debug plots.

        """
        # find most recent point in history conforming to current Lmin
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            print("wandered out of L constraint; resetting", ui[0])
            ui, Li = None, None

        if Li is None and self.history:
            # try to resume from a previous point above the current contour
            for j, (uj, Lj) in enumerate(self.history[::-1]):
                if Lj > Lmin and region.inside(uj.reshape((1,-1))):
                    ui, Li = uj, Lj
                    #print("recovering at point %d/%d " % (j+1, len(self.history)))
                    self.last = ui, Li

                    #pj = transform(uj.reshape((1, -1)))
                    #Lj2 = loglike(pj)[0]
                    #assert Lj2 > Lmin, (Lj2, Lj, uj, pj)

                    break
            pass

        # select starting point
        if Li is None:
            self.new_chain(region)
            # choose a new random starting point
            #mask = region.inside(us)
            #assert mask.any(), ("One of the live points does not satisfies the current region!",
            #    region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(len(us))
            self.starti = i
            ui = us[i,:]
            # print("starting at", ui[0])
            #assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[i]
            self.history.append((ui.copy(), Li.copy()))
            del i

        while True:
            unew = self.move(ui, region, ndraw=ndraw, plot=plot)
            if plot:
                plt.plot([ui[0], unew[:,0]], [ui[1], unew[:,1]], '-', color='k', lw=0.5)
                plt.plot(ui[0], ui[1], 'd', color='r', ms=4)
                plt.plot(unew[:,0], unew[:,1], 'x', color='r', ms=4)
            mask = np.logical_and(unew > 0, unew < 1).all(axis=1)
            unew = unew[mask,:]
            nc = 0
            if self.region_filter:
                mask = inside_region(region, unew, ui)
                if mask.any():
                    i = np.where(mask)[0][0]
                else:
                    self.adjust_outside_region()
                    continue
            else:
                i = 0

            if len(unew) == 0:
                self.adjust_outside_region()
                continue
            break
        
        unew = unew[i,:]
        pnew = transform(unew.reshape((1, -1)))
        Lnew = loglike(pnew)[0]
        nc = 1
        if Lnew > Lmin:
            if plot:
                plt.plot(unew[0], unew[1], 'o', color='g', ms=4)
            self.adjust_accept(True, unew, pnew, Lnew, nc)
        else:
            self.adjust_accept(False, unew, pnew, Lnew, nc)

        if len(self.history) > self.nsteps:
            # print("made %d steps" % len(self.history), Lnew, Lmin)
            u, L = self.history[-1]
            p = transform(u.reshape((1, -1)))[0]
            self.finalize_chain(region=region, Lmin=Lmin, Ls=Ls)
            return u, p, L, nc

        # do not have a independent sample yet
        return None, None, None, nc


class CubeMHSampler(StepSampler):
    """Simple step sampler, staggering around in cube space."""

    def move(self, ui, region, ndraw=1, plot=False):
        """Move in cube space."""
        # propose in that direction
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        unew = ui.reshape((1, -1)) + jitter
        return unew


class RegionMHSampler(StepSampler):
    """Simple step sampler, staggering around in transformLayer space."""

    def move(self, ui, region, ndraw=1, plot=False):
        """Move in transformLayer space."""
        ti = region.transformLayer.transform(ui)
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        tnew = ti.reshape((1, -1)) + jitter
        unew = region.transformLayer.untransform(tnew)
        return unew


class CubeSliceSampler(StepSampler):
    """Slice sampler, respecting the region."""

    def new_chain(self, region=None):
        """Starts a new path, reset slice."""
        self.interval = None
        self.found_left = False
        self.found_right = False
        self.axis_index = 0
        
        self.history = []
        self.last = None, None
        self.nrejects = 0

    def generate_direction(self, ui, region):
        """Start in a new direction, by choosing a random parameter."""
        return generate_cube_oriented_direction(ui, region)

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        """Adjust proposal given that we have been `accepted` at a new point after `nc` calls."""
        v, left, right, u = self.interval
        if not self.found_left:
            if accepted:
                self.interval = (v, left * 2, right, u)
            else:
                self.found_left = True
        elif not self.found_right:
            if accepted:
                self.interval = (v, left, right * 2, u)
            else:
                self.found_right = True
                # adjust scale
                if -left > self.next_scale or right > self.next_scale:
                    self.next_scale *= 1.1
                else:
                    self.next_scale /= 1.1
                #print("adjusting after accept...", self.next_scale)
        else:
            if accepted:
                # start with a new interval next time
                self.interval = None

                self.last = unew, Lnew
                self.history.append((unew.copy(), Lnew.copy()))
            else:
                self.nrejects += 1
                # shrink current interval
                if u == 0:
                    pass
                elif u < 0:
                    left = u
                elif u > 0:
                    right = u

                self.interval = (v, left, right, u)

    def adjust_outside_region(self):
        """Adjust proposal given that we landed outside region."""
        self.adjust_accept(False, unew=None, pnew=None, Lnew=None, nc=0)

    def move(self, ui, region, ndraw=1, plot=False):
        """Advance the slice sampling move."""
        if self.interval is None:
            v = self.generate_direction(ui, region)

            # expand direction until it is surely outside
            left = -self.scale
            right = self.scale
            self.found_left = False
            self.found_right = False
            u = 0

            self.interval = (v, left, right, u)

        else:
            v, left, right, u = self.interval

        if plot:
            plt.plot([(ui + v * left)[0], (ui + v * right)[0]],
                [(ui + v * left)[1], (ui + v * right)[1]],
                ':o', color='k', lw=2, alpha=0.3)

        # shrink direction if outside
        if not self.found_left:
            xj = ui + v * left

            if not self.region_filter or inside_region(region, xj.reshape((1, -1)), ui):
                return xj.reshape((1, -1))
            else:
                self.found_left = True

        if not self.found_right:
            xj = ui + v * right

            if not self.region_filter or inside_region(region, xj.reshape((1, -1)), ui):
                return xj.reshape((1, -1))
            else:
                self.found_right = True

                # adjust scale
                if -left > self.next_scale or right > self.next_scale:
                    self.next_scale *= 1.1
                else:
                    self.next_scale /= 1.1
                #print("adjusting scale...", self.next_scale)

        while True:
            u = np.random.uniform(left, right)
            xj = ui + v * u

            if not self.region_filter or inside_region(region, xj.reshape((1, -1)), ui):
                self.interval = (v, left, right, u)
                return xj.reshape((1, -1))
            else:
                if u < 0:
                    left = u
                else:
                    right = u
                self.interval = (v, left, right, u)


class RegionSliceSampler(CubeSliceSampler):
    """Slice sampler, randomly picking region axes."""

    def generate_direction(self, ui, region):
        """Choose a random axis from region.transformLayer."""
        return generate_region_oriented_direction(ui, region, tscale=self.scale, scale=None)


class RegionSequentialSliceSampler(CubeSliceSampler):
    """Slice sampler, sequentially iterating region axes."""

    def generate_direction(self, ui, region, scale=1):
        """Choose from region.transformLayer the next axis iteratively."""
        ndim = len(ui)
        ti = region.transformLayer.transform(ui)

        # choose axis in transformed space:
        j = self.axis_index % ndim
        self.axis_index = j + 1
        tv = np.zeros(ndim)
        tv[j] = 1.0
        # convert back to unit cube space:
        uj = region.transformLayer.untransform(ti + tv * 1e-3)
        v = uj - ui
        v *= scale / (v**2).sum()**0.5
        return v


class BallSliceSampler(CubeSliceSampler):
    """Hit & run sampler. Choose random directions in space."""

    def generate_direction(self, ui, region):
        """Choose a isotropically random direction from region.transformLayer."""
        return generate_random_direction(ui, region)


class RegionBallSliceSampler(CubeSliceSampler):
    """Hit & run sampler. Choose random directions according to region."""

    def generate_direction(self, ui, region):
        """Choose a isotropically random direction from region.transformLayer."""
        return generate_region_random_direction(ui, region)


class SpeedVariableRegionSliceSampler(CubeSliceSampler):
    """Slice sampler, in region axes.

    Updates only some dimensions at a time, completely user-definable.
    """

    def __init__(self, step_matrix):
        """Initialise sampler.

        Parameters
        -----------
        step_matrix: matrix or list of slices

            **if a bool matrix of shape (n_steps, n_dims):**

            Each row of the matrix indicates which parameters
            should be updated.

            Example::

                [[True, True], [False, True], [False, True]]

            This would update the first parameter 1/3 times, and the second
            parameters every time. Three steps are made until the point
            is considered independent.

            For a full update in every step, use::

                np.ones((n_steps, n_dims), dtype=bool)

            **if a list of slices:**

            Each entry indicates which parameters should be updated.

            Example::

                [Ellipsis, slice(2,10), slice(5,10)]

            This would update the first parameter 1/3 times, parameters
            2-9 2/3 times and parameter 5-9 in every step.
            Three steps are made until the point is considered independent.

        """
        nsteps = len(step_matrix)

        StepSampler.__init__(self, nsteps=nsteps)
        self.reset()
        self.step_matrix = step_matrix

    def generate_direction(self, ui, region, scale=1):
        """Generate a slice sampling direction, using only some of the axes."""
        ndim = len(ui)
        ti = region.transformLayer.transform(ui)

        # choose random axis in transformed space
        j = np.random.randint(ndim)
        tv = np.zeros(ndim)
        tv[j] = 1.0
        # convert back to unit cube space:
        uj = region.transformLayer.untransform(ti + tv * 1e-3)
        uk = ui.copy()

        j = self.axis_index % self.nsteps
        self.axis_index = j + 1
        # only update active dimensions
        active_dims = self.step_matrix[j]
        # project uj onto ui. vary only active dimensions
        uk[active_dims] = uj[active_dims]

        v = uk - ui
        v *= scale / (v**2).sum()**0.5
        return v

