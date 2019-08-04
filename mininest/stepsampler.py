"""

MCMC-like step sampling within a region

"""


import numpy as np

class StepSampler(object):
    """
    Simple step sampler, staggering around
    """
    def __init__(self, nsteps, max_rejects=-5):
        self.history = []
        self.nsteps = nsteps
        if max_rejects < 0:
            self.max_rejects = (-max_rejects) * self.nsteps
        else:
            self.max_rejects = max_rejects
        
        self.nrejects = 0
        self.scale = 1.0
        self.last = None, None
    
    def __str__(self):
        return type(self).__name__ + '(%d steps)' % self.nsteps
    
    def move(self, ui, region, ndraw=1, plot=False):
        raise NotImplementedError()
    
    def adjust_outside_region(self):
        #print("ineffective proposal scale (%e). shrinking..." % self.scale)
        self.scale *= 0.1
        assert self.scale > 0
        self.last = None, None
    
    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        if accepted:
            self.scale *= 1.04
            self.last = unew, Lnew
            self.history.append((unew, Lnew))
        else:
            self.scale /= 1.04
            self.nrejects += 1

    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=40, plot=False):
        
        # find most recent point in history conforming to current Lmin
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            #print("wandered out of L constraint; resetting", ui[0])
            ui, Li = None, None
        
        if Li is not None and not region.inside(ui.reshape((1,-1))):
            # region was updated and we are not inside anymore 
            # so reset
            ui, Li = None, None
        
        if Li is None and self.history:
            # try to resume from a previous point above the current contour
            for uj, Lj in self.history[::-1]:
                if Lj >= Lmin and region.inside(uj.reshape((1,-1))):
                    ui, Li = uj, Lj
                    break
        
        if self.max_rejects > 0 and (self.nrejects > self.max_rejects):
            # we are somehow stuck and not going anywhere
            # so reset
            ui, Li = None, None
        
        # select starting point
        if Li is None:
            # choose a new random starting point
            mask = region.inside(us)
            assert mask.any(), ("None of the live points satisfies the current region!", 
                region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(mask.sum())
            self.starti = i
            ui = us[mask,:][i]
            #print("starting at", ui[0])
            assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[mask][i]
            self.nrejects = 0
            self.history.append((ui, Li))
        
        unew = self.move(ui, region, ndraw=ndraw)
        mask = np.logical_and(unew > 0, unew < 1).all(axis=1)
        unew = unew[mask,:]
        mask = region.inside(unew)
        nc = 0
        
        if mask.any():
            i = np.where(mask)[0][0]
            unew = unew[i,:]
            pnew = transform(unew)
            Lnew = loglike(pnew)
            nc = 1
            if Lnew >= Lmin:
                self.adjust_accept(True, unew, pnew, Lnew, nc)
                if len(self.history) >= self.nsteps:
                    #print("made %d steps" % len(self.history))
                    self.history = []
                    self.last = None, None
                    return unew, pnew, Lnew, nc
            else:
                self.adjust_accept(False, unew, pnew, Lnew, nc)
        else:
            self.adjust_outside_region()
        
        # do not have a independent sample yet
        return None, None, None, nc

class CubeMHSampler(StepSampler):
    """
    Simple step sampler, staggering around
    """
    def move(self, ui, region, ndraw=1, plot=False):
        # propose in that direction
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        unew = ui.reshape((1, -1)) + jitter
        return unew

class RegionMHSampler(StepSampler):
    """
    Simple step sampler, staggering around
    """
    def move(self, ui, region, ndraw=1, plot=False):
        ti = region.transformLayer.transform(ui)
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        tnew = ti.reshape((1, -1)) + jitter
        unew = region.transformLayer.untransform(tnew)
        return unew

class DESampler(StepSampler):
    """
    Simple step sampler using as directions the differences 
    between two randomly chosen live points.
    """
    def move(self, ui, region, ndraw=1, plot=False):
        # choose direction
        # avoid drawing the two exact same points (no direction)
        # avoid drawing the starting point (to avoid linear combinations)
        j = np.random.randint(len(region.u) - 1)
        if j >= self.starti:
            j += 1
        k = np.random.randint(len(region.u) - 2)
        if k >= self.starti:
            k += 1
        if k >= j:
            k += 1
        direction = region.u[j,:] - region.u[k,:]
        assert (direction != 0).all(), (j, k, direction, region.u[j,:], region.u[k,:])
        
        # propose in that direction
        jitter = np.random.normal(0, 1e-4 / len(ui), size=(ndraw, len(ui)))
        direction = direction + jitter * (direction**2).sum()**0.5
        vector = np.random.normal(size=(ndraw, 1)) * direction * self.scale
        unew = ui.reshape((1, -1)) + vector + jitter
        return unew


class CubeSliceSampler(StepSampler):
    """
    Slice sampler, respecting the region
    """
    def __init__(self, nsteps, max_rejects=-5):
        StepSampler.__init__(self, nsteps=nsteps, max_rejects=-5)
        self.interval = None

    def generate_direction(self, ui, region):
        ndim = len(ui)
        # choose axis
        j = np.random.randint(ndim)
        # use doubling procedure to identify left and right maxima borders
        v = np.zeros(ndim)
        v[j] = 1.0
        return v

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        if accepted:
            # start with a new interval next time
            self.interval = None
            
            self.last = unew, Lnew
            self.history.append((unew, Lnew))
        else:
            self.nrejects += 1
            # continue on current interval
            pass

    def adjust_outside_region(self):
        pass
    
    def move(self, ui, region, ndraw=1, plot=False):
        if self.interval is None:
            v = self.generate_direction(ui, region)
            
            # expand direction until it is surely outside
            left = -self.scale
            right = self.scale
            while True:
                xj = ui + v * left
                if not region.inside(xj.reshape((1, -1))):
                    break
                left *= 2
                
            while True:
                xj = ui + v * right
                if not region.inside(xj.reshape((1, -1))):
                    break
                right *= 2
            self.scale = max(-left, right)
            self.interval = (v, left, right, 0)
        else:
            v, left, right, u = self.interval
            # check if we rejected last time, and shrink corresponding side
            if u == 0:
                pass
            elif u < 0:
                left = u
            elif u > 0:
                right = u
        
        # shrink direction if outside
        while True:
            u = np.random.uniform(left, right)
            xj = ui + v * u
            if region.inside(xj.reshape((1, -1))):
                self.interval = (v, left, right, u)
                #if plot:
                #    plot.plot(
                #        [ui[0] + v[0] * left, ui[0] + v[0] * right], 
                #        [ui[1] + v[1] * left, ui[1] + v[1] * right],
                #        '-', color='r', alpha=0.2)
                return xj.reshape((1, -1))
            else:
                if u < 0:
                    left = u
                else:
                    right = u
                self.interval = (v, left, right, u)


class RegionSliceSampler(CubeSliceSampler):
    """
    Slice sampler, in region axes
    """
    def generate_direction(self, ui, region):
        ndim = len(ui)
        ti = region.transformLayer.transform(ui)
        
        # choose axis in transformed space:
        j = np.random.randint(ndim)
        tv = np.zeros(ndim)
        tv[j] = 1.0
        # convert back to unit cube space:
        uj = region.transformLayer.untransform(ti + tv * 1e-3)
        v = uj - ui
        return v
