"""

MCMC-like step sampling within a region

"""


import numpy as np

class DESampler(object):
    """
    Simple step sampler using as directions the differences 
    between two randomly chosen live points.
    """
    def __init__(self, nsteps=5):
        self.history = []
        self.nsteps = nsteps
        self.scale = 1.0
        self.last = None, None
    
    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=40):
        
        # find most recent point in history conforming to current Lmin
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            ui, Li = None, None
        
        if Li is None and self.history:
            pass
            #for uj, Lj in self.history[::-1]:
            #    if Lj >= Lmin:
            #        ui, Li = uj, Lj
            #        break
        
        if Li is not None and not region.inside(ui.reshape((1,-1))):
            ui, Li = None, None
        
        i = len(us)
        # select starting point
        if Li is None:
            # choose a new random starting point
            mask = region.inside(us)
            assert mask.any(), ("None of the live points satisfies the current region!", 
                region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(mask.sum())
            ui = us[mask,:][i]
            assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[mask][i]
            self.history.append((ui, Li))
        
        # choose direction
        # avoid drawing the two exact same points (no direction)
        # avoid drawing the starting point (to avoid linear combinations)
        upoints = region.u
        j = np.random.randint(len(region.u) - 1)
        if j >= i:
            j += 1
        k = np.random.randint(len(region.u) - 2)
        if k >= i:
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
                self.scale *= 1.01
                self.last = unew, Lnew
                self.history.append((unew, Lnew))
                if len(self.history) >= self.nsteps:
                    #print("made %d steps" % len(self.history))
                    self.history = []
                    self.last = None, None
                    return unew, pnew, Lnew, nc
            else:
                self.scale /= 1.01
        else:
            #print("bad proposal %e; shrinking..." % self.scale)
            self.scale *= 0.1
            assert self.scale > 0
            self.last = None, None
        
        # do not have a independent sample yet
        return None, None, None, nc

class CubeMHSampler(object):
    """
    Simple step sampler, staggering around
    """
    def __init__(self, nsteps=5):
        self.history = []
        self.nsteps = nsteps
        self.scale = 1.0
        self.last = None, None
    
    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=40):
        
        # find most recent point in history conforming to current Lmin
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            print("wandered out of L constraint; resetting", ui[i,0])
            ui, Li = None, None
        
        if Li is not None and not region.inside(ui.reshape((1,-1))):
            ui, Li = None, None
        
        # select starting point
        if Li is None:
            # choose a new random starting point
            mask = region.inside(us)
            assert mask.any(), ("None of the live points satisfies the current region!", 
                region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(mask.sum())
            ui = us[mask,:][i]
            #print("starting at", ui[0])
            assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[mask][i]
            self.history.append((ui, Li))
        
        # propose in that direction
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        unew = ui.reshape((1, -1)) + jitter
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
                self.scale *= 1.01
                self.last = unew, Lnew
                self.history.append((unew, Lnew))
                if len(self.history) >= self.nsteps:
                    #print("made %d steps" % len(self.history))
                    self.history = []
                    self.last = None, None
                    return unew, pnew, Lnew, nc
            else:
                self.scale /= 1.01
        else:
            print("bad proposal %e; shrinking..." % self.scale)
            self.scale *= 0.1
            assert self.scale > 0
            self.last = None, None
        
        # do not have a independent sample yet
        return None, None, None, nc

class RegionMHSampler(object):
    """
    Simple step sampler, staggering around
    """
    def __init__(self, nsteps=5):
        self.history = []
        self.nsteps = nsteps
        self.scale = 1.0
        self.last = None, None
    
    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=40):
        
        # find most recent point in history conforming to current Lmin
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            print("wandered out of L constraint; resetting", ui[i,0])
            ui, Li = None, None
        
        if Li is not None and not region.inside(ui.reshape((1,-1))):
            ui, Li = None, None
        
        # select starting point
        if Li is None:
            # choose a new random starting point
            mask = region.inside(us)
            assert mask.any(), ("None of the live points satisfies the current region!", 
                region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(mask.sum())
            ui = us[mask,:][i]
            #print("starting at", ui[0])
            assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[mask][i]
            self.history.append((ui, Li))
        
        ti = region.transformLayer.transform(ui)
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        tnew = ti.reshape((1, -1)) + jitter
        
        unew = region.transformLayer.untransform(tnew)
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
                self.scale *= 1.01
                self.last = unew, Lnew
                self.history.append((unew, Lnew))
                if len(self.history) >= self.nsteps:
                    #print("made %d steps" % len(self.history))
                    self.history = []
                    self.last = None, None
                    return unew, pnew, Lnew, nc
            else:
                self.scale /= 1.01
        else:
            print("bad proposal %e; shrinking..." % self.scale)
            self.scale *= 0.1
            assert self.scale > 0
            self.last = None, None
        
        # do not have a independent sample yet
        return None, None, None, nc

