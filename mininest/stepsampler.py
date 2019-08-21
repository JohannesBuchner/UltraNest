"""

MCMC-like step sampling within a region

"""


import numpy as np

def generate_cube_oriented_direction(ui, region):
    ndim = len(ui)
    # choose axis
    j = np.random.randint(ndim)
    # use doubling procedure to identify left and right maxima borders
    v = np.zeros(ndim)
    v[j] = 1.0
    return v


def generate_region_oriented_direction(ui, region, scale=1):
    ndim = len(ui)
    ti = region.transformLayer.transform(ui)
    
    # choose axis in transformed space:
    j = np.random.randint(ndim)
    tv = np.zeros(ndim)
    tv[j] = 1.0
    # convert back to unit cube space:
    uj = region.transformLayer.untransform(ti + tv * 1e-3)
    v = uj - ui
    v *= scale / (v**2).sum()**0.5
    return v

def generate_region_random_direction(ui, region, scale=1):
    ndim = len(ui)
    ti = region.transformLayer.transform(ui)
    
    # choose axis in transformed space:
    ti = np.random.normal(ti, scale)
    # convert back to unit cube space:
    uj = region.transformLayer.untransform(ti)
    v = uj - ui
    return v


class StepSampler(object):
    """
    Simple step sampler, staggering around
    Scales proposal towards a 50% acceptance rate
    """
    def __init__(self, nsteps):
        """
        nsteps: int
            number of accepted steps until the sample is considered independent
        """
        self.history = []
        self.nsteps = nsteps
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
    
    def reset(self):
        self.nrejects = 0

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
            self.reset()
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
    def __init__(self, nsteps):
        """
        see StepSampler.__init__ documentation
        """
        StepSampler.__init__(self, nsteps=nsteps)
        self.interval = None

    def generate_direction(self, ui, region):
        return generate_cube_oriented_direction(ui, region)

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
        return generate_region_oriented_direction(ui, region)


from mininest.samplingpath import SamplingPath, ContourSamplingPath


class SamplingPathSliceSampler(StepSampler):
    """
    Slice sampler, respecting the region, on the sampling path
    """
    def __init__(self, nsteps):
        """
        see StepSampler.__init__ documentation
        """
        StepSampler.__init__(self, nsteps=nsteps)
        self.interval = None
        self.path = None

    def generate_direction(self, ui, region, scale=1):
        return generate_region_oriented_direction(ui, region, scale=scale)

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
            v = self.generate_direction(ui, region, scale=self.scale)
            self.path = ContourSamplingPath(SamplingPath(ui, v, 0.0), 
                region)
            
            if not (ui > 0).all() or not (ui < 1).all() or not region.inside(ui.reshape((1, -1))):
                assert False, ui
            
            # unit hypercube diagonal gives a reasonable maximum path length
            maxlength = len(ui)**0.5
            
            # expand direction until it is surely outside
            left = -1
            right = +1
            while abs(left*self.scale) < maxlength:
                xj, vj = self.path.extrapolate(left)
                if not (xj > 0).all() or not (xj < 1).all() or not region.inside(xj.reshape((1, -1))):
                    break
                #self.path.add(left, xj, vj, 0.0)
                left *= 2
            
            while abs(right * self.scale) < maxlength:
                xj, _ = self.path.extrapolate(right)
                if not (xj > 0).all() or not (xj < 1).all() or not region.inside(xj.reshape((1, -1))):
                    break
                #self.path.add(right, xj, vj, 0.0)
                right *= 2
            
            scale = max(-left, right)
            #print("scale %f gave %d %d " % (self.scale, left, right))
            if scale < 5:
                self.scale /= 1.1
            #if scale > 100:
            #    self.scale *= 1.1
            
            assert self.scale > 1e-10, self.scale
            self.interval = (left, right, None)
        else:
            left, right, mid = self.interval
            # we rejected mid, and shrink corresponding side
            if mid < 0:
                left = mid
            elif mid > 0:
                right = mid
        
        # shrink direction if outside
        while True:
            mid = np.random.randint(left, right + 1)
            #print("interpolating %d - %d - %d" % (left, mid, right), 
            #    self.path.points)
            if mid == 0:
                _, xj, _, _ = self.path.points[0]
            else:
                xj, _ = self.path.extrapolate(mid)
            
            if region.inside(xj.reshape((1, -1))):
                self.interval = (left, right, mid)
                return xj.reshape((1, -1))
            else:
                if mid < 0:
                    left = mid
                else:
                    right = mid
                self.interval = (left, right, mid)


class SamplingPathStepSampler(StepSampler):
    """
    Step sampler on a sampling path
    """
    def __init__(self, nresets, nsteps):
        """
        nresets: int
            after this many iterations, select a new direction
        nsteps: int
            how many steps to make in total
        """
        StepSampler.__init__(self, nsteps=nsteps)
        #self.lasti = None
        self.path = None
        self.nresets = nresets
        self.scale = 0.1
        self.istep = 0
        self.iresets = 0
    
    def __str__(self):
        return type(self).__name__ + '(%d steps, %d resets)' % (self.nsteps, self.nresets)

    def reset(self):
        self.nrejects = 0
        self.istep = 0

    def generate_direction(self, ui, region, scale):
        return generate_region_random_direction(ui, region, scale=scale)

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        #print("step %d, scale %f, %s" % (self.istep, self.scale, accepted))
        if self.istep % self.nresets == 0:
            #print("triggering re-orientation")
            # reset path so we go in a new direction
            self.path = None
        
        self.istep += 1
        
        if accepted:
            # start at new point next time
            self.lasti = self.nexti
            self.history.append((unew, Lnew))
            
            # hypercube diagonal sets a sane maximum scale
            maxlength = len(unew)**0.5
            if self.scale < maxlength:
                self.scale *= 1.01
        else:
            self.nrejects += 1
            self.scale /= 1.1
            assert self.scale > 0, (self.scale, self.istep, self.nrejects)
            # continue on current point

    def adjust_outside_region(self):
        if self.istep % self.nresets == 0:
            #print("triggering re-orientation")
            # reset path so we go in a new direction
            self.path = None
        
        self.istep += 1
        
        self.scale /= 1.1
        assert self.scale > 0
    
    def move(self, ui, region, ndraw=1, plot=False):
        if self.path is None:
            v = self.generate_direction(ui, region, scale=self.scale)
            assert (v**2).sum() > 0, (v, self.scale)
            self.path = ContourSamplingPath(SamplingPath(ui, v, 0.0), 
                region)
            
            if not (ui > 0).all() or not (ui < 1).all() or not region.inside(ui.reshape((1, -1))):
                assert False, ui
            
            #print(ui, v)
            # unit hypercube diagonal gives a reasonable maximum path length
            self.lasti = 0
        
        self.nexti = self.lasti + np.random.randint(0, 2) * 2 - 1
        
        #print("evaluating at", self.nexti)
        if self.nexti == 0:
            _, x, _, _ = self.path.points[0]
        else:
            x, _ = self.path.extrapolate(self.nexti)
        #print("returning", x)
        return x.reshape((1, -1))


from mininest.flatnuts import ClockedSimpleStepSampler, ClockedStepSampler, ClockedBisectSampler #, ClockedNUTSSampler


class OtherSamplerProxy(object):
    """
    Proxy for ClockedSamplers
    """
    def __init__(self, nnewdirections, nsteps, epsilon=0.1, sampler='steps'):
        """
        nsteps: int
            number of accepted steps until the sample is considered independent
        """
        self.nsteps = nsteps
        self.samplername = sampler
        self.sampler = None
        self.epsilon = epsilon
        self.last = None, None
        self.Llast = None
        self.ncalls = 0
        self.nnewdirections = nnewdirections
        self.nreflections = 0
        self.nreverses = 0
        self.nsteps_done = 0
    
    def __str__(self):
        return 'Proxy[%s](%dx%d steps)' % (self.samplername, self.nnewdirections, self.nsteps)
    
    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        if self.sampler is not None:
            self.nreflections += self.sampler.nreflections
            self.nreverses += self.sampler.nreverses
            points = self.sampler.points
            # range
            ilo, _, _, _ = min(points)
            ihi, _, _, _ = max(points)
            self.nsteps_done += ihi - ilo
        
        #if self.nsteps <= 1:
        #    return
        
        # what is good? lots of reflections or reverses, large range
        
        # what is bad?
        # very narrow range
        print("point range: %d    %d reverses, %d reflections, %d nsteps, %d restarts   epsilon=%f" % (
            self.nsteps_done, self.nreverses, self.nreflections, self.nsteps_done, self.nrestarts, self.epsilon))
        #if irange <= 2:
        #    self.epsilon /= 2.0
        
        # or straight path with no reflections
        #if self.nrestarts > 10:
        #    self.epsilon /= 1.5
        #if self.nrestarts > 2:
        #    self.epsilon /= 1.01
        #if self.nrestarts < 2:
        #    self.epsilon *= 1.01
        
        assert self.epsilon > 0, self.epsilon
        
        #if irange <= 2:
        #    # path too long, shorten a bit
        #    self.epsilon /= 1.1
        #if nreflections < len(unew):
        #    # path too short, lengthen a bit
        #    self.epsilon *= 1.1
        if max(self.nreflections, self.nreverses) > self.nnewdirections * self.nsteps * 0.05:
            # path too long, shorten a bit
            self.epsilon /= 1.1
        else:
            self.epsilon *= 1.01
    
    def adjust_outside_region(self, *args, **kwargs):
        pass
    
    def startup(self, region, us, Ls):
        #print("starting from scratch...")
        # choose a new random starting point
        mask = region.inside(us)
        assert mask.all(), ("None of the live points satisfies the current region!", 
            region.maxradiussq, region.u, region.unormed, us)
        i = np.random.randint(mask.sum())
        self.starti = i
        ui = us[mask,:][i]
        assert np.logical_and(ui > 0, ui < 1).all(), ui
        Li = Ls[mask][i]
        self.last = ui, Li
        self.Llast = None
        self.ncalls = 0
        self.nrestarts = 0

        self.nreflections = 0
        self.nreverses = 0
        self.nsteps_done = 0
    
    def start_direction(self, region):
        ui, Li = self.last
        # choose random direction
        tt = np.random.normal(region.transformLayer.transform(ui), self.epsilon)
        v = region.transformLayer.untransform(tt) - ui
        self.nrestarts += 1
        
        if self.sampler is not None:
            self.nreflections += self.sampler.nreflections
            self.nreverses += self.sampler.nreverses
            points = self.sampler.points
            # range
            ilo, _, _, _ = min(points)
            ihi, _, _, _ = max(points)
            self.nsteps_done += ihi - ilo
        
        if self.sampler is None or True:
            samplingpath = SamplingPath(ui, v, Li)
            contourpath = ContourSamplingPath(samplingpath, region)
            if self.samplername == 'simple':
                self.sampler = ClockedSimpleStepSampler(contourpath)
                self.sampler.set_nsteps(self.nsteps)
            elif self.samplername == 'steps':
                self.sampler = ClockedStepSampler(contourpath)
                self.sampler.set_nsteps(self.nsteps)
            elif self.samplername == 'bisect':
                self.sampler = ClockedBisectSampler(contourpath)
                self.sampler.set_nsteps(self.nsteps)
            else:
                assert False
            self.sampler.log = False
    
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
        
        if Li is None or self.sampler is None:
            self.startup(region, us, Ls)
            self.start_direction(region)

        sample, is_independent = self.sampler.next(self.Llast)
        if sample is None: # nothing to do
            print("ran out of things to do.")
            assert False, (sample, is_independent)
            self.start_direction(region)
            return None, None, None, 0
        
        if is_independent:
            unew, Lnew = sample
            assert np.isfinite(unew).all(), unew
            assert np.isfinite(Lnew).all(), Lnew
            # done, reset:
            #print("got a sample:", unew)
            if self.nrestarts >= self.nnewdirections:
                xnew = transform(unew)
                self.adjust_accept(True, unew, xnew, Lnew, self.ncalls)
                self.sampler = None
                self.last = None, None
                return unew, xnew, Lnew, 0
            else:
                self.last = unew, Lnew
                self.start_direction(region)
                self.Llast = None
                return None, None, None, 0
        else:
            unew = sample
            xnew = transform(unew)
            self.Llast = loglike(xnew)
            nc = 1
            self.ncalls += 1
            if self.Llast > Lmin:
                self.last = unew, self.Llast
            else:
                self.Llast = None
            return None, None, None, nc
    
    def move(self, startu, region, plot=None):
        Ls = np.zeros(1)
        if self.sampler is None:
            self.startup(region, startu.reshape((1,-1)), Ls)
            self.start_direction(region)
            self.Llast = None
        self.sampler.plot = True
        self.sampler.log = True
        while True:
            sample, is_independent = self.sampler.next(self.Llast)
            print("sample proposed in move:", sample)
            if sample is None:
                #self.startup(region, startu.reshape((1,-1)), Ls)
                self.Llast = None
                self.start_direction(region)
            else:
                if is_independent:
                    unew = sample[0].reshape((1, -1))
                else:
                    unew = sample.reshape((1, -1))
                
                if region.inside(unew):
                    self.Llast = 0.
                    self.last = unew[0], self.Llast
                else:
                    self.Llast = None
                print("sample eval:", self.Llast, is_independent)
                return unew
    
