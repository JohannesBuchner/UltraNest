"""

MCMC-like step sampling within a region

"""


import numpy as np

class StepSampler(object):
    """
    Simple step sampler, staggering around
    Scales proposal towards a 50% acceptance rate
    """
    def __init__(self, nsteps):
        """
        nsteps: int
            number of accepted steps until the sample is considered independent
        
        max_rejects: int
            If the likelihood threshold cannot be be reached after a 
            large number of tries, it is perhaps better to restart from 
            a new starting point.
            
            max_rejects>0 sets the number of tries until a new starting point
            is selected.
            max_rejects=0: deactivate this (keep trying forever)
            max_rejects<0: equivalent to max_rejects = -max_rejects * nsteps
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
    def __init__(self, nsteps):
        """
        see StepSampler.__init__ documentation
        """
        StepSampler.__init__(self, nsteps=nsteps)
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


def distances(l, o, r=1):
    """
    Compute sphere-line intersection
    
    l: direction vector (line starts at 0)
    o: center of sphere (coordinate vector)
    r: radius of sphere (float)
    
    returns (tpos, tneg), the positive and negative coordinate along the l vector where r is intersected.
    If no intersection, throws AssertError
    """
    loc = (l * o).sum()
    osqrnorm = (o**2).sum()
    #print(loc.shape, loc.shape, osqrnorm.shape)
    rootterm =  loc**2 - osqrnorm + r**2
    # make sure we are crossing the sphere
    assert (rootterm > 0).all(), rootterm 
    return -loc + rootterm**0.5, -loc - rootterm**0.5

def gen_unit_vectors(N, d):
    """
    Generates N random unit vectors in d dimensions
    """
    vec = np.random.normal(size=(N, d))
    norm = (vec**2).sum(axis=1)**0.5
    assert norm.shape == (N,), norm.shape
    return vec / norm.reshape((N, 1))

def gen_unit_vector(d):
    """
    Generates a random unit vectors in d dimensions
    """
    vec = np.random.normal(size=(d))
    norm = (vec**2).sum()**0.5
    return vec / norm

def isunitlength(vec):
    """
    Verifies that vec is of unit length.
    """
    norm = (vec**2).sum()**0.5
    assert np.isclose(norm, 1), norm

def angle(a, b):
    """
    Compute the dot product between vectors a and b
    The arccos of it would give an actual angle.
    """
    return (a*b).sum()


class FlatNUTS(object):
    """
    Flat version of the No-U-Turn Sampler.
    
    Uses MLFriends ellipsoids for reflections.
    """
    def __init__(self, plot=False):
        # initial step width
        self.epsilon = 1./3.
        self.reset()
        self.plot = plot
    
    def reset(self):
        if self.shrink_epsilon:
            self.epsilon *= 0.5
        
        self.state = 'start'
        self.path = []
        self.fwd_possible = True
        self.rwd_possible = True
        self.nevals = 0
        self.shrink_epsilon = False
        self.nreflections = 0
        self.expansions = []
        self.bisects = []
        self.reflect_at = []
        self.tree_depth = 0
        self.stop_doubling = False
    
    def __str__(self):
        return type(self).__name__
    
    def generate_direction(self, region, ui):
        """ choose a direction vector. 
        Uses the region transformation to sample according to the local metric. """
        
        ta = self.region.transformLayer.transform(ui)
        tb = np.random.normal(ta, 1e-6)
        b = self.region.transformLayer.untransform(tb)
        return b - ui
    
    def find_starting_point(self, region, us, Ls):
        """ select a random starting point and direction """
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
        vi = self.generate_direction()
        return ui, vi, Li
    
    def expand_to_step_if_possible(self, i):
        if i > 0 and self.fwd_possible:
            sign = 1
            fwd = True
            starti, startx, startv = max(self.points)
            if starti >= i:
                # already done
                return False
            deltai = i - starti
        elif self.rwd_possible:
            sign = -1
            fwd = False
            starti, startx, startv = min(self.points)
            if starti <= i:
                # already done
                return False
            deltai = i - starti
        else:
            # we are stuck now, and have to hope that the 
            # caller does not expect us to have filled that point
            return False
            #assert False, (i, self.fwd_possible, self.rwd_possible)
        
        #print("trying to expand to", i, " which is %d away" % deltai)
        xi = startx + startv * deltai * self.epsilon
        
        # wrap with unit cube here
        # 
        # 
        
        is_inside, Li = self.evaluate_point(xi)
        
        if is_inside:
            # can jump there directly
            #print("   trivial by direct jump")
            self.points.append((i, xi, startv, Li))
        else:
            if abs(deltai) > 1:
                # should find edge
                self.bisects.append([starti, i])
        return True
    
    def bisect_next(self):
        starti, i = self.bisects.pop(0)
        if i > starti:
            fwd = True
            sign = 1
        else:
            fwd = False
            sign = -1
        
        deltai = i - starti
        assert abs(deltai) > 1, deltai
        starti, startx, startv = [(i, x, v) for i, x, v in self.points if i == starti]
        
        midi = (i + starti) // 2
        deltai = i - midi
        
        xi = startx + startv * deltai * self.epsilon
        
        # wrap with unit cube here
        # 
        # 
        
        is_inside, Li = self.evaluate_point(xi)
        
        if is_inside:
            # add to path
            self.path.append((midi, xi, startv, Li))
            # new interval is midi,i
            if abs(midi - i) > 1:
                self.bisects.append([midi, i])
            else:
                # we found the end point: midi
                self.reflect_at.append(midi)
                pass
        else:
            # new interval is starti, midi
            if abs(midi - starti) > 1:
                self.bisects.append([starti, midi])
            else:
                # we found that starti is the end point
                self.reflect_at.append(starti)
                pass
        
        return True
    
    def nuts_doubling(self):
        self.tree_depth = 0
        stop = False
        
        validrange = (0, 0)
        ilo, ihi = max(self.path)[0], min(self.path)[0]
        
        rwd = np.random.randint(2) == 1
        if self.tree_depth > 7:
            print("NUTS step: tree depth %d, %s" % (self.tree_depth, "rwd" if rwd else "fwd"))
        if rwd:
            nexti = self.left_state[0] - 2**self.tree_depth
            if nexti < ilo:
                self.expansions.append(nexti)
                return False
            self.left_state, _, newrange, newstop = self.build_tree(self.left_state, self.tree_depth, rwd=rwd)
        else:   
            nexti = self.right_state[0] + 2**self.tree_depth
            if nexti > ihi:
                self.expansions.append(nexti)
                return False
            #print("  building fwd tree...")
            _, self.right_state, newrange, newstop = self.build_tree(self.right_state, self.tree_depth, rwd=rwd)
        
        if not newstop:
            validrange = (min(validrange[0], newrange[0]), max(validrange[1], newrange[1]))
            #print("  new range: %d..%d" % (validrange[0], validrange[1]))
            
            ileft, xleft, vleft = self.left_state
            iright, xright, vright = self.right_state
            #if self.plot: plt.plot([xleft[0], xright[0]], [xleft[1] + (j+1)*0.02, xright[1] + (j+1)*0.02], '--')
            #if j > 5:
            #   print("  first-to-last arrow", ileft, iright, xleft, xright, xright-xleft, " velocities:", vright, vleft)
            #   print("  stopping criteria: ", newstop, angle(xright-xleft, vleft), angle(xright-xleft, vright))
            stop = newstop or angle(xright-xleft, vleft) <= 0 or angle(xright-xleft, vright) <= 0
            
            self.tree_depth += 1
            if self.tree_depth > 3:
                # check whether both ends of the tree are at the end of the path
                if validrange[0] < min(self.points)[0] and validrange[1] > max(self.points)[0]:
                    print("Stopping stuck NUTS")
                    print("starting point was: ", self.points[0])
                    stop = True
                #if j > 7:
                #   print("starting point was: ", self.points[0])
                #   print("Stopping after %d levels" % j)
                #   break
        if stop:
            # switch to interpolation mode
            return self.sample_chain_point(validrange[0], validrange[1])
        
        else:
            # keep doubling!
            pass
    
    def build_tree(self, startstate, j, rwd):
        """
        Build sub-trees of depth j in direction rwd
        
        startstate: (i, x, v) state information of first node
        j: int height of the tree
        rwd: bool whether we go backward
        """
        if j == 0:
            # base case: go forward one step
            i = startstate[0] + (-1 if rwd else +1)
            #self.expand_to_step(i)
            #print("  build_tree@%d" % i, rwd)
            xi, vi, onpath = self.interpolate_point(i)
            if self.plot: plt.plot(xi[0], xi[1], 'x', color='gray')
            # this is a good state, so return it
            return (i, xi, vi), (i, xi, vi), (i,i), False
        
        # recursion-build the left and right subtrees
        (ileft, xleft, vleft), (iright, xright, vright), rangea, stopa = self.build_tree(startstate, j-1, rwd)
        if stopa:
            #print("  one subtree already terminated; returning")
            #plt.plot([xright[0], xleft[0]], [xright[1], xleft[1]], ':', color='navy')
            return (ileft, xleft, vleft), (iright, xright, vright), (ileft,iright), stopa
        
        if rwd:
            # go back
            (ileft, xleft, vleft), _, rangeb, stopb = self.build_tree((ileft, xleft, vleft), j-1, rwd)
        else:
            _, (iright, xright, vright), rangeb, stopb = self.build_tree((iright, xright, vright), j-1, rwd)
        #print("  subtree termination at %d" % j, stopa, stopb, angle(xright-xleft, vleft), angle(xright-xleft, vright), angle(vleft, vright))
        #plt.plot([xright[0], xleft[0]], [xright[1], xleft[1]], ':', color='gray')
        # NUTS criterion: start to end vector must point in the same direction as velocity at end-point
        # additional criterion: start and end velocities must point in opposite directions
        stop = stopa or stopb or angle(xright-xleft, vleft) <= 0 or angle(xright-xleft, vright) <= 0 or angle(vleft, vright) <= 0
        return (ileft, xleft, vleft), (iright, xright, vright), (ileft,iright), stop
    
    def move(self, ui, region, ndraw=1, plot=False):
        self.region = region
        self.sample_points = ui
        
        if self.state == 'start':
            # start in a random direction. Add starting point to path
            ui, vi, Li = self.find_starting_point(region, us, Ls)
            self.path = [(0, ui, vi, Li)]
            self.left_state = self.path[0]
            self.right_state = self.path[0]
            self.state = 'prepare-left'
            # fall through, because no evaluation yet
        
        # pre-compute some directions, so we do not make very small
        # steps with unnecessary evaluations in the beginning
        if self.state == 'prepare-left':
            had_eval = self.expand_to_step_if_possible(i = 10)
            self.state = 'prepare-right'
            if had_eval:
                return
        
        if self.state == 'prepare-right':
            had_eval = self.expand_to_step_if_possible(i = -10)
            self.state = 'bisect'
            if had_eval:
                return
        
        # ok, we are now ready to begin. Start bisecting if we 
        # have anything to bisect, so that we know the current path
        # and whether any sides are closed
        if self.state == 'bisect':
            if len(self.bisects) > 0:
                had_eval = self.bisect_next()
                if had_eval:
                    return
            else:
                self.state = 'expand'
        
        # expand
        if self.state == 'expand':
            # use NUTS doubling procedure
        
        
            
            
            outsidei = self.bisect(0, startx, startv, deltai, offseti=starti)
            self.nreflections += 1
            #print("bisecting gave reflection point", outsidei, "(+", starti, ")")
            xj = startx + startv * outsidei * self.epsilon
            if self.plot: plt.plot(xj[0], xj[1], 'xr')
            vk = self.reverse(xj, startv * sign) * sign
            xk = xj + vk * sign * self.epsilon
            self.nevals += 1
            if self.is_inside(xk):
                self.points.append((outsidei + starti, xk, vk))
                if continue_after_reflection or angle(vk, startv) > 0:
                    self.expand_to_step(i) # recurse
            pass
        
    def reverse(self, reflpoint, v):
        """
        Reflect off the surface at reflpoint going in direction v
        
        returns the new direction.
        """
        # check which ellipses contain reflpoint
        bpts = self.region.transformLayer.transform(reflpoint)
        idnearby = np.empty(len(self.region.unormed), dtype=int)
        find_nearby(bpts, self.region.unormed, self.maxradiussq, idnearby)
        mask = idnearby >= 0
        if not mask.any():
            # the reflection point is not in the region.
            # that means epsilon is too large
            
            # to handle this correctly, we must reverse (i.e., stop)
            self.shrink_epsilon = True
            return -v
        
        sphere_centers = self.region.u[mask,:]
        tsphere_centers = self.region.unormed[mask,:]
        
        # ok, we found some live points that contain the reflpoint
        # we scale their spheres to touch reflpoint
        # and compute the tangent there.
        tt = get_sphere_tangents(tsphere_centers, bpts)
        assert tt.shape == tsphere_centers.shape, (tt.shape, tsphere_centers.shape)
        
        # convert back to u space
        t = self.region.transformLayer.untransform(tt * 1e-3 + tsphere_centers) - sphere_centers
        assert t.shape == sphere_centers.shape, (t.shape, sphere_centers.shape)
        
        # compute new vector 
        normal = -t / (t**2).sum(axis=1)**0.5
        assert normal.shape == t.shape, (normal.shape, t.shape)
        
        mask_forward = (normal * v).sum(axis=1) > 0
        if not mask_forward.any():
            # none of the reflections point forward.
            # reverse.
            return -v
        
        # now down-select the ones that are forward reflections
        normal = normal[mask_forward,:]
        # chose one at random
        j = np.random.randint(len(normal))
        normal = normal[j,:]
        isunitlength(normal)
        isunitlength(v)
        
        vnew = v - 2 * angle(normal, v) * normal
        assert vnew.shape == v.shape, (vnew.shape, v.shape)
        isunitlength(vnew)
        return vnew
    
    def expand_to_step(self, i, continue_after_reflection=True):
        """
        Run steps forward or backward to step i (can be positive or 
        negative, 0 is the starting point), if possible.
        
        Tries to jump ahead if possible, and bisect otherwise.
        This avoid having to make all steps in between.
        """
        if i > 0 and self.fwd_possible:
            sign = 1
            fwd = True
            starti, startx, startv = max(self.points)
            if starti >= i:
                # already done
                return
            deltai = i - starti
        elif self.rwd_possible:
            sign = -1
            fwd = False
            starti, startx, startv = min(self.points)
            if starti <= i:
                # already done
                return
            deltai = i - starti
        else:
            # we are stuck now, and have to hope that the 
            # caller does not expect us to have filled that point
            return 
            #assert False, (i, self.fwd_possible, self.rwd_possible)
        
        #print("trying to expand to", i, " which is %d away" % deltai)
        xi = startx + startv * deltai * self.epsilon
        
        self.nevals += 1
        
        if self.is_inside(xi):
            # can jump there directly
            #print("   trivial by direct jump")
            self.points.append((i, xi, startv))
            return
        
        outsidei = self.bisect(0, startx, startv, deltai, offseti=starti)
        self.nreflections += 1
        #print("bisecting gave reflection point", outsidei, "(+", starti, ")")
        xj = startx + startv * outsidei * self.epsilon
        if self.plot: plt.plot(xj[0], xj[1], 'xr')
        vk = self.reverse(xj, startv * sign) * sign
        xk = xj + vk * sign * self.epsilon
        self.nevals += 1
        if self.is_inside(xk):
            self.points.append((outsidei + starti, xk, vk))
            if continue_after_reflection or angle(vk, startv) > 0:
                self.expand_to_step(i) # recurse
        else:
            #print("   could not come back inside", xk)
            if self.plot: plt.plot(xk[0], xk[1], 's', color='r')
            if fwd:
                self.fwd_possible = False
            else:
                self.rwd_possible = False
    
    
    def expand_onestep(self, fwd=True):
        """
        Make a single step forward (if fwd=True) or backwards)
        from the current state (stored in self.points)
        """
        if fwd:
            starti, startx, startv, _ = max(self.points)
            sign = 1
        else:
            starti, startx, startv, _ = min(self.points)
            sign = -1
        
        j = starti + 1*sign
        v = startv
        xj = startx + v * sign * self.epsilon
        
        #print("proposed step:", startx, "->", xj)
        
        is_inside, Lj = self.evaluate_point(xj)
        
        if is_inside(xj):
            # Everything ok, we keep going
            #print("  inside")
            self.points.append((j, xj, v, Lj))
            return True
        else:
            # We stepped outside, so now we need to reflect
            if self.plot: plt.plot(xj[0], xj[1], 'xr')
            vk = self.reverse(xj, v * sign) * sign
            #print("  outside; reflecting velocity", v, vk)
            xk = xj + vk * sign * self.epsilon
            self.nreflections += 1
            #print("  trying new point,", xk)
            self.nevals += 1
            if self.is_inside(xk):
                #print("  back inside!")
                self.points.append((j, xk, vk))
                return True
            else:
                if self.plot: plt.plot(xk[0], xk[1], 'sr')
                #print("  nope.")
                if fwd:
                    self.fwd_possible = False
                else:
                    self.rwd_possible = False
                return False
    
    def path_plot(self, color='blue'):
        self.points.sort()
        x0 = [x[0] for i, x, v in self.points]
        x1 = [x[1] for i, x, v in self.points]
        plt.plot(x0, x1, 'o-', color=color, mfc='None', ms=8)
        x0 = [x[0] for i, x, v in self.points if i == 0]
        x1 = [x[1] for i, x, v in self.points if i == 0]
        plt.plot(x0, x1, 's', mec='k', mfc='None', ms=10)

class BisectSampler(StepSampler):
    """
    Step sampler that does not require each step to be evaluated
    """
        
    def expand_to_step(self, i, continue_after_reflection=True):
        """
        Run steps forward or backward to step i (can be positive or 
        negative, 0 is the starting point), if possible.
        
        Tries to jump ahead if possible, and bisect otherwise.
        This avoid having to make all steps in between.
        """
        if i > 0 and self.fwd_possible:
            sign = 1
            fwd = True
            starti, startx, startv = max(self.points)
            if starti >= i:
                # already done
                return
            deltai = i - starti
        elif self.rwd_possible:
            sign = -1
            fwd = False
            starti, startx, startv = min(self.points)
            if starti <= i:
                # already done
                return
            deltai = i - starti
        else:
            # we are stuck now, and have to hope that the 
            # caller does not expect us to have filled that point
            return 
            #assert False, (i, self.fwd_possible, self.rwd_possible)
        
        #print("trying to expand to", i, " which is %d away" % deltai)
        xi = startx + startv * deltai * self.epsilon
        self.nevals += 1
        if self.is_inside(xi):
            # can jump there directly
            #print("   trivial by direct jump")
            self.points.append((i, xi, startv))
            return
        outsidei = self.bisect(0, startx, startv, deltai, offseti=starti)
        self.nreflections += 1
        #print("bisecting gave reflection point", outsidei, "(+", starti, ")")
        xj = startx + startv * outsidei * self.epsilon
        if self.plot: plt.plot(xj[0], xj[1], 'xr')
        vk = self.reverse(xj, startv * sign) * sign
        xk = xj + vk * sign * self.epsilon
        self.nevals += 1
        if self.is_inside(xk):
            self.points.append((outsidei + starti, xk, vk))
            if continue_after_reflection or angle(vk, startv) > 0:
                self.expand_to_step(i) # recurse
        else:
            #print("   could not come back inside", xk)
            if self.plot: plt.plot(xk[0], xk[1], 's', color='r')
            if fwd:
                self.fwd_possible = False
            else:
                self.rwd_possible = False
    
    def interpolate_point(self, i):
        """
        Given our sparsely sampled track (stored in .points),
        potentially with reflections, 
        extract the corrdinates of the point with index i.
        That point may not have been evaluated.
        """
        points_before = [(j, xj, vj) for j, xj, vj in self.points if j <= i]
        points_after  = [(j, xj, vj) for j, xj, vj in self.points if j >= i]
        
        # check if the point after is really after i
        if len(points_after) == 0 and not self.fwd_possible:
            # the path cannot continue, and i does not exist.
            #print("    interpolate_point %d: the path cannot continue fwd, and i does not exist." % i)
            j, xj, vj = max(points_before)
            return xj, vj, False
        
        # check if the point before is really before i
        if len(points_before) == 0 and not self.rwd_possible:
            # the path cannot continue, and i does not exist.
            k, xk, vk = min(points_after)
            #print("    interpolate_point %d: the path cannot continue rwd, and i does not exist." % i)
            return xk, vk, False
        
        j, xj, vj = max(points_before)
        k, xk, vk = min(points_after)
        
        #print("    interpolate_point %d between %d-%d" % (i, j, k))
        if j == i: # we have this exact point in the chain
            return xj, vj, True
        assert not k == i # otherwise the above would be true too
        
        # expand_to_step explores each reflection in detail, so
        # any points with change in v should have j == i
        # therefore we can assume:
        assert (vj == vk).all()
        # the new point is then just a linear interpolation
        w = (i - k) * 1. / (j - k)
        xl = xj * w + (1 - w) * xk
        return xl, vj, True
    
class NUTSSampler(BisectSampler):
    """
    No-U-turn sampler (NUTS) on flat surfaces.
    
    see nuts_step function.
    """
    
    def nuts_step(self):
        """
        Alternatingly doubles the number of steps to forward and backward 
        direction (which may include reflections, see StepSampler and
        BisectSampler).
        When track returns (start and end of tree point toward each other),
        terminates and returns a random point on that track.
        """
        # this is (0, x0, v0) in both cases
        left_state = self.points[0]
        right_state = self.points[0]
        
        # pre-explore a bit (until reflection or the number of steps)
        # this avoids doing expand_to_step with small step numbers later
        self.expand_to_step(-10, continue_after_reflection=False)
        self.expand_to_step(+10, continue_after_reflection=False)
        
        stop = False
        
        j = 0 # tree depth
        validrange = (0, 0)
        while not stop:
            rwd = np.random.randint(2) == 1
            if j > 7:
                print("NUTS step: tree depth %d, %s" % (j, "rwd" if rwd else "fwd"))
            if rwd:
                self.expand_to_step(left_state[0] - 2**j)
                #print("  building rwd tree...")
                left_state, _, newrange, newstop = self.build_tree(left_state, j, rwd=rwd)
            else:   
                self.expand_to_step(right_state[0] + 2**j)
                #print("  building fwd tree...")
                _, right_state, newrange, newstop = self.build_tree(right_state, j, rwd=rwd)
            
            if not newstop:
                validrange = (min(validrange[0], newrange[0]), max(validrange[1], newrange[1]))
                #print("  new range: %d..%d" % (validrange[0], validrange[1]))
            
            ileft, xleft, vleft = left_state
            iright, xright, vright = right_state
            if self.plot: plt.plot([xleft[0], xright[0]], [xleft[1] + (j+1)*0.02, xright[1] + (j+1)*0.02], '--')
            #if j > 5:
            #   print("  first-to-last arrow", ileft, iright, xleft, xright, xright-xleft, " velocities:", vright, vleft)
            #   print("  stopping criteria: ", newstop, angle(xright-xleft, vleft), angle(xright-xleft, vright))
            stop = newstop or angle(xright-xleft, vleft) <= 0 or angle(xright-xleft, vright) <= 0
            
            j = j + 1
            if j > 3:
                # check whether both ends of the tree are at the end of the path
                if validrange[0] < min(self.points)[0] and validrange[1] > max(self.points)[0]:
                    print("Stopping stuck NUTS")
                    print("starting point was: ", self.points[0])
                    break
                #if j > 7:
                #   print("starting point was: ", self.points[0])
                #   print("Stopping after %d levels" % j)
                #   break
            
        
        return self.sample_chain_point(validrange[0], validrange[1])
    
    def sample_chain_point(self, a, b):
        """
        Gets a point on the track between a and b (inclusive)
        """
        if self.plot: 
            for i in range(a, b+1):
                xi, vi, onpath = self.interpolate_point(i)
                plt.plot(xi[0], xi[1], '+', color='g')
        while True:
            i = np.random.randint(a, b+1)
            xi, vi, onpath = self.interpolate_point(i)
            #print("NUTS sampled point:", xi, (i, a, b))
            if not onpath:
                continue
            #if (i, xi, vi) not in self.points:
            return xi
    






    
    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=40, plot=False):
        # select starting point
        Li = None
        
        if Li is None:
            ui, vi, Li = self.find_starting_point(region, us, Ls)
            self.path.append((0, ui, vi, Li))
        
        
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
