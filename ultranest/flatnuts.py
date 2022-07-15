"""
FLATNUTS is a implementation of No-U-turn sampler 
for nested sampling assuming a flat prior space (hyper-cube u-space).

This is highly experimental. It is similar to NoGUTS and suffers from 
the same stability problems.

Directional sampling within regions.

Work in unit cube space. assume a step size.

1. starting from a live point
2. choose a random direction based on whitened space metric
3. for forward and backward direction:

  1. find distance where leaving spheres (surely outside)
  2. bisect the step that leads out of the likelihood threshold
  3. can we scatter forward?

     - if we stepped outside the unit cube, use normal to the parameter(s) we stepped out from
     - if gradient available, use it at first outside point
     - for each sphere that contains the last inside point:

       - resize so that first outside point is on the surface, get tangential vector there
         (this vector is just the difference between sphere center and last inside point)
       - compute reflection of direction vector with tangential plane
     - choose a forward reflection at random (if any)

  3.4) test if next point is inside again. If yes, continue NUTS

NUTS: 
  - alternatingly double the number of steps to the forward or backward side
  - build a tree; terminate when start and end directions are not forward any more
  - choose a end point at random out of the sequence

If the number of steps on any straight line is <10 steps, make step size smaller
If the number of steps on any straight line is >100 steps, make step size slightly bigger

Parameters:
 - Number of NUTS tracks (has to be user-tuned to ensure sufficiently independent samples; starting from 1, look when Z does not change anymore)
 - Step size (self-adjusting)

Benefit of this algorithm:
 - insensitive to step size
 - insensitive to dimensionality (sqrt scaling), better than slice sampling
 - takes advantage of region information, can accelerate low-d problems as well
Drawbacks:
 - inaccurate reflections degrade dimensionality scaling
 - more complex to implement than slice sampling

"""


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from .samplingpath import angle, extrapolate_ahead


class SingleJumper(object):
    """ Jump on step at a time. If unsuccessful, reverse direction. """
    def __init__(self, stepsampler, nsteps=0):
        self.stepsampler = stepsampler
        self.direction = +1
        assert nsteps > 0
        self.nsteps = nsteps
        self.isteps = 0
        self.currenti = 0
        self.naccepts = 0
        self.nrejects = 0
    
    def prepare_jump(self):
        target = self.currenti + self.direction
        self.stepsampler.set_nsteps(target)
    
    def check_gaps(self, gaps):
        # gaps cannot happen, because we make each jump explicitly
        pass
    # then user runs stepsampler until it is done
    
    def make_jump(self, gaps={}):
        target = self.currenti + self.direction
        pointi = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.stepsampler.points if j == target]
        accept = len(pointi) > 0
        if accept:
            self.currenti = target
            self.naccepts += 1
        else:
            pointi = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.stepsampler.points if j == self.currenti]
            # reverse
            self.direction *= -1
            self.nrejects += 1
        
        self.isteps += 1
        return pointi[0][1], pointi[0][3]


class DirectJumper(object):
    """ Jump to n steps immediately. If unsuccessful, takes rest in other direction. """
    def __init__(self, stepsampler, nsteps, log=False):
        self.stepsampler = stepsampler
        self.direction = +1
        assert nsteps > 0
        self.nsteps = nsteps
        self.isteps = 0
        self.currenti = 0
        self.naccepts = 0
        self.nrejects = 0
        self.log = log
    
    def prepare_jump(self):
        target = self.currenti + self.nsteps
        self.stepsampler.set_nsteps(target)
    
    # then user runs stepsampler until it is done
    def check_gaps(self, gaps):
        pointi = {j: (xj, Lj) for j, xj, vj, Lj in self.stepsampler.points}
        ilo, ihi = min(pointi.keys()), max(pointi.keys())
        currenti = self.currenti
        direction = self.direction
        for isteps in range(self.nsteps):
            target = currenti + direction
            accept = ilo <= target <= ihi and not gaps.get(target, False)
            if accept:
                currenti = target
                if self.log:
                    print("accepted jump %d->%d" % (self.currenti, target), 'fwd' if self.direction == 1 else 'rwd')
            else:
                # reverse
                if self.log:
                    print("rejected jump %d->%d" % (self.currenti, target), 'fwd' if self.direction == 1 else 'rwd')
                direction *= -1
        
        if self.log: print("--> %d" % currenti)
        # double-check that final point is OK:
        # if we already evaluated it, it is OK
        if currenti in pointi:
            return None, None
        
        if currenti in gaps:
            assert gaps[currenti] == False, "could not have jumped into a known gap"
            return None, None
        
        xj, vj, Lj, onpath = self.stepsampler.contourpath.interpolate(currenti)
        if Lj is not None:
            return None, None
        
        if self.log: print("    checking for gap ...")
        # otherwise ask caller to verify it and call us again with
        # gaps[i] = True if outside, gaps[i] = False if OK
        return xj, currenti
    
    def make_jump(self, gaps={}):
        pointi = {j: (xj, Lj) for j, xj, vj, Lj in self.stepsampler.points}
        ilo, ihi = min(pointi.keys()), max(pointi.keys())
        
        for self.isteps in range(self.nsteps):
            target = self.currenti + self.direction
            accept = ilo <= target <= ihi and not gaps.get(target, False)
            if accept:
                if self.log:
                    print("accepted jump %d->%d" % (self.currenti, target), 'fwd' if self.direction == 1 else 'rwd')
                self.currenti = target
                self.naccepts += 1
            else:
                if self.log:
                    print("rejected jump %d->%d" % (self.currenti, target), 'fwd' if self.direction == 1 else 'rwd')
                # reverse
                self.direction *= -1
                self.nrejects += 1
        self.isteps += 1
        
        return pointi[self.currenti]


class IntervalJumper(object):
    """ Use interval to choose final point randomly """
    def __init__(self, stepsampler, nsteps):
        self.stepsampler = stepsampler
        self.direction = +1
        assert nsteps >= 0
        self.nsteps = nsteps
        self.isteps = 0
        self.currenti = 0
        self.naccepts = 0
        self.nrejects = 0
    
    def prepare_jump(self):
        target = self.currenti + self.nsteps
        self.stepsampler.set_nsteps(target)
        self.stepsampler.set_nsteps(-target)
    
    # then user runs stepsampler until it is done
    
    def make_jump(self):
        pointi = {j: (xj, Lj) for j, xj, vj, Lj in self.stepsampler.points}
        ilo, ihi = min(pointi.keys()), max(pointi.keys())
        a, b = self.nutssampler.validrange
        nused = b - a
        # these were not used:
        ntotal = ihi - ilo
        
        # count the number of accepts and rejects
        self.naccepts = nused
        self.nrejects = ntotal - nused
        
        return None

class ClockedSimpleStepSampler(object):
    """
    Find a new point with a series of small steps
    """
    def __init__(self, contourpath, plot=False, log=False):
        """
        Starts a sampling track from x in direction v.
        is_inside is a function that returns true when a given point is inside the volume

        epsilon gives the step size in direction v.
        samples, if given, helps choose the gradient -- To be removed
        plot: if set to true, make some debug plots
        """
        self.contourpath = contourpath
        self.points = self.contourpath.points
        self.nreflections = 0
        self.nreverses = 0
        self.plot = plot
        self.log = log
        self.reset()
    
    def reset(self):
        self.goals = []
    
    def reverse(self, reflpoint, v, plot=False):
        """
        Reflect off the surface at reflpoint going in direction v
        
        returns the new direction.
        """
        normal = self.contourpath.gradient(reflpoint, plot=plot)
        if normal is None:
            #assert False
            return -v
        
        vnew = v - 2 * angle(normal, v) * normal
        if self.log: print("    new direction:", vnew)
        assert vnew.shape == v.shape, (vnew.shape, v.shape)
        assert np.isclose(norm(vnew), norm(v)), (vnew, v, norm(vnew), norm(v))
        #isunitlength(vnew)
        if plot:
            plt.plot([reflpoint[0], (-v + reflpoint)[0]], [reflpoint[1], (-v + reflpoint)[1]], '-', color='k', lw=2, alpha=0.5)
            plt.plot([reflpoint[0], (vnew + reflpoint)[0]], [reflpoint[1], (vnew + reflpoint)[1]], '-', color='k', lw=3)
        return vnew
    
    def set_nsteps(self, i):
        self.goals.insert(0, ('sample-at', i))
    
    def is_done(self):
        return self.goals == []
    
    def expand_onestep(self, fwd, transform, loglike, Lmin):
        """ Helper interface, make one step (forward fwd=True or backward fwd=False) """
        if fwd:
            starti, _, _, _ = max(self.points)
            i = starti + 1
        else:
            starti, _, _, _ = min(self.points)
            i = starti - 1
        return self.expand_to_step(i, transform, loglike, Lmin)

    def expand_to_step(self, nsteps, transform, loglike, Lmin):
        """ Helper interface, go to step nstep """
        self.set_nsteps(nsteps)
        return self.get_independent_sample(transform, loglike, Lmin)

    def get_independent_sample(self, transform, loglike, Lmin):
        """ Helper interface, call next() until a independent sample is returned """
        Llast = None
        while True:
            sample, is_independent = self.next(Llast)
            if sample is None:
                return None, None
            if is_independent:
                unew, Lnew = sample
                return unew, Lnew
            else:
                unew = sample
                xnew = transform(unew)
                Llast = loglike(xnew)
                if Llast < Lmin:
                    Llast = None


class ClockedStepSampler(ClockedSimpleStepSampler):
    """
    Find a new point with a series of small steps
    """

    def continue_sampling(self, i):
        if i > 0 and self.contourpath.samplingpath.fwd_possible \
        or i < 0 and self.contourpath.samplingpath.rwd_possible:
            # we are not done:
            self.goals.insert(0, ('expand-to', i))
            self.goals.append(('sample-at', i))
        else:
            # we are not done, but cannot reach the goal.
            # reverse. Find position from where to reverse
            if i > 0:
                starti, _, _, _ = max(self.points)
                reversei = starti + 1
            else:
                starti, _, _, _ = min(self.points)
                reversei = starti - 1
            if self.log: print("reversing at %d..." % starti)
            # how many steps are missing?
            self.nreverses += 1
            deltai = i - starti
            # request one less because one step is spent on
            # the outside try
            #if self.log: print("   %d steps to do at %d -> [from %d, delta=%d] targeting %d." % (
            #    i - starti, starti, reversei, deltai, reversei - deltai))
            # make this many steps in the other direction
            self.goals.append(('sample-at', reversei - deltai))
    
    def expand_to(self, i):
        if i > 0 and self.contourpath.samplingpath.fwd_possible:
            starti, startx, startv, _ = max(self.points)
            if i > starti:
                if self.log: print("going forward...", i, starti)
                j = starti + 1
                xj, v = self.contourpath.extrapolate(j)
                if j != i: # ultimate goal not reached yet
                    self.goals.insert(0, ('expand-to', i))
                self.goals.insert(0, ('eval-at', j, xj, v, +1))
                return xj, False
            else:
                if self.log: print("already done...", i, starti)
                # we are already done
                pass
        elif i < 0 and self.contourpath.samplingpath.rwd_possible:
            starti, startx, startv, _ = min(self.points)
            if i < starti:
                if self.log: print("going backwards...", i, starti)
                j = starti - 1
                xj, v = self.contourpath.extrapolate(j)
                if j != i: # ultimate goal not reached yet
                    self.goals.insert(0, ('expand-to', i))
                self.goals.insert(0, ('eval-at', j, xj, v, -1))
                return xj, False
            else:
                if self.log: print("already done...", i, starti)
                # we are already done
                pass
        else:
            # we are trying to go somewhere we cannot.
            # skip to other goals
            pass
    
    def eval_at(self, j, xj, v, sign, Llast):
        if Llast is not None:
            # we can go about our merry way.
            self.contourpath.add(j, xj, v, Llast)
        else:
            # We stepped outside, so now we need to reflect
            self.nreflections += 1
            if self.log: print("reflecting:", xj, v)
            if self.plot: plt.plot(xj[0], xj[1], 'xr')
            vk = self.reverse(xj, v * sign, plot=self.plot) * sign
            if self.log: print("new direction:", vk)
            xk, vk = extrapolate_ahead(sign, xj, vk, contourpath=self.contourpath)
            if self.log: print("reflection point:", xk)
            self.goals.insert(0, ('reflect-at', j, xk, vk, sign))
            return xk, False
    
    def reflect_at(self, j, xk, vk, sign, Llast):
        self.nreflections += 1
        if Llast is not None:
            # we can go about our merry way.
            self.contourpath.add(j, xk, vk, Llast)
        else:
            # we are stuck and have to give up this direction
            if self.plot: plt.plot(xk[0], xk[1], 's', mfc='None', mec='r', ms=10)
            if sign == 1:
                self.contourpath.samplingpath.fwd_possible = False
            else:
                self.contourpath.samplingpath.rwd_possible = False

    
    def next(self, Llast=None):
        """
        Run steps forward or backward to step i (can be positive or 
        negative, 0 is the starting point) 
        """
        if self.log: print("next() call", Llast)
        while self.goals:
            if self.log: print("goals: ", self.goals)
            goal = self.goals.pop(0)
            if goal[0] == 'sample-at':
                i = goal[1]
                assert Llast is None

                if not self.contourpath.samplingpath.fwd_possible \
                and  not self.contourpath.samplingpath.rwd_possible \
                and len(self.points) == 1:
                    # we are stuck and cannot move.
                    # return the starting point as our best effort
                    starti, startx, startv, startL = self.points[0]
                    if self.log: print("stuck! returning start point", starti)
                    return (startx, startL), True

                # find point
                # here we assume all intermediate points have been sampled
                pointi = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.points if j == i]
                if len(pointi) != 0:
                    # return the previously sampled point
                    _, xj, _, Lj = pointi[0]
                    if self.log: print("returning point", i)
                    return (xj, Lj), True
                
                self.continue_sampling(i)
            
            elif goal[0] == 'expand-to':
                i = goal[1]
                ret = self.expand_to(i)
                if ret is not None:
                    return ret
            
            elif goal[0] == 'eval-at':
                _, j, xj, v, sign = goal
                ret = self.eval_at(j, xj, v, sign, Llast)
                Llast = None
                if ret is not None:
                    return ret
            
            elif goal[0] == 'reflect-at':
                _, j, xk, vk, sign = goal
                self.reflect_at(j, xk, vk, sign, Llast)
                Llast = None
            
            else:
                assert False, goal
        
        return None, False

class ClockedBisectSampler(ClockedStepSampler):
    """
    Step sampler that does not require each step to be evaluated
    """
    
    def continue_sampling(self, i):
        if i > 0:
            starti, _, _, _ = max(self.points)
            #fwd = True
            inside = i < starti
            more_possible = self.contourpath.samplingpath.fwd_possible
        else:
            starti, _, _, _ = min(self.points)
            #fwd = False
            inside = starti < i
            more_possible = self.contourpath.samplingpath.rwd_possible
        
        if inside:
            # interpolate point on track
            xj, vj, Lj, onpath = self.contourpath.interpolate(i)
            if self.log: print("target is on track, returning interpolation at %d..." % i, xj, Lj)
            return (xj, Lj), True
        elif more_possible:
            # we are not done:
            self.goals.insert(0, ('expand-to', i))
            if self.log: print("not done yet, continue expanding to %d..." % i)
            self.goals.append(('sample-at', i))
        else:
            # we are not done, but cannot reach the goal.
            # reverse. Find position from where to reverse
            if i > 0:
                starti, _, _, _ = max(self.points)
                reversei = starti + 1
            else:
                starti, _, _, _ = min(self.points)
                reversei = starti - 1
            if self.log: print("reversing at %d..." % starti)
            # how many steps are missing?
            self.nreverses += 1
            deltai = i - starti
            # request one less because one step is spent on
            # the outside try
            if self.log: print("   %d steps to do at %d -> [from %d, delta=%d] targeting %d." % (
                i - starti, starti, reversei, deltai, reversei - deltai))
            # make this many steps in the other direction
            self.goals.append(('sample-at', reversei - deltai))

    def expand_to(self, j):
        # check if we already tried 
        
        if j > 0 and self.contourpath.samplingpath.fwd_possible:
            #print("going forward...", j)
            starti, startx, startv, _ = max(self.points)
            if j > starti:
                xj, v = self.contourpath.extrapolate(j)
                self.goals.insert(0, ('bisect', starti, startx, startv, None, None, None, j, xj, v, +1))
                #self.goals.append(goal)
                return xj, False
            else:
                # we are already done
                if self.log: print("done going to", j, starti)
                pass
        elif j < 0 and self.contourpath.samplingpath.rwd_possible:
            #print("going backward...", j)
            starti, startx, startv, _ = min(self.points)
            if j < starti:
                xj, v = self.contourpath.extrapolate(j)
                self.goals.insert(0, ('bisect', starti, startx, startv, None, None, None, j, xj, v, -1))
                #self.goals.append(goal)
                return xj, False
            else:
                # we are already done
                if self.log: print("done going to", j)
                pass
        else:
            # we are trying to go somewhere we cannot.
            # skip to other goals
            if self.log: print("cannot go there", j)
            pass
    
    def bisect_at(self, lefti, leftx, leftv, midi, midx, midv, righti, rightx, rightv, sign, Llast):
        # Bisect to find first point outside
        
        # left is inside (i: index, x: coordinate, v: direction)
        # mid is the middle just evaluated (if not None)
        # right is outside
        if self.log: print("bisecting ...", lefti, midi, righti)
        
        if midi is None:
            # check if right is actually outside
            if Llast is None:
                # yes it is. continue below
                pass
            else:
                # right is actually inside
                # so we successfully jumped all the way successfully
                if self.log: print("successfully went all the way in one jump!")
                self.contourpath.add(righti, rightx, rightv, Llast)
                Llast = None
                return
        else:
            # shrink interval based on previous evaluation point
            if Llast is not None:
                #print("   inside.  updating interval %d-%d" % (midi, righti))
                lefti, leftx, leftv = midi, midx, midv
                self.contourpath.add(midi, midx, midv, Llast)
                Llast = None
            else:
                #print("   outside. updating interval %d-%d" % (lefti, midi))
                righti, rightx, rightv = midi, midx, midv
        
        # we need to bisect. righti was outside
        midi = (righti + lefti) // 2
        if midi == lefti or midi == righti:
            # we are done bisecting. right is the first point outside
            if self.log: print("  bisecting gave reflection point", righti, rightx, rightv)
            if self.plot: plt.plot(rightx[0], rightx[1], 'xr')
            # compute reflected direction
            vk = self.reverse(rightx, rightv * sign, plot=self.plot) * sign
            if self.log: print("  reversing there", rightv)
            # go from reflection point one step in that direction
            # that is our new point
            xk, vk = extrapolate_ahead(sign, rightx, vk, contourpath=self.contourpath)
            if self.log: print("  making one step from", rightx, rightv, '-->', xk, vk)
            self.nreflections += 1
            if self.log: print("  trying new point,", xk)
            self.goals.insert(0, ('reflect-at', righti, xk, vk, sign))
            return xk, False
        else:
            if self.log: print("  continue bisect at", midi)
            # we should evaluate the middle point
            midx, midv = extrapolate_ahead(midi - lefti, leftx, leftv, contourpath=self.contourpath)
            # continue bisecting
            self.goals.insert(0, ('bisect', lefti, leftx, leftv, midi, midx, midv, righti, rightx, rightv, sign))
            return midx, False
    
    
    def next(self, Llast=None):
        """
        Run steps forward or backward to step i (can be positive or 
        negative, 0 is the starting point) 
        """
        if self.log: print()
        if self.log: print("next() call", Llast)
        while self.goals:
            if self.log: print("goals: ", self.goals)
            goal = self.goals.pop(0)

            if goal[0] == 'sample-at':
                i = goal[1]
                assert Llast is None

                if not self.contourpath.samplingpath.fwd_possible and not self.contourpath.samplingpath.rwd_possible \
                    and len(self.points) == 1:
                    # we are stuck and cannot move.
                    # return the starting point as our best effort
                    if self.log: print("stuck! returning start point.")
                    starti, startx, startv, startL = self.points[0]
                    return (startx, startL), True

                # check if point already sampled
                pointi = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.points if j == i]

                if len(pointi) == 1:
                    # return the previously sampled point
                    _, xj, _, Lj = pointi[0]
                    return (xj, Lj), True
                
                self.continue_sampling(i)
            
            elif goal[0] == 'expand-to':
                ret = self.expand_to(goal[1])
                if ret is not None:
                    return ret

            elif goal[0] == 'bisect':
                _, lefti, leftx, leftv, midi, midx, midv, righti, rightx, rightv, sign = goal
                ret = self.bisect_at(lefti, leftx, leftv, midi, midx, midv, righti, rightx, rightv, sign, Llast)
                Llast = None
                if ret is not None:
                    return ret
            
            elif goal[0] == 'reflect-at':
                _, j, xk, vk, sign = goal
                self.reflect_at(j, xk, vk, sign, Llast)
                Llast = None
            else:
                assert False, goal
            
        return None, False

class ClockedNUTSSampler(ClockedBisectSampler):
    """
    No-U-turn sampler (NUTS) on flat surfaces.
    
    """
    
    def reset(self):
        self.goals = []
        self.left_state = self.points[0][:3]
        self.right_state = self.points[0][:3]
        self.left_warmed_up = False
        self.right_warmed_up = False
        self.tree_built = False
        self.validrange = (0, 0)
        self.tree_depth = 0
        self.current_direction = np.random.randint(2) == 1
    
    def next(self, Llast=None):
        """
        Alternatingly doubles the number of steps to forward and backward 
        direction (which may include reflections, see StepSampler and
        BisectSampler).
        When track returns (start and end of tree point toward each other),
        terminates and returns a random point on that track.
        """
        while not self.tree_built:
            if self.log: print("continue building tree")
            rwd = self.current_direction
            
            if True or self.tree_depth > 7:
                print("NUTS step: tree depth %d, %s" % (self.tree_depth, "rwd" if rwd else "fwd"))
            
            
            # make sure the path is prepared for the desired tree
            if rwd:
                goal = ('expand-to', self.left_state[0] - 2**self.tree_depth)
            else:
                goal = ('expand-to', self.right_state[0] + 2**self.tree_depth)
            
            if goal not in self.goals:
                self.goals.append(goal)
            
            # work down any open tasks
            while self.goals:
                sample, is_independent = ClockedBisectSampler.next(self, Llast=Llast)
                Llast = None
                if sample is not None:
                    return sample, is_independent
            
            # now check if terminating
            if rwd:
                self.left_state, _, newrange, newstop = self.build_tree(self.left_state, self.tree_depth, rwd=rwd)
            else:   
                _, self.right_state, newrange, newstop = self.build_tree(self.right_state, self.tree_depth, rwd=rwd)
            
            if not newstop:
                self.validrange = (min(self.validrange[0], newrange[0]), max(self.validrange[1], newrange[1]))
                print("  new NUTS range: %d..%d" % (self.validrange[0], self.validrange[1]))
            
            ileft, xleft, vleft = self.left_state
            iright, xright, vright = self.right_state
            if self.plot: plt.plot([xleft[0], xright[0]], [xleft[1] + (self.tree_depth+1)*0.02, xright[1] + (self.tree_depth+1)*0.02], '--')
            #if j > 5:
            #   print("  first-to-last arrow", ileft, iright, xleft, xright, xright-xleft, " velocities:", vright, vleft)
            #   print("  stopping criteria: ", newstop, angle(xright-xleft, vleft), angle(xright-xleft, vright))
            
            # avoid U-turns:
            stop = newstop or angle(xright - xleft, vleft) <= 0 or angle(xright - xleft, vright) <= 0
            
            # stop when we cannot continue in any direction
            stop = stop and (self.contourpath.samplingpath.fwd_possible or self.contourpath.samplingpath.rwd_possible)
            
            if stop:
                self.tree_built = True
            else:
                self.tree_depth = self.tree_depth + 1
                self.current_direction = np.random.randint(2) == 1
        
        # Tree was built, we only need to sample from it
        print("sampling between", self.validrange)
        return self.sample_chain_point(self.validrange[0], self.validrange[1])
    
    def sample_chain_point(self, a, b):
        """
        Gets a point on the track between a and b (inclusive)
        returns tuple ((point coordinates, likelihood), is_independent)
            where is_independent is always True
        
        """
        if self.plot: 
            for i in range(a, b+1):
                xi, vi, Li, onpath = self.contourpath.interpolate(i)
                plt.plot(xi[0], xi[1], '_ ', color='b', ms=10, mew=2)
        
        while True:
            i = np.random.randint(a, b+1)
            xi, vi, Li, onpath = self.contourpath.interpolate(i)
            if not onpath: 
                continue
            return (xi, Li), True
    
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
            #print("  build_tree@%d" % i, rwd, self.contourpath.samplingpath.fwd_possible, self.contourpath.samplingpath.rwd_possible)
            xi, vi, _, _ = self.contourpath.interpolate(i)
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
