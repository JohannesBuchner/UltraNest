"""

FLATNUTS
=========

Directional sampling within regions.

Work in unit cube space. assume a step size
1) starting from a live point
2) choose a random direction based on whitened space metric
3) for forward and backward direction:
  3.1) find distance where leaving spheres (surely outside)
  3.2) bisect the step that leads out of the likelihood threshold
  3.3) can we scatter forward?
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

def nearest_box_intersection_line(ray_origin, ray_direction, fwd=True):
    """
    Compute intersection of a line (ray) and a unit box (0:1 in all axes)
    
    Based on
    http://www.iquilezles.org/www/articles/intersectors/intersectors.htm
    
    ray_origin: starting point of line
    ray_direction: line direction vector
    
    returns: p, t, i
    p: intersection point
    t: intersection point distance from ray_origin in units in ray_direction
    i: axes which change direction at pN
    
    To continue forward traversing at the reflection point use:
    
    while True:
        # update current point x
        x, _, i = box_line_intersection(x, v)
        # change direction
        v[i] *= -1
    
    """
    
    # make sure ray starts inside the box
    
    assert (ray_origin >= 0).all(), ray_origin
    assert (ray_origin <= 1).all(), ray_origin
    
    # step size
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1./ ray_direction
        n = m * (ray_origin - 0.5)
        k = np.abs(m) * 0.5
        # line coordinates of intersection
        # find first intersecting coordinate
        if fwd:
            t2 = -n + k
            tF = np.nanmin(t2)
            iF = np.where(t2 == tF)[0]
        else:
            t1 = -n - k
            tF = np.nanmax(t1)
            iF = np.where(t1 == tF)[0]
    
    pF = ray_origin + ray_direction * tF
    eps = 1e-6
    assert (pF >= -eps).all(), pF
    assert (pF <= 1 + eps).all(), pF
    pF[pF < 0] = 0
    pF[pF > 1] = 1
    return pF, tF, iF

def box_line_intersection(ray_origin, ray_direction):
    """ return intersections of a line with the unit cube, in both sides """
    pF, tF, iF = nearest_box_intersection_line(ray_origin, ray_direction, fwd=True)
    pN, tN, iN = nearest_box_intersection_line(ray_origin, ray_direction, fwd=False)
    if tN > tF or tF < 0:
        assert False, "no intersection"
    return (pN, tN, iN), (pF, tF, iF)

def linear_steps_with_reflection(ray_origin, ray_direction, t, wrapped_dims=None):
    """ go t steps in direction ray_direction from ray_origin,
    but reflect off the unit cube if encountered. In any case, 
    the distance should be t * ray_direction.
    
    Returns (new_point, new_direction)
    """
    if t == 0:
        return ray_origin, ray_direction
    if t < 0:
        new_point, new_direction = linear_steps_with_reflection(ray_origin, -ray_direction, -t)
        return new_point, -new_direction
    
    if wrapped_dims is not None:
        reflected = np.zeros(len(ray_origin), dtype=bool)
    
    tleft = 1.0 * t
    while True:
        p, t, i = nearest_box_intersection_line(ray_origin, ray_direction, fwd=True)
        #print(p, t, i, ray_origin, ray_direction)
        assert np.isfinite(p).all()
        assert t >= 0, t
        if tleft <= t: # stopping before reaching any border
            assert np.all(ray_origin + tleft * ray_direction >= 0), (ray_origin, tleft, ray_direction)
            assert np.all(ray_origin + tleft * ray_direction <= 1), (ray_origin, tleft, ray_direction)
            return ray_origin + tleft * ray_direction, ray_direction
        # go to reflection point
        ray_origin = p
        assert np.isfinite(ray_origin).all(), ray_origin
        # reflect
        ray_direction = ray_direction.copy()
        if wrapped_dims is None:
            ray_direction[i] *= -1
        else:
            # if we already once bumped into that (wrapped) axis, 
            # do not continue but return this as end point
            if np.logical_and(reflected[i], wrapped_dims[i]).any():
                return ray_origin, ray_direction
            
            # note which axes we already flipped
            reflected[i] = True
            
            # in wrapped axes, we can keep going. Otherwise, reflects
            ray_direction[i] *= np.where(wrapped_dims[i], 1, -1)
            
            # in the i axes, we should wrap the coordinates
            assert np.logical_or(np.isclose(ray_origin[i], 1), np.isclose(ray_origin[i], 0)).all(), ray_origin[i]
            ray_origin[i] = np.where(wrapped_dims[i], 1 - ray_origin[i], ray_origin[i])
        
        assert np.isfinite(ray_direction).all(), ray_direction
        # reduce remaining distance
        tleft -= t

def get_sphere_tangents(sphere_center, edge_point):
    """ Assume a sphere centered at sphere_center with radius 
    so that edge_point is on the surface. At edge_point, in 
    which direction does the normal vector point? 
    
    Returns vector pointing to the sphere center.
    """
    arrow = sphere_center - edge_point
    return arrow / norm(arrow, axis=1).reshape((-1, 1))
    
def get_sphere_tangent(sphere_center, edge_point):
    """ Assume a sphere centered at sphere_center with radius 
    so that edge_point is on the surface. At edge_point, in 
    which direction does the normal vector point? 
    
    Returns vector pointing to the sphere center.
    """
    arrow = sphere_center - edge_point
    return arrow / norm(arrow)

def reflect(v, normal):
    """ reflect vector v off a normal vector, return new direction vector """
    return v - 2 * (normal * v).sum() * normal

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

def isunitlength(vec):
    """
    Verifies that vec is of unit length.
    """
    assert np.isclose(norm(vec), 1), norm(vec)

def angle(a, b):
    """
    Compute the dot product between vectors a and b
    The arccos of it would give an actual angle.
    """
    #anorm = (a**2).sum()**0.5
    #bnorm = (b**2).sum()**0.5
    return (a*b).sum() # / anorm / bnorm

def extrapolate_ahead(di, xj, vj):
    """
    Make di steps of size vj from xj.
    Reflect off unit cube if necessary.
    """
    assert di == int(di)
    return linear_steps_with_reflection(xj, vj, di)

class SamplingPath(object):
    def __init__(self, x0, v0, L0):
        self.reset(x0, v0, L0)
    
    def add(self, i, x0, v0, L0):
        assert L0 is not None
        self.points.append((i, x0, v0, L0))
    
    def reset(self, x0, v0, L0):
        self.points = []
        self.add(0, x0, v0, L0)
        self.fwd_possible = True
        self.rwd_possible = True
    
    def plot(self, **kwargs):
        x = np.array([x for i, x, v, L in sorted(self.points)])
        p, = plt.plot(x[:,0], x[:,1], 'o ', **kwargs)
        ilo, _, _, _ = min(self.points)
        ihi, _, _, _ = max(self.points)
        x = np.array([self.interpolate(i)[0] for i in range(ilo, ihi+1)])
        kwargs['color'] = p.get_color()
        plt.plot(x[:,0], x[:,1], 'o-', ms=4, mfc='None', **kwargs)
    
    def interpolate(self, i):
        """
        Interpolate a point on the path
        
        Given our sparsely sampled track (stored in .points),
        potentially with reflections, 
        extract the corrdinates of the point with index i.
        That point may not have been evaluated.
        """
        
        points_before = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.points if j <= i]
        points_after  = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.points if j >= i]
        
        # check if the point after is really after i
        if len(points_after) == 0 and not self.fwd_possible:
            # the path cannot continue, and i does not exist.
            #print("    interpolate_point %d: the path cannot continue fwd, and i does not exist." % i)
            j, xj, vj, Lj = max(points_before)
            return xj, vj, Lj, False
        
        # check if the point before is really before i
        if len(points_before) == 0 and not self.rwd_possible:
            # the path cannot continue, and i does not exist.
            k, xk, vk, Lk = min(points_after)
            #print("    interpolate_point %d: the path cannot continue rwd, and i does not exist." % i)
            return xk, vk, Lk, False
        
        if len(points_before) == 0 or len(points_after) == 0:
            #return None, None, None, False
            raise KeyError("can not extrapolate outside path")
        
        j, xj, vj, Lj = max(points_before)
        k, xk, vk, Lk = min(points_after)
        
        #print("    interpolate_point %d between %d-%d" % (i, j, k))
        if j == i: # we have this exact point in the chain
            return xj, vj, Lj, True
        
        assert not k == i # otherwise the above would be true too
        
        # expand_to_step explores each reflection in detail, so
        # any points with change in v should have j == i
        # therefore we can assume:
        # assert (vj == vk).all()
        # this ^ is not true, because reflections on the unit cube can
        # occur, and change v without requiring a intermediate point.
        
        # j....i....k
        xl1, vj1 = extrapolate_ahead(i - j, xj, vj)
        xl2, vj2 = extrapolate_ahead(i - k, xk, vk)
        assert np.allclose(xl1, xl2), (xl1, xl2, i, j, k, xj, vj, xk, vk)
        assert np.allclose(vj1, vj2), (xl1, vj1, xl2, vj2, i, j, k, xj, vj, xk, vk)
        xl = xl1
        
        #xl = interpolate_between_two_points(i, xj, j, xk, k)
        # the new point is then just a linear interpolation
        #w = (i - k) * 1. / (j - k)
        #xl = xj * w + (1 - w) * xk
        
        return xl, vj, None, True
        
    def extrapolate(self, i):
        """
        Advance beyond the current path, extrapolate from the end point.
        
        i: index on path.
        
        returns coordinates of the new point.
        """
        if i >= 0:
            j, xj, vj, Lj = max(self.points)
            deltai = i - j
            assert deltai > 0, ("should be extrapolating", i, j)
        else:
            j, xj, vj, Lj = min(self.points)
            deltai = i - j
            assert deltai < 0, ("should be extrapolating", i, j)
        
        newpoint = extrapolate_ahead(deltai, xj, vj)
        #print((deltai, j, xj, vj), newpoint)
        return newpoint



class ContourSamplingPath(object):
    def __init__(self, samplingpath, region):
        self.samplingpath = samplingpath
        self.points = self.samplingpath.points
        self.region = region
    
    def add(self, i, x, v, L):
        self.samplingpath.add(i, x, v, L)
    
    def gradient(self, reflpoint, v, plot=False):
        """
        reflpoint: 
            point outside the likelihood contour, reflect there
        v:
            previous direction vector
        return:
            gradient vector (normal of ellipsoid)
        
        Finds spheres enclosing the reflpoint, and chooses their mean
        as the direction to go towards. If no spheres enclose the 
        reflpoint, use nearest sphere.
        
        v is not used, because that would break detailed balance.
        
        Considerations:
           - in low-d, we want to focus on nearby live point spheres
             The border traced out is fairly accurate, at least in the
             normal away from the inside.
             
           - in high-d, reflpoint is contained by all live points,
             and none of them point to the center well. Because the
             sampling is poor, the "region center" position
             will be very stochastic.
        """
        if plot:
            plt.plot(reflpoint[0], reflpoint[1], '+ ', color='k', ms=10)
        
        # check which the reflections the ellipses would make
        region = self.region
        bpts = region.transformLayer.transform(reflpoint.reshape((1,-1)))
        dist = ((bpts - region.unormed)**2).sum(axis=1)
        nearby = dist < region.maxradiussq
        assert nearby.shape == (len(region.unormed),), (nearby.shape, len(region.unormed))
        if not nearby.any():
            nearby = dist == dist.min()
        sphere_centers = region.u[nearby,:]

        tsphere_centers = region.unormed[nearby,:]
        nlive, ndim = region.unormed.shape
        assert tsphere_centers.shape[1] == ndim, (tsphere_centers.shape, ndim)
        
        # choose mean among those points
        tsphere_center = tsphere_centers.mean(axis=0)
        assert tsphere_center.shape == (ndim,), (tsphere_center.shape, ndim)
        tt = get_sphere_tangent(tsphere_center, bpts.flatten())
        assert tt.shape == tsphere_center.shape, (tt.shape, tsphere_center.shape)
        
        # convert back to u space
        sphere_center = region.transformLayer.untransform(tsphere_center)
        t = region.transformLayer.untransform(tt * 1e-3 + tsphere_center) - sphere_center
        
        if plot:
            tt_all = get_sphere_tangent(tsphere_centers, bpts)
            t_all = region.transformLayer.untransform(tt_all * 1e-3 + tsphere_centers) - sphere_centers
            plt.plot(sphere_centers[:,0], sphere_centers[:,1], 'o ', mfc='None', mec='b', ms=10, mew=1)
            for si, ti in zip(sphere_centers, t_all):
                plt.plot([si[0], ti[0] * 1000 + si[0]], [si[1], ti[1] * 1000 + si[1]], color='gray', alpha=0.5)
            plt.plot(sphere_center[0], sphere_center[1], '^ ', mfc='None', mec='g', ms=10, mew=3)
            plt.plot([sphere_center[0], t[0] * 1000 + sphere_center[0]], [sphere_center[1], t[1] * 1000 + sphere_center[1]], color='gray')

        # compute new vector
        normal = t / norm(t)
        isunitlength(normal)
        assert normal.shape == t.shape, (normal.shape, t.shape)
        
        return normal
        
class ClockedStepSampler(object):
    """
    Find a new point with a series of small steps
    """
    def __init__(self, contourpath, epsilon=0.1, plot=False):
        """
        Starts a sampling track from x in direction v.
        is_inside is a function that returns true when a given point is inside the volume
        
        epsilon gives the step size in direction v.
        samples, if given, helps choose the gradient -- To be removed
        plot: if set to true, make some debug plots
        """
        self.epsilon_too_large = False
        self.contourpath = contourpath
        self.points = self.contourpath.points
        self.epsilon = epsilon
        self.nevals = 0
        self.nreflections = 0
        self.plot = plot
        self.reset()
    
    def reset(self):
        self.goals = []
    
    def reverse(self, reflpoint, v, plot=False):
        """
        Reflect off the surface at reflpoint going in direction v
        
        returns the new direction.
        """
        normal = self.contourpath.gradient(reflpoint, v, plot=plot)
        if normal is None:
            #assert False
            return -v
        
        vnew = v - 2 * angle(normal, v) * normal
        print("    new direction:", vnew)
        assert vnew.shape == v.shape, (vnew.shape, v.shape)
        assert np.isclose(norm(vnew), norm(v)), (vnew, v, norm(vnew), norm(v))
        #isunitlength(vnew)
        if plot:
            plt.plot([reflpoint[0], (-v + reflpoint)[0]], [reflpoint[1], (-v + reflpoint)[1]], '-', color='k', lw=2, alpha=0.5)
            plt.plot([reflpoint[0], (vnew + reflpoint)[0]], [reflpoint[1], (vnew + reflpoint)[1]], '-', color='k', lw=3)
        return vnew
    
    def set_nsteps(self, i):
        self.goals.insert(0, ('sample-at', i))
    
    def next(self, Llast=None):
        """
        Run steps forward or backward to step i (can be positive or 
        negative, 0 is the starting point) 
        """
        print("next() call", Llast)
        while self.goals:
            print("goals: ", self.goals)
            goal = self.goals.pop(0)
            if goal[0] == 'sample-at':
                i = goal[1]
                assert Llast is None
                # find point
                # here we assume all intermediate points have been sampled
                pointi = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.points if j == i]
                if len(pointi) == 0:
                    if i > 0 and self.contourpath.samplingpath.fwd_possible \
                    or i < 0 and self.contourpath.samplingpath.rwd_possible:
                        # we are not done:
                        self.goals.insert(0, ('expand-to', i))
                        self.goals.append(goal)
                        continue
                    else:
                        # we are not done, but cannot reach the goal.
                        # reverse. Find position from where to reverse
                        if i > 0:
                            starti, _, _, _ = max(self.points)
                        else:
                            starti, _, _, _ = min(self.points)
                        print("reversing at %d..." % starti)
                        # how many steps are missing?
                        deltai = i - starti
                        print("   %d steps to do at %d -> targeting %d." % (deltai, starti, starti - deltai))
                        # make this many steps in the other direction
                        self.goals.append(('sample-at', starti - deltai))
                        continue
                else:
                    # return the previously sampled point
                    _, xj, _, Lj = pointi[0]
                    return (xj, Lj), True
            
            elif goal[0] == 'expand-to':
                i = goal[1]
                if i > 0 and self.contourpath.samplingpath.fwd_possible:
                    starti, startx, startv, _ = max(self.points)
                    if i > starti:
                        print("going forward...", i, starti)
                        j = starti + 1
                        xj, v = self.contourpath.samplingpath.extrapolate(j)
                        if j != i: # ultimate goal not reached yet
                            self.goals.insert(0, goal)
                        self.goals.insert(0, ('eval-at', j, xj, v, +1))
                        return xj, False
                    else:
                        print("already done...", i, starti)
                        # we are already done
                        pass
                elif i < 0 and self.contourpath.samplingpath.rwd_possible:
                    starti, startx, startv, _ = min(self.points)
                    if i < starti:
                        print("going backwards...", i, starti)
                        j = starti - 1
                        xj, v = self.contourpath.samplingpath.extrapolate(j)
                        if j != i: # ultimate goal not reached yet
                            self.goals.insert(0, goal)
                        self.goals.insert(0, ('eval-at', j, xj, v, -1))
                        return xj, False
                    else:
                        print("already done...", i, starti)
                        # we are already done
                        pass
                else:
                    # we are trying to go somewhere we cannot.
                    # skip to other goals
                    pass
            
            elif goal[0] == 'eval-at':
                _, j, xj, v, sign = goal
                if Llast is not None:
                    # we can go about our merry way.
                    self.contourpath.add(j, xj, v, Llast)
                    Llast = None
                    continue
                else:
                    # We stepped outside, so now we need to reflect
                    if self.plot: plt.plot(xj[0], xj[1], 'xr')
                    print("reflecting:", xj, v)
                    vk = self.reverse(xj, v * sign, plot=self.plot) * sign
                    print("new direction:", vk)
                    xk, vk = extrapolate_ahead(sign, xj, vk)
                    print("reflection point:", xk)
                    self.goals.insert(0, ('reflect-at', j, xk, vk, sign))
                    return xk, False
            
            elif goal[0] == 'reflect-at':
                _, j, xk, vk, sign = goal
                if Llast is not None:
                    # we can go about our merry way.
                    self.contourpath.add(j, xk, vk, Llast)
                    Llast = None
                    continue
                else:
                    # we are stuck and have to give up this direction
                    if self.plot: plt.plot(xk[0], xk[1], 's', mfc='None', mec='r', ms=10)
                    if sign == 1:
                        self.contourpath.samplingpath.fwd_possible = False
                    else:
                        self.contourpath.samplingpath.rwd_possible = False
                    continue
            else:
                assert False, goal
        
        return None, False

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
    
class ClockedBisectSampler(ClockedStepSampler):
    """
    Step sampler that does not require each step to be evaluated
    """
    
    def next(self, Llast=None):
        """
        Run steps forward or backward to step i (can be positive or 
        negative, 0 is the starting point) 
        """
        print()
        print("next() call", Llast)
        while self.goals:
            print("goals: ", self.goals)
            goal = self.goals.pop(0)

            if goal[0] == 'sample-at':
                i = goal[1]
                assert Llast is None
                # check if point already sampled
                pointi = [(j, xj, vj, Lj) for j, xj, vj, Lj in self.points if j == i]

                if len(pointi) == 0:
                    if i > 0:
                        starti, _, _, _ = max(self.points)
                        fwd = True
                        inside = i < starti
                        more_possible = self.contourpath.samplingpath.fwd_possible
                    else:
                        starti, _, _, _ = min(self.points)
                        fwd = False
                        inside = starti < i
                        more_possible = self.contourpath.samplingpath.rwd_possible
                    
                    if inside:
                        # interpolate point on track
                        xj, vj, Lj, onpath = self.contourpath.samplingpath.interpolate(i)
                        return (xj, Lj), True
                    elif more_possible:
                        # we are not done:
                        self.goals.insert(0, ('expand-to', i))
                        self.goals.append(goal)
                        continue
                    elif not fwd and self.contourpath.samplingpath.fwd_possible \
                    or       fwd and self.contourpath.samplingpath.rwd_possible:
                        #print("reversing...", i, self.contourpath.samplingpath.rwd_possible,
                        #    self.contourpath.samplingpath.fwd_possible)
                        # we are not done, but cannot reach the goal.
                        # reverse. Find position from where to reverse
                        if i > 0:
                            starti, _, _, _ = max(self.points)
                        else:
                            starti, _, _, _ = min(self.points)
                        print("reversing at %d..." % starti)
                        # how many steps are missing?
                        deltai = i - starti
                        #print("   %d steps to do at %d -> targeting %d." % (deltai, starti, starti - deltai))
                        # make this many steps in the other direction
                        #print("reversing...", i, starti, deltai, starti - deltai)
                        self.goals.append(('sample-at', starti - deltai))
                        continue
                    else:
                        # we are not done, but cannot reach the goal.
                        # move on to next goal (if any)
                        continue
                        
                else:
                    # return the previously sampled point
                    _, xj, _, Lj = pointi[0]
                    return (xj, Lj), True
            
            elif goal[0] == 'expand-to':
                j = goal[1]
                # check if we already tried 
                
                if j > 0 and self.contourpath.samplingpath.fwd_possible:
                    #print("going forward...", j)
                    starti, startx, startv, _ = max(self.points)
                    if j > starti:
                        xj, v = self.contourpath.samplingpath.extrapolate(j)
                        self.goals.insert(0, ('bisect', starti, startx, startv, None, None, None, j, xj, v, +1))
                        #self.goals.append(goal)
                        return xj, False
                    else:
                        # we are already done
                        print("done going to", j, starti)
                        pass
                elif j < 0 and self.contourpath.samplingpath.rwd_possible:
                    #print("going backward...", j)
                    starti, startx, startv, _ = min(self.points)
                    if j < starti:
                        xj, v = self.contourpath.samplingpath.extrapolate(j)
                        self.goals.insert(0, ('bisect', starti, startx, startv, None, None, None, j, xj, v, -1))
                        #self.goals.append(goal)
                        return xj, False
                    else:
                        # we are already done
                        print("done going to", j)
                        pass
                else:
                    # we are trying to go somewhere we cannot.
                    # skip to other goals
                    print("cannot go there", j)
                    pass

            elif goal[0] == 'bisect':
                # Bisect to find first point outside
                
                # left is inside (i: index, x: coordinate, v: direction)
                # mid is the middle just evaluated (if not None)
                # right is outside
                _, lefti, leftx, leftv, midi, midx, midv, righti, rightx, rightv, sign = goal
                print("bisecting ...", lefti, midi, righti)
                
                if midi is None:
                    # check if right is actually outside
                    if Llast is None:
                        # yes it is. continue below
                        pass
                    else:
                        # right is actually inside
                        # so we successfully jumped all the way successfully
                        print("successfully went all the way in one jump!")
                        self.contourpath.add(righti, rightx, rightv, Llast)
                        Llast = None
                        continue
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
                    print("  bisecting gave reflection point", righti, rightx, rightv)
                    if self.plot: plt.plot(rightx[0], rightx[1], 'xr')
                    # compute reflected direction
                    vk = self.reverse(rightx, rightv * sign, plot=self.plot) * sign
                    print("  reversing there", rightv)
                    # go from reflection point one step in that direction
                    # that is our new point
                    xk, vk = extrapolate_ahead(sign, rightx, vk)
                    print("  making one step from", rightx, rightv, '-->', xk, vk)
                    self.nreflections += 1
                    print("  trying new point,", xk)
                    self.goals.insert(0, ('reflect-at', righti, xk, vk, sign))
                    return xk, False
                else:
                    print("  continue bisect at", midi)
                    # we should evaluate the middle point
                    midx, midv = extrapolate_ahead(midi - lefti, leftx, leftv)
                    # continue bisecting
                    self.goals.insert(0, ('bisect', lefti, leftx, leftv, midi, midx, midv, righti, rightx, rightv, sign))
                    return midx, False
            
            elif goal[0] == 'reflect-at':
                _, j, xk, vk, sign = goal
                
                if Llast is not None:
                    # we can go about our merry way.
                    #print("  inside", j)
                    self.contourpath.add(j, xk, vk, Llast)
                    Llast = None
                    continue
                else:
                    #print("  reflection failed", j, xk, vk)
                    # we are stuck and have to give up this direction
                    if self.plot: plt.plot(xk[0], xk[1], 's', mfc='None', mec='r', ms=10)
                    if sign == 1:
                        self.contourpath.samplingpath.fwd_possible = False
                    else:
                        self.contourpath.samplingpath.rwd_possible = False
                    Llast = None
                    continue
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
        """
        if not self.left_warmed_up:
            print("warming up left.")
            goal = ('expand-to', -10)
            if goal not in self.goals:
                self.goals.append(goal)
            while self.goals:
                if self.goals[0][0] == 'reflect-at':
                    break
                sample, is_independent = ClockedBisectSampler.next(self, Llast=Llast)
                Llast = None
                if sample is not None:
                    return sample, is_independent
            print("left warmed up.")
            self.left_warmed_up = True
            self.goals = []
        
        if not self.right_warmed_up:
            print("warming up right.")
            goal = ('expand-to', 10)
            if goal not in self.goals:
                self.goals.append(goal)
            while self.goals:
                if self.goals[0][0] == 'reflect-at':
                    break
                sample, is_independent = ClockedBisectSampler.next(self, Llast=Llast)
                Llast = None
                if sample is not None:
                    return sample, is_independent
            print("right warmed up.")
            self.right_warmed_up = True
            self.goals = []
        """
        
        while not self.tree_built:
            print("continue building tree")
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
                xi, vi, Li, onpath = self.contourpath.samplingpath.interpolate(i)
                plt.plot(xi[0], xi[1], '_ ', color='b', ms=10, mew=2)
        
        while True:
            i = np.random.randint(a, b+1)
            xi, vi, Li, onpath = self.contourpath.samplingpath.interpolate(i)
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
            xi, vi, _, _ = self.contourpath.samplingpath.interpolate(i)
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

