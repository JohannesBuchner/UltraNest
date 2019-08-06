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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

def linear_steps_with_reflection(ray_origin, ray_direction, t):
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
    
    tleft = 1.0 * t
    while True:
        p, t, i = nearest_box_intersection_line(ray_origin, ray_direction, fwd=True)
        assert np.isfinite(p).all()
        assert t > 0, t
        if tleft <= t: # stopping before reaching any border
            return ray_origin + tleft * ray_direction, ray_direction
        # go to reflection point
        ray_origin = p
        assert np.isfinite(ray_origin).all(), ray_origin
        # reflect
        ray_direction = ray_direction.copy()
        ray_direction[i] *= -1
        assert np.isfinite(ray_direction).all(), ray_direction
        # reduce remaining distance
        tleft -= t

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

def extrapolate_ahead(di, xj, vj):
    """
    Make di steps of size vj from xj.
    Reflect off unit cube if necessary.
    """
    #return xj + di * vj
    return linear_steps_with_reflection(xj, vj, di)

class SamplingPath(object):
    def __init__(self, x0, v0, L0):
        self.points = []
        self.add(0, x0, v0, L0)
        self.fwd_possible = True
        self.rwd_possible = True
    
    def add(self, i, x0, v0, L0):
        self.points.append((i, x0, v0, L0))
    
    
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
        assert (vj == vk).all()
        
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
            deltai = j - i
            assert deltai > 0, ("should be extrapolating", i, j)
        else:
            j, xj, vj, Lj = min(self.points)
            deltai = j - i
            assert deltai < 0, ("should be extrapolating", i, j)
        
        return extrapolate_ahead(deltai, xj, vj)


class StepSampler(object):
    """
    Find a new point with a series of small steps
    """
    def __init__(self, x, v, gradient, is_inside, epsilon=0.1, samples=None, plot=False):
        """
        Starts a sampling track from x in direction v.
        is_inside is a function that returns true when a given point is inside the volume
        
        epsilon gives the step size in direction v.
        samples, if given, helps choose the gradient -- To be removed
        plot: if set to true, make some debug plots
        """
        self.is_inside = is_inside
        self.gradient = gradient
        self.fwd_possible = True
        self.rwd_possible = True
        self.path = SamplingPath((x, v, L))
        self.points = self.path.points
        self.epsilon = epsilon
        self.nevals = 0
        self.nreflections = 0
        self.plot = plot
        self.sample_points = samples
    
    def set_center(self, sample_points):
        """
        update "samples"; to be removed
        """
        self.sample_points = sample_points
    
    def reverse(self, reflpoint, v):
        """
        Reflect off the surface at reflpoint going in direction v
        
        returns the new direction.
        """
        normal = self.gradient(reflpoint)
        vnew = v - 2 * angle(normal, v) * normal
        isunitlength(vnew)
        return vnew
        
    
    def expand_to_step(self, i):
        """
        Run steps forward or backward to step i (can be positive or 
        negative, 0 is the starting point) 
        """
        if i > 0 and self.fwd_possible:
            starti, startx, startv = max(self.points)
            for j in range(starti, i+1):
                if not self.expand_onestep():
                    break
        elif self.rwd_possible:
            starti, startx, startv = min(self.points)
            for j in range(starti, i, -1):
                if not self.expand_onestep(fwd=False):
                    break
    
    def expand_onestep(self, fwd=True):
        """
        Make a single step forward (if fwd=True) or backwards)
        from the current state (stored in self.points)
        """
        
        if fwd:
            starti, startx, startv = max(self.points)
            sign = 1
        else:
            starti, startx, startv = min(self.points)
            sign = -1
        
        j = starti + 1*sign
        v = startv
        self.path.extrapolate(j)
        xj = startx + v * sign * self.epsilon
        #print("proposed step:", startx, "->", xj)
        self.nevals += 1
        if self.is_inside(xj):
            # Everything ok, we keep going
            #print("  inside")
            self.points.append((j, xj, v))
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
    def bisect(self, left, leftx, leftv, right, offseti):
        """
        Bisect to find first point outside
        left is the index of the point still inside
        leftx is its coordinate
        
        right is the index of the point already outside
        rightx is its coordinate
        
        offseti is an offset to the indeces to be applied before storing the point
        
        """
        # left is always inside
        # right is always outside
        while True:
            mid = (right + left) // 2
            #print("bisect: interval %d-%d-%d (+%d)" % (left,mid,right, offseti))
            if mid == left or mid == right:
                break
            midx = leftx + mid * self.epsilon * leftv
            self.nevals += 1
            if self.is_inside(midx):
                #print("   inside.  updating interval %d-%d" % (mid, right))
                self.points.append((mid+offseti, midx, leftv))
                left = mid
            else:
                #print("   outside. updating interval %d-%d" % (left, mid))
                right = mid
        return right
        
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

# tests to do:
#  - try different test functions: symmetric gaussian=circle, simplex, correlated gaussian
# samplers:
#  - simple single steps
#  - bisect
#  - simple NUTS: use simple steps to build the tree
#  - NUTS+bisect: build the tree sparsely
#  - NUTS+bisect+stopper+: if a single reflection is reversing, go all the way to the end


if __name__ == '__main__':
    def is_inside(x):
        return (x**2).sum() < 1
    
    import sys
    seed = int(sys.argv[1])
    np.random.seed(seed)
    d = 2
    start     = gen_unit_vector(d) * np.random.uniform()**(1./d)
    direction = gen_unit_vector(d)

    isunitlength(direction)

    plt.figure(figsize=(5,5))
    sampler = SliceSampler(start, direction, is_inside, plot=True)
    x = sampler.slice_step()
    print("slice sampler:", sampler.nevals)

    #sampler = StepSampler(start, direction, is_inside)
    #sampler.expand_to_step(-10)
    #sampler.expand_to_step(+10)
    #sampler.plot(color='gray')

    #sampler = BisectSampler(start, direction, is_inside)
    #sampler.expand_to_step(-10)
    #sampler.expand_to_step(+10)
    #sampler.plot()
    print()
    print("----")

    sampler = NUTSSampler(start, direction, is_inside, plot=True)
    xi = sampler.nuts_step()
    plt.plot(xi[0], xi[1], 's', mfc='None', mec='r', ms=10)
    sampler.path_plot()
    print("NUTS sampler:", sampler.nevals)

    rect = Ellipse((0,0), 1*2, 1*2, edgecolor='k', linewidth=3, facecolor='none')
    plt.gca().add_artist(rect)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig("stepsampler_test1.pdf", bbox_inches='tight')
    plt.close()

