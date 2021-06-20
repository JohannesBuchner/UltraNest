"""Sparsely sampled, virtual sampling path.

Supports reflections at unit cube boundaries, and regions.
"""


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def nearest_box_intersection_line(ray_origin, ray_direction, fwd=True):
    r"""Compute intersection of a line (ray) and a unit box (0:1 in all axes).

    Based on
    http://www.iquilezles.org/www/articles/intersectors/intersectors.htm

    To continue forward traversing at the reflection point use::

        while True:
            # update current point x
            x, _, i = box_line_intersection(x, v)
            # change direction
            v[i] *= -1

    Parameters
    -----------
    ray_origin: vector
        starting point of line
    ray_direction: vector
        line direction vector

    Returns
    --------
    p: vector
        intersection point
    t: float
        intersection point distance from ray\_origin in units in ray\_direction
    i: int
        axes which change direction at pN

    """
    # make sure ray starts inside the box
    assert (ray_origin >= 0).all(), ray_origin
    assert (ray_origin <= 1).all(), ray_origin
    assert ((ray_direction**2).sum()**0.5 > 1e-200).all(), ray_direction

    # step size
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1. / ray_direction
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
    assert (pF >= -eps).all(), (pF, ray_origin, ray_direction)
    assert (pF <= 1 + eps).all(), (pF, ray_origin, ray_direction)
    pF[pF < 0] = 0
    pF[pF > 1] = 1
    return pF, tF, iF


def box_line_intersection(ray_origin, ray_direction):
    """Find intersections of a line with the unit cube, in both sides.

    Returns
    --------
    left: nearest_box_intersection_line return value
        from negative direction
    right: nearest_box_intersection_line return value
        from positive direction

    """
    pF, tF, iF = nearest_box_intersection_line(ray_origin, ray_direction, fwd=True)
    pN, tN, iN = nearest_box_intersection_line(ray_origin, ray_direction, fwd=False)
    if tN > tF or tF < 0:
        assert False, "no intersection"
    return (pN, tN, iN), (pF, tF, iF)


def linear_steps_with_reflection(ray_origin, ray_direction, t, wrapped_dims=None):
    """Go `t` steps in direction `ray_direction` from `ray_origin`.

    Reflect off the unit cube if encountered, respecting wrapped dimensions.
    In any case, the distance should be ``t * ray_direction``.

    Returns
    --------
    new_point: vector
        end point
    new_direction: vector
        new direction.

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
        # print(p, t, i, ray_origin, ray_direction)
        assert np.isfinite(p).all()
        assert t >= 0, t
        if tleft <= t:  # stopping before reaching any border
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


def get_sphere_tangent(sphere_center, edge_point):
    """Compute tangent at sphere surface point.

    Assume a sphere centered at sphere_center with radius
    so that edge_point is on the surface. At edge_point, in
    which direction does the normal vector point?

    Returns
    --------
    tangent: vector
        vector pointing to the sphere center.

    """
    arrow = sphere_center - edge_point
    return arrow / norm(arrow)


def get_sphere_tangents(sphere_center, edge_point):
    """Compute tangent at sphere surface point.

    Assume a sphere centered at sphere_center with radius
    so that edge_point is on the surface. At edge_point, in
    which direction does the normal vector point?

    This function is vectorized and handles arrays of arguments.

    Returns
    --------
    tangent: vector
        vector pointing to the sphere center.

    """
    arrow = sphere_center - edge_point
    return arrow / norm(arrow, axis=1).reshape((-1, 1))


def reflect(v, normal):
    """Reflect vector ``v`` off a ``normal`` vector, return new direction vector."""
    return v - 2 * (normal * v).sum() * normal


def distances(l, o, r=1):
    """Compute sphere-line intersection.

    Parameters
    -----------
    l: vector
        direction vector (line starts at 0)
    o: vector
        center of sphere (coordinate vector)
    r: float
        radius of sphere

    Returns
    --------
    tpos, tneg: floats
        the positive and negative coordinate along the `l` vector where `r` is intersected.
        If no intersection, throws AssertError.

    """
    loc = (l * o).sum()
    osqrnorm = (o**2).sum()
    # print(loc.shape, loc.shape, osqrnorm.shape)
    rootterm = loc**2 - osqrnorm + r**2
    # make sure we are crossing the sphere
    assert (rootterm > 0).all(), rootterm
    return -loc + rootterm**0.5, -loc - rootterm**0.5


def isunitlength(vec):
    """Verify that `vec` is of unit length."""
    assert np.isclose(norm(vec), 1), norm(vec)


def angle(a, b):
    """Compute dot product between vectors `a` and `b`.

    The arccos of the return value would give an actual angle.
    """
    # anorm = (a**2).sum()**0.5
    # bnorm = (b**2).sum()**0.5
    return (a * b).sum()  # / anorm / bnorm


def extrapolate_ahead(dj, xj, vj, contourpath=None):
    """Make `di` steps of size `vj` from `xj`.

    Reflect off unit cube if necessary.
    """
    assert dj == int(dj)

    # optimistically try to go there directly

    xk, vk = linear_steps_with_reflection(xj, vj, dj)

    return xk, vk  # deactivate feature below

    if contourpath is None:
        return xk, vk

    # check if we can already tell that the point will be outside
    region = contourpath.region
    if contourpath.region.inside(xk.reshape((1, -1))):
        return xk, vk

    # find first point outside region
    sign = 1 if dj > 0 else -1
    d = np.arange(0, dj, sign)
    first_point_outside = dj, xk, vk
    for di in d:
        xi, vi = linear_steps_with_reflection(xj, vj, di)
        if not region.inside(xk.reshape((1, -1))):
            first_point_outside = di, xi, vi
            break

    # reflect at this point (first outside)
    dout, reflpoint, v = first_point_outside

    if dout == 0:
        # already the starting point is outside.
        # return extrapolation and hope caller handles it
        return xk, vk

    # reversing:
    normal = contourpath.gradient(reflpoint)  # , v * sign)
    if normal is None:
        vnew = -v
    else:
        vnew = (v - 2 * angle(normal, v) * normal) * sign
    assert vnew.shape == v.shape, (vnew.shape, v.shape)
    assert np.isclose(norm(vnew), norm(v)), (vnew, v, norm(vnew), norm(v))

    # make one step (xl replaces first_point_outside/reflpoint)
    xl, vl = linear_steps_with_reflection(reflpoint, vnew, sign)
    # how many steps are still to do?
    dleft = dj - dout

    # make that many step in that direction, by recursing.

    # it is possible that this point is also outside. The next iteration
    # will catch that case.

    return extrapolate_ahead(dleft, xl, vl, contourpath=contourpath)


def interpolate(i, points, fwd_possible, rwd_possible, contourpath=None):
    """Interpolate a point on the path indicated by `points`.

    Given a sparsely sampled track (stored in .points),
    potentially encountering reflections,
    extract the corrdinates of the point with index `i`.
    That point may not have been evaluated yet.

    Parameters
    -----------
    i: int
        position on track to return.
    points: list of tuples (index, coordinate, direction, loglike)
        points on the path
    fwd_possible: bool
        whether the path could be extended in the positive direction.
    rwd_possible: bool
        whether the path could be extended in the negative direction.
    contourpath: ContourPath
        Use region to reflect. Not used at the moment.

    """
    points_before = [(j, xj, vj, Lj) for j, xj, vj, Lj in points if j <= i]
    points_after = [(j, xj, vj, Lj) for j, xj, vj, Lj in points if j >= i]

    # check if the point after is really after i
    if len(points_after) == 0 and not fwd_possible:
        # the path cannot continue, and i does not exist.
        # print("    interpolate_point %d: the path cannot continue fwd, and i does not exist." % i)
        j, xj, vj, Lj = max(points_before)
        return xj, vj, Lj, False

    # check if the point before is really before i
    if len(points_before) == 0 and not rwd_possible:
        # the path cannot continue, and i does not exist.
        k, xk, vk, Lk = min(points_after)
        # print("    interpolate_point %d: the path cannot continue rwd, and i does not exist." % i)
        return xk, vk, Lk, False

    if len(points_before) == 0 or len(points_after) == 0:
        # return None, None, None, False
        raise KeyError("cannot extrapolate outside path")

    j, xj, vj, Lj = max(points_before)
    k, xk, vk, Lk = min(points_after)

    # print("    interpolate_point %d between %d-%d" % (i, j, k))
    if j == i:  # we have this exact point in the chain
        return xj, vj, Lj, True

    assert not k == i  # otherwise the above would be true too

    # expand_to_step explores each reflection in detail, so
    # any points with change in v should have j == i
    # therefore we can assume:
    # assert (vj == vk).all()
    # this ^ is not true, because reflections on the unit cube can
    # occur, and change v without requiring a intermediate point.

    # j....i....k
    xl1, vj1 = extrapolate_ahead(i - j, xj, vj, contourpath=contourpath)
    xl2, vj2 = extrapolate_ahead(i - k, xk, vk, contourpath=contourpath)
    assert np.allclose(xl1, xl2), (xl1, xl2, i, j, k, xj, vj, xk, vk)
    assert np.allclose(vj1, vj2), (xl1, vj1, xl2, vj2, i, j, k, xj, vj, xk, vk)
    xl = xl1

    # xl = interpolate_between_two_points(i, xj, j, xk, k)
    # the new point is then just a linear interpolation
    # w = (i - k) * 1. / (j - k)
    # xl = xj * w + (1 - w) * xk

    return xl, vj, None, True


class SamplingPath(object):
    """Path described by a (potentially sparse) sequence of points.

    Convention of the stored point tuple ``(i, x, v, L)``:
    `i`: index (0 is starting point)
    `x`: point
    `v`: direction
    `L`: loglikelihood value
    """

    def __init__(self, x0, v0, L0):
        """Initialise with path starting point.

        Starting point (`x0`), direction (`v0`) and
        loglikelihood value (`L0`) of the path. Is given index 0.
        """
        self.reset(x0, v0, L0)

    def add(self, i, xi, vi, Li):
        """Add point `xi`, direction `vi` and value `Li` with index `i` to the path."""
        assert Li is not None
        assert len(xi.shape) == 1, (xi, xi.shape)
        assert len(vi.shape) == 1, (vi, vi.shape)
        assert len(np.shape(Li)) == 0, (Li, Li.shape)
        self.points.append((i, xi, vi, Li))

    def reset(self, x0, v0, L0):
        """Reset path, start from ``x0, v0, L0``."""
        self.points = []
        self.add(0, x0, v0, L0)
        self.fwd_possible = True
        self.rwd_possible = True

    def plot(self, **kwargs):
        """Plot the current path.

        Only uses first two dimensions.
        """
        x = np.array([x for i, x, v, L in sorted(self.points)])
        p, = plt.plot(x[:,0], x[:,1], 'o ', **kwargs)
        ilo, _, _, _ = min(self.points)
        ihi, _, _, _ = max(self.points)
        x = np.array([self.interpolate(i)[0] for i in range(ilo, ihi + 1)])
        kwargs['color'] = p.get_color()
        plt.plot(x[:,0], x[:,1], 'o-', ms=4, mfc='None', **kwargs)

    def interpolate(self, i):
        """Interpolate point with index `i` on path."""
        return interpolate(i, self.points, fwd_possible=self.fwd_possible, rwd_possible=self.rwd_possible)

    def extrapolate(self, i):
        """Advance beyond the current path, extrapolate from the end point.

        Parameters
        -----------
        i: int
            index on path.

        returns
        --------
        coords: vector
            coordinates of the new point.

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
        return newpoint


class ContourSamplingPath(object):
    """Region-aware form of the sampling path.

    Uses region points to guess a likelihood contour gradient.
    """

    def __init__(self, samplingpath, region):
        """Initialise with `samplingpath` and `region`."""
        self.samplingpath = samplingpath
        self.points = self.samplingpath.points
        self.region = region

    def add(self, i, x, v, L):
        """Add point `xi`, direction `vi` and value `Li` with index `i` to the path."""
        self.samplingpath.add(i, x, v, L)

    def interpolate(self, i):
        """Interpolate point with index `i` on path."""
        return interpolate(
            i, self.samplingpath.points,
            fwd_possible=self.samplingpath.fwd_possible,
            rwd_possible=self.samplingpath.rwd_possible,
            contourpath=self)

    def extrapolate(self, i):
        """Advance beyond the current path, extrapolate from the end point.

        Parameters
        -----------
        i: int
            index on path.

        returns
        --------
        coords: vector
            coordinates of the new point.

        """
        if i >= 0:
            j, xj, vj, Lj = max(self.samplingpath.points)
            deltai = i - j
            assert deltai > 0, ("should be extrapolating", i, j)
        else:
            j, xj, vj, Lj = min(self.samplingpath.points)
            deltai = i - j
            assert deltai < 0, ("should be extrapolating", i, j)

        newpoint = extrapolate_ahead(deltai, xj, vj, contourpath=self)
        return newpoint

    def gradient(self, reflpoint, plot=False):
        """Compute gradient approximation.

        Finds spheres enclosing the `reflpoint`, and chooses their mean
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

        Parameters
        -----------
        reflpoint: vector
            point outside the likelihood contour, reflect there
        v: vector
            previous direction vector

        Returns
        ---------
        gradient: vector
            normal of ellipsoid

        """
        if plot:
            plt.plot(reflpoint[0], reflpoint[1], '+ ', color='k', ms=10)

        # check which the reflections the ellipses would make
        region = self.region
        bpts = region.transformLayer.transform(reflpoint.reshape((1,-1)))
        dist = ((bpts - region.unormed)**2).sum(axis=1)
        assert dist.shape == (len(region.unormed),), (dist.shape, len(region.unormed))
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

            """
            # plot in transformed space too:
            origax = plt.gca()
            ax = plt.axes([0.7, 0.7, 0.27, 0.27])
            plt.plot(bpts[:,0], bpts[:,1], '+ ', color='k', ms=10)
            plt.plot(region.unormed[:,0], region.unormed[:,1], 'x ', color='k', ms=4)
            plt.plot(tsphere_centers[:,0], tsphere_centers[:,1], 'o ', mfc='None', mec='b', ms=10, mew=1)
            for si, ti in zip(tsphere_centers, tt_all):
                plt.plot([si[0], ti[0] + si[0]], [si[1], ti[1] + si[1]], '--', lw=2, color='gray', alpha=0.5)
            plt.plot(tsphere_center[0], tsphere_center[1], '^ ', mfc='None', mec='g', ms=10, mew=3)
            plt.plot([tsphere_center[0], tt[0] + tsphere_center[0]],
                [tsphere_center[1], tt[1] + tsphere_center[1]], lw=1, color='gray')
            plt.sca(origax)
            """

            plt.plot(sphere_centers[:,0], sphere_centers[:,1], 'o ', mfc='None', mec='b', ms=10, mew=1)
            if not (dist < region.maxradiussq).any():
                plt.plot(sphere_centers[:,0], sphere_centers[:,1], 's ', mfc='None', mec='b', ms=10, mew=1)
            for si, ti in zip(sphere_centers, t_all):
                plt.plot([si[0], ti[0] * 1000 + si[0]], [si[1], ti[1] * 1000 + si[1]], '--', lw=2, color='gray', alpha=0.5)
            plt.plot(sphere_center[0], sphere_center[1], '^ ', mfc='None', mec='g', ms=10, mew=3)
            plt.plot([sphere_center[0], t[0] * 1000 + sphere_center[0]], [sphere_center[1], t[1] * 1000 + sphere_center[1]], color='gray')

        # compute new vector
        normal = t / norm(t)
        isunitlength(normal)
        assert normal.shape == t.shape, (normal.shape, t.shape)

        return normal
