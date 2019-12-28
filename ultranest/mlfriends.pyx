# cython: language_level=3
# ,profile=True
"""Construct and sample from region.

Implements MLFriends efficiently, with transformation layers and clustering.
"""

import numpy as np
cimport numpy as np
from numpy import pi
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def count_nearby(np.ndarray[np.float_t, ndim=2] apts,
    np.ndarray[np.float_t, ndim=2] bpts,
    np.float_t radiussq,
    np.ndarray[np.int64_t, ndim=1] nnearby
):
    """Count the number of points in `a` within square radius `radiussq` for each point `b` in `bpts`.

    The number is written to `nnearby` (of same length as bpts).
    """
    cdef int na = apts.shape[0]
    cdef int nb = bpts.shape[0]
    cdef int ndim = apts.shape[1]
    #assert ndim == bpts.shape[1]
    #assert nnearby.shape[0] == nb

    cdef int i, j
    cdef np.float_t d

    # go through the unselected points and find the worst case
    for j in range(nb):
        # find the nearest selected point
        nnearby[j] = 0
        for i in range(na):
            d = 0
            for k in range(ndim):
                d += (apts[i,k] - bpts[j,k])**2
            if d <= radiussq:
                nnearby[j] += 1

    #return nnearby


@cython.boundscheck(False)
@cython.wraparound(False)
def find_nearby(np.ndarray[np.float_t, ndim=2] apts,
    np.ndarray[np.float_t, ndim=2] bpts,
    np.float_t radiussq,
    np.ndarray[np.int64_t, ndim=1] nnearby
):
    """Gets the index of a point in `a` within square radius `radiussq`, for each point `b` in `bpts`.

    The number is written to `nnearby` (of same length as `bpts`).
    If none is found, -1 is returned.
    """
    cdef int na = apts.shape[0]
    cdef int nb = bpts.shape[0]
    cdef int ndim = apts.shape[1]
    #assert ndim == bpts.shape[1]
    #assert nnearby.shape[0] == nb

    cdef int i, j
    cdef np.float_t d

    # go through the unselected points and find the worst case
    for j in range(nb):
        # find the nearest selected point
        nnearby[j] = -1
        for i in range(na):
            d = 0.0
            for k in range(ndim):
                d += (apts[i,k] - bpts[j,k])**2
            if d <= radiussq:
                nnearby[j] = i
                break

    #return nnearby


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float compute_maxradiussq(np.ndarray[np.float_t, ndim=2] apts, np.ndarray[np.float_t, ndim=2] bpts):
    """Measure shortest euclidean distance to any point in `apts`, for each point `b` in `bpts`.

    Returns the square of the maximum over these.
    """
    cdef int na = apts.shape[0]
    cdef int nb = bpts.shape[0]
    cdef int ndim = apts.shape[1]
    #assert ndim == bpts.shape[1]
    #assert f.dtype == np.float_t and g.dtype == np.float_t

    cdef int i, j
    cdef np.float_t d
    cdef np.float_t mind = 1e300
    cdef np.float_t maxd = 0

    # go through the unselected points and find the worst case
    for j in range(nb):

        # find the nearest selected point
        mind = 1e300
        for i in range(na):
            d = 0
            for k in range(ndim):
                d += (apts[i,k] - bpts[j,k])**2
            mind = min(mind, d)

        maxd = max(maxd, mind)

    return maxd


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_mean_pair_distance(
    np.ndarray[np.float_t, ndim=2] pts,
    np.ndarray[np.int64_t, ndim=1] clusterids
):
    """Count the number of points in `a` within square radius `radiussq` for each point `b` in `bpts`.

    The number is written to `nnearby` (of same length as bpts).
    """
    cdef int na = pts.shape[0]
    cdef int ndim = pts.shape[1]

    cdef int i, j
    cdef np.float_t total_dist = 0.0
    cdef np.float_t pair_dist
    cdef int Npairs = 0

    # go through the unselected points and find the worst case
    for j in range(na):
        if clusterids[j] == 0:
            continue
        # find the nearest selected point
        for i in range(j):
            # only consider points in the same cluster
            if clusterids[j] == clusterids[i]:
                pair_dist = 0.0
                for k in range(ndim):
                    pair_dist += (pts[i,k] - pts[j,k])**2
                total_dist += pair_dist**0.5
                Npairs += 1
    
    assert np.isfinite(total_dist), total_dist
    return total_dist / Npairs





def update_clusters(upoints, tpoints, maxradiussq, clusterids=None):
    """Clusters `upoints`, so that clusters are distinct if no member pair is within a radius of sqrt(`maxradiussq`)

    clusterids are the cluster indices of each point
    clusterids re-uses the existing ids to assign new cluster ids

    clustering is performed on a transformed coordinate space (`tpoints`).
    Returned values are based on upoints.

    Returns
    ---------
    nclusters: int
        the number of clusters found, which is also clusterids.max()
    new_clusterids: array of int
        the new clusterids for each point
    overlapped_points:
        upoints with their cluster centers subtracted.

    """
    #print("clustering with maxradiussq %f..." % maxradiussq)
    assert upoints.shape == tpoints.shape
    clusteridxs = np.zeros(len(tpoints), dtype=int)
    currentclusterid = 1
    i = 0
    if clusterids is None:
        clusterids = np.zeros(len(tpoints), dtype=int)
    else:
        # avoid issues when old clusterids are from a longer array
        clusterids = clusterids[:len(tpoints)]
        existing = clusterids == currentclusterid
        if existing.any():
            i = np.where(existing)[0][0]

    clusteridxs[i] = currentclusterid
    while True:
        # compare known members to unassociated
        nonmembermask = clusteridxs == 0
        if not nonmembermask.any():
            # everyone has been assigned -> done!
            break

        nonmembers = tpoints[nonmembermask,:]
        idnearby = np.empty(len(nonmembers), dtype=int)
        members = tpoints[clusteridxs == currentclusterid,:]
        find_nearby(members, nonmembers, maxradiussq, idnearby)
        #print('merging %d into cluster %d of size %d' % (np.count_nonzero(nnearby), currentclusterid, len(members)))

        if (idnearby >= 0).any():
            # place into cluster
            newmembers = nonmembermask
            newmembers[nonmembermask] = idnearby >= 0
            #print('adding', newmembers.sum())
            clusteridxs[newmembers] = currentclusterid
        else:
            # start a new cluster
            currentclusterid = currentclusterid + 1
            i = np.where(nonmembermask)[0][0]
            if clusterids is not None:
                existing = clusterids == currentclusterid
                if existing.any():
                    i = np.where(existing)[0][0]

            clusteridxs[i] = currentclusterid

    assert (clusteridxs > 0).all()
    nclusters = len(np.unique(clusteridxs))
    #assert np.all(np.unique(clusteridxs) == np.arange(nclusters)+1), (np.unique(clusteridxs), nclusters, np.arange(nclusters)+1)
    if nclusters == 1:
        overlapped_upoints = upoints
    else:
        overlapped_upoints = np.empty_like(upoints)
        for idx in np.unique(clusteridxs):
            group_upoints = upoints[clusteridxs == idx,:]
            if len(group_upoints) > 1:
                # center on group mean
                group_mean = group_upoints.mean(axis=0).reshape((1,-1))
            else:
                # if a single point, the differences would be zero,
                # not giving weight to this being an outlier.
                # so use the mean of the entire point population instead
                group_mean = upoints.mean(axis=0).reshape((1,-1))
            overlapped_upoints[clusteridxs == idx,:] = group_upoints - group_mean
    #print("clustering done, %d clusters" % nclusters)
    #if nclusters > 1:
    #    np.savetxt("clusters%d.txt" % nclusters, upoints)
    #    np.savetxt("clusters%d_radius.txt" % nclusters, [maxradiussq])
    return nclusters, clusteridxs, overlapped_upoints



def make_eigvals_positive(a, targetprod):
    """For the symmetric square matrix ``a``, increase any zero eigenvalues
    to fulfill the given target product of eigenvalues.

    Returns a (possibly) new matrix."""

    assert np.isfinite(a).all(), a
    try:
        w, v = np.linalg.eigh(a)  # Use eigh because we assume a is symmetric.
    except np.linalg.LinAlgError as e:
        print(a, targetprod)
        raise e
    mask = w < 1.e-10
    if np.any(mask):
        nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
        nzeros = mask.sum()  # number of zero eigenvalues
        w[mask] = (targetprod / nzprod) ** (1./nzeros)  # adjust zero eigvals
        a = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))  # re-form cov

    return a

def bounding_ellipsoid(x, minvol=0.):
    """Calculate bounding ellipsoid containing a set of points x.

    Parameters
    ----------
    x : (npoints, ndim) ndarray
        Coordinates of uniformly sampled points.
    pointvol : float, optional
        Used to set a minimum bound on the ellipsoid volume when
        minvol is True.

    Returns
    -------
    mean and covariance of points

    """
    # Function taken from nestle, MIT licensed, (C) kbarbary

    npoints, ndim = x.shape

    # Calculate covariance of points
    ctr = np.mean(x, axis=0)
    delta = x - ctr
    cov = np.cov(delta, rowvar=0)
    assert np.isfinite(cov).all(), (cov, x)
    if ndim == 1:
        cov = np.atleast_2d(cov)

    # For a ball of uniformly distributed points, the covariance will be
    # smaller than r^2 by a factor of 1/(n+2) [see, e.g.,
    # http://mathoverflow.net/questions/35276/
    # covariance-of-points-distributed-in-a-n-ball]. In nested sampling,
    # we are supposing the points are uniformly distributed within
    # an ellipse, so the same factor holds. Expand `cov`
    # to compensate for that when defining the ellipse matrix:
    cov *= (ndim + 2)

    # Ensure that ``cov`` is nonsingular.
    # It can be singular when the ellipsoid has zero volume, which happens
    # when npoints <= ndim or when enough points are linear combinations
    # of other points. (e.g., npoints = ndim+1 but one point is a linear
    # combination of others). When this happens, we expand the ellipse
    # in the zero dimensions to fulfill the volume expected from
    # ``pointvol``.
    #targetprod = (npoints * pointvol / vol_prefactor(ndim))**2
    cov = make_eigvals_positive(cov, minvol)

    return ctr, cov


class ScalingLayer(object):
    """Simple transformation layer that only shifts and scales each axis."""

    def __init__(self, mean=0, std=1, nclusters=1, wrapped_dims=[], clusterids=None):
        """Initialise layer."""
        self.mean = mean
        self.std = std
        self.nclusters = nclusters
        self.wrapped_dims = wrapped_dims
        self.has_wraps = len(wrapped_dims) > 0
        self.clusterids = clusterids

    def optimize_wrap(self, points):
        """Find largest gap in live points and wrap parameter space there.

        For example, if a wrapped axis has::

            |*****           ****|

        it would identify the middle, subtract it, so that the new space
        is::

            |      ********      |

        Does nothing if there are no wrapped/circular parameters.
        """
        if not self.has_wraps:
            return points

        N, ndims = points.shape
        self.wrap_cuts = []
        for i in self.wrapped_dims:
            # find largest gap
            vals = np.pad(points[:,i], 1, mode='constant', constant_values=(0,1))
            vals.sort()
            assert vals[0] == 0
            assert vals[-1] == 1
            deltas = vals[1:] - vals[:-1]
            j = deltas.argmax()

            # wrap between i, i+1
            cut = (vals[j] + vals[j+1]) / 2.
            self.wrap_cuts.append(cut)

    def wrap(self, points):
        """Wrap points for circular parameters."""
        if not self.has_wraps:
            return points
        wpoints = points.copy().reshape((-1, points.shape[-1]))
        for i, cut in zip(self.wrapped_dims, self.wrap_cuts):
            wpoints[:,i] = np.fmod(wpoints[:,i] + (1 - cut), 1)
        return wpoints

    def unwrap(self, wpoints):
        """Undo wrapping for circular parameters."""
        if not self.has_wraps:
            return wpoints
        points = wpoints.copy().reshape((-1, wpoints.shape[-1]))
        for i, cut in zip(self.wrapped_dims, self.wrap_cuts):
            points[:,i] = np.fmod(points[:,i] + cut, 1)
        return points

    def optimize(self, points, centered_points, clusterids=None, minvol=0.):
        """Optimize layer.

        Estimates mean and std of points.
        """
        self.optimize_wrap(points)
        wrapped_points = self.wrap(points)
        self.mean = wrapped_points.mean(axis=0).reshape((1,-1))
        self.std = centered_points.std(axis=0).reshape((1,-1))
        self.volscale = np.product(self.std)
        self.set_clusterids(clusterids=clusterids, npoints=len(points))

    def set_clusterids(self, clusterids=None, npoints=None):
        """Updates the cluster id assigned to each point."""
        if clusterids is None and self.clusterids is None and npoints is not None:
            # for the beginning, set cluster ids to one for all points
            clusterids = np.ones(npoints, dtype=int)
        if clusterids is not None:
            # if we have a value, update
            self.clusterids = clusterids

    def create_new(self, upoints, maxradiussq, minvol=0.):
        """Learn next layer from this optimized layer's clustering.

        Returns a new, optimized ScalingLayer.
        """
        # perform clustering in transformed space
        uwpoints = self.wrap(upoints)
        tpoints = self.transform(upoints)
        nclusters, clusteridxs, overlapped_uwpoints = update_clusters(uwpoints, tpoints, maxradiussq, self.clusterids)
        #clusteridxs = track_clusters(clusteridxs, self.clusterids)
        s = ScalingLayer(nclusters=nclusters, wrapped_dims=self.wrapped_dims, clusterids=clusteridxs)
        s.optimize(upoints, overlapped_uwpoints)
        return s

    def transform(self, u):
        """Transform points from cube space to a whitened space."""
        if self.has_wraps:
            w = self.wrap(u)
        else:
            w = u
        return ((w - self.mean) / self.std).reshape(u.shape)

    def untransform(self, ww):
        """Transform points from whitened space back to cube space."""
        w = (ww * self.std) + self.mean
        if self.has_wraps:
            u = self.unwrap(w).reshape(ww.shape)
        else:
            u = w.reshape(ww.shape)
        return u

class AffineLayer(ScalingLayer):
    """Affine whitening transformation.

    Learns the covariance of points.
    """

    def __init__(self, ctr=0, T=1, invT=1, nclusters=1, wrapped_dims=[], clusterids=None):
        """Initialise layer.

        The parameters are optional and can be learned from points with :meth:`optimize`

        Parameters
        ----------
        ctr: vector
            Center of points
        T: matrix
            transformation matrix
        invT: matrix
            inverse transformation matrix
        nclusters: int
            number of clusters
        wrapped_dims: array of bools
            indicates which parameter axes are circular.
        clusterids: array of int
            cluster id for each point

        """
        self.ctr = ctr
        self.T = T
        self.invT = invT
        self.nclusters = nclusters
        self.wrapped_dims = wrapped_dims
        self.has_wraps = len(wrapped_dims) > 0
        self.clusterids = clusterids

    def optimize(self, points, centered_points, clusterids=None, minvol=0.):
        """Optimize layer.

        Estimates covariance of `centered_points`. `minvol` sets the
        smallest allowed size of the covariance to avoid numerical
        collapse.
        """
        self.optimize_wrap(points)
        wrapped_points = self.wrap(points)
        self.ctr = np.mean(wrapped_points, axis=0)
        cov = np.cov(centered_points, rowvar=0)
        cov *= (len(self.ctr) + 2)
        self.cov = cov
        eigval, eigvec = np.linalg.eigh(cov)
        #if not (eigval > 0).all():
        #    raise np.linalg.LinAlgError("Points on a hyperplane")
        #assert (eigval > 0).all(), (eigval, eigvec, cov, points, centered_points)
        #eigvalmin = np.product(eigval[eigval > 0]) / minvol
        eigvalmin = eigval.max() * 1e-40
        eigval[eigval < eigvalmin] = eigvalmin
        a = np.linalg.inv(cov)
        self.volscale = np.linalg.det(a)**-0.5

        #Lambda = np.diag(eigval)
        self.T = eigvec * eigval**-0.5
        self.invT = np.linalg.inv(self.T)
        #print('transform used:', self.T, self.invT, 'cov:', cov, 'eigen:', eigval, eigvec)
        self.set_clusterids(clusterids=clusterids, npoints=len(points))

    def create_new(self, upoints, maxradiussq, minvol=0.):
        """Learn next layer from this optimized layer's clustering.

        Returns a new, optimized AffineLayer.
        """
        # perform clustering in transformed space
        uwpoints = self.wrap(upoints)
        tpoints = self.transform(upoints)
        nclusters, clusteridxs, overlapped_uwpoints = update_clusters(uwpoints, tpoints, maxradiussq, self.clusterids)
        #clusteridxs = track_clusters(clusteridxs, self.clusterids)
        s = AffineLayer(nclusters=nclusters, wrapped_dims=self.wrapped_dims, clusterids=clusteridxs)
        s.optimize(upoints, overlapped_uwpoints, minvol=minvol)
        return s

    def transform(self, u):
        """Transform points from cube space to a whitened space."""
        if self.has_wraps:
            w = self.wrap(u)
        else:
            w = u
        return np.dot(w - self.ctr, self.T)

    def untransform(self, ww):
        """Transform points from whitened space back to cube space."""
        w = np.dot(ww, self.invT) + self.ctr
        if self.has_wraps:
            u = self.unwrap(w).reshape(ww.shape)
        else:
            u = w.reshape(ww.shape)
        return u



class MLFriends(object):
    """MLFriends region.

    Defines a region around nested sampling live points for

    1. checking whether a proposed point likely also fulfills the
       likelihood constraints
    2. proposing new points.

    Learns geometry of region from existing live points.
    """

    def __init__(self, u, transformLayer):
        """Initialise region.

        Parameters
        -----------
        u: array of vectors
            live points
        transformLayer: ScalingLayer or AffineLayer
            whitenin layer

        """
        if not np.logical_and(u > 0, u < 1).all():
            raise ValueError("not all u values are between 0 and 1: %s" % u[~np.logical_and(u > 0, u < 1).all()])

        self.u = u
        self.set_transformLayer(transformLayer)

        self.sampling_methods = [
            self.sample_from_transformed_boundingbox,
            self.sample_from_boundingbox,
            self.sample_from_points,
            self.sample_from_wrapping_ellipsoid
        ]
        self.current_sampling_method = self.sample_from_boundingbox

    def estimate_volume(self):
        """Estimate the order of magnitude of the volume around a single point
        given the current transformLayer and

        Does not account for:
        * the number of live points
        * their overlap
        * the intersection with the unit cube borders
        """
        r = self.maxradiussq**0.5
        N, ndim = self.u.shape
        # how large is a sphere of size r in untransformed coordinates?
        return np.log(self.transformLayer.volscale) + np.log(r) * ndim #+ np.log(vol_prefactor(ndim))

    def set_transformLayer(self, transformLayer):
        """Update transformation layer. Invalidates attribute `maxradius`."""
        self.transformLayer = transformLayer
        self.unormed = self.transformLayer.transform(self.u)
        assert np.isfinite(self.unormed).all(), (self.unormed, self.u)
        self.bbox_lo = self.unormed.min(axis=0)
        self.bbox_hi = self.unormed.max(axis=0)
        self.maxradiussq = None

    def compute_maxradiussq(self, nbootstraps=50):
        """Return MLFriends radius after `nbootstraps` bootstrapping rounds"""
        N, ndim = self.u.shape
        selected = np.empty(N, dtype=bool)
        maxd = 0

        for i in range(nbootstraps):
            idx = np.random.randint(N, size=N)
            selected[:] = False
            selected[idx] = True
            a = self.unormed[selected,:]
            b = self.unormed[~selected,:]

            # compute distances from a to b
            maxd = max(maxd, compute_maxradiussq(a, b))

        assert maxd > 0, (maxd, self.u)
        return maxd

    def compute_enlargement(self, nbootstraps=50, minvol=0., rng=np.random):
        """Return MLFriends radius and ellipsoid enlargement after `nbootstraps` bootstrapping rounds.

        The wrapping ellipsoid covariance is determined in each bootstrap round.
        """
        N, ndim = self.u.shape
        assert np.isfinite(self.unormed).all(), self.unormed
        selected = np.empty(N, dtype=bool)
        maxd = 0.0
        maxf = 0.0

        for i in range(nbootstraps):
            idx = rng.randint(N, size=N)
            selected[:] = False
            selected[idx] = True
            ta = self.unormed[selected,:]
            tb = self.unormed[~selected,:]
            ua = self.u[selected,:]
            ub = self.u[~selected,:]

            # compute distances from a to b
            maxd = max(maxd, compute_maxradiussq(ta, tb))

            # compute enlargement of bounding ellipsoid
            ctr, cov = bounding_ellipsoid(ua, minvol=minvol)
            a = np.linalg.inv(cov)  # inverse covariance
            # compute expansion factor
            delta = ub - ctr
            f = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta).max()
            assert np.isfinite(f), (ctr, cov, self.unormed, f, delta, a)
            maxf = max(maxf, f)

        assert maxd > 0, (maxd, self.u, self.unormed)
        assert maxf > 0, (maxf, self.u, self.unormed)
        return maxd, maxf

    def sample_from_points(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Chooses randomly from points and their ellipsoids.
        """
        N, ndim = self.u.shape
        # generate points near random existing points
        idx = np.random.randint(N, size=nsamples)
        v = np.random.normal(size=(nsamples, ndim))
        v *= (np.random.uniform(size=nsamples)**(1./ndim) / np.linalg.norm(v, axis=1)).reshape((-1, 1))
        v = self.unormed[idx,:] + v * self.maxradiussq**0.5

        # count how many are around
        nnearby = np.empty(nsamples, dtype=int)
        count_nearby(self.unormed, v, self.maxradiussq, nnearby)
        vmask = np.random.uniform(high=nnearby) < 1
        w = self.transformLayer.untransform(v[vmask,:])
        wmask = np.logical_and(w > 0, w < 1).all(axis=1)
        wmask[wmask] = self.inside_ellipsoid(w[wmask])

        return w[wmask,:], idx[vmask][wmask]

    def sample_from_boundingbox(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from bounding box around region.
        """
        N, ndim = self.u.shape
        # draw from unit cube in prior space
        u = np.random.uniform(size=(nsamples, ndim))
        wmask = self.inside_ellipsoid(u)
        # check if inside region in transformed space
        v = self.transformLayer.transform(u[wmask,:])
        idnearby = np.empty(len(v), dtype=int)
        find_nearby(self.unormed, v, self.maxradiussq, idnearby)
        vmask = idnearby >= 0
        return u[wmask,:][vmask,:], idnearby[vmask]

    def sample_from_transformed_boundingbox(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from bounding box around region (in whitened space).
        """
        N, ndim = self.u.shape
        # draw from rectangle in transformed space
        v = np.random.uniform(self.bbox_lo - self.maxradiussq, self.bbox_hi + self.maxradiussq, size=(nsamples, ndim))
        idnearby = np.empty(nsamples, dtype=int)
        find_nearby(self.unormed, v, self.maxradiussq, idnearby)
        vmask = idnearby >= 0

        # check if inside unit cube
        w = self.transformLayer.untransform(v[vmask,:])
        wmask = np.logical_and(w > 0, w < 1).all(axis=1)
        wmask[wmask] = self.inside_ellipsoid(w[wmask])

        return w[wmask,:], idnearby[vmask][wmask]

    def sample_from_wrapping_ellipsoid(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from wrapping ellipsoid and filters with region.
        """
        N, ndim = self.u.shape
        # draw from rectangle in transformed space

        z = np.random.normal(size=(nsamples, ndim))
        assert ((z**2).sum(axis=1) > 0).all(), (z**2).sum(axis=1)
        z /= ((z**2).sum(axis=1)**0.5).reshape((nsamples, 1))
        assert self.enlarge > 0, self.enlarge
        u = z * self.enlarge**0.5 * np.random.uniform(size=(nsamples, 1))**(1./ndim)

        w = self.ellipsoid_center + np.einsum('ij,kj->ki', self.ellipsoid_axes, u)
        #assert self.inside_ellipsoid(w).all()

        wmask = np.logical_and(w > 0, w < 1).all(axis=1)
        v = self.transformLayer.transform(w[wmask,:])
        idnearby = np.empty(len(v), dtype=int)
        find_nearby(self.unormed, v, self.maxradiussq, idnearby)
        vmask = idnearby >= 0

        return w[wmask,:][vmask,:], idnearby[vmask]

    def sample(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Switches automatically between the `sampling_methods` (attribute).
        """
        samples, idx = self.current_sampling_method(nsamples=nsamples)
        if len(samples) == 0:
            # no result, choose another method
            self.current_sampling_method = self.sampling_methods[np.random.randint(len(self.sampling_methods))]
            #print("switching to %s" % self.current_sampling_method)
        return samples, idx

    def inside(self, pts):
        """Check if inside region.

        Parameters
        ----------
        pts: array of vectors
            Points to check

        Returns
        ---------
        is_inside: array of bools
            True if inside MLFriends region and wrapping ellipsoid,
            for each point in *pts*.

        """
        bpts = self.transformLayer.transform(pts)
        idnearby = np.empty(len(pts), dtype=int)
        find_nearby(self.unormed, bpts, self.maxradiussq, idnearby)
        mask = idnearby >= 0

        # additionally require points to be inside bounding ellipsoid
        mask[mask] = self.inside_ellipsoid(pts[mask,:])
        return mask

    def create_ellipsoid(self, minvol=0.0):
        """Create wrapping ellipsoid and store its center and covariance."""
        assert self.enlarge is not None
        # compute enlargement of bounding ellipsoid
        ctr, cov = bounding_ellipsoid(self.u, minvol=minvol)
        a = np.linalg.inv(cov)

        self.ellipsoid_center = ctr
        self.ellipsoid_invcov = a
        self.ellipsoid_cov = cov

        l, v = np.linalg.eigh(a)
        self.ellipsoid_axlens = 1. / np.sqrt(l)
        self.ellipsoid_axes = np.dot(v, np.diag(self.ellipsoid_axlens))


    def inside_ellipsoid(self, u):
        """Check if inside wrapping ellipsoid.

        Parameters
        ----------
        u: array of vectors
            Points to check

        Returns
        ---------
        is_inside: array of bools
            True if inside wrapping ellipsoid, for each point in `pts`.

        """
        # to disable wrapping ellipsoid
        #return np.ones(len(u), dtype=bool)

        # compute distance vector to center
        d = u - self.ellipsoid_center
        # distance in normalised coordates: vector . matrix . vector
        # where the matrix is the ellipsoid inverse covariance
        r = np.einsum('ij,jk,ik->i', d, self.ellipsoid_invcov, d)
        # (r <= 1) means inside
        return r <= self.enlarge

    def compute_mean_pair_distance(self):
        return compute_mean_pair_distance(self.unormed, self.transformLayer.clusterids)
        
        
