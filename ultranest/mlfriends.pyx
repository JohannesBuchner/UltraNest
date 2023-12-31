# cython: language_level=3,annotate=True,profile=True,fast_fail=True,warning_errors=True
"""
Region construction methods
---------------------------

Construct and sample from regions of neighbourhoods around the live points.
Includes

* an efficient implementation of MLFriends, with transformation layers and clustering.
  * RadFriends: Buchner (2014) https://arxiv.org/abs/1407.5459
  * MLFriends: Buchner (2019) https://arxiv.org/abs/1707.04476
* a single-ellipsoid region (Mukherjee et al., 2006, https://arxiv.org/abs/astro-ph/0508461)
* a very fast single-ellipsoid, axis-aligned region, for use with step-samplers in high dimensions

"""

import numpy as np
cimport numpy as np
from numpy import pi
cimport cython
from cython.cimports.libc.math import sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
cdef count_nearby(
    np.ndarray[np.float_t, ndim=2] apts,
    np.ndarray[np.float_t, ndim=2] bpts,
    np.float_t radiussq,
    np.ndarray[np.int_t, ndim=1] nnearby
):
    """Count the number of points in ``apts`` within square radius ``radiussq`` for each point ``b`` in `bpts``.

    The number is written to ``nnearby`` (of same length as ``bpts``).

    Parameters
    ----------
    apts: array
        points
    bpts: array
        points
    radiussq: float
        square of the MLFriends radius
    nnearby: array of ints
        The result will be written here.
    """
    cdef size_t na = apts.shape[0]
    cdef size_t nb = bpts.shape[0]
    cdef size_t ndim = apts.shape[1]

    cdef unsigned long i, j
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


@cython.boundscheck(False)
@cython.wraparound(False)
def find_nearby(
    np.ndarray[np.float_t, ndim=2] apts,
    np.ndarray[np.float_t, ndim=2] bpts,
    np.float_t radiussq,
    np.ndarray[np.int_t, ndim=1] nnearby
):
    """Gets the index of a point in `a` within square radius `radiussq`, for each point `b` in `bpts`.

    The number is written to `nnearby` (of same length as `bpts`).
    If none is found, -1 is written.

    Parameters
    ----------
    apts: array
        points
    bpts: array
        points
    radiussq: float
        square of the MLFriends radius
    nnearby: array of ints
        The result will be written here.

    """
    cdef size_t na = apts.shape[0]
    cdef size_t nb = bpts.shape[0]
    cdef size_t ndim = apts.shape[1]

    cdef unsigned long i, j
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float compute_maxradiussq(np.ndarray[np.float_t, ndim=2] apts, np.ndarray[np.float_t, ndim=2] bpts):
    """Measure the euclidean distance to the nearest point in `apts`, for each point `b` in `bpts`.

    Parameters
    ----------
    apts: array
        points
    bpts: array
        points

    Returns
    ---------
    array containing the square euclidean distance for each point in bpts.
    """
    cdef size_t na = apts.shape[0]
    cdef size_t nb = bpts.shape[0]
    cdef size_t ndim = apts.shape[1]

    cdef unsigned long i, j
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
    np.ndarray[np.int_t, ndim=1] clusterids
):
    """Compute the average distance between pairs of points.
    Pairs from different clusters are excluded in the computation.

    Parameters
    ----------
    pts: array
        points
    clusterids: array of ints or None
        for each point, index of the associated cluster.

    Returns
    ---------
    mean distance between point pairs.
    """
    cdef size_t na = pts.shape[0]
    cdef size_t ndim = pts.shape[1]

    cdef unsigned long i, j
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
                total_dist += sqrt(pair_dist)
                Npairs += 1

    assert np.isfinite(total_dist), total_dist
    return total_dist / Npairs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _update_clusters(
    np.ndarray[np.float_t, ndim=2] upoints,
    np.ndarray[np.float_t, ndim=2] tpoints,
    np.float_t maxradiussq,
    np.ndarray[np.int_t, ndim=1] clusterids,
):
    """same signature as ``update_clusters()``, see there."""
    assert upoints.shape[0] == tpoints.shape[0], ('different number of points', upoints.shape[0], tpoints.shape[0])
    assert upoints.shape[1] == tpoints.shape[1], ('different dimensionality of points', upoints.shape[1], tpoints.shape[1])
    clusteridxs = np.zeros(len(tpoints), dtype=int)
    currentclusterid = 1
    i = 0
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
        # print('merging %d into cluster %d of size %d' % (np.count_nonzero(nnearby), currentclusterid, len(members)))

        if (idnearby >= 0).any():
            # place into cluster
            newmembers = nonmembermask
            newmembers[nonmembermask] = idnearby >= 0
            # print('adding', newmembers.sum())
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
    # assert np.all(np.unique(clusteridxs) == np.arange(nclusters)+1), (np.unique(clusteridxs), nclusters, np.arange(nclusters)+1)
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

    return nclusters, clusteridxs, overlapped_upoints


@cython.boundscheck(False)
@cython.wraparound(False)
def update_clusters(
    np.ndarray[np.float_t, ndim=2] upoints,
    np.ndarray[np.float_t, ndim=2] tpoints,
    np.float_t maxradiussq,
    clusterids=None,
):
    """Clusters `upoints`, so that clusters are distinct if no
    member pair is within a radius of sqrt(`maxradiussq`).

    Parameters
    ----------
    upoints: array
        points (in u-space)
    tpoints: array
        points (in t-space)
    maxradiussq: float
        square of the MLFriends radius
    clusterids: array of ints or None
        for each point, index of the associated cluster.

    Returns
    ---------
    nclusters: int
        the number of clusters found, which is also clusterids.max()
    new_clusterids: array of int
        the new clusterids for each point
    overlapped_points:
        upoints with their cluster centers subtracted.

    The existing cluster ids are re-used when assigning new clusters,
    if possible.
    Clustering is performed on a transformed coordinate space (`tpoints`).
    Returned values are based on upoints.
    """
    if clusterids is None:
        clusterids = np.zeros(len(tpoints), dtype=int)
    return _update_clusters(upoints, tpoints, maxradiussq, clusterids)


@cython.boundscheck(False)
@cython.wraparound(False)
def make_eigvals_positive(
    np.ndarray[np.float_t, ndim=2] a,
    np.float_t targetprod
):
    """For the symmetric square matrix ``a``, increase any zero eigenvalues
    to fulfill a target product of eigenvalues.

    Parameters
    ----------
    a: array
        covariance matrix
    targetprod: array
        target product of eigenvalues

    Returns
    ---------
    covariance matrix
    """

    assert np.isfinite(a).all(), a
    try:
        w, v = np.linalg.eigh(a)  # Use eigh because we assume a is symmetric.
    except np.linalg.LinAlgError as e:
        print(a, targetprod)
        raise e
    mask = w < max(1.e-10, 1e-300**(1. / len(a)))
    if np.any(mask):
        nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
        nzeros = mask.sum()  # number of zero eigenvalues
        w[mask] = (targetprod / nzprod) ** (1. / nzeros)  # adjust zero eigvals
        a = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))  # re-form cov

    return a


@cython.boundscheck(False)
@cython.wraparound(False)
def bounding_ellipsoid(
    np.ndarray[np.float_t, ndim=2] x,
    np.float_t minvol=0.
):
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

    ndim = x.shape[1]

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
    if minvol > 0:
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
        """Optimization for wrapped/circular parameters.
        Does nothing if there are no wrapped/circular parameters.

        Parameters
        ----------
        points: array
            points to use for optimization (in u-space)

        Find largest gap in live points and wrap parameter space there.

        For example, if a wrapped axis has::

            |*****           ****|

        it would identify the middle, subtract it, so that the new space
        is::

            |      ********      |
        """
        if not self.has_wraps:
            return

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

        Parameters
        ----------
        points: array
            points to use for optimize (in u-space)
        centered_points: array
            points with their cluster center subtracted
        clusterids: array of ints
            for each point, which cluster they belong to
        minvol:
            ignored
        """
        self.optimize_wrap(points)
        wrapped_points = self.wrap(points)
        self.mean = wrapped_points.mean(axis=0).reshape((1,-1))
        self.std = centered_points.std(axis=0).reshape((1,-1))
        self.axes = np.diag(self.std[0])
        self.logvolscale = np.sum(np.log(self.std))
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

        Parameters
        ----------
        upoints: array
            points to use for optimize (in u-space)
        maxradiussq: float
            square of the MLFriends radius
        minvol: float
            Minimum volume to regularize sample covariance

        Returns
        ---------
        A new, optimized ScalingLayer.
        """
        # perform clustering in transformed space
        uwpoints = self.wrap(upoints)
        tpoints = self.transform(upoints)
        nclusters, clusteridxs, overlapped_uwpoints = update_clusters(uwpoints, tpoints, maxradiussq, self.clusterids)
        # clusteridxs = track_clusters(clusteridxs, self.clusterids)
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

    For learning the next layer's covariance, the clustering
    is considered: the sample covariance is computed after subtracting
    the cluster mean. This co-centers all clusters and gets
    the average cluster shape, avoiding learning a covariance dominated
    by the distance between clusters.
    """

    def __init__(self, ctr=0, T=1, invT=1, nclusters=1, wrapped_dims=[], clusterids=None):
        """Initialise layer.

        The parameters are optional and can be learned from points with :meth:`optimize`

        Parameters
        ----------
        ctr: vector
            Center of points
        T: matrix
            Transformation matrix. This matrix whitens the points 
            to a unit Gaussian.
        invT: matrix
            Inverse transformation matrix. For transforming a unit
            Gaussian into something with the sample cov.
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

        Estimates covariance of ``centered_points``. ``minvol`` sets the
        smallest allowed size of the covariance to avoid numerical
        collapse.

        Parameters
        ----------
        points: array
            points to use for optimize (in u-space)
        centered_points: array
            points with their cluster center subtracted
        clusterids: array of ints
            for each point, which cluster they belong to
        minvol: float
            Minimum volume to regularize sample covariance
        """
        self.optimize_wrap(points)
        wrapped_points = self.wrap(points)
        # point center
        self.ctr = np.mean(wrapped_points, axis=0)
        # compute sample covariance
        cov = np.cov(centered_points, rowvar=0)
        cov *= (len(self.ctr) + 2)
        self.cov = cov
        # Eigen decomposition of the covariance, with numerical stability
        eigval, eigvec = np.linalg.eigh(cov)
        eigvalmin = eigval.max() * 1e-40
        eigval[eigval < eigvalmin] = eigvalmin
        # Try explicit inversion; if this fails, the error is escalated.
        a = np.linalg.inv(cov)
        # log-volume of the space
        self.logvolscale = np.linalg.slogdet(a)[1] * -0.5

        # Transformation matrix with the correct scale
        # this matrix whitens the points to a unit Gaussian.
        self.T = eigvec * eigval**-0.5
        # Inverse transformation matrix, for transforming a unit 
        # Gaussian into something with the sample cov.
        self.invT = np.linalg.inv(self.T)
        # These also are the principle axes of the space
        self.axes = self.invT
        # print('transform used:', self.T, self.invT, 'cov:', cov, 'eigen:', eigval, eigvec)
        self.set_clusterids(clusterids=clusterids, npoints=len(points))

    def create_new(self, upoints, maxradiussq, minvol=0.):
        """Learn next layer from this optimized layer's clustering.

        Parameters
        ----------
        upoints: array
            points to use for optimize (in u-space)
        maxradiussq: float
            square of the MLFriends radius
        minvol: float
            Minimum volume to regularize sample covariance

        Returns
        ---------
        A new, optimized AffineLayer.
        """
        # perform clustering in transformed space
        uwpoints = self.wrap(upoints)
        tpoints = self.transform(upoints)
        nclusters, clusteridxs, overlapped_uwpoints = update_clusters(uwpoints, tpoints, maxradiussq, self.clusterids)
        # clusteridxs = track_clusters(clusteridxs, self.clusterids)
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

class MaxPrincipleGapAffineLayer(AffineLayer):
    """Affine whitening transformation.

    For learning the next layer's covariance, the clustering
    and principal axis is considered: 
    the sample covariance is computed after subtracting
    the cluster mean. All points are projected onto the line
    defined by the principle axis vector starting from the origin.
    Then, on the sorted line positions, the largest gap is identified.
    All points before the gap are mean-subtracted, and all points
    after the gap are mean-subtracted. Then, the final
    sample covariance is computed. This should give a more "local"
    covariance, even in the case where clusters could not yet be 
    clearly identified.
    """

    def create_new(self, upoints, maxradiussq, minvol=0.):
        """Learn next layer from this optimized layer's clustering.

        Parameters
        ----------
        upoints: array
            points to use for optimize (in u-space)
        maxradiussq: float
            square of the MLFriends radius
        minvol: float
            Minimum volume to regularize sample covariance

        Returns
        ---------
        A new, optimized MaxPrincipleGapAffineLayer.
        """
        # perform clustering in transformed space
        uwpoints = self.wrap(upoints)
        tpoints = self.transform(upoints)
        nclusters, clusteridxs, overlapped_uwpoints = update_clusters(uwpoints, tpoints, maxradiussq, self.clusterids)

        cov = np.cov(overlapped_uwpoints, rowvar=0)
        cov *= (len(self.ctr) + 2)
        eigval, eigvec = np.linalg.eigh(cov)
        # identify principle axis
        principal_vector = eigvec[:, -1]
        # project all overlapped_uwpoints onto principle axis,
        # obtaining position on line
        t = np.dot(overlapped_uwpoints - overlapped_uwpoints.mean(axis=0).reshape((1,-1)), principal_vector)
        # sort positions, identify largest gap
        tsorted = np.sort(t)
        tgapindex = np.argmax(np.diff(tsorted))
        # compute center of largest gap
        tsep = (tsorted[tgapindex] + tsorted[tgapindex + 1]) / 2
        # assign point to left and right cluster
        left_cluster = t < tsep
        # subtract the respective cluster mean from overlapped_uwpoints
        left_mean = overlapped_uwpoints[left_cluster, :].mean(axis=0)
        right_mean = overlapped_uwpoints[~left_cluster, :].mean(axis=0)
        halved_overlapped_uwpoints = overlapped_uwpoints.copy()
        halved_overlapped_uwpoints[left_cluster, :] -= left_mean
        halved_overlapped_uwpoints[~left_cluster, :] -= right_mean

        # re-optimize with the new subtracted points
        s = MaxPrincipleGapAffineLayer(nclusters=nclusters, wrapped_dims=self.wrapped_dims, clusterids=clusteridxs)
        s.optimize(upoints, halved_overlapped_uwpoints, minvol=minvol)
        return s



def vol_prefactor(np.int_t n):
    """Volume constant for an ``n``-dimensional sphere.

    for ``n`` even:  $$    (2pi)^(n    /2) / (2 * 4 * ... * n)$$
    for ``n`` odd :  $$2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)$$

    Parameters
    ----------
    n: int
        dimensionality

    Returns
    ---------
    volume (float)
    """
    if n % 2 == 0:
        f = 1.
        i = 2
    else:
        f = 2.
        i = 3

    while i <= n:
        f *= 2. / i * pi
        i += 2

    return f


def _inside_ellipsoid(
    np.ndarray[np.float_t, ndim=2] points,
    np.ndarray[np.float_t, ndim=1] ellipsoid_center,
    np.ndarray[np.float_t, ndim=2] ellipsoid_invcov,
    np.float_t square_radius
):
    """Check if inside ellipsoid

    Parameters
    ----------
    points: array of vectors
        Points to check
    ellipsoid_center: vector
        center of ellipsoid
    ellipsoid_invcov: matrix
        inverse covariance matrix
    square_radius: float
        square radius

    Returns
    ---------
    is_inside: array of bools
        True if inside wrapping for each point in ``points``.
    """
    # compute distance vector to center
    d = points - ellipsoid_center
    # distance in normalised coordates: vector . matrix . vector
    # where the matrix is the ellipsoid inverse covariance
    r = np.einsum('ij,jk,ik->i', d, ellipsoid_invcov, d)
    # (r <= 1) means inside
    return r <= square_radius


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
            whitening layer

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
        self.vol_prefactor = vol_prefactor(self.u.shape[1])

    def estimate_volume(self):
        """Estimate the order of magnitude of the volume around a single point
        given the current transformLayer.

        Does not account for:
        * the number of live points
        * their overlap
        * the intersection with the unit cube borders

        Returns
        -------
        volume: float
            Volume
        """
        r = self.maxradiussq**0.5
        N, ndim = self.u.shape
        # how large is a sphere of size r in untransformed coordinates?
        return self.transformLayer.logvolscale + np.log(r) * ndim  #+ np.log(vol_prefactor(ndim))

    def set_transformLayer(self, transformLayer):
        """Update transformation layer. Invalidates attribute `maxradius`.

        Parameters
        ----------
        transformLayer: ScalingLayer or AffineLayer
            t-space transformation layer

        """
        self.transformLayer = transformLayer
        self.unormed = self.transformLayer.transform(self.u)
        assert np.isfinite(self.unormed).all(), (self.unormed, self.u)
        self.bbox_lo = self.unormed.min(axis=0)
        self.bbox_hi = self.unormed.max(axis=0)
        self.maxradiussq = None

    def compute_maxradiussq(self, nbootstraps=50):
        """Run MLFriends bootstrapping

        Parameters
        ----------
        nbootstraps: int
            number of bootstrapping rounds

        Returns
        -------
        square radius that safely encloses all live points.
        """
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
        """Return MLFriends radius and ellipsoid enlargement using bootstrapping.

        The wrapping ellipsoid covariance is determined in each bootstrap round.

        Parameters
        ----------
        nbootstraps: int
            number of bootstrapping rounds
        minvol: float
            minimum volume to enforce to wrapping ellipsoid
        rng:
            random number generator

        Returns
        -------
        max_distance: float
            square radius of MLFriends algorithm
        max_radius: float
            square radius of enclosing ellipsoid.
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
            if selected.all() or not selected.any():
                continue

            # compute distances from a to b
            maxd = max(maxd, compute_maxradiussq(
                self.unormed[selected,:],
                self.unormed[~selected,:]))

            # compute enlargement of bounding ellipsoid
            ctr, cov = bounding_ellipsoid(self.u[selected,:], minvol=minvol)
            a = np.linalg.inv(cov)  # inverse covariance
            # compute expansion factor
            delta = self.u[~selected,:] - ctr
            #f = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta).max()
            f = np.einsum('ij,jk,ik->i', delta, a, delta).max()
            assert np.isfinite(f), (ctr, cov, self.unormed, f, delta, a)
            if not f > 0:
                raise np.linalg.LinAlgError("Distances are not positive")
            maxf = max(maxf, f)

        assert maxd > 0, (maxd, self.u, self.unormed)
        assert maxf > 0, (maxf, self.u, self.unormed)
        return maxd, maxf

    def sample_from_points(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Chooses randomly from points and their ellipsoids.

        Parameters as described in *sample()*.
        """
        N, ndim = self.u.shape
        # generate points near random existing points
        idx = np.random.randint(N, size=nsamples)
        v = np.random.normal(size=(nsamples, ndim))
        v *= (np.random.uniform(size=nsamples)**(1. / ndim) / np.linalg.norm(v, axis=1)).reshape((-1, 1))
        v = self.unormed[idx,:] + v * self.maxradiussq**0.5

        # count how many are around
        nnearby = np.empty(nsamples, dtype=int)
        count_nearby(self.unormed, v, self.maxradiussq, nnearby)
        vmask = np.random.uniform(high=nnearby) < 1
        w = self.transformLayer.untransform(v[vmask,:])
        wmask = np.logical_and(w > 0, w < 1).all(axis=1)
        wmask[wmask] = self.inside_ellipsoid(w[wmask])

        return w[wmask,:]

    def sample_from_boundingbox(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from bounding box around region.

        Parameters as described in *sample()*.
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
        return u[wmask,:][vmask,:]

    def sample_from_transformed_boundingbox(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from bounding box around region (in whitened space).

        Parameters as described in *sample()*.
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

        return w[wmask,:]

    def sample_from_wrapping_ellipsoid(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from wrapping ellipsoid and filters with region.

        Parameters as described in ``sample()``.
        """
        N, ndim = self.u.shape
        # draw from rectangle in transformed space

        z = np.random.normal(size=(nsamples, ndim))
        assert ((z**2).sum(axis=1) > 0).all(), (z**2).sum(axis=1)
        z /= ((z**2).sum(axis=1)**0.5).reshape((nsamples, 1))
        assert self.enlarge > 0, self.enlarge
        u = z * self.enlarge**0.5 * np.random.uniform(size=(nsamples, 1))**(1./ndim)

        w = self.ellipsoid_center + np.dot(u, self.ellipsoid_axes_T)
        #assert self.inside_ellipsoid(w).all()

        wmask = np.logical_and(w > 0, w < 1).all(axis=1)
        v = self.transformLayer.transform(w[wmask,:])
        idnearby = np.empty(len(v), dtype=int)
        find_nearby(self.unormed, v, self.maxradiussq, idnearby)
        vmask = idnearby >= 0

        return w[wmask,:][vmask,:]

    def sample(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Switches automatically between the ``sampling_methods`` (attribute).

        Parameters
        ----------
        nsamples: int
            number of samples to draw

        Returns
        -------
        samples: array of shape (nsamples, dimension)
            samples drawn
        idx: array of integers (nsamples)
            index of a point nearby (MLFriends.u)
        """
        samples = self.current_sampling_method(nsamples=nsamples)
        if len(samples) == 0:
            # no result, choose another method
            self.current_sampling_method = self.sampling_methods[np.random.randint(len(self.sampling_methods))]
            # print("switching to %s" % self.current_sampling_method)
        return samples

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
            for each point in ``pts``.

        """
        # require points to be inside bounding ellipsoid
        mask = self.inside_ellipsoid(pts)

        if mask.any():
            # additionally require points to be near neighbours
            bpts = self.transformLayer.transform(pts[mask,:])
            idnearby = np.empty(len(bpts), dtype=int)
            find_nearby(self.unormed, bpts, self.maxradiussq, idnearby)
            mask[mask] = idnearby >= 0

        return mask

    def create_ellipsoid(self, minvol=0.0):
        """Create wrapping ellipsoid and store its center and covariance.

        Parameters
        ----------
        minvol: float
            If positive, make sure ellipsoid has at least this volume.
        """
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
        self.ellipsoid_axes_T = self.ellipsoid_axes.transpose()

        l2, v2 = np.linalg.eigh(cov)
        self.ellipsoid_inv_axlens = 1. / np.sqrt(l2)
        self.ellipsoid_inv_axes = np.dot(v2, np.diag(self.ellipsoid_inv_axlens))


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
        return _inside_ellipsoid(u, self.ellipsoid_center, self.ellipsoid_invcov, self.enlarge)

    def compute_mean_pair_distance(self):
        return compute_mean_pair_distance(self.unormed, self.transformLayer.clusterids)


class RobustEllipsoidRegion(MLFriends):
    """Ellipsoidal region.

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
            whitening layer

        """
        if not np.logical_and(u > 0, u < 1).all():
            raise ValueError("not all u values are between 0 and 1: %s" % u[~np.logical_and(u > 0, u < 1).all()])

        self.u = u
        self.set_transformLayer(transformLayer)

        self.sampling_methods = [
            #self.sample_from_transformed_boundingbox,
            self.sample_from_boundingbox,
            self.sample_from_wrapping_ellipsoid
        ]
        self.current_sampling_method = self.sample_from_boundingbox
        self.vol_prefactor = vol_prefactor(self.u.shape[1])

    def sample_from_boundingbox(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from bounding box around region.

        Parameters as described in *sample()*.
        """
        N, ndim = self.u.shape
        # draw from unit cube in prior space
        u = np.random.uniform(size=(nsamples, ndim))
        wmask = self.inside_ellipsoid(u)
        return u[wmask,:]

    def sample_from_transformed_boundingbox(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from bounding box around region (in whitened space).

        Parameters as described in *sample()*.
        """
        N, ndim = self.u.shape
        # draw from rectangle in transformed space
        v = np.random.uniform(self.bbox_lo - self.maxradiussq, self.bbox_hi + self.maxradiussq, size=(nsamples, ndim))

        # check if inside unit cube
        w = self.transformLayer.untransform(v)
        wmask = np.logical_and(w > 0, w < 1).all(axis=1)
        wmask[wmask] = self.inside_ellipsoid(w[wmask])

        return w[wmask,:]

    def sample_from_wrapping_ellipsoid(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Draws uniformly from wrapping ellipsoid and filters with region.

        Parameters as described in ``sample()``.
        """
        N, ndim = self.u.shape
        # draw from rectangle in transformed space

        z = np.random.normal(size=(nsamples, ndim))
        assert ((z**2).sum(axis=1) > 0).all(), (z**2).sum(axis=1)
        z /= ((z**2).sum(axis=1)**0.5).reshape((nsamples, 1))
        assert self.enlarge > 0, self.enlarge
        u = z * self.enlarge**0.5 * np.random.uniform(size=(nsamples, 1))**(1./ndim)

        w = self.ellipsoid_center + np.dot(u, self.ellipsoid_axes_T)
        #assert self.inside_ellipsoid(w).all()

        wmask = np.logical_and(w > 0, w < 1).all(axis=1)
        return w[wmask,:]

    def sample(self, nsamples=100):
        """Draw uniformly sampled points from MLFriends region.

        Switches automatically between the ``sampling_methods`` (attribute).

        Parameters
        ----------
        nsamples: int
            number of samples to draw

        Returns
        -------
        samples: array of shape (nsamples, dimension)
            samples drawn
        idx: array of integers (nsamples)
            index of a point nearby (MLFriends.u)
        """
        samples = self.current_sampling_method(nsamples=nsamples)
        if len(samples) == 0:
            # no result, choose another method
            self.current_sampling_method = self.sampling_methods[np.random.randint(len(self.sampling_methods))]
            #print("switching to %s" % self.current_sampling_method)
        return samples

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
            for each point in ``pts``.

        """
        # require points to be inside bounding ellipsoid
        return self.inside_ellipsoid(pts)

    def compute_enlargement(self, nbootstraps=50, minvol=0., rng=np.random):
        """Return MLFriends radius and ellipsoid enlargement using bootstrapping.

        The wrapping ellipsoid covariance is determined in each bootstrap round.

        Parameters
        ----------
        nbootstraps: int
            number of bootstrapping rounds
        minvol: float
            minimum volume to enforce to wrapping ellipsoid
        rng:
            random number generator

        Returns
        -------
        max_distance: float
            square radius of MLFriends algorithm
        max_radius: float
            square radius of enclosing ellipsoid.
        """
        N, ndim = self.u.shape
        if N < ndim + 1:
            raise FloatingPointError('not enough live points to compute covariance')
        assert np.isfinite(self.unormed).all(), self.unormed
        selected = np.empty(N, dtype=bool)
        maxd = 1e300
        maxf = 0.0

        for i in range(nbootstraps):
            idx = rng.randint(N, size=N)
            selected[:] = False
            selected[idx] = True

            # compute enlargement of bounding ellipsoid
            ctr, cov = bounding_ellipsoid(self.u[selected,:])
            a = np.linalg.inv(cov)  # inverse covariance
            # compute expansion factor
            delta = self.u[~selected,:] - ctr
            #f = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta).max()
            f = np.einsum('ij,jk,ik->i', delta, a, delta).max()
            assert np.isfinite(f), (ctr, cov, self.unormed, f, delta, a)
            if not f > 0:
                raise np.linalg.LinAlgError("Distances are not positive")
            maxf = max(maxf, f)

        assert maxd > 0, (maxd, self.u, self.unormed)
        assert maxf > 0, (maxf, self.u, self.unormed)
        return maxd, maxf

    def estimate_volume(self):
        """Estimate the volume of the ellipsoid.

        Does not account for the intersection with the unit cube borders.

        Returns
        -------
        logvolume: float
            logarithm of the volume.
        """
        ndim = len(self.ellipsoid_cov)
        sign, logvol = np.linalg.slogdet(self.ellipsoid_cov)
        if sign > 0:
            return logvol + ndim * np.log(self.enlarge)
        else:
            return -1e300


class SimpleRegion(RobustEllipsoidRegion):
    """Axis-aligned ellipsoidal region.

    Defines a region around nested sampling live points for

    1. checking whether a proposed point likely also fulfills the
       likelihood constraints
    2. proposing new points.

    Learns geometry of region from existing live points.
    """

    def create_ellipsoid(self, minvol=0.0):
        """Create wrapping ellipsoid and store its center and covariance.

        Parameters
        ----------
        minvol: float
            If positive, make sure ellipsoid has at least this volume.
        """
        assert self.enlarge is not None
        # compute enlargement of bounding ellipsoid
        ctr = np.mean(self.u, axis=0)
        var = np.var(self.u, axis=0)
        a = np.diag(1. / var)
        cov = np.diag(var)

        self.ellipsoid_center = ctr
        self.ellipsoid_invcov = a
        self.ellipsoid_cov = cov

        l, v = np.linalg.eigh(a)
        self.ellipsoid_axlens = 1. / np.sqrt(l)
        self.ellipsoid_axes = np.dot(v, np.diag(self.ellipsoid_axlens))
        self.ellipsoid_axes_T = self.ellipsoid_axes.transpose()

        l2, v2 = np.linalg.eigh(cov)
        self.ellipsoid_inv_axlens = 1. / np.sqrt(l2)
        self.ellipsoid_inv_axes = np.dot(v2, np.diag(self.ellipsoid_inv_axlens))


    def compute_enlargement(self, nbootstraps=50, minvol=0., rng=np.random):
        """Return MLFriends radius and ellipsoid enlargement using bootstrapping.

        The wrapping ellipsoid covariance is determined in each bootstrap round.

        Parameters
        ----------
        nbootstraps: int
            number of bootstrapping rounds
        minvol: float
            minimum volume to enforce to wrapping ellipsoid
        rng:
            random number generator

        Returns
        -------
        max_distance: float
            square radius of MLFriends algorithm
        max_radius: float
            square radius of enclosing ellipsoid.
        """
        N, ndim = self.u.shape
        assert np.isfinite(self.u).all(), self.u
        assert np.isfinite(self.unormed).all(), self.unormed
        selected = np.empty(N, dtype=bool)
        maxd = 1e300
        maxf = 0.0
        if N < ndim + 1:
            raise FloatingPointError('not enough live points to compute variance')

        for i in range(nbootstraps):
            idx = rng.randint(N, size=N)
            selected[:] = False
            selected[idx] = True

            # compute enlargement of bounding ellipsoid
            ctr = np.mean(self.u[selected,:], axis=0)
            var = np.var(self.u[selected,:], axis=0)
            # compute expansion factor
            f = np.sum((self.u[~selected,:] - ctr.reshape((1, -1)))**2 / var, axis=0).max()
            assert np.isfinite(f), (self.u, ctr, var, self.unormed, f)
            if not f > 0:
                raise np.linalg.LinAlgError("Distances are not positive")
            maxf = max(maxf, f)

        assert maxd > 0, (maxd, self.u, self.unormed)
        assert maxf > 0, (maxf, self.u, self.unormed)
        return maxd, maxf


class WrappingEllipsoid(object):
    """Ellipsoid which safely wraps points."""

    def __init__(self, u):
        """Initialise region.

        Parameters
        -----------
        u: array of vectors
            live points
        """
        self.u = u
        # allow some parameters to have exactly the same value
        # this can occur with grid / categorical parameters
        self.variable_dims = np.std(self.u, axis=0) > 0
        if self.variable_dims.all():
            self.variable_dims = Ellipsis

    def compute_enlargement(self, nbootstraps=50, rng=np.random):
        """Return ellipsoid enlargement after `nbootstraps` bootstrapping rounds.

        The wrapping ellipsoid covariance is determined in each bootstrap round.
        """
        N = len(self.u)
        v = self.u[:,self.variable_dims]
        selected = np.empty(N, dtype=bool)
        maxf = 0.0

        for i in range(nbootstraps):
            idx = rng.randint(N, size=N)
            selected[:] = False
            selected[idx] = True
            ua = v[selected,:]
            ub = v[~selected,:]

            # compute enlargement of bounding ellipsoid
            ctr, cov = bounding_ellipsoid(ua)
            a = np.linalg.inv(cov)  # inverse covariance
            # compute expansion factor
            delta = ub - ctr
            f = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta).max()
            if not f > 0:
                raise np.linalg.LinAlgError("Distances are not positive")
            maxf = max(maxf, f)

        assert maxf > 0, (maxf, self.u, self.active_dims)
        return maxf

    def create_ellipsoid(self, minvol=0.0):
        """Create wrapping ellipsoid and store its center and covariance."""
        assert self.enlarge is not None
        # compute enlargement of bounding ellipsoid
        ctr, cov = bounding_ellipsoid(self.u[:,self.variable_dims], minvol=minvol)
        a = np.linalg.inv(cov)

        self.ellipsoid_center = ctr
        self.ellipsoid_invcov = a
        self.ellipsoid_cov = cov

        l, v = np.linalg.eigh(a)
        self.ellipsoid_axlens = 1. / np.sqrt(l)
        self.ellipsoid_axes = np.dot(v, np.diag(self.ellipsoid_axlens))

    def update_center(self, ctr):
        """Update ellipsoid center, considering fixed dimensions.

        Parameters
        ----------
        ctr: vector
            new center

        """
        if self.variable_dims is Ellipsis:
            self.ellipsoid_center = ctr
        else:
            self.ellipsoid_center = ctr[self.variable_dims]

    def inside(self, u):
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
        # check the variable subspace with the ellipsoid
        inside_variable = _inside_ellipsoid(u[:,self.variable_dims], self.ellipsoid_center, self.ellipsoid_invcov, self.enlarge)
        if self.variable_dims is Ellipsis:
            return inside_variable
        else:
            # the remaining dims must be exactly equal
            inside_fixed = np.all(self.u[0, ~self.variable_dims] == u[:,~self.variable_dims], axis=1)
            return np.logical_and(inside_fixed, inside_variable)
