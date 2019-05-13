#
# cython: language_level=3
import numpy as np
cimport numpy as np
from numpy import pi

def count_nearby(np.ndarray[np.float_t, ndim=2] apts, 
    np.ndarray[np.float_t, ndim=2] bpts, 
    np.float_t radiussq, 
    np.ndarray[np.int64_t, ndim=1] nnearby
):
    """
    For each point b in bpts
    Count the number of points in a within radisu radiussq.
    
    The number is written to nnearby (of same length as bpts).
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
            if d < radiussq:
                nnearby[j] += 1
        
    return nnearby


def find_nearby(np.ndarray[np.float_t, ndim=2] apts, 
    np.ndarray[np.float_t, ndim=2] bpts, 
    np.float_t radiussq, 
    np.ndarray[np.int64_t, ndim=1] nnearby
):
    """
    For each point b in bpts
    Count the number of points in a within radisu radiussq.
    
    The number is written to nnearby (of same length as bpts).
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
            d = 0
            for k in range(ndim):
                d += (apts[i,k] - bpts[j,k])**2
            if d < radiussq:
                nnearby[j] = i
                break
        
    return nnearby


def compute_maxradiussq(np.ndarray[np.float_t, ndim=2] apts, np.ndarray[np.float_t, ndim=2] bpts):
    """
    For each point b in bpts measure shortest euclidean distance to any point in apts.
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

def track_clusters(newclusterids, oldclusterids):
    # if all members of an old cluster are assigned to only one new cluster id
    # reuse that id
    # otherwise, give new id
    return newclusterids
    relabel_map = {}
    c = 0
    for c in np.unique(oldclusterids):
        if c == 0:
            continue
        newname = newclusterids[c == oldclusterids]
        newcs, cts = np.unique(newname, return_counts=True)
        assert (newcs != 0).all(), newcs
        #mask = newcs != 0
        #newcs = newcs[mask]
        #cts = cts[mask]
        
        # reuse old ID from majority
        relabel_map[newcs[cts.argmax()]] = c
    
    for newc in np.unique(newclusterids):
        nextc = newc
        while nextc in relabel_map.values():
            nextc += 1
        relabel_map[newc] = nextc
    
    mergedclusterids = newclusterids.copy()
    for newc, nextc in relabel_map.items():
        mergedclusterids[newclusterids == newc] = nextc
    
    return mergedclusterids


def update_clusters(upoints, points, maxradiussq):
    """
    clusters points, so that clusters are distinct if no member pair is within a radius of sqrt(maxradiussq)
    clusterids are the cluster indices of each point
    clusterids re-uses the existing ids to assign new cluster ids
    
    returns (nclusters, new_clusterids, overlapped_points)
    
    nclusters: the number of clusters found, which is also clusterids.max()
    
    new_clusterids: new clusterids
    
    overlapped_points: 
    The point coordinates are subtracted from the cluster centers,
    i.e., then points contains the clusters overlapped at the origin.
    
    """
    print("clustering with maxradiussq %f..." % maxradiussq)
    clusteridxs = np.zeros(len(points), dtype=int)
    #currentclusterid = clusterids[clusterids > 0].min()
    #i = np.where(clusterids == currentclusterid)[0][0]
    currentclusterid = 1
    i = 0
    
    clusteridxs[i] = currentclusterid
    while True:
        # compare known members to unassociated
        nonmembermask = clusteridxs == 0
        if not nonmembermask.any():
            # everyone has been assigned -> done!
            break
        
        nonmembers = points[nonmembermask,:]
        nnearby = np.zeros(len(nonmembers), dtype=int)
        members = points[clusteridxs == currentclusterid,:]
        find_nearby(members, nonmembers, maxradiussq, nnearby)
        #print('merging %d into cluster %d of size %d' % (np.count_nonzero(nnearby), currentclusterid, len(members)))
        
        if (nnearby >= 0).any():
            # place into cluster
            newmembers = nonmembermask
            newmembers[nonmembermask] = nnearby >= 0
            #print('adding', newmembers.sum())
            clusteridxs[newmembers] = currentclusterid
        else:
            # start a new cluster
            currentclusterid += 1
            i = np.where(nonmembermask)[0][0]
            
            clusteridxs[i] = currentclusterid
    
    nclusters = np.max(clusteridxs)
    if nclusters == 1:
        overlapped_points = upoints
    else:
        i = 0
        overlapped_points = np.empty_like(points)
        for idx in np.unique(clusteridxs):
            group_points = upoints[clusteridxs == idx,:]
            group_mean = group_points.mean(axis=0).reshape((1,-1))
            j = i + len(group_points)
            overlapped_points[i:j,:] = group_points - group_mean
            i = j
    print("clustering done, %d clusters" % nclusters)
    return nclusters, clusteridxs, overlapped_points

class ScalingLayer(object):
    def __init__(self, mean=0, std=1, nclusters=1, wrapped_dims=[]):
        self.mean = mean
        self.std = std
        self.nclusters = nclusters
        self.wrapped_dims = wrapped_dims
        self.has_wraps = len(wrapped_dims) == 0
    
    def optimize_wrap(self, points):
        if not self.has_wraps:
            return points
        
        N, ndims = points.shape
        self.wrap_cuts = []
        for i in self.wrapped_dims:
            # find largest gap
            vals = np.pad(points[:,i], 1, mode='constant', constant_values=(0,1))
            vals.sort()
            deltas = vals[1:] - vals[:-1]
            j = deltas.argmax()
            if j == 0 or j == N - 2: 
                # wrap is at 0 or 1, no wrapping needs to be done.
                continue
            
            # wrap between i, i+1
            cut = (vals[i] + vals[i+1]) / 2.
            self.wrap_cuts.append(cut)
    
    def wrap(self, points):
        if not self.has_wraps:
            return points
        wpoints = np.atleast_2d(points).copy()
        for i, cut in zip(self.wrapped_dims, self.wrap_cuts):
            wpoints[:,i] = np.fmod(wpoints[:,i] + cut, 1)
        return wpoints

    def unwrap(self, wpoints):
        if not self.has_wraps:
            return wpoints
        points = np.atleast_2d(wpoints).copy()
        for i, cut in zip(self.wrapped_dims, self.wrap_cuts):
            points[:,i] = np.fmod(points[:,i] + 1 - cut, 1)
        return points
    
    def optimize(self, points, clusterids=None):
        self.optimize_wrap(points)
        wrapped_points = self.wrap(points)
        self.mean = wrapped_points.mean(axis=0)
        self.std = wrapped_points.std(axis=0)
        if clusterids is None:
            clusterids = np.ones(len(wrapped_points), dtype=int)
        
        self.clusterids = clusterids
        self.volscale = np.product(self.std)
    
    def create_new(self, upoints, maxradiussq):
        # perform clustering in transformed space
        points = self.transform(upoints)
        nclusters, clusteridxs, overlapped_points = update_clusters(points, upoints, maxradiussq)
        clusteridxs = track_clusters(clusteridxs, self.clusterids)
        s = ScalingLayer(nclusters=nclusters, wrapped_dims=self.wrapped_dims)
        s.optimize(overlapped_points, clusterids=clusteridxs)
        return s
        
    def transform(self, u):
        w = self.wrap(u)
        return ((w - self.mean.reshape((1,-1))) / self.std.reshape((1,-1))).reshape(u.shape)
    
    def untransform(self, ww):
        uu = self.unwrap(ww)
        return (ww * self.std.reshape((1,-1)) + self.mean.reshape((1,-1))).reshape(uu.shape)

class AffineLayer(ScalingLayer):
    def __init__(self, ctr=0, T=1, invT=1, nclusters=1, wrapped_dims=[]):
        self.ctr = ctr
        self.T = T
        self.invT = invT
        self.nclusters = nclusters
        self.wrapped_dims = wrapped_dims
    
    def optimize(self, points, clusterids=None):
        self.optimize_wrap(points)
        wrapped_points = self.wrap(points)
        self.ctr = np.mean(wrapped_points, axis=0)
        cov = np.cov(wrapped_points, rowvar=0)
        eigval, eigvec = np.linalg.eigh(cov)
        a = np.linalg.inv(cov)
        self.volscale = np.linalg.det(a)**-0.5
        Lambda = np.diag(eigval)
        Phi = eigvec
        self.T = eigvec
        eigvecI = np.linalg.inv(eigvec)
        self.invT = eigvecI
        
        if clusterids is None:
            clusterids = np.ones(len(wrapped_points), dtype=int)
        self.clusterids = clusterids

    def create_new(self, upoints, maxradiussq):
        # perform clustering in transformed space
        points = self.transform(upoints)
        clusteridxs = np.zeros(len(points), dtype=int)
        nclusters, clusteridxs, overlapped_points = update_clusters(points, upoints, maxradiussq)
        clusteridxs = track_clusters(clusteridxs, self.clusterids)
        s = AffineLayer(nclusters=nclusters, wrapped_dims=self.wrapped_dims)
        s.optimize(overlapped_points, clusterids=clusteridxs)
        return s
    
    def transform(self, u):
        w = self.wrap(u)
        return np.dot(w - self.ctr, self.T)
    
    def untransform(self, ww):
        uu = self.unwrap(ww)
        return np.dot(uu, self.invT) + self.ctr


class MLFriends(object):
    def __init__(self, u, transformLayer):
        self.u = u
        self.transformLayer = transformLayer
        self.transformLayer.optimize(self.u)
        self.unormed = self.transformLayer.transform(self.u)
        self.maxradiussq = 1e300
        self.bbox_lo = self.unormed.min(axis=0)
        self.bbox_hi = self.unormed.max(axis=0)
        self.current_sampling_method = self.sample_from_boundingbox
        self.sampling_methods = [self.sample_from_points, self.sample_from_transformed_boundingbox, self.sample_from_boundingbox]
        self.sampling_statistics = np.zeros((len(self.sampling_methods), 2), dtype=int)
    
    def estimate_volume(self):
        r = self.maxradiussq**0.5
        N, ndim = self.u.shape
        # how large is a sphere of size r in untransformed coordinates?
        return self.transformLayer.volscale * r**ndim #* vol_prefactor(ndim)
    
    def update_transform(self):
        """
        Update transformation layer
        Invalidates maxradius
        """
        self.transformLayer.update(self.u, self.maxradiussq)
        self.unormed = self.transformLayer.transform(self.u)
        self.maxradiussq = 1e300
    
    def compute_maxradiussq(self, nbootstraps=50):
        """
        Return MLFriends radius after nbootstraps bootstrapping rounds
        """
        N, ndim = self.u.shape
        selected = np.empty(N, dtype=bool)
        #idx = np.arange(N)
        maxd = 0
        for i in range(nbootstraps):
            idx = np.random.randint(N, size=N)
            #selidx = np.unique()
            #selected = np.in1d(idx, selidx)
            selected[:] = True
            selected[idx] = False
            a = self.unormed[selected,:]
            b = self.unormed[~selected,:]
            # compute distances from a to b
            maxd = max(maxd, compute_maxradiussq(a, b))
            #print('maxd:', maxd)
        return maxd
    
    def sample_from_points(self, nsamples=100):
        """
        Draw uniformly sampled points from MLFriends region
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

        return w[wmask,:], idx[vmask][wmask]
    
    def sample_from_boundingbox(self, nsamples=100):
        N, ndim = self.u.shape
        # draw from unit cube in prior space
        u = np.random.uniform(size=(nsamples, ndim))
        # check if inside region in transformed space
        v = self.transformLayer.transform(u)
        nnearby = np.empty(nsamples, dtype=int)
        find_nearby(self.unormed, v, self.maxradiussq, nnearby)
        vmask = nnearby >= 0
        return u[vmask,:], nnearby[vmask]
    
    def sample_from_transformed_boundingbox(self, nsamples=100):
        N, ndim = self.u.shape
        # draw from rectangle in transformed space
        v = np.random.uniform(self.bbox_lo - self.maxradiussq, self.bbox_hi + self.maxradiussq, size=(nsamples, ndim))
        nnearby = np.empty(nsamples, dtype=int)
        find_nearby(self.unormed, v, self.maxradiussq, nnearby)
        vmask = nnearby >= 0
        # check if inside unit cube
        w = self.transformLayer.untransform(v[vmask,:])
        wmask = np.logical_and(w > 0, w < 1).all(axis=1)

        return w[wmask,:], nnearby[vmask][wmask]
    
    def sample(self, nsamples=100):
        samples, idx = self.current_sampling_method(nsamples=nsamples)
        if len(samples) == 0:
            # no result, choose another method
            self.current_sampling_method = self.sampling_methods[np.random.randint(len(self.sampling_methods))]
            print("switching to %s" % self.current_sampling_method)
        return samples, idx
        
        frac = (self.sampling_statistics[:,0] + 1.) / (self.sampling_statistics[:,1] + 1.)
        frac /= frac.sum()
        i = np.random.choice(len(frac), p=frac)
        m = self.sampling_methods[i]
        samples, idx = m(nsamples=nsamples)
        #print("using %s" % m, frac, '%.2f' % (len(samples) * 100. / nsamples))
        self.sampling_statistics[i,0] += len(samples)
        self.sampling_statistics[i,1] += nsamples
        return samples, idx
        
    
    def inside(self, pts):
        bpts = self.transformLayer.transform(pts)
        nnearby = np.empty(len(pts), dtype=int)
        find_nearby(self.unormed, bpts, self.maxradiussq, nnearby)
        return nnearby != 0
