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
    assert ndim == bpts.shape[1]
    assert nnearby.shape[0] == nb

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


def compute_maxradiussq(np.ndarray[np.float_t, ndim=2] apts, np.ndarray[np.float_t, ndim=2] bpts):
    """
    For each point b in bpts measure shortest euclidean distance to any point in apts.
    Returns the square of the maximum over these.
    """
    cdef int na = apts.shape[0]
    cdef int nb = bpts.shape[0]
    cdef int ndim = apts.shape[1]
    assert ndim == bpts.shape[1]
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


def update_clusters(upoints, points, clusterids, maxradiussq):
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
    currentclusterid = 1
    clusteridxs = np.zeros(len(points))
    i = np.where(clusterids == currentclusterid)[0][0]
    
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
        count_nearby(members, nonmembers, maxradiussq, nnearby)
        #print('merging into cluster', currentclusterid, maxradiussq**0.5, len(members), members, len(nonmembers), nonmembers)
        
        if (nnearby > 0).any():
            # place into cluster
            newmembers = nonmembermask
            newmembers[nonmembermask] = nnearby > 0
            #print('adding', newmembers.sum())
            clusteridxs[newmembers] = currentclusterid
        else:
            # start a new cluster
            currentclusterid += 1
            i = np.where(nonmembermask)[0][0]
            
            # sticky label from last round, if possible
            newmembers = nonmembermask
            newmembers[nonmembermask] = clusterids[nonmembermask] == currentclusterid
            if newmembers.any():
                i = np.where(newmembers)[0][0]
            
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
    
    return nclusters, clusteridxs, overlapped_points

class ScalingLayer(object):
    def __init__(self, mean=0, std=1, nclusters=1):
        self.mean = mean
        self.std = std
        self.nclusters = nclusters
    
    def optimize(self, points, clusterids=None):
        self.mean = points.mean(axis=0)
        self.std = points.std(axis=0)
        if clusterids is None:
            clusterids = np.ones(len(points), dtype=int)
        
        self.clusterids = clusterids
        self.volscale = np.product(self.std)
    
    def create_new(self, upoints, maxradiussq):
        # perform clustering in transformed space
        points = self.transform(upoints)
        nclusters, clusteridxs, overlapped_points = update_clusters(points, upoints, self.clusterids, maxradiussq)
        s = ScalingLayer(nclusters=nclusters)
        s.optimize(overlapped_points, clusterids=clusteridxs)
        return s
        
    def transform(self, u):
        return (u - self.mean.reshape((1,-1))) / self.std.reshape((1,-1))
    
    def untransform(self, uu):
        return uu * self.std.reshape((1,-1)) + self.mean.reshape((1,-1))


def vol_prefactor(n):
    """Volume constant for an n-dimensional sphere:

    for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
    for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
    """
    if n % 2 == 0:
        f = 1.
        i = 2
        while i <= n:
            f *= (2. / i * pi)
            i += 2
    else:
        f = 2.
        i = 3
        while i <= n:
            f *= (2. / i * pi)
            i += 2

    return f

class AffineLayer(ScalingLayer):
    def __init__(self, ctr=0, T=1, invT=1, nclusters=1):
        self.ctr = ctr
        self.T = T
        self.invT = invT
        self.nclusters = nclusters
    
    def optimize(self, points, clusterids=None):
        self.ctr = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=0)
        eigval, eigvec = np.linalg.eigh(cov)
        a = np.linalg.inv(cov)
        self.volscale = np.linalg.det(a)**-0.5
        Lambda = np.diag(eigval)
        Phi = eigvec
        self.T = eigvec
        eigvecI = np.linalg.inv(eigvec)
        self.invT = eigvecI
        
        if clusterids is None:
            clusterids = np.ones(len(points), dtype=int)
        self.clusterids = clusterids

    def create_new(self, upoints, maxradiussq):
        # perform clustering in transformed space
        points = self.transform(upoints)
        clusteridxs = np.zeros(len(points))
        nclusters, clusteridxs, overlapped_points = update_clusters(points, upoints, self.clusterids, maxradiussq)
        s = AffineLayer(nclusters=nclusters)
        s.optimize(overlapped_points, clusterids=clusteridxs)
        return s
    
    def transform(self, u):
        return np.dot(u - self.ctr, self.T)
    
    def untransform(self, uu):
        return np.dot(uu, self.invT) + self.ctr

    def copy(self):
        s = AffineLayer(ctr=self.ctr.copy(), T=self.T.copy(), invT=self.invT.copy(), nclusters=self.nclusters)
        s.clusterids = self.clusterids.copy()
        return s


class MLFriends(object):
    def __init__(self, u, transformLayer):
        self.u = u
        self.transformLayer = transformLayer
        self.transformLayer.optimize(self.u)
        self.unormed = self.transformLayer.transform(self.u)
        self.maxradiussq = 1e300
    
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
        idx = np.arange(N)
        maxd = 0
        for i in range(nbootstraps):
            selidx = np.unique(np.random.randint(N, size=N))
            selected = np.in1d(idx, selidx)
            a = self.unormed[selected,:]
            b = self.unormed[~selected,:]
            # compute distances from a to b
            maxd = max(maxd, compute_maxradiussq(a, b))
            #print('maxd:', maxd)
        return maxd
    
    def sample(self, nsamples=100):
        """
        Draw uniformly sampled points from MLFriends region
        """
        N, ndim = self.u.shape
        # generate points near random existing points
        idx = np.random.randint(N, size=nsamples)
        v = np.random.normal(size=(nsamples, ndim))
        v *= (np.random.uniform(size=nsamples)**(1./ndim) / np.linalg.norm(v, axis=1)).reshape((-1, 1))
        v = self.unormed[idx,:] + v * self.maxradiussq**0.5
        w = self.transformLayer.untransform(v)
        umask = np.logical_and(w > 0, w < 1).all(axis=1)

        # count how many are around
        nnearby = np.empty(nsamples, dtype=int)
        count_nearby(self.unormed, v, self.maxradiussq, nnearby)
        mask = np.logical_and(umask, np.random.uniform(high=nnearby) < 1)

        return w[mask,:], idx[mask]
    
    def sample_boundingbox(self, nsamples=100):
        N, ndim = self.u.shape
        # draw from unit cube
        v = np.random.uniform(size=(nsamples, ndim))
        # check if inside
        mask = self.inside(v)
        return v[mask,:]
    
    def inside(self, pts):
        bpts = self.transformLayer.transform(pts)
        nnearby = np.empty(len(pts), dtype=int)
        count_nearby(self.unormed, bpts, self.maxradiussq, nnearby)

        return nnearby > 0
