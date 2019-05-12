import numpy as np
cimport numpy as np

def count_nearby(np.ndarray[np.float_t, ndim=2] apts, 
    np.ndarray[np.float_t, ndim=2] bpts, 
    np.float_t radiussq, 
    np.ndarray[np.int64_t, ndim=1] nnearby
):
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


class TransformationLayer(object):
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.nclusters = 1
    
    def fast_update(self, points):
        self.mean = points.mean(axis=0)
        self.std = points.std(axis=0)
        self.clusterids = np.ones(len(self.mean), dtype=int)
    
    def update(self, upoints, maxradiussq):
        # find clusters
        
        # perform clustering in transformed space
        points = self.transform(upoints)
        clusteridxs = np.zeros(len(points))
        currentclusterid = 1
        i = np.where(self.clusterids == currentclusterid)[0][0]
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
                newmembers[nonmembermask] = self.clusterids[nonmembermask] == currentclusterid
                if newmembers.any():
                    i = np.where(newmembers)[0][0]
                
                clusteridxs[i] = currentclusterid
        
        nclusters = np.max(clusteridxs)
        self.nclusters = nclusters
        if nclusters == 1:
            self.fast_update(upoints)
        else:
            i = 0
            overlapped_points = np.empty_like(points)
            for idx in np.unique(clusteridxs):
                group_points = upoints[clusteridxs == idx,:]
                group_mean = group_points.mean(axis=0).reshape((1,-1))
                j = i + len(group_points)
                overlapped_points[i:j,:] = group_points - group_mean
                i = j
            self.fast_update(overlapped_points)
        
        self.clusterids = clusteridxs
        
    def transform(self, u):
        return (u - self.mean.reshape((1,-1))) / self.std.reshape((1,-1))
    
    def untransform(self, uu):
        return uu * self.std.reshape((1,-1)) + self.mean.reshape((1,-1))

class MLFriends(object):
    def __init__(self, u):
        self.u = u
        self.transformLayer = TransformationLayer()
        self.transformLayer.fast_update(self.u)
        self.unormed = self.transformLayer.transform(self.u)
        self.maxradiussq = 1e300
        self.ellscale = 0
        self.ell = None
    
    def update_transform(self):
        """ 
        Update transformation layer
        Invalidates maxradius
        """
        self.transformLayer.update(self.u, self.maxradiussq)
        self.unormed = self.transformLayer.transform(self.u)
        self.maxradiussq = 1e300
        self.ellscale = 1e300
        self.ell = nestle.bounding_ellipsoid(self.unormed)
    
    def compute_ellscale(self, nbootstraps=50):
        """
        Return MLFriends radius after nbootstraps bootstrapping rounds
        """
        N, ndim = self.u.shape
        idx = np.arange(N)
        ellscale = 1
        for i in range(nbootstraps):
            selidx = np.unique(np.random.randint(N, size=N))
            selected = np.in1d(idx, selidx)
            a = self.unormed[selected,:]
            b = self.unormed[~selected,:]
            # compute distances from a to b
            ell = nestle.bounding_ellipsoid(a)
            d = b - ell.ctr
            dist = np.dot(np.dot(d, ell.a), d)
            ellscale = max(ellscale, dist)
        return ellscale
    
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
		
    def sample_boundingellipse(self, nsamples=100):
        N, ndim = self.u.shape
        # draw from ellipse
        v = np.random.normal(size=(nsamples, ndim))
        v *= (np.random.uniform(size=nsamples)**(1./ndim) / np.linalg.norm(v, axis=1)).reshape((-1, 1))
        w = self.ell.ctr + np.dot(self.ell.axes, v) * self.ellscale**0.5
        # check if inside
        mask = self.inside(v)
        return w[mask,:]
		
    
    def inside(self, pts):
        bpts = self.transformLayer.transform(pts)
        nnearby = np.empty(len(pts), dtype=int)
        count_nearby(self.unormed, bpts, self.maxradiussq, nnearby)

        d = bpts - self.ell.ctr
        inell = np.dot(np.dot(d, ell.a), d) <= self.ellscale

        return np.logical_and(nnearby > 0, inell)
