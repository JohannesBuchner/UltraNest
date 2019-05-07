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

class MLFriends(object):
    def __init__(self, u):
        self.u = u
        self.mean = u.mean(axis=0)
        self.std = u.std(axis=0)
        #self.mean *= 0
        #self.std = self.std * 0 + 1
        self.unormed = (u - self.mean) / self.std
        self.maxradiussq = 1e300

    def compute_maxradiussq(self, nbootstraps=1):
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
        return maxd
    
    def sample(self, nsamples=100):
        N, ndim = self.u.shape
        idx = np.random.randint(N, size=nsamples)
        v = np.random.normal(size=(nsamples, ndim))
        v /= np.linalg.norm(v, axis=0)
        v = self.unormed[idx,:] + v
        
        # count how many are around
        nnearby = np.empty(nsamples, dtype=int)
        count_nearby(self.unormed, v, self.maxradiussq, nnearby)
        mask = np.random.uniform(high=nnearby) < 1
        
        w = v[mask,:] * self.std.reshape((1,-1)) + self.mean.reshape((1,-1))
        return w
    
    def inside(self, pts):
        bpts = (pts - self.mean.reshape((1,-1))) / self.std.reshape((1,-1))
        nnearby = np.empty(len(pts), dtype=int)
        count_nearby(self.unormed, bpts, self.maxradiussq, nnearby)
        return nnearby > 0
