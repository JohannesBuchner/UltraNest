# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython state machine for snowballing nested sampling: sampling and dead
point release, with HDF5-backed completed data.

Threads (lineages) of live points are advanced in batch advances: each
batch advance replaces the P lowest-likelihood thread ends, producing one
replacement walk per thread (a pluggable direction proposal, stepping out
and bisection, nsteps accepted steps).  Every completed walk produces one
dead point: the threshold it was replaced at, the walk end (replacement
point, likelihood), the walk origin (start point, its likelihood) and the
raw jump distance.

Dead point release (burial gate): a dead point at threshold L is
released once every live point sits above L - all threads have advanced
beyond that threshold.  The front (the lowest live point) is monotone
within a rotation, so release order equals threshold order and a small
reorder buffer holds dead points until the front passes them.  The
released dead points are handed to the caller, which feeds them to the
nested sampling integrator (ultranest.netiter.MultiCounter) and stores
them.

Storage: completed dead points are written once to a growing HDF5 dataset
(the growing approach of ultranest.store.HDF5PointStore) and are not kept
in memory.

Widening appends threads (fresh prior draws as ends) for the next
snowball rotation.

The direction proposal is pluggable: ``stepsampler`` is a function
``direction(u, region, scale=1) -> w`` (e.g. the vectorized direction
proposals in ultranest.stepfuncs), called with the walk origin and the
region object.  The default is the built-in mixture of differential-
evolution directions over the origin-eligible thread ends and random
region axes.
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, srand
from libc.math cimport sqrt, log, log1p, exp, INFINITY

ctypedef np.float64_t f64
ctypedef np.int64_t i64

cdef inline double _uniform():
    return rand() / (2147483647.0 + 1.0)

cdef int BUF_CHUNK = 256       # release buffer growth
cdef int BUR_CHUNK = 1024      # burial sequence growth
cdef int WRITE_CHUNK = 512     # HDF5 write batching


class DeadPointStore:
    """Growing HDF5 storage for completed dead points.

    One 'points' dataset grows along axis 0 as needed (the growing
    approach of ultranest.store.HDF5PointStore), appended in batches.
    Columns: threshold, L, walk-origin L, jump, order, thread, storage
    row, u (d), walk-origin u (d).
    """
    FILES_OPENED = []

    def __init__(self, filepath, d, **h5_file_args):
        import h5py
        self.d = int(d)
        self.ncols = 7 + 2 * self.d
        h5_file_args['mode'] = h5_file_args.get('mode', 'a')
        for i, (filepath2, fileobj2) in enumerate(DeadPointStore.FILES_OPENED):
            if filepath == filepath2:
                fileobj2.close()
                DeadPointStore.FILES_OPENED.pop(i)
        self.fileobj = h5py.File(filepath, **h5_file_args)
        DeadPointStore.FILES_OPENED.append((filepath, self.fileobj))
        if 'points' not in self.fileobj:
            self.fileobj.create_dataset(
                'points', dtype=float, shape=(0, self.ncols),
                maxshape=(None, self.ncols))
        self.nrows, ncols = self.fileobj['points'].shape
        if ncols != self.ncols:
            raise IOError("store '%s' has %d columns, expected %d"
                          % (filepath, ncols, self.ncols))

    def add_rows(self, rows):
        """Append an (n, ncols) array; returns the first row index used."""
        rows = np.asarray(rows, dtype=float)
        n = rows.shape[0]
        if n == 0:
            return self.nrows
        ds = self.fileobj['points']
        ds.resize(self.nrows + n, axis=0)
        ds[self.nrows:self.nrows + n, :] = rows
        first = self.nrows
        self.nrows += n
        return first

    def read(self, row_indices):
        """Rows at the given (any-order, unique) indices, (n, ncols)."""
        idx = np.asarray(row_indices, dtype=np.int64)
        order = np.argsort(idx, kind='stable')
        data = np.empty((idx.shape[0], self.ncols))
        data[order, :] = self.fileobj['points'][idx[order], :]
        return data

    def read_all(self):
        """All stored rows, in storage order."""
        return np.array(self.fileobj['points'][:self.nrows])

    def flush(self):
        """Flush file to disk."""
        self.fileobj.flush()

    def close(self):
        """Close file."""
        self.fileobj.close()


class MemoryPointStore:
    """In-memory fallback with the DeadPointStore interface (testing)."""

    def __init__(self, d):
        self.d = int(d)
        self.ncols = 7 + 2 * self.d
        self._data = np.zeros((0, self.ncols))
        self.nrows = 0

    def add_rows(self, rows):
        rows = np.asarray(rows, dtype=float)
        first = self.nrows
        if rows.shape[0]:
            self._data = np.concatenate([self._data, rows])
            self.nrows = self._data.shape[0]
        return first

    def read(self, row_indices):
        return np.array(self._data[np.asarray(row_indices, dtype=np.int64)])

    def read_all(self):
        return np.array(self._data)

    def flush(self):
        pass

    def close(self):
        pass


cdef class SnowballState:
    cdef public int K, P, d, nsteps
    cdef public long ndead
    cdef public bint integrate
    cdef public double logZ, logVrem, h, Lmax
    cdef public np.ndarray _headu, _headL, _headorig, _fill
    cdef np.ndarray _wu, _wl, _wphase, _wv, _wprop
    cdef np.ndarray _wleft, _wright, _wuoff, _wscale, _wnext
    cdef np.ndarray _wsrcu, _wsrcl, _wtgt, _wLmin, _wfresh
    cdef np.ndarray _whistu, _whistL, _whistn
    cdef public object region, stepsampler
    cdef np.ndarray _buf_lmin, _buf_L, _buf_u, _buf_ou, _buf_oL
    cdef np.ndarray _buf_jump, _buf_order, _buf_thread, _buf_row
    cdef int _buf_n, _buf_alloc
    cdef public object store
    cdef np.ndarray _write_buf
    cdef int _write_n, _write_alloc
    # custom integrator (integrate=True): earlier rotations' dead points
    # (re-buried when the front passes them again), sorted by threshold
    cdef np.ndarray _old_lmin, _old_rows, _old_threads
    cdef long _old_ptr
    # custom integrator: this rotation's released cells (threshold order)
    cdef np.ndarray _burial_rows, _buried_lmin, _burial_threads
    cdef long _bur_n, _bur_alloc
    cdef double _invK, _logwidth_step

    def __init__(self, int K, int P, int d, int nsteps=1, int seed=0,
                 storepath=None, mode='w', integrate=False):
        self.K = K
        self.P = P
        self.d = d
        self.nsteps = nsteps
        self.integrate = integrate
        srand(seed)
        self._headu = np.zeros((K, d))
        self._headL = np.full(K, -1e300)
        self._headorig = np.full(K, -INFINITY)
        self._fill = np.zeros(K, dtype=np.int64)
        self._wu = np.zeros((P, d))
        self._wl = np.full(P, -1e300)
        self._wphase = np.full(P, -1, dtype=np.int64)
        self._wv = np.zeros((P, d))
        self._wprop = np.zeros((P, d))
        self._wleft = np.ones(P)
        self._wright = np.ones(P)
        self._wuoff = np.zeros(P)
        self._wscale = np.ones(P)
        self._wnext = np.ones(P)
        self._wsrcu = np.zeros((P, d))
        self._wsrcl = np.full(P, -1e300)
        self._wtgt = np.zeros(P, dtype=np.int64)
        self._wLmin = np.zeros(P)
        self._wfresh = np.zeros(P, dtype=np.int64)
        self._whistu = np.zeros((P, nsteps + 1, d))
        self._whistL = np.zeros((P, nsteps + 1))
        self._whistn = np.zeros(P, dtype=np.int64)
        self.region = None
        self.stepsampler = None
        self._buf_alloc = BUF_CHUNK
        self._buf_n = 0
        self._buf_lmin = np.zeros(BUF_CHUNK)
        self._buf_L = np.zeros(BUF_CHUNK)
        self._buf_u = np.zeros((BUF_CHUNK, d))
        self._buf_ou = np.zeros((BUF_CHUNK, d))
        self._buf_oL = np.zeros(BUF_CHUNK)
        self._buf_jump = np.zeros(BUF_CHUNK)
        self._buf_order = np.zeros(BUF_CHUNK, dtype=np.int64)
        self._buf_thread = np.zeros(BUF_CHUNK, dtype=np.int64)
        self._buf_row = np.zeros(BUF_CHUNK, dtype=np.int64)
        if storepath is None:
            self.store = MemoryPointStore(d)
        else:
            self.store = DeadPointStore(storepath, d, mode=mode)
        self._write_alloc = WRITE_CHUNK
        self._write_n = 0
        self._write_buf = np.zeros((WRITE_CHUNK, self.store.ncols))
        self.ndead = 0
        # custom integrator bookkeeping (only used with integrate=True)
        self._old_lmin = np.zeros(0)
        self._old_rows = np.zeros(0, dtype=np.int64)
        self._old_threads = np.zeros(0, dtype=np.int64)
        self._old_ptr = 0
        self._bur_alloc = BUR_CHUNK
        self._bur_n = 0
        self._burial_rows = np.zeros(BUR_CHUNK, dtype=np.int64)
        self._buried_lmin = np.zeros(BUR_CHUNK)
        self._burial_threads = np.zeros(BUR_CHUNK, dtype=np.int64)
        self.logZ = -1e300
        self.logVrem = 0.0
        self.h = 0.0
        self.Lmax = -1e300
        self._invK = 1.0 / K
        self._logwidth_step = log1p(-exp(-self._invK))

    def set_heads(self, u, L):
        """Initialise the thread ends from prior draws (origin -inf)."""
        self._headu = np.ascontiguousarray(u, dtype=np.float64)
        self._headL = np.ascontiguousarray(L, dtype=np.float64)
        self._headorig[:] = -INFINITY
        self._fill[:] = 0
        self._buf_n = 0

    def widen(self, u_add, L_add):
        """Append threads (fresh prior draws as ends) for the next
        snowball rotation.  With ``integrate=True``, the released cells
        of this rotation (plus any earlier ones the front did not pass
        again) become the earlier-rotation set that the next rotation
        re-buries at the new width, and the integrator accumulation is
        reset."""
        cdef int K_old = self.K
        cdef int K_add = len(u_add)
        self._headu = np.vstack(
            [self._headu, np.ascontiguousarray(u_add, dtype=np.float64)])
        self._headL = np.concatenate(
            [self._headL, np.ascontiguousarray(L_add, dtype=np.float64)])
        self._headorig = np.concatenate(
            [self._headorig, np.full(K_add, -INFINITY)])
        self._fill = np.concatenate(
            [self._fill, np.zeros(K_add, dtype=np.int64)])
        self.K = K_old + K_add
        if self.integrate:
            new_lmin = np.concatenate([self._old_lmin[self._old_ptr:],
                                       self._buried_lmin[:self._bur_n]])
            new_rows = np.concatenate([self._old_rows[self._old_ptr:],
                                       self._burial_rows[:self._bur_n]])
            new_threads = np.concatenate([self._old_threads[self._old_ptr:],
                                          self._burial_threads[:self._bur_n]])
            order = np.argsort(new_lmin, kind='stable')
            self._old_lmin = np.ascontiguousarray(new_lmin[order])
            self._old_rows = np.ascontiguousarray(new_rows[order],
                                                  dtype=np.int64)
            self._old_threads = np.ascontiguousarray(new_threads[order],
                                                     dtype=np.int64)
            self._old_ptr = 0
            self._bur_n = 0
            self.logZ = -1e300
            self.logVrem = 0.0
            self.h = 0.0
            self._invK = 1.0 / self.K
            self._logwidth_step = log1p(-exp(-self._invK))

    def get_heads(self):
        return self._headu, self._headL

    def set_region(self, region):
        """Set the region object (passed to the direction proposal)."""
        self.region = region

    def any_active(self):
        return bool((self._wphase >= 0).any())

    @cython.cdivision(True)
    cdef int _draw_start(self, double Lmin):
        """Random thread end index with origin Lmin <= Lmin and L > Lmin,
        or -1.  Such ends are draws from the region above their origin,
        which contains the region above Lmin, so they are
        target-distributed starts for a walk at Lmin."""
        cdef int tries = 0
        cdef int i = 0
        while tries < 10000:
            i = <int>(_uniform() * self.K)
            if i >= self.K:
                i = self.K - 1
            if self._headorig[i] <= Lmin and self._headL[i] > Lmin:
                return i
            tries += 1
        return -1

    @cython.cdivision(True)
    cdef void _generate_direction(self, int p):
        """Direction for walker p: the pluggable ``stepsampler`` proposal
        (called with the walk origin and the region), or the built-in
        mixture of differential-evolution directions over the
        origin-eligible thread ends (50%) and random region axes (50%)."""
        cdef int d = self.d
        cdef int a, b, q, trial
        cdef bint degenerate
        cdef double Lmin = self._wLmin[p]
        if self.stepsampler is not None and self.region is not None:
            w = self.stepsampler(
                np.asarray(self._wu[p, :]), self.region, 1.0)
            for q in range(d):
                self._wv[p, q] = w[q]
            return
        if _uniform() < 0.5:
            for trial in range(20):
                a = self._draw_start(Lmin)
                b = self._draw_start(Lmin)
                if a < 0 or b < 0:
                    continue
                degenerate = True
                for q in range(d):
                    if self._headu[a, q] != self._headu[b, q]:
                        degenerate = False
                if degenerate:
                    continue
                for q in range(d):
                    self._wv[p, q] = self._headu[a, q] - self._headu[b, q]
                return
        # region-oriented: a random axis
        axes = self.region.transformLayer.axes \
            if self.region is not None else np.eye(d)
        q = <int>(_uniform() * d)
        if q >= d:
            q = d - 1
        for a in range(d):
            self._wv[p, a] = axes[q, a]

    def start_round(self):
        """Select the P lowest-likelihood thread ends and start a
        replacement walk for each from an origin-eligible thread end."""
        cdef f64[:, ::1] headu = self._headu
        cdef f64[::1] headL = self._headL
        cdef f64[:, ::1] wu = self._wu
        cdef f64[::1] wl = self._wl
        cdef i64[::1] phase = self._wphase
        cdef f64[:, ::1] srcu = self._wsrcu
        cdef f64[::1] srcl = self._wsrcl
        cdef i64[::1] tgt = self._wtgt
        cdef f64[::1] wLmin = self._wLmin
        cdef i64[::1] histn = self._whistn
        cdef f64[:, :, ::1] histu = self._whistu
        cdef f64[:, ::1] histL = self._whistL
        cdef i64[::1] fresh = self._wfresh
        cdef int P = self.P
        cdef int d = self.d
        cdef int p, q, i
        cdef double Lmin
        tsel = np.argpartition(self._headL, min(self.P, self.K - 1))[:self.P]
        tsel = np.ascontiguousarray(tsel, dtype=np.int64)
        cdef i64[::1] tv = tsel
        for p in range(P):
            tgt[p] = tv[p]
            Lmin = headL[tgt[p]]
            wLmin[p] = Lmin
            i = self._draw_start(Lmin)
            if i < 0:
                i = tgt[p]  # degenerate: the target itself (origin < Lmin)
            for q in range(d):
                wu[p, q] = headu[i, q]
                srcu[p, q] = headu[i, q]
                histu[p, 0, q] = headu[i, q]
            wl[p] = headL[i]
            srcl[p] = headL[i]
            histL[p, 0] = headL[i]
            histn[p] = 1
            phase[p] = 0
            fresh[p] = True

    def propose(self):
        """One proposal per active walker; returns the (P, d) array (idle
        walkers carry their current point; the caller evaluates all)."""
        cdef f64[:, ::1] wu = self._wu
        cdef i64[::1] phase = self._wphase
        cdef f64[:, ::1] wv = self._wv
        cdef f64[:, ::1] wprop = self._wprop
        cdef f64[::1] wleft = self._wleft
        cdef f64[::1] wright = self._wright
        cdef f64[::1] wuoff = self._wuoff
        cdef f64[::1] scale = self._wscale
        cdef i64[::1] fresh = self._wfresh
        cdef int P = self.P
        cdef int d = self.d
        cdef int p, q
        cdef double u
        if self.stepsampler is not None and self.region is not None:
            # vectorized direction proposal: one batch call for the fresh
            # walkers
            fresh_idx = [p for p in range(P)
                         if phase[p] >= 0 and fresh[p]]
            if fresh_idx:
                us = np.asarray([self._wu[p, :] for p in fresh_idx])
                ws = np.asarray(self.stepsampler(us, self.region, 1.0))
                for j, p in enumerate(fresh_idx):
                    for q in range(d):
                        self._wv[p, q] = ws[j, q]
        for p in range(P):
            if phase[p] >= 0 and fresh[p]:
                if not (self.stepsampler is not None
                        and self.region is not None):
                    self._generate_direction(p)
                wleft[p] = -scale[p]
                wright[p] = scale[p]
                wuoff[p] = 0.0
                phase[p] = 0
                fresh[p] = False
            if phase[p] == 0:
                for q in range(d):
                    wprop[p, q] = wu[p, q] + wv[p, q] * wleft[p]
            elif phase[p] == 1:
                for q in range(d):
                    wprop[p, q] = wu[p, q] + wv[p, q] * wright[p]
            elif phase[p] == 2:
                u = wleft[p] + (wright[p] - wleft[p]) * _uniform()
                wuoff[p] = u
                for q in range(d):
                    wprop[p, q] = wu[p, q] + wv[p, q] * u
            else:
                for q in range(d):
                    wprop[p, q] = wu[p, q]
        return self._wprop

    cdef void _grow_burial(self):
        cdef long add = BUR_CHUNK
        self._burial_rows = np.concatenate(
            [self._burial_rows, np.zeros(add, dtype=np.int64)])
        self._buried_lmin = np.concatenate([self._buried_lmin, np.zeros(add)])
        self._burial_threads = np.concatenate(
            [self._burial_threads, np.zeros(add, dtype=np.int64)])
        self._bur_alloc += add

    cdef void _grow_buffer(self):
        cdef int add = BUF_CHUNK
        self._buf_lmin = np.concatenate([self._buf_lmin, np.zeros(add)])
        self._buf_L = np.concatenate([self._buf_L, np.zeros(add)])
        self._buf_u = np.concatenate([self._buf_u, np.zeros((add, self.d))])
        self._buf_ou = np.concatenate([self._buf_ou, np.zeros((add, self.d))])
        self._buf_oL = np.concatenate([self._buf_oL, np.zeros(add)])
        self._buf_jump = np.concatenate([self._buf_jump, np.zeros(add)])
        self._buf_order = np.concatenate(
            [self._buf_order, np.zeros(add, dtype=np.int64)])
        self._buf_thread = np.concatenate(
            [self._buf_thread, np.zeros(add, dtype=np.int64)])
        self._buf_row = np.concatenate(
            [self._buf_row, np.zeros(add, dtype=np.int64)])
        self._buf_alloc += add

    cdef void _flush_store(self):
        if self._write_n > 0:
            self.store.add_rows(np.asarray(self._write_buf[:self._write_n]))
            self._write_n = 0

    def receive(self, Lnew):
        """Update the walker states with the proposal likelihoods.
        Completed walks replace their thread's end and stage one dead
        point (threshold, walk end, walk origin, raw jump, creation
        order) in the release buffer."""
        cdef f64[::1] Lnewv = np.ascontiguousarray(Lnew, dtype=np.float64)
        cdef f64[:, ::1] headu = self._headu
        cdef f64[::1] headL = self._headL
        cdef f64[::1] headorig = self._headorig
        cdef i64[::1] fill = self._fill
        cdef f64[:, ::1] wu = self._wu
        cdef f64[::1] wl = self._wl
        cdef i64[::1] phase = self._wphase
        cdef f64[:, ::1] wprop = self._wprop
        cdef f64[::1] wleft = self._wleft
        cdef f64[::1] wright = self._wright
        cdef f64[::1] wuoff = self._wuoff
        cdef f64[::1] scale = self._wscale
        cdef f64[::1] nxt = self._wnext
        cdef f64[:, ::1] srcu = self._wsrcu
        cdef f64[::1] srcl = self._wsrcl
        cdef i64[::1] tgt = self._wtgt
        cdef f64[::1] wLmin = self._wLmin
        cdef i64[::1] fresh = self._wfresh
        cdef f64[:, :, ::1] histu = self._whistu
        cdef f64[:, ::1] histL = self._whistL
        cdef i64[::1] histn = self._whistn
        cdef int P = self.P
        cdef int d = self.d
        cdef int nsteps = self.nsteps
        cdef int p, q, k2
        cdef bint inside, accepted
        cdef double uoff, nudge, ns, jump
        cdef long b, order
        for p in range(P):
            if phase[p] < 0:
                continue
            inside = True
            for q in range(d):
                if not (0.0 < wprop[p, q] < 1.0):
                    inside = False
            accepted = inside and Lnewv[p] > wLmin[p]
            if phase[p] == 0:
                if accepted:
                    wleft[p] = wleft[p] * 2
                else:
                    phase[p] = 1
            elif phase[p] == 1:
                if accepted:
                    wright[p] = wright[p] * 2
                else:
                    phase[p] = 2
                    if -wleft[p] > nxt[p] or wright[p] > nxt[p]:
                        nxt[p] = nxt[p] * 1.1
                    else:
                        nxt[p] = nxt[p] / 1.1
            else:
                if accepted:
                    for q in range(d):
                        wu[p, q] = wprop[p, q]
                        histu[p, histn[p], q] = wprop[p, q]
                    wl[p] = Lnewv[p]
                    histL[p, histn[p]] = Lnewv[p]
                    histn[p] = histn[p] + 1
                    fresh[p] = True
                else:
                    uoff = wuoff[p]
                    if uoff == 0.0:
                        pass
                    elif uoff < 0.0:
                        wleft[p] = uoff
                    else:
                        wright[p] = uoff
            if histn[p] > nsteps:
                # walk complete: replace the thread end, stage one dead
                # point in the release buffer
                if self._buf_n >= self._buf_alloc:
                    self._grow_buffer()
                b = self._buf_n
                jump = 0.0
                for q in range(d):
                    jump = jump + (histu[p, histn[p] - 1, q]
                                   - srcu[p, q]) ** 2
                jump = sqrt(jump)
                self._buf_lmin[b] = wLmin[p]
                self._buf_L[b] = histL[p, histn[p] - 1]
                self._buf_oL[b] = srcl[p]
                self._buf_jump[b] = jump
                for q in range(d):
                    self._buf_u[b, q] = histu[p, histn[p] - 1, q]
                    self._buf_ou[b, q] = srcu[p, q]
                order = 0
                for k2 in range(self.K):
                    if headL[k2] < histL[p, histn[p] - 1]:
                        order = order + 1
                self._buf_order[b] = order
                self._buf_thread[b] = tgt[p]
                self._buf_row[b] = fill[tgt[p]]
                self._buf_n = b + 1
                for q in range(d):
                    headu[tgt[p], q] = histu[p, histn[p] - 1, q]
                headL[tgt[p]] = histL[p, histn[p] - 1]
                headorig[tgt[p]] = wLmin[p]
                fill[tgt[p]] = fill[tgt[p]] + 1
                self.ndead = self.ndead + 1
                # finalise the proposal scale as SliceSampler.finalize_chain
                nudge = 1.1 ** (1.0 / nsteps)
                ns = nxt[p]
                if ns > scale[p] * nudge ** 10:
                    ns = scale[p] * nudge ** 10
                elif ns < scale[p] / nudge ** 10:
                    ns = scale[p] / nudge ** 10
                nxt[p] = ns
                scale[p] = ns
                phase[p] = -1

    def release(self):
        """Release the dead points the front has passed, in threshold
        order: a dead point is released once every live point sits above
        its threshold.  Writes them to the store (once) and returns
        (threads, values, rows): for each released dead point, its thread
        index, its threshold (the node value for the tree) and its store
        row (the node id; the walk end's u is at that row).

        With ``integrate=True``, the released dead points -- and the
        earlier rotations' dead points the front passes again -- are also
        integrated with 1/K strips (the custom integrator: logZ, logVrem,
        h) and recorded for :py:meth:`get_records`.
        """
        self._flush_store()
        cdef f64[::1] headL = self._headL
        cdef int K = self.K
        cdef double front = headL[0]
        cdef int i
        for i in range(1, K):
            if headL[i] < front:
                front = headL[i]
        ready = np.nonzero(
            np.asarray(self._buf_lmin[:self._buf_n]) < front)[0]
        cdef long n_buf = ready.shape[0]
        cdef long n_old = 0
        cdef long old_left = self._old_ptr
        cdef long nold = self._old_lmin.shape[0]
        if self.integrate and old_left < nold:
            n_old = <long>np.searchsorted(
                np.asarray(self._old_lmin[old_left:]), front, side='left')
        if n_buf == 0 and n_old == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0),
                    np.zeros(0, dtype=np.int64))
        cdef int d = self.d
        # merged threshold order: the earlier rotations' cells (indices
        # into _old_lmin) first, then the buffer cells
        if self.integrate:
            all_lmin = np.concatenate([
                np.asarray(self._old_lmin[old_left:old_left + n_old]),
                np.asarray(self._buf_lmin)[ready]])
            all_src = np.concatenate([
                np.zeros(n_old, dtype=np.int64),
                np.ones(n_buf, dtype=np.int64)])
            all_idx = np.concatenate([
                np.arange(old_left, old_left + n_old, dtype=np.int64),
                ready.astype(np.int64)])
        else:
            all_lmin = np.asarray(self._buf_lmin)[ready]
            all_src = np.ones(n_buf, dtype=np.int64)
            all_idx = ready.astype(np.int64)
        ordv = np.argsort(all_lmin, kind='stable').astype(np.int64)
        cdef i64[::1] ordm = ordv
        cdef i64[::1] srcm = np.ascontiguousarray(all_src)
        cdef i64[::1] idxm = np.ascontiguousarray(all_idx)
        cdef f64[::1] lmm = np.ascontiguousarray(all_lmin)
        cdef long total = all_lmin.shape[0]
        cdef long s, o, b, store_row
        cdef double L, logwidth, logwt, mm, logZ_new
        cdef double logZ = self.logZ
        cdef double logVrem = self.logVrem
        cdef double hh = self.h
        cdef double step = self._logwidth_step
        cdef double invK = self._invK
        out_threads = np.zeros(total, dtype=np.int64)
        out_values = np.zeros(total, dtype=np.float64)
        out_rows = np.zeros(total, dtype=np.int64)
        keep = None
        if n_buf > 0:
            keep = np.ones(self._buf_n, dtype=bool)
            keep[ready] = False
        for s in range(total):
            o = ordm[s]
            L = lmm[o]
            if srcm[o] == 0:
                # an earlier rotation's dead point, re-released at the
                # current width: its store row and thread are known
                b = idxm[o]
                out_threads[s] = self._old_threads[b]
                out_values[s] = L
                out_rows[s] = self._old_rows[b]
                self._old_ptr += 1
            else:
                b = idxm[o]
                store_row = self.store.nrows + self._write_n
                self._write_buf[self._write_n, 0] = self._buf_lmin[b]
                self._write_buf[self._write_n, 1] = self._buf_L[b]
                self._write_buf[self._write_n, 2] = self._buf_oL[b]
                self._write_buf[self._write_n, 3] = self._buf_jump[b]
                self._write_buf[self._write_n, 4] = <double>self._buf_order[b]
                self._write_buf[self._write_n, 5] = <double>self._buf_thread[b]
                self._write_buf[self._write_n, 6] = <double>self._buf_row[b]
                for i in range(d):
                    self._write_buf[self._write_n, 7 + i] = self._buf_u[b, i]
                    self._write_buf[self._write_n, 7 + d + i] = self._buf_ou[b, i]
                self._write_n += 1
                out_threads[s] = self._buf_thread[b]
                out_values[s] = L
                out_rows[s] = store_row
            if self.integrate:
                # custom integrator: 1/K strip in threshold order
                logwidth = logVrem + step
                logwt = logwidth + L
                if logwt > logZ:
                    mm = logwt
                else:
                    mm = logZ
                logZ_new = mm + log(exp(logZ - mm) + exp(logwt - mm))
                hh = exp(logwt - logZ_new) * L \
                    + exp(logZ - logZ_new) * (hh + logZ) - logZ_new
                logZ = logZ_new
                logVrem -= invK
                if self._bur_n >= self._bur_alloc:
                    self._grow_burial()
                self._burial_rows[self._bur_n] = out_rows[s]
                self._buried_lmin[self._bur_n] = L
                self._burial_threads[self._bur_n] = out_threads[s]
                self._bur_n += 1
        self.logZ = logZ
        self.logVrem = logVrem
        self.h = hh
        if n_buf > 0:
            surv = np.nonzero(keep)[0]
            n_keep = surv.shape[0]
            self._buf_lmin[:n_keep] = self._buf_lmin[surv]
            self._buf_L[:n_keep] = self._buf_L[surv]
            self._buf_oL[:n_keep] = self._buf_oL[surv]
            self._buf_jump[:n_keep] = self._buf_jump[surv]
            self._buf_order[:n_keep] = self._buf_order[surv]
            self._buf_thread[:n_keep] = self._buf_thread[surv]
            self._buf_row[:n_keep] = self._buf_row[surv]
            self._buf_u[:n_keep] = self._buf_u[surv]
            self._buf_ou[:n_keep] = self._buf_ou[surv]
            self._buf_n = n_keep
        # the released rows are read back by the caller immediately
        self._flush_store()
        return (out_threads, out_values, out_rows)

    def check(self, double frac_remain):
        """frac_remain termination criterion on the custom integrator's
        incrementally maintained values (only meaningful with
        ``integrate=True``)."""
        self.Lmax = float(np.max(self._headL))
        return bool((self.Lmax + self.logVrem) - self.logZ
                    < log(frac_remain))

    def get_records(self):
        """The custom integrator's released dead points of the latest
        rotation, in threshold order: threshold, replacement likelihood
        and point, walk origin (point, likelihood), raw jump, creation
        order and bookkeeping position x = -lnX (only meaningful with
        ``integrate=True``)."""
        self._flush_store()
        cdef long n = self._bur_n
        cdef int d = self.d
        if n == 0:
            return (np.zeros(0), np.zeros(0), np.zeros((0, d)),
                    np.zeros((0, d)), np.zeros(0), np.zeros(0),
                    np.zeros(0, dtype=np.int64), np.zeros(0))
        x = (np.arange(n, dtype=float) + 1.0) * self._invK
        lmin = np.array(self._buried_lmin[:n])
        data = self.store.read(self._burial_rows[:n])
        return (lmin, np.array(data[:, 1]), np.array(data[:, 7:7 + d]),
                np.array(data[:, 7 + d:7 + 2 * d]), np.array(data[:, 2]),
                np.array(data[:, 3]), np.array(data[:, 4], dtype=np.int64),
                x)

    def flush(self):
        """Flush the store to disk."""
        self._flush_store()
        self.store.flush()

    def close(self):
        """Flush and close the store."""
        self._flush_store()
        self.store.close()
