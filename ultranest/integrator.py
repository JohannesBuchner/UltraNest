"""Ultranest calculates the Bayesian evidence and posterior samples of arbitrary models."""

# Some parts are from the Nestle library by Kyle Barbary (https://github.com/kbarbary/nestle)
# Some parts are from the nnest library by Adam Moss (https://github.com/adammoss/nnest)

from __future__ import print_function, division

import os
import sys
import csv
import json
import operator
import time
import warnings

from numpy import log, exp, logaddexp
import numpy as np

from .utils import create_logger, make_run_dir, resample_equal, vol_prefactor, vectorize, listify as _listify
from .utils import is_affine_transform, normalised_kendall_tau_distance, distributed_work_chunk_size
from ultranest.mlfriends import MLFriends, AffineLayer, ScalingLayer, find_nearby, WrappingEllipsoid
from .store import HDF5PointStore, TextPointStore, NullPointStore
from .viz import get_default_viz_callback
from .ordertest import UniformOrderAccumulator
from .netiter import PointPile, SingleCounter, MultiCounter, BreadthFirstIterator, TreeNode, count_tree_between, find_nodes_before, logz_sequence
from .netiter import dump_tree, combine_results
from .hotstart import get_auxiliary_contbox_parameterization


__all__ = ['ReactiveNestedSampler', 'NestedSampler', 'read_file', 'warmstart_from_similar_file']


def _get_cumsum_range(pi, dp):
    """Compute quantile indices from probabilities.

    Parameters
    ------------
    pi: array
        probability of each item.
    dp: float
        Quantile (between 0 and 0.5).

    Returns
    ---------
    index_lo: int
        Index of the item corresponding to quantile ``dp``.
    index_hi: int
        Index of the item corresponding to quantile ``1-dp``.
    """
    ci = pi.cumsum()
    ilo = np.where(ci > dp)[0]
    ilo = ilo[0] if len(ilo) > 0 else 0
    ihi = np.where(ci < 1. - dp)[0]
    ihi = ihi[-1] if len(ihi) > 0 else -1
    return ilo, ihi


def _sequentialize_width_sequence(minimal_widths, min_width):
    """Turn a list of required tree width into an ordered sequence.

    Parameters
    ------------
    minimal_widths: list of (Llo, Lhi, width)
        Defines the required width between Llo and Lhi.
    min_width: int
        Minimum width everywhere.

    Returns
    ---------
    Lsequence: list of (L, width)
        A sequence of L points and the expected tree width at and above it.

    """
    Lpoints = np.unique(_listify(
        [-np.inf], [L for L, _, _ in minimal_widths],
        [L for _, L, _ in minimal_widths], [np.inf]))
    widths = np.ones(len(Lpoints)) * min_width

    for Llo, Lhi, width in minimal_widths:
        # all Lpoints within that range should be maximized to width
        # mask = np.logical_and(Lpoints >= Llo, Lpoints <= Lhi)
        # the following allows segments to specify -inf..L ranges
        mask = ~np.logical_or(Lpoints < Llo, Lpoints > Lhi)
        widths[mask] = np.where(widths[mask] < width, width, widths[mask])

    # the width has to monotonically increase to the maximum from both sides
    # so we fill up any intermediate dips
    max_width = widths.max()
    mid = np.where(widths == max_width)[0][0]
    widest = 0
    for i in range(mid):
        widest = widths[i] = max(widest, widths[i])
    widest = 0
    for i in range(len(widths) - 1, mid, -1):
        widest = widths[i] = max(widest, widths[i])

    return list(zip(Lpoints, widths))


def _explore_iterator_batch(explorer, pop, x_dim, num_params, pointpile, batchsize=1):
    batch = []

    while True:
        next_node = explorer.next_node()
        if next_node is None:
            break
        rootid, node, (_, active_rootids, active_values, active_node_ids) = next_node
        Lmin = node.value
        children = []

        _, row = pop(Lmin)
        if row is not None:
            logl = row[1]
            u = row[3:3 + x_dim]
            v = row[3 + x_dim:3 + x_dim + num_params]

            assert u.shape == (x_dim,)
            assert v.shape == (num_params,)
            assert logl > Lmin
            children.append((u, v, logl))
            child = pointpile.make_node(logl, u, v)
            node.children.append(child)

        batch.append((Lmin, active_values.copy(), children))
        if len(batch) >= batchsize:
            yield batch
            batch = []
        explorer.expand_children_of(rootid, node)
    if len(batch) > 0:
        yield batch


def resume_from_similar_file(
    log_dir, x_dim, loglikelihood, transform,
    max_tau=0, verbose=False, ndraw=400
):
    """
    Change a stored UltraNest run to a modified loglikelihood/transform.

    Parameters
    ----------
    log_dir: str
        Folder containing results
    x_dim: int
        number of dimensions
    loglikelihood: function
        new likelihood function
    transform: function
        new transform function
    verbose: bool
        show progress
    ndraw: int
        set to >1 if functions can take advantage of vectorized computations
    max_tau: float
        Allowed dissimilarity in the live point ordering, quantified as
        normalised Kendall tau distance.

        max_tau=0 is the very conservative choice of stopping the warm start
        when the live point order differs.
        Near 1 are completely different live point orderings.
        Values in between permit mild disorder.

    Returns
    ----------
    sequence: dict
        contains arrays storing for each iteration estimates of:

            * logz: log evidence estimate
            * logzerr: log evidence uncertainty estimate
            * logvol: log volume estimate
            * samples_n: number of live points
            * logwt: log weight
            * logl: log likelihood

    final: dict
        same as ReactiveNestedSampler.results and
        ReactiveNestedSampler.run return values

    """
    import h5py
    filepath = os.path.join(log_dir, 'results', 'points.hdf5')
    filepath2 = os.path.join(log_dir, 'results', 'points.hdf5.new')
    fileobj = h5py.File(filepath, 'r')
    _, ncols = fileobj['points'].shape
    num_params = ncols - 3 - x_dim

    points = fileobj['points'][:]
    fileobj.close()
    del fileobj

    pointstore2 = HDF5PointStore(filepath2, ncols, mode='w')
    stack = list(enumerate(points))

    pointpile = PointPile(x_dim, num_params)
    pointpile2 = PointPile(x_dim, num_params)

    def pop(Lmin):
        """Find matching sample from points file."""
        # look forward to see if there is an exact match
        # if we do not use the exact matches
        #   this causes a shift in the loglikelihoods
        for i, (idx, next_row) in enumerate(stack):
            row_Lmin = next_row[0]
            L = next_row[1]
            if row_Lmin <= Lmin and L > Lmin:
                idx, row = stack.pop(i)
                return idx, row
        return None, None

    roots = []
    roots2 = []
    initial_points_u = []
    initial_points_v = []
    initial_points_logl = []
    while True:
        _, row = pop(-np.inf)
        if row is None:
            break
        logl = row[1]
        u = row[3:3 + x_dim]
        v = row[3 + x_dim:3 + x_dim + num_params]
        initial_points_u.append(u)
        initial_points_v.append(v)
        initial_points_logl.append(logl)

    v2 = transform(np.array(initial_points_u, ndmin=2, dtype=float))
    assert np.allclose(v2, initial_points_v), 'transform inconsistent, cannot resume'
    logls_new = loglikelihood(v2)

    for u, v, logl, logl_new in zip(initial_points_u, initial_points_v, initial_points_logl, logls_new):
        roots.append(pointpile.make_node(logl, u, v))
        roots2.append(pointpile2.make_node(logl_new, u, v))
        pointstore2.add(_listify([-np.inf, logl_new, 0.0], u, v), 1)

    batchsize = ndraw
    explorer = BreadthFirstIterator(roots)
    explorer2 = BreadthFirstIterator(roots2)
    main_iterator2 = SingleCounter()
    main_iterator2.Lmax = logls_new.max()
    good_state = True

    indices1, indices2 = np.meshgrid(np.arange(len(logls_new)), np.arange(len(logls_new)))
    last_good_like = -1e300
    last_good_state = 0
    epsilon = 1 + 1e-6
    niter = 0
    for batch in _explore_iterator_batch(explorer, pop, x_dim, num_params, pointpile, batchsize=batchsize):
        assert len(batch) > 0
        batch_u = np.array([u for _, _, children in batch for u, _, _ in children], ndmin=2, dtype=float)
        if batch_u.size > 0:
            assert batch_u.shape[1] == x_dim, batch_u.shape
            batch_v = np.array([v for _, _, children in batch for _, v, _ in children], ndmin=2, dtype=float)
            # print("calling likelihood with %d points" % len(batch_u))
            v2 = transform(batch_u)
            assert batch_v.shape[1] == num_params, batch_v.shape
            assert np.allclose(v2, batch_v), 'transform inconsistent, cannot resume'
            logls_new = loglikelihood(batch_v)
        else:
            # no new points
            logls_new = []

        j = 0
        for Lmin, active_values, children in batch:

            next_node2 = explorer2.next_node()
            rootid2, node2, (active_nodes2, _, active_values2, _) = next_node2
            Lmin2 = float(node2.value)

            # in the tails of distributions it can happen that two points are out of order
            # but that may not be very important
            # in the interest of practicality, we allow this and only stop the
            # warmstart copying when some bulk of points differ.
            # in any case, warmstart should not be considered safe, but help iterating
            # and a final clean run is needed to finalise the results.
            if len(active_values) != len(active_values2):
                if verbose == 2:
                    print("stopping, number of live points differ (%d vs %d)" % (len(active_values), len(active_values2)))
                    good_state = False
                break

            if len(active_values) != len(indices1):
                indices1, indices2 = np.meshgrid(np.arange(len(active_values)), np.arange(len(active_values2)))
            tau = normalised_kendall_tau_distance(active_values, active_values2, indices1, indices2)
            order_consistent = tau <= max_tau
            if order_consistent and len(active_values) > 10 and len(active_values) > 10:
                good_state = True
            elif not order_consistent:
                good_state = False
            else:
                # maintain state
                pass
            if verbose == 2:
                print(niter, tau)
            if good_state:
                # print("        (%.1e)   L=%.1f" % (last_good_like, Lmin2))
                # assert last_good_like < Lmin2, (last_good_like, Lmin2)
                last_good_like = Lmin2
                last_good_state = niter
            else:
                # interpolate a increasing likelihood
                # in the hope that the step size is smaller than
                # the likelihood increase
                Lmin2 = last_good_like
                node2.value = Lmin2
                last_good_like = last_good_like * epsilon
                break

            for u, v, logl_old in children:
                logl_new = logls_new[j]
                j += 1

                # print(j, Lmin2, '->', logl_new, 'instead of', Lmin, '->', [c.value for c in node2.children])
                child2 = pointpile2.make_node(logl_new, u, v)
                node2.children.append(child2)
                if logl_new > Lmin2:
                    pointstore2.add(_listify([Lmin2, logl_new, 0.0], u, v), 1)
                else:
                    if verbose == 2:
                        print("cannot use new point because it would decrease likelihood (%.1f->%.1f)" % (Lmin2, logl_new))
                    # good_state = False
                    # break

            main_iterator2.passing_node(node2, active_nodes2)

            niter += 1
            if verbose:
                sys.stderr.write("%d...\r" % niter)

            explorer2.expand_children_of(rootid2, node2)

        if not good_state:
            break
        if main_iterator2.logZremain < main_iterator2.logZ and not good_state:
            # stop as the results diverged already
            break

    if verbose:
        sys.stderr.write("%d/%d iterations salvaged (%.2f%%).\n" % (
            last_good_state + 1, len(points), (last_good_state + 1) * 100. / len(points)))
    # delete the ones at the end from last_good_state onwards
    # assert len(pointstore2.fileobj['points']) == niter, (len(pointstore2.fileobj['points']), niter)
    mask = pointstore2.fileobj['points'][:,0] <= last_good_like
    points2 = pointstore2.fileobj['points'][:][mask,:]
    del pointstore2.fileobj['points']
    pointstore2.fileobj.create_dataset(
        'points', dtype=np.float64,
        shape=(0, pointstore2.ncols), maxshape=(None, pointstore2.ncols))
    pointstore2.fileobj['points'].resize(len(points2), axis=0)
    pointstore2.fileobj['points'][:] = points2
    pointstore2.close()
    del pointstore2

    os.replace(filepath2, filepath)


def _update_region_bootstrap(region, nbootstraps, minvol=0., comm=None, mpi_size=1):
    """
    Update *region* with *nbootstraps* rounds of excluding points randomly.

    Stiffen ellipsoid size using the minimum volume *minvol*.

    If the mpi communicator *comm* is not None, use MPI to distribute
    the bootstraps over the *mpi_size* processes.
    """
    assert nbootstraps > 0, nbootstraps
    # catch potential errors so MPI syncing still works
    e = None
    try:
        r, f = region.compute_enlargement(
            minvol=minvol,
            nbootstraps=max(1, nbootstraps // mpi_size))
    except np.linalg.LinAlgError as e1:
        e = e1
        r, f = np.nan, np.nan

    if comm is not None:
        recv_maxradii = comm.gather(r, root=0)
        recv_maxradii = comm.bcast(recv_maxradii, root=0)
        # if there are very many processors, we may have more
        # rounds than requested, leading to slowdown
        # thus we throw away the extra ones
        r = np.max(recv_maxradii[:nbootstraps])
        recv_enlarge = comm.gather(f, root=0)
        recv_enlarge = comm.bcast(recv_enlarge, root=0)
        f = np.max(recv_enlarge[:nbootstraps])

    if not np.isfinite(r) and not np.isfinite(r):
        # reraise error if needed
        if e is None:
            raise np.linalg.LinAlgError("compute_enlargement failed")
        else:
            raise e

    region.maxradiussq = r
    region.enlarge = f
    return r, f


class NestedSampler(object):
    """Simple Nested sampler for reference."""

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 derived_param_names=[],
                 resume='subfolder',
                 run_num=None,
                 log_dir='logs/test',
                 num_live_points=1000,
                 vectorized=False,
                 wrapped_params=[],
                 ):
        """Set up nested sampler.

        Parameters
        -----------
        param_names: list of str, names of the parameters.
            Length gives dimensionality of the sampling problem.
        loglike: function
            log-likelihood function.
            Receives multiple parameter vectors, returns vector of likelihood.
        transform: function
            parameter transform from unit cube to physical parameters.
            Receives multiple cube vectors, returns multiple parameter vectors.
        derived_param_names: list of str
            Additional derived parameters created by transform. (empty by default)
        log_dir: str
            where to store output files
        resume: 'resume', 'overwrite' or 'subfolder'
            if 'overwrite', overwrite previous data.
            if 'subfolder', create a fresh subdirectory in log_dir.
            if 'resume' or True, continue previous run if available.
        wrapped_params: list of bools
            indicating whether this parameter wraps around (circular parameter).
        num_live_points: int
            Number of live points
        vectorized: bool
            If true, loglike and transform function can receive arrays
            of points.
        run_num: int
            unique run number. If None, will be automatically incremented.
        

        """
        self.paramnames = param_names
        x_dim = len(self.paramnames)
        self.num_live_points = num_live_points
        self.sampler = 'nested'
        self.x_dim = x_dim
        self.derivedparamnames = derived_param_names
        num_derived = len(self.derivedparamnames)
        self.num_params = x_dim + num_derived
        self.volfactor = vol_prefactor(self.x_dim)
        if wrapped_params is None:
            self.wrapped_axes = []
        else:
            self.wrapped_axes = np.where(wrapped_params)[0]

        assert resume or resume in ('overwrite', 'subfolder', 'resume'), "resume should be one of 'overwrite' 'subfolder' or 'resume'"
        append_run_num = resume == 'subfolder'
        resume = resume == 'resume' or resume

        if not vectorized:
            transform = vectorize(transform)
            loglike = vectorize(loglike)

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        u = np.random.uniform(size=(2, self.x_dim))
        p = self.transform(u)
        assert p.shape == (2, self.num_params), ("Error in transform function: returned shape is %s, expected %s" % (p.shape, (2, self.num_params)))
        logl = loglike(p)
        assert np.logical_and(u > 0, u < 1).all(), ("Error in transform function: u was modified!")
        assert logl.shape == (2,), ("Error in loglikelihood function: returned shape is %s, expected %s" % (p.shape, (2, self.num_params)))
        assert np.isfinite(logl).all(), ("Error in loglikelihood function: returned non-finite number: %s for input u=%s p=%s" % (logl, u, p))

        def safe_loglike(x):
            """Call likelihood function safely wrapped to avoid non-finite values."""
            x = np.asarray(x)
            logl = loglike(x)
            assert np.isfinite(logl).all(), (
                'User-provided loglikelihood returned non-finite value:',
                logl[~np.isfinite(logl)][0],
                "for input value:",
                x[~np.isfinite(logl),:][0,:])
            return logl

        self.loglike = safe_loglike

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
        except Exception:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = self.mpi_rank == 0
        self.log_to_disk = self.log and log_dir is not None

        if self.log and log_dir is not None:
            self.logs = make_run_dir(log_dir, run_num, append_run_num=append_run_num)
            log_dir = self.logs['run_dir']
        else:
            log_dir = None

        self.logger = create_logger(__name__ + '.' + type(self).__name__, log_dir=log_dir)

        if self.log:
            self.logger.info('Num live points [%d]', self.num_live_points)

        if self.log_to_disk:
            # self.pointstore = TextPointStore(os.path.join(self.logs['results'], 'points.tsv'), 2 + self.x_dim + self.num_params)
            self.pointstore = HDF5PointStore(
                os.path.join(self.logs['results'], 'points.hdf5'),
                3 + self.x_dim + self.num_params, mode='a' if resume else 'w')
        else:
            self.pointstore = NullPointStore(3 + self.x_dim + self.num_params)

    def run(
            self,
            update_interval_iter=None,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=0.001,
            max_iters=None):
        """Explore parameter space.

        Parameters
        ----------
        update_interval_iter:
            Update region after this many iterations.
        update_interval_ncall:
            Update region after update_interval_ncall likelihood calls.
        log_interval:
            Update stdout status line every log_interval iterations
        dlogz:
            Target evidence uncertainty.
        max_iters:
            maximum number of integration iterations.

        """
        if update_interval_ncall is None:
            update_interval_ncall = max(1, round(self.num_live_points))

        if update_interval_iter is None:
            if update_interval_ncall == 0:
                update_interval_iter = max(1, round(self.num_live_points))
            else:
                update_interval_iter = max(1, round(0.2 * self.num_live_points))

        if log_interval is None:
            log_interval = max(1, round(0.2 * self.num_live_points))
        else:
            log_interval = round(log_interval)
            if log_interval < 1:
                raise ValueError("log_interval must be >= 1")

        viz_callback = get_default_viz_callback()

        prev_u = []
        prev_v = []
        prev_logl = []
        if self.log:
            # try to resume:
            self.logger.info('Resuming...')
            for i in range(self.num_live_points):
                _, row = self.pointstore.pop(-np.inf)
                if row is not None:
                    prev_logl.append(row[1])
                    prev_u.append(row[3:3 + self.x_dim])
                    prev_v.append(row[3 + self.x_dim:3 + self.x_dim + self.num_params])
                else:
                    break

            prev_u = np.array(prev_u)
            prev_v = np.array(prev_v)
            prev_logl = np.array(prev_logl)

            num_live_points_missing = self.num_live_points - len(prev_logl)
        else:
            num_live_points_missing = -1

        if self.use_mpi:
            num_live_points_missing = self.comm.bcast(num_live_points_missing, root=0)
            prev_u = self.comm.bcast(prev_u, root=0)
            prev_v = self.comm.bcast(prev_v, root=0)
            prev_logl = self.comm.bcast(prev_logl, root=0)

        use_point_stack = True

        assert num_live_points_missing >= 0
        if num_live_points_missing > 0:
            if self.use_mpi:
                # self.logger.info('Using MPI with rank [%d]', self.mpi_rank)
                if self.mpi_rank == 0:
                    active_u = np.random.uniform(size=(num_live_points_missing, self.x_dim))
                else:
                    active_u = np.empty((num_live_points_missing, self.x_dim), dtype=np.float64)
                active_u = self.comm.bcast(active_u, root=0)
            else:
                active_u = np.random.uniform(size=(num_live_points_missing, self.x_dim))
            active_v = self.transform(active_u)

            if self.use_mpi:
                if self.mpi_rank == 0:
                    chunks = [[] for _ in range(self.mpi_size)]
                    for i, chunk in enumerate(active_v):
                        chunks[i % self.mpi_size].append(chunk)
                else:
                    chunks = None
                data = self.comm.scatter(chunks, root=0)
                active_logl = self.loglike(data)
                recv_active_logl = self.comm.gather(active_logl, root=0)
                recv_active_logl = self.comm.bcast(recv_active_logl, root=0)
                active_logl = np.concatenate(recv_active_logl, axis=0)
            else:
                active_logl = self.loglike(active_v)

            if self.log_to_disk:
                for i in range(num_live_points_missing):
                    self.pointstore.add(
                        _listify([-np.inf, active_logl[i], 0.], active_u[i,:], active_v[i,:]),
                        num_live_points_missing)

            if len(prev_u) > 0:
                active_u = np.concatenate((prev_u, active_u))
                active_v = np.concatenate((prev_v, active_v))
                active_logl = np.concatenate((prev_logl, active_logl))
            assert active_u.shape == (self.num_live_points, self.x_dim)
            assert active_v.shape == (self.num_live_points, self.num_params)
            assert active_logl.shape == (self.num_live_points,)
        else:
            active_u = prev_u
            active_v = prev_v
            active_logl = prev_logl

        saved_u = []
        saved_v = []  # Stored points for posterior results
        saved_logl = []
        saved_logwt = []
        h = 0.0  # Information, initially 0.
        logz = -1e300  # ln(Evidence Z), initially Z=0
        logvol = log(1.0 - exp(-1.0 / self.num_live_points))
        logz_remain = np.max(active_logl)
        fraction_remain = 1.0
        ncall = num_live_points_missing  # number of calls we already made
        first_time = True
        if self.x_dim > 1:
            transformLayer = AffineLayer(wrapped_dims=self.wrapped_axes)
        else:
            transformLayer = ScalingLayer(wrapped_dims=self.wrapped_axes)
        transformLayer.optimize(active_u, active_u)
        region = MLFriends(active_u, transformLayer)

        if self.log:
            self.logger.info('Starting sampling ...')
        ib = 0
        samples = []
        ndraw = 100
        it = 0
        next_update_interval_ncall = -1
        next_update_interval_iter = -1

        while max_iters is None or it < max_iters:

            # Worst object in collection and its weight (= volume * likelihood)
            worst = np.argmin(active_logl)
            logwt = logvol + active_logl[worst]

            # Update evidence Z and information h.
            logz_new = np.logaddexp(logz, logwt)
            h = (exp(logwt - logz_new) * active_logl[worst] + exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new

            # Add worst object to samples.
            saved_u.append(np.array(active_u[worst]))
            saved_v.append(np.array(active_v[worst]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[worst])

            # expected_vol = np.exp(-it / self.num_live_points)

            # The new likelihood constraint is that of the worst object.
            loglstar = active_logl[worst]

            if ncall > next_update_interval_ncall and it > next_update_interval_iter:

                if first_time:
                    nextregion = region
                else:
                    # rebuild space
                    # print()
                    # print("rebuilding space...", active_u.shape, active_u)
                    nextTransformLayer = transformLayer.create_new(active_u, region.maxradiussq)
                    nextregion = MLFriends(active_u, nextTransformLayer)

                # print("computing maxradius...")
                r, f = _update_region_bootstrap(nextregion, 30, 0., self.comm if self.use_mpi else None, self.mpi_size)

                nextregion.maxradiussq = r
                nextregion.enlarge = f
                # force shrinkage of volume
                # this is to avoid re-connection of dying out nodes
                if nextregion.estimate_volume() < region.estimate_volume():
                    region = nextregion
                    transformLayer = region.transformLayer
                region.create_ellipsoid(minvol=exp(-it / self.num_live_points) * self.volfactor)

                if self.log:
                    viz_callback(
                        points=dict(u=active_u, p=active_v, logl=active_logl),
                        info=dict(
                            it=it, ncall=ncall, logz=logz, logz_remain=logz_remain,
                            paramnames=self.paramnames + self.derivedparamnames,
                            logvol=logvol),
                        region=region, transformLayer=transformLayer)
                    self.pointstore.flush()

                next_update_interval_ncall = ncall + update_interval_ncall
                next_update_interval_iter = it + update_interval_iter
                first_time = False

            while True:
                if ib >= len(samples) and use_point_stack:
                    # root checks the point store
                    next_point = np.zeros((1, 3 + self.x_dim + self.num_params))

                    if self.log_to_disk:
                        _, stored_point = self.pointstore.pop(loglstar)
                        if stored_point is not None:
                            next_point[0,:] = stored_point
                        else:
                            next_point[0,:] = -np.inf
                        use_point_stack = not self.pointstore.stack_empty

                    if self.use_mpi:  # and informs everyone
                        use_point_stack = self.comm.bcast(use_point_stack, root=0)
                        next_point = self.comm.bcast(next_point, root=0)

                    # assert not use_point_stack

                    # unpack
                    likes = next_point[:,1]
                    samples = next_point[:,3:3 + self.x_dim]
                    samplesv = next_point[:,3 + self.x_dim:3 + self.x_dim + self.num_params]
                    # skip if we already know it is not useful
                    ib = 0 if np.isfinite(likes[0]) else 1

                while ib >= len(samples):
                    # get new samples
                    ib = 0

                    nc = 0
                    u = region.sample(nsamples=ndraw)
                    nu = u.shape[0]
                    if nu == 0:
                        v = np.empty((0, self.x_dim))
                        logl = np.empty((0,))
                    else:
                        v = self.transform(u)
                        logl = self.loglike(v)
                        nc += nu
                        accepted = logl > loglstar
                        u = u[accepted,:]
                        v = v[accepted,:]
                        logl = logl[accepted]
                        # father = father[accepted]

                    # collect results from all MPI members
                    if self.use_mpi:
                        recv_samples = self.comm.gather(u, root=0)
                        recv_samplesv = self.comm.gather(v, root=0)
                        recv_likes = self.comm.gather(logl, root=0)
                        recv_nc = self.comm.gather(nc, root=0)
                        recv_samples = self.comm.bcast(recv_samples, root=0)
                        recv_samplesv = self.comm.bcast(recv_samplesv, root=0)
                        recv_likes = self.comm.bcast(recv_likes, root=0)
                        recv_nc = self.comm.bcast(recv_nc, root=0)
                        samples = np.concatenate(recv_samples, axis=0)
                        samplesv = np.concatenate(recv_samplesv, axis=0)
                        likes = np.concatenate(recv_likes, axis=0)
                        ncall += sum(recv_nc)
                    else:
                        samples = np.array(u)
                        samplesv = np.array(v)
                        likes = np.array(logl)
                        ncall += nc

                    if self.log:
                        for ui, vi, logli in zip(samples, samplesv, likes):
                            self.pointstore.add(
                                _listify([loglstar, logli, 0.0], ui, vi),
                                ncall)

                if likes[ib] > loglstar:
                    active_u[worst] = samples[ib, :]
                    active_v[worst] = samplesv[ib,:]
                    active_logl[worst] = likes[ib]

                    # if we keep the region informed about the new live points
                    # then the region follows the live points even if maxradius is not updated
                    region.u[worst,:] = active_u[worst]
                    region.unormed[worst,:] = region.transformLayer.transform(region.u[worst,:])

                    # if we track the cluster assignment, then in the next round
                    # the ids with the same members are likely to have the same id
                    # this is imperfect
                    # transformLayer.clusterids[worst] = transformLayer.clusterids[father[ib]]
                    # so we just mark the replaced ones as "unassigned"
                    transformLayer.clusterids[worst] = 0
                    ib = ib + 1
                    break
                else:
                    ib = ib + 1

            # Shrink interval
            logvol -= 1.0 / self.num_live_points
            logz_remain = np.max(active_logl) - it / self.num_live_points
            fraction_remain = np.logaddexp(logz, logz_remain) - logz

            if it % log_interval == 0 and self.log:
                # nicelogger(self.paramnames, active_u, active_v, active_logl, it, ncall, logz, logz_remain, region=region)
                sys.stdout.write('Z=%.1g+%.1g | Like=%.1g..%.1g | it/evals=%d/%d eff=%.4f%%  \r' % (
                    logz, logz_remain, loglstar, np.max(active_logl), it,
                    ncall, np.inf if ncall == 0 else it * 100 / ncall))
                sys.stdout.flush()

                # if efficiency becomes low, bulk-process larger arrays
                ndraw = max(128, min(16384, round((ncall + 1) / (it + 1) / self.mpi_size)))

            # Stopping criterion
            if fraction_remain < dlogz:
                break
            it = it + 1

        logvol = -len(saved_v) / self.num_live_points - log(self.num_live_points)
        for i in range(self.num_live_points):
            logwt = logvol + active_logl[i]
            logz_new = np.logaddexp(logz, logwt)
            h = (exp(logwt - logz_new) * active_logl[i] + exp(logz - logz_new) * (h + logz) - logz_new)
            logz = logz_new
            saved_u.append(np.array(active_u[i]))
            saved_v.append(np.array(active_v[i]))
            saved_logwt.append(logwt)
            saved_logl.append(active_logl[i])

        saved_u = np.array(saved_u)
        saved_v = np.array(saved_v)
        saved_wt = exp(np.array(saved_logwt) - logz)
        saved_logl = np.array(saved_logl)
        logzerr = np.sqrt(h / self.num_live_points)

        if self.log_to_disk:
            with open(os.path.join(self.logs['results'], 'final.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['niter', 'ncall', 'logz', 'logzerr', 'h'])
                writer.writerow([it + 1, ncall, logz, logzerr, h])
            self.pointstore.close()

        if not self.use_mpi or self.mpi_rank == 0:
            print()
            print("niter: {:d}\n ncall: {:d}\n nsamples: {:d}\n logz: {:6.3f} +/- {:6.3f}\n h: {:6.3f}"
                  .format(it + 1, ncall, len(saved_v), logz, logzerr, h))

        self.results = dict(
            samples=resample_equal(saved_v, saved_wt / saved_wt.sum()),
            ncall=ncall, niter=it, logz=logz, logzerr=logzerr,
            weighted_samples=dict(
                upoints=saved_u, points=saved_v, weights=saved_wt,
                logweights=saved_logwt, logl=saved_logl),
        )

        return self.results

    def print_results(self):
        """Give summary of marginal likelihood and parameters."""
        print()
        print('logZ = %(logz).3f +- %(logzerr).3f' % self.results)

        print()
        for i, p in enumerate(self.paramnames + self.derivedparamnames):
            v = self.results['samples'][:,i]
            sigma = v.std()
            med = v.mean()
            if sigma == 0:
                i = 3
            else:
                i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join(['    %-20s' + fmt + " +- " + fmt])
            print(fmts % (p, med, sigma))

    def plot(self):
        """Make corner plot."""
        if self.log_to_disk:
            import matplotlib.pyplot as plt
            import corner
            data = np.array(self.results['weighted_samples']['points'])
            weights = np.array(self.results['weighted_samples']['weights'])
            cumsumweights = np.cumsum(weights)

            mask = cumsumweights > 1e-4

            corner.corner(
                data[mask,:], weights=weights[mask],
                labels=self.paramnames + self.derivedparamnames,
                show_titles=True)
            plt.savefig(os.path.join(self.logs['plots'], 'corner.pdf'), bbox_inches='tight')
            plt.close()


def warmstart_from_similar_file(
    usample_filename,
    param_names,
    loglike,
    transform,
    vectorized=False,
    min_num_samples=50
):
    """Warmstart from a previous run.

    Usage::

        aux_paramnames, aux_log_likelihood, aux_prior_transform, vectorized = warmstart_from_similar_file(
            'model1/chains/weighted_post_untransformed.txt', parameters, log_likelihood_with_background, prior_transform)

        aux_sampler = ReactiveNestedSampler(aux_paramnames, aux_log_likelihood, transform=aux_prior_transform,vectorized=vectorized)
        aux_sampler.run()
        posterior_samples = aux_results['samples'][:,-1]

    See :py:func:`ultranest.hotstart.get_auxiliary_contbox_parameterization`
    for more information.

    The remaining parameters have the same meaning as in :py:class:`ReactiveNestedSampler`.

    Parameters
    ------------
    usample_filename: str
        'directory/chains/weighted_post_untransformed.txt'
        contains posteriors in u-space (untransformed) of a previous run.
        Columns are weight, logl, param1, param2, ...
    min_num_samples: int
        minimum number of samples in the usample_filename file required.
        Too few samples will give a poor approximation.

    Other Parameters
    -----------------
    param_names: list
    loglike: function
    transform: function
    vectorized: bool

    Returns
    ---------
    aux_param_names: list
        new parameter list
    aux_loglikelihood: function
        new loglikelihood function
    aux_transform: function
        new prior transform function
    vectorized: bool
        whether the new functions are vectorized
    """
    # load samples
    try:
        with open(usample_filename) as f:
            old_param_names = f.readline().lstrip('#').strip().split()
            auxiliary_usamples = np.loadtxt(f)
    except IOError:
        warnings.warn('not hot-resuming, could not load file "%s"' % usample_filename)
        return param_names, loglike, transform, vectorized

    ulogl = auxiliary_usamples[:,1]
    uweights_full = auxiliary_usamples[:,0] * np.exp(ulogl - ulogl.max())
    mask = uweights_full > 0
    uweights = uweights_full[mask]
    uweights /= uweights.sum()
    upoints = auxiliary_usamples[mask,2:]
    del auxiliary_usamples

    nsamples = len(upoints)
    if nsamples < min_num_samples:
        raise ValueError('file "%s" has too few samples (%d) to hot-resume' % (usample_filename, nsamples))

    # check that the parameter meanings have not changed
    if old_param_names != ['weight', 'logl'] + param_names:
        raise ValueError('file "%s" has parameters %s, expected %s, cannot hot-resume.' % (usample_filename, old_param_names, param_names))

    return get_auxiliary_contbox_parameterization(
        param_names, loglike=loglike, transform=transform,
        vectorized=vectorized,
        upoints=upoints,
        uweights=uweights,
    )


class ReactiveNestedSampler(object):
    """Nested sampler with reactive exploration strategy.

    Storage & resume capable, optionally MPI parallelised.
    """

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 derived_param_names=[],
                 wrapped_params=None,
                 resume='subfolder',
                 run_num=None,
                 log_dir=None,
                 num_test_samples=2,
                 draw_multiple=True,
                 num_bootstraps=30,
                 vectorized=False,
                 ndraw_min=128,
                 ndraw_max=65536,
                 storage_backend='hdf5',
                 warmstart_max_tau=-1,
                 ):
        """Initialise nested sampler.

        Parameters
        -----------
        param_names: list of str, names of the parameters.
            Length gives dimensionality of the sampling problem.

        loglike: function
            log-likelihood function.
            Receives multiple parameter vectors, returns vector of likelihood.
        transform: function
            parameter transform from unit cube to physical parameters.
            Receives multiple cube vectors, returns multiple parameter vectors.

        derived_param_names: list of str
            Additional derived parameters created by transform. (empty by default)

        log_dir: str
            where to store output files
        resume: 'resume', 'resume-similar', 'overwrite' or 'subfolder'

            if 'overwrite', overwrite previous data.

            if 'subfolder', create a fresh subdirectory in log_dir.

            if 'resume' or True, continue previous run if available.
            Only works when dimensionality, transform or likelihood are consistent.

            if 'resume-similar', continue previous run if available.
            Only works when dimensionality and transform are consistent.
            If a likelihood difference is detected, the existing likelihoods
            are updated until the live point order differs.
            Otherwise, behaves like resume.

        run_num: int or None
            If resume=='subfolder', this is the subfolder number.
            Automatically increments if set to None.

        wrapped_params: list of bools
            indicating whether this parameter wraps around (circular parameter).

        num_test_samples: int
            test transform and likelihood with this number of
            random points for errors first. Useful to catch bugs.

        vectorized: bool
            If true, loglike and transform function can receive arrays
            of points.

        draw_multiple: bool
            If efficiency goes down, dynamically draw more points
            from the region between `ndraw_min` and `ndraw_max`.
            If set to False, few points are sampled at once.

        ndraw_min: int
            Minimum number of points to simultaneously propose.
            Increase this if your likelihood makes vectorization very cheap.

        ndraw_max: int
            Maximum number of points to simultaneously propose.
            Increase this if your likelihood makes vectorization very cheap.
            Memory allocation may be slow for extremely high values.

        num_bootstraps: int
            number of logZ estimators and MLFriends region
            bootstrap rounds.

        storage_backend: str or class
            Class to use for storing the evaluated points (see ultranest.store)
            'hdf5' is strongly recommended. 'tsv' and 'csv' are also possible.

        warmstart_max_tau: float
            Maximum disorder to accept when resume='resume-similar';
            Live points are reused as long as the live point order
            is below this normalised Kendall tau distance.
            Values from 0 (highly conservative) to 1 (extremely negligent).
        """
        self.paramnames = param_names
        x_dim = len(self.paramnames)

        self.sampler = 'reactive-nested'
        self.x_dim = x_dim
        self.transform_layer_class = AffineLayer if x_dim > 1 else ScalingLayer
        self.derivedparamnames = derived_param_names
        self.num_bootstraps = int(num_bootstraps)
        num_derived = len(self.derivedparamnames)
        self.num_params = x_dim + num_derived
        if wrapped_params is None:
            self.wrapped_axes = []
        else:
            assert len(wrapped_params) == self.x_dim, ("wrapped_params has the number of entries:", wrapped_params, ", expected", self.x_dim)
            self.wrapped_axes = np.where(wrapped_params)[0]

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
                self._setup_distributed_seeds()
        except Exception:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = self.mpi_rank == 0
        self.log_to_disk = self.log and log_dir is not None
        self.log_to_pointstore = self.log_to_disk

        assert resume in (True, 'overwrite', 'subfolder', 'resume', 'resume-similar'), \
            "resume should be one of 'overwrite' 'subfolder', 'resume' or 'resume-similar'"
        append_run_num = resume == 'subfolder'
        resume_similar = resume == 'resume-similar'
        resume = resume in ('resume-similar', 'resume', True)

        if self.log and log_dir is not None:
            self.logs = make_run_dir(log_dir, run_num, append_run_num=append_run_num)
            log_dir = self.logs['run_dir']
        else:
            log_dir = None

        if self.log:
            self.logger = create_logger('ultranest', log_dir=log_dir)
            self.logger.debug('ReactiveNestedSampler: dims=%d+%d, resume=%s, log_dir=%s, backend=%s, vectorized=%s, nbootstraps=%s, ndraw=%s..%s' % (
                x_dim, num_derived, resume, log_dir, storage_backend, vectorized,
                num_bootstraps, ndraw_min, ndraw_max,
            ))
        self.root = TreeNode(id=-1, value=-np.inf)

        self.pointpile = PointPile(self.x_dim, self.num_params)
        if self.log_to_pointstore:
            storage_filename = os.path.join(self.logs['results'], 'points.' + storage_backend)
            storage_num_cols = 3 + self.x_dim + self.num_params
            if storage_backend == 'tsv':
                self.pointstore = TextPointStore(storage_filename, storage_num_cols)
                self.pointstore.delimiter = '\n'
            elif storage_backend == 'csv':
                self.pointstore = TextPointStore(storage_filename, storage_num_cols)
                self.pointstore.delimiter = ','
            elif storage_backend == 'hdf5':
                self.pointstore = HDF5PointStore(storage_filename, storage_num_cols, mode='a' if resume else 'w')
            else:
                # use custom backend
                self.pointstore = storage_backend
        else:
            self.pointstore = NullPointStore(3 + self.x_dim + self.num_params)
        self.ncall = self.pointstore.ncalls
        self.ncall_region = 0

        if not vectorized:
            if transform is not None:
                transform = vectorize(transform)
            loglike = vectorize(loglike)
            draw_multiple = False

        self.draw_multiple = draw_multiple
        self.ndraw_min = ndraw_min
        self.ndraw_max = ndraw_max
        self.build_tregion = transform is not None
        if not self._check_likelihood_function(transform, loglike, num_test_samples):
            assert self.log_to_disk
            if resume_similar and self.log_to_disk:
                assert storage_backend == 'hdf5', 'resume-similar is only supported for HDF5 files'
                assert 0 <= warmstart_max_tau <= 1, 'warmstart_max_tau parameter needs to be set to a value between 0 and 1'
                # close
                self.pointstore.close()
                del self.pointstore
                # rewrite points file
                if self.log:
                    self.logger.info('trying to salvage points from previous, different run ...')
                resume_from_similar_file(
                    log_dir, x_dim, loglike, transform,
                    ndraw=ndraw_min if vectorized else 1,
                    max_tau=warmstart_max_tau, verbose=False)
                self.pointstore = HDF5PointStore(
                    os.path.join(self.logs['results'], 'points.hdf5'),
                    3 + self.x_dim + self.num_params, mode='a' if resume else 'w')
            elif resume:
                raise Exception("Cannot resume because loglikelihood function changed, "
                                "unless resume=resume-similar. To start from scratch, delete '%s'." % (log_dir))
        self._set_likelihood_function(transform, loglike, num_test_samples)
        self.stepsampler = None

    def _setup_distributed_seeds(self):
        if not self.use_mpi:
            return
        seed = 0
        if self.mpi_rank == 0:
            seed = np.random.randint(0, 1000000)

        seed = self.comm.bcast(seed, root=0)
        if self.mpi_rank > 0:
            # from http://arxiv.org/abs/1005.4117
            seed = int(abs(((seed * 181) * ((self.mpi_rank - 83) * 359)) % 104729))
            # print('setting seed:', self.mpi_rank, seed)
            np.random.seed(seed)

    def _check_likelihood_function(self, transform, loglike, num_test_samples):
        """Test the `transform` and `loglike`lihood functions.

        `num_test_samples` samples are used to check whether they work and give the correct output.

        returns whether the most recently stored point (if any)
        still returns the same likelihood value.
        """
        # do some checks on the likelihood function
        # this makes debugging easier by failing early with meaningful errors

        # if we are resuming, check that last sample still gives same result
        num_resume_test_samples = 0
        if num_test_samples and not self.pointstore.stack_empty:
            num_resume_test_samples = 1
            num_test_samples -= 1

        if num_test_samples > 0:
            # test with num_test_samples random points
            u = np.random.uniform(size=(num_test_samples, self.x_dim))
            p = transform(u) if transform is not None else u
            assert np.shape(p) == (num_test_samples, self.num_params), (
                "Error in transform function: returned shape is %s, expected %s" % (
                    np.shape(p), (num_test_samples, self.num_params)))
            logl = loglike(p)
            assert np.logical_and(u > 0, u < 1).all(), (
                "Error in transform function: u was modified!")
            assert np.shape(logl) == (num_test_samples,), (
                "Error in loglikelihood function: returned shape is %s, expected %s" % (np.shape(logl), (num_test_samples,)))
            assert np.isfinite(logl).all(), (
                "Error in loglikelihood function: returned non-finite number: %s for input u=%s p=%s" % (logl, u, p))

        if not self.pointstore.stack_empty and num_resume_test_samples > 0:
            # test that last sample gives the same likelihood value
            _, lastrow = self.pointstore.stack[-1]
            assert len(lastrow) == 3 + self.x_dim + self.num_params, (
                "Cannot resume: problem has different dimensionality",
                len(lastrow), (2, self.x_dim, self.num_params))
            lastL = lastrow[1]
            lastu = lastrow[3:3 + self.x_dim]
            u = lastu.reshape((1, -1))
            lastp = lastrow[3 + self.x_dim:3 + self.x_dim + self.num_params]
            if self.log:
                self.logger.debug("Testing resume consistency: %s: u=%s -> p=%s -> L=%s ", lastrow, lastu, lastp, lastL)
            p = transform(u) if transform is not None else u
            if not np.allclose(p.flatten(), lastp) and self.log:
                self.logger.warning(
                    "Trying to resume from previous run, but transform function gives different result: %s gave %s, now %s",
                    lastu, lastp, p.flatten())
            assert np.allclose(p.flatten(), lastp), (
                "Cannot resume because transform function changed. "
                "To start from scratch, delete '%s'." % (self.logs['run_dir']))
            logl = loglike(p).flatten()[0]
            if not np.isclose(logl, lastL) and self.log:
                self.logger.warning(
                    "Trying to resume from previous run, but likelihood function gives different result: %s gave %s, now %s",
                    lastu.flatten(), lastL, logl)
            return np.isclose(logl, lastL)
        return True

    def _set_likelihood_function(self, transform, loglike, num_test_samples, make_safe=False):
        """Store the transform and log-likelihood functions.

        if make_safe is set, make functions safer by accepting misformed
        return shapes and non-finite likelihood values.
        """

        def safe_loglike(x):
            """Safe wrapper of likelihood function."""
            x = np.asarray(x)
            if len(x.shape) == 1:
                assert x.shape[0] == self.x_dim
                x = np.expand_dims(x, 0)
            logl = loglike(x)
            if len(logl.shape) == 0:
                logl = np.expand_dims(logl, 0)
            logl[np.logical_not(np.isfinite(logl))] = -1e100
            return logl

        if make_safe:
            self.loglike = safe_loglike
        else:
            self.loglike = loglike

        if transform is None:
            self.transform = lambda x: x
        elif make_safe:
            def safe_transform(x):
                """Safe wrapper of transform function."""
                x = np.asarray(x)
                if len(x.shape) == 1:
                    assert x.shape[0] == self.x_dim
                    x = np.expand_dims(x, 0)
                return transform(x)
            self.transform = safe_transform
        else:
            self.transform = transform

        lims = np.ones((2, self.x_dim))
        lims[0,:] = 1e-6
        lims[1,:] = 1 - 1e-6
        self.transform_limits = self.transform(lims).transpose()

        self.volfactor = vol_prefactor(self.x_dim)

    def _widen_nodes(self, weighted_parents, weights, nnodes_needed, update_interval_ncall):
        """Ensure that at parents have `nnodes_needed` live points (parallel arcs).

        If not, fill up by sampling.
        """
        ndone = len(weighted_parents)
        if ndone == 0:
            if self.log:
                self.logger.info('No parents, so widening roots')
            self._widen_roots(nnodes_needed)
            return {}

        # select parents with weight 1/parent_weights
        p = 1. / np.array(weights)
        if (p == p[0]).all():
            parents = weighted_parents
        else:
            # preferentially select nodes with few parents, as those
            # have most weight
            i = np.random.choice(len(weighted_parents), size=nnodes_needed, p=p / p.sum())
            if self.use_mpi:
                i = self.comm.bcast(i, root=0)

            parents = [weighted_parents[ii] for ii in i]

        del weighted_parents, weights

        # sort from low to high
        parents.sort(key=operator.attrgetter('value'))
        Lmin = parents[0].value
        if np.isinf(Lmin):
            # some of the parents were born by sampling from the entire
            # prior volume. So we can efficiently apply a solution:
            # expand the roots
            if self.log:
                self.logger.info('parent value is -inf, so widening roots')
            self._widen_roots(nnodes_needed)
            return {}

        # double until we reach the necessary points
        # this is probably 1, from (2K - K) / K
        nsamples = int(np.ceil((nnodes_needed - ndone) / len(parents)))

        if self.log:
            self.logger.info('Will add %d live points (x%d) at L=%.1g ...', nnodes_needed - ndone, nsamples, Lmin)

        # add points where necessary (parents can have multiple entries)
        target_min_num_children = {}
        for n in parents:
            orign = target_min_num_children.get(n.id, len(n.children))
            target_min_num_children[n.id] = orign + nsamples

        return target_min_num_children

    def _widen_roots(self, nroots):
        """Ensure root has `nroots` children.

        Sample from prior to fill up (if needed).
        """
        if self.log and len(self.root.children) > 0:
            self.logger.info('Widening roots to %d live points (have %d already) ...', nroots, len(self.root.children))
        nnewroots = nroots - len(self.root.children)
        if nnewroots <= 0:
            # nothing to do
            return

        prev_u = []
        prev_v = []
        prev_logl = []
        prev_rowid = []

        if self.log and self.use_point_stack:
            # try to resume:
            # self.logger.info('Resuming...')
            for i in range(nnewroots):
                rowid, row = self.pointstore.pop(-np.inf)
                if row is None:
                    break
                prev_logl.append(row[1])
                prev_u.append(row[3:3 + self.x_dim])
                prev_v.append(row[3 + self.x_dim:3 + self.x_dim + self.num_params])
                prev_rowid.append(rowid)

        if self.log:
            prev_u = np.array(prev_u)
            prev_v = np.array(prev_v)
            prev_logl = np.array(prev_logl)

            num_live_points_missing = nnewroots - len(prev_logl)
        else:
            num_live_points_missing = -1

        if self.use_mpi:
            num_live_points_missing = self.comm.bcast(num_live_points_missing, root=0)
            prev_u = self.comm.bcast(prev_u, root=0)
            prev_v = self.comm.bcast(prev_v, root=0)
            prev_logl = self.comm.bcast(prev_logl, root=0)

        assert num_live_points_missing >= 0
        if self.log and num_live_points_missing > 0:
            self.logger.info('Sampling %d live points from prior ...', num_live_points_missing)
        if num_live_points_missing > 0:
            num_live_points_todo = distributed_work_chunk_size(num_live_points_missing, self.mpi_rank, self.mpi_size)
            self.ncall += num_live_points_missing

            if num_live_points_todo > 0:
                active_u = np.random.uniform(size=(num_live_points_todo, self.x_dim))
                active_v = self.transform(active_u)
                active_logl = self.loglike(active_v)
            else:
                active_u = np.empty((0, self.x_dim))
                active_v = np.empty((0, self.num_params))
                active_logl = np.empty((0,))

            if self.use_mpi:
                recv_samples = self.comm.gather(active_u, root=0)
                recv_samplesv = self.comm.gather(active_v, root=0)
                recv_likes = self.comm.gather(active_logl, root=0)
                recv_samples = self.comm.bcast(recv_samples, root=0)
                recv_samplesv = self.comm.bcast(recv_samplesv, root=0)
                recv_likes = self.comm.bcast(recv_likes, root=0)

                active_u = np.concatenate(recv_samples, axis=0)
                active_v = np.concatenate(recv_samplesv, axis=0)
                active_logl = np.concatenate(recv_likes, axis=0)

            assert active_logl.shape == (num_live_points_missing,), (active_logl.shape, num_live_points_missing)

            if self.log_to_pointstore:
                for i in range(num_live_points_missing):
                    rowid = self.pointstore.add(_listify(
                        [-np.inf, active_logl[i], 0.0],
                        active_u[i,:],
                        active_v[i,:]), 1)

            if len(prev_u) > 0:
                active_u = np.concatenate((prev_u, active_u))
                active_v = np.concatenate((prev_v, active_v))
                active_logl = np.concatenate((prev_logl, active_logl))
            assert active_u.shape == (nnewroots, self.x_dim), (active_u.shape, nnewroots, self.x_dim, num_live_points_missing, len(prev_u))
            assert active_v.shape == (nnewroots, self.num_params), (active_v.shape, nnewroots, self.num_params, num_live_points_missing, len(prev_u))
            assert active_logl.shape == (nnewroots,), (active_logl.shape, nnewroots)
        else:
            active_u = prev_u
            active_v = prev_v
            active_logl = prev_logl

        roots = [self.pointpile.make_node(logl, u, p) for u, p, logl in zip(active_u, active_v, active_logl)]
        if len(active_u) > 4:
            self.build_tregion = not is_affine_transform(active_u, active_v)
        self.root.children += roots

    def _adaptive_strategy_advice(self, Lmin, parallel_values, main_iterator, minimal_widths, frac_remain, Lepsilon):
        """Check if integration is done.

        Returns range where more sampling is needed

        Returns
        --------
        Llo: float
            lower log-likelihood bound, nan if done
        Lhi: float
            lower log-likelihood bound, nan if done

        Parameters
        -----------
        Lmin: float
            current loglikelihood threshold
        parallel_values: array of floats
            loglikelihoods of live points
        main_iterator: BreadthFirstIterator
            current tree exploration iterator
        minimal_widths: list
            current width required
        frac_remain: float
            maximum fraction of integral in remainder for termination
        Lepsilon: float
            loglikelihood accuracy threshold

        """
        Ls = parallel_values.copy()
        Ls.sort()
        # Ls = [node.value] + [n.value for rootid2, n in parallel_nodes]
        Lmax = Ls[-1]
        Lmin = Ls[0]

        # all points the same, stop
        if Lmax - Lmin < Lepsilon:
            return np.nan, np.nan

        # max remainder contribution is Lmax + weight, to be added to main_iterator.logZ
        # the likelihood that would add an equal amount as main_iterator.logZ is:
        logZmax = main_iterator.logZremain
        Lnext = logZmax - (main_iterator.logVolremaining + log(frac_remain)) - log(len(Ls))
        L1 = Ls[1] if len(Ls) > 1 else Ls[0]
        Lmax1 = np.median(Ls)
        Lnext = max(min(Lnext, Lmax1), L1)

        # if the remainder dominates, return that range
        if main_iterator.logZremain > main_iterator.logZ:
            return Lmin, Lnext

        if main_iterator.remainder_fraction > frac_remain:
            return Lmin, Lnext

        return np.nan, np.nan

    def _find_strategy(self, saved_logl, main_iterator, dlogz, dKL, min_ess):
        """Ask each strategy which log-likelihood interval needs more exploration.

        Returns
        -------
        (Llo_Z, Lhi_Z): floats
            interval where dlogz strategy requires more samples.
        (Llo_KL, Lhi_KL): floats
            interval where posterior uncertainty strategy requires more samples.
        (Llo_ess, Lhi_ess): floats
            interval where effective sample strategy requires more samples.

        Parameters
        ----------
        saved_logl: array of float
            loglikelihood values in integration
        main_iterator: BreadthFirstIterator
            current tree exploration iterator
        dlogz: float
            required logZ accuracy (smaller is stricter)
        dKL: float
            required Kulback-Leibler information gain between bootstrapped
            nested sampling incarnations (smaller is stricter).
        min_ess: float
            required number of effective samples (higher is stricter).

        """
        saved_logl = np.asarray(saved_logl)
        logw = np.asarray(main_iterator.logweights) + saved_logl.reshape((-1,1)) - main_iterator.all_logZ
        ref_logw = logw[:,0].reshape((-1,1))
        other_logw = logw[:,1:]

        Llo_ess = np.inf
        Lhi_ess = -np.inf
        w = exp(ref_logw.flatten())
        w /= w.sum()
        ess = len(w) / (1.0 + ((len(w) * w - 1)**2).sum() / len(w))
        if ess < min_ess:
            samples = np.random.choice(len(w), p=w, size=min_ess)
            Llo_ess = saved_logl[samples].min()
            Lhi_ess = saved_logl[samples].max()
        if self.log and Lhi_ess > Llo_ess:
            self.logger.info("Effective samples strategy wants to improve: %.2f..%.2f (ESS = %.1f, need >%d)",
                             Llo_ess, Lhi_ess, ess, min_ess)
        elif self.log and min_ess > 0:
            self.logger.info("Effective samples strategy satisfied (ESS = %.1f, need >%d)",
                             ess, min_ess)

        # compute KL divergence
        with np.errstate(invalid='ignore'):
            KL = np.where(np.isfinite(other_logw), exp(other_logw) * (other_logw - ref_logw), 0)
        KLtot = KL.sum(axis=0)
        dKLtot = np.abs(KLtot - KLtot.mean())
        p = np.where(KL > 0, KL, 0)
        p /= p.sum(axis=0).reshape((1, -1))

        Llo_KL = np.inf
        Lhi_KL = -np.inf
        for i, (pi, dKLi, logwi) in enumerate(zip(p.transpose(), dKLtot, other_logw)):
            if dKLi > dKL:
                ilo, ihi = _get_cumsum_range(pi, 1. / 400)
                # ilo and ihi are most likely missing in this iterator
                # --> select the one before/after in this iterator
                ilos = np.where(np.isfinite(logwi[:ilo]))[0]
                ihis = np.where(np.isfinite(logwi[ihi:]))[0]
                ilo2 = ilos[-1] if len(ilos) > 0 else 0
                ihi2 = (ihi + ihis[0]) if len(ihis) > 0 else -1
                # self.logger.info('   - KL[%d] = %.2f: need to improve near %.2f..%.2f --> %.2f..%.2f' % (
                #  i, dKLi, saved_logl[ilo], saved_logl[ihi], saved_logl[ilo2], saved_logl[ihi2]))
                Llo_KL = min(Llo_KL, saved_logl[ilo2])
                Lhi_KL = max(Lhi_KL, saved_logl[ihi2])

        if self.log and Lhi_KL > Llo_KL:
            self.logger.info("Posterior uncertainty strategy wants to improve: %.2f..%.2f (KL: %.2f+-%.2f nat, need <%.2f nat)",
                             Llo_KL, Lhi_KL, KLtot.mean(), dKLtot.max(), dKL)
        elif self.log:
            self.logger.info("Posterior uncertainty strategy is satisfied (KL: %.2f+-%.2f nat, need <%.2f nat)",
                             KLtot.mean(), dKLtot.max(), dKL)

        Nlive_min = 0
        p = exp(logw)
        p /= p.sum(axis=0).reshape((1, -1))
        deltalogZ = np.abs(main_iterator.all_logZ[1:] - main_iterator.logZ)

        tail_fraction = w[np.asarray(main_iterator.istail)].sum() / w.sum()
        logzerr_tail = logaddexp(log(tail_fraction) + main_iterator.logZ, main_iterator.logZ) - main_iterator.logZ
        maxlogzerr = max(main_iterator.logZerr, deltalogZ.max(), main_iterator.logZerr_bs)
        if maxlogzerr > dlogz:
            if logzerr_tail > maxlogzerr:
                if self.log:
                    self.logger.info("logz error is dominated by tail. Decrease frac_remain to make progress.")
            # very convervative estimation using all iterations
            # this punishes short intervals with many live points
            niter_max = len(saved_logl)
            Nlive_min = int(np.ceil(niter_max**0.5 / dlogz))
            if self.log:
                self.logger.debug("  conservative estimate says at least %d live points are needed to reach dlogz goal", Nlive_min)

            # better estimation:

            # get only until where logz bulk is (random sample here)
            itmax = np.random.choice(len(w), p=w)
            # back out nlive sequence (width changed by (1 - exp(-1/N))*(exp(-1/N)) )
            logweights = np.array(main_iterator.logweights[:itmax])
            with np.errstate(divide='ignore', invalid='ignore'):
                widthratio = 1 - np.exp(logweights[1:,0] - logweights[:-1,0])
                nlive = 1. / np.log((1 - np.sqrt(1 - 4 * widthratio)) / (2 * widthratio))
                nlive[~(nlive > 1)] = 1

            # build iteration groups
            nlive_sets, niter = np.unique(nlive.astype(int), return_counts=True)
            if self.log:
                self.logger.debug(
                    "  number of live points vary between %.0f and %.0f, most (%d/%d iterations) have %d",
                    nlive.min(), nlive.max(), niter.max(), itmax, nlive_sets[niter.argmax()])
            for nlive_floor in nlive_sets:
                # estimate error if this was the minimum nlive applied
                nlive_adjusted = np.where(nlive_sets < nlive_floor, nlive_floor, nlive_sets)
                deltalogZ_expected = (niter / nlive_adjusted**2.0).sum()**0.5
                if deltalogZ_expected < dlogz:
                    # achievable with Nlive_min
                    Nlive_min = int(nlive_floor)
                    if self.log:
                        self.logger.debug("  at least %d live points are needed to reach dlogz goal", Nlive_min)
                    break

        if self.log and Nlive_min > 0:
            self.logger.info(
                "Evidency uncertainty strategy wants %d minimum live points (dlogz from %.2f to %.2f, need <%s)",
                Nlive_min, deltalogZ.mean(), deltalogZ.max(), dlogz)
        elif self.log:
            self.logger.info(
                "Evidency uncertainty strategy is satisfied (dlogz=%.2f, need <%s)",
                (main_iterator.logZerr_bs**2 + logzerr_tail**2)**0.5, dlogz)
        if self.log:
            self.logger.info(
                '  logZ error budget: single: %.2f bs:%.2f tail:%.2f total:%.2f required:<%.2f',
                main_iterator.logZerr, main_iterator.logZerr_bs, logzerr_tail,
                (main_iterator.logZerr_bs**2 + logzerr_tail**2)**0.5, dlogz)

        return Nlive_min, (Llo_KL, Lhi_KL), (Llo_ess, Lhi_ess)

    def _refill_samples(self, Lmin, ndraw, nit):
        """Get new samples from region."""
        nc = 0
        u = self.region.sample(nsamples=ndraw)
        assert np.logical_and(u > 0, u < 1).all(), (u)
        nu = u.shape[0]
        if nu == 0:
            v = np.empty((0, self.num_params))
            logl = np.empty((0,))
            accepted = np.empty(0, dtype=bool)
        else:
            if nu > 1 and not self.draw_multiple:
                # peel off first if multiple evaluation is not supported
                nu = 1
                u = u[:1,:]

            v = self.transform(u)
            logl = np.ones(nu) * -np.inf

            if self.tregion is not None:
                # check wrapping ellipsoid in transformed space
                accepted = self.tregion.inside(v)
                nt = accepted.sum()
            else:
                # if undefined, all pass; rarer branch
                accepted = np.ones(nu, dtype=bool)
                nt = nu

            if nt > 0:
                logl[accepted] = self.loglike(v[accepted, :])
                nc += nt
            accepted = logl > Lmin

            # print("it: %4d ndraw: %d -> %d -> %d -> %d " % (nit, ndraw, nu, nt, accepted.sum()))

        if not self.sampling_slow_warned and nit * ndraw >= 100000 and nit > 20:
            warning_message1 = ("Sampling from region seems inefficient (%d/%d accepted in iteration %d). " % (accepted.sum(), ndraw, nit))
            warning_message2 = "To improve efficiency, modify the transformation so that the current live points%s are ellipsoidal, " + \
                "or use a stepsampler, or set frac_remain to a lower number (e.g., 0.5) to terminate earlier."
            if self.log_to_disk:
                debug_filename = os.path.join(self.logs['extra'], 'sampling-stuck-it%d')
                np.savez(
                    debug_filename + '.npz',
                    u=self.region.u, unormed=self.region.unormed,
                    maxradiussq=self.region.maxradiussq,
                    sample_u=u, sample_v=v, sample_logl=logl)
                np.savetxt(debug_filename + '.csv', self.region.u, delimiter=',')
                warning_message = warning_message1 + (warning_message2 % (' (stored for you in %s.csv)' % debug_filename))
            else:
                warning_message = warning_message1 + warning_message2 % ''
            warnings.warn(warning_message)
            logl_region = self.loglike(self.transform(self.region.u))
            if (logl_region == Lmin).all():
                raise ValueError(
                    "Region cannot sample a higher point. "
                    "All remaining live points have the same value.")
            if not (logl_region > Lmin).any():
                raise ValueError(
                    "Region cannot sample a higher point. "
                    "Perhaps you are resuming from a different problem?"
                    "Delete the output files and start again.")
            self.sampling_slow_warned = True

        self.ncall_region += ndraw
        return u[accepted,:], v[accepted,:], logl[accepted], nc, 0

    def _create_point(self, Lmin, ndraw, active_u, active_values):
        """Draw a new point above likelihood threshold `Lmin`.

        Parameters
        -----------
        Lmin: float
            loglikelihood threshold to draw above
        ndraw: float
            number of points to try to sample at once
        active_u: array of floats
            current live points
        active_values: array
            loglikelihoods of current live points

        """
        assert self.region.inside(active_u).any(), \
            ("None of the live points satisfies the current region!",
             self.region.maxradiussq, self.region.u, self.region.unormed, active_u,
             getattr(self.region, 'bbox_lo'),
             getattr(self.region, 'bbox_hi'),
             getattr(self.region, 'ellipsoid_cov'),
             getattr(self.region, 'ellipsoid_center'),
             getattr(self.region, 'ellipsoid_invcov'),
             getattr(self.region, 'ellipsoid_cov'),
             )

        nit = 0
        while True:
            ib = self.ib
            if ib >= len(self.samples) and self.use_point_stack:
                # root checks the point store
                next_point = np.zeros((1, 3 + self.x_dim + self.num_params)) * np.nan
                # print("1", self.mpi_rank, next_point)

                if self.log_to_pointstore:
                    _, stored_point = self.pointstore.pop(Lmin)
                    if stored_point is not None:
                        next_point[0,:] = stored_point
                    else:
                        next_point[0,:] = -np.inf
                    # print("2", self.mpi_rank, next_point)
                    self.use_point_stack = not self.pointstore.stack_empty

                if self.use_mpi:  # and informs everyone
                    self.use_point_stack = self.comm.bcast(self.use_point_stack, root=0)
                    # print("3", self.mpi_rank, next_point)
                    next_point = self.comm.bcast(next_point, root=0)

                # unpack
                if np.ndim(next_point) != 2:
                    print("XXXX ", self.mpi_rank, next_point, self.use_point_stack)
                self.likes = next_point[:,1]
                self.samples = next_point[:,3:3 + self.x_dim]
                self.samplesv = next_point[:,3 + self.x_dim:3 + self.x_dim + self.num_params]
                # skip if we already know it is not useful
                ib = 0 if np.isfinite(self.likes[0]) else 1

            use_stepsampler = self.stepsampler is not None
            while ib >= len(self.samples):
                ib = 0
                if use_stepsampler:
                    u, v, logl, nc = self.stepsampler.__next__(
                        self.region,
                        transform=self.transform, loglike=self.loglike,
                        Lmin=Lmin, us=active_u, Ls=active_values,
                        ndraw=ndraw, tregion=self.tregion)
                    quality = self.stepsampler.nsteps
                else:
                    u, v, logl, nc, quality = self._refill_samples(Lmin, ndraw, nit)
                nit += 1

                if logl is None:
                    u = np.empty((0, self.x_dim))
                    v = np.empty((0, self.num_params))
                    logl = np.empty((0,))
                elif u.ndim == 1:
                    assert np.logical_and(u > 0, u < 1).all(), (u)
                    u = u.reshape((1, self.x_dim))
                    v = v.reshape((1, self.num_params))
                    logl = logl.reshape((1,))

                if self.use_mpi:
                    recv_samples = self.comm.gather(u, root=0)
                    recv_samplesv = self.comm.gather(v, root=0)
                    recv_likes = self.comm.gather(logl, root=0)
                    recv_nc = self.comm.gather(nc, root=0)
                    recv_samples = self.comm.bcast(recv_samples, root=0)
                    recv_samplesv = self.comm.bcast(recv_samplesv, root=0)
                    recv_likes = self.comm.bcast(recv_likes, root=0)
                    recv_nc = self.comm.bcast(recv_nc, root=0)
                    self.samples = np.concatenate(recv_samples, axis=0)
                    self.samplesv = np.concatenate(recv_samplesv, axis=0)
                    self.likes = np.concatenate(recv_likes, axis=0)
                    self.ncall += sum(recv_nc)
                else:
                    self.samples = u
                    self.samplesv = v
                    self.likes = logl
                    self.ncall += nc

                if self.log:
                    for ui, vi, logli in zip(self.samples, self.samplesv, self.likes):
                        self.pointstore.add(
                            _listify([Lmin, logli, quality], ui, vi),
                            self.ncall)

            if self.likes[ib] > Lmin:
                u = self.samples[ib, :]
                assert np.logical_and(u > 0, u < 1).all(), (u)
                p = self.samplesv[ib, :]
                logl = self.likes[ib]

                self.ib = ib + 1
                return u, p, logl
            else:
                self.ib = ib + 1

    def _update_region(
        self, active_u, active_node_ids,
        bootstrap_rootids=None, active_rootids=None,
        nbootstraps=30, minvol=0., active_p=None
    ):
        """Build a new MLFriends region from `active_u`, and wrapping ellipsoid.

        Both are safely built using bootstrapping, so that the
        region can be used for sampling and rejecting points.
        If MPI is enabled, this computation is parallelised.

        If active_p is not None, a wrapping ellipsoid is built also
        in the user-transformed parameter space.

        Parameters
        -----------
        active_u: array of floats
            current live points
        active_node_ids: 2d array of ints
            which bootstrap initialisation the points belong to.
        active_rootids: 2d array of ints
            roots active in each bootstrap initialisation
        bootstrap_rootids: array of ints
            bootstrap samples. if None, they are drawn fresh.
        nbootstraps: int
            number of bootstrap rounds
        active_p: array of floats
            current live points, in user-transformed space
        minvol: float
            expected current minimum volume of region.

        Returns
        --------
        updated: bool
            True if update was made, False if previous region remained.

        """
        assert nbootstraps > 0
        updated = False
        if self.region is None:
            # if self.log:
            #    self.logger.debug("building first region ...")
            self.transformLayer = self.transform_layer_class(wrapped_dims=self.wrapped_axes)
            self.transformLayer.optimize(active_u, active_u, minvol=minvol)
            self.region = self.region_class(active_u, self.transformLayer)
            self.region_nodes = active_node_ids.copy()
            assert self.region.maxradiussq is None

            _update_region_bootstrap(self.region, nbootstraps, minvol, self.comm if self.use_mpi else None, self.mpi_size)
            self.region.create_ellipsoid(minvol=minvol)
            # if self.log:
            #     self.logger.debug("building first region ... r=%e, f=%e" % (r, f))
            updated = True

            # verify correctness:
            # self.region.create_ellipsoid(minvol=minvol)
            # assert self.region.inside(active_u).all(), self.region.inside(active_u).mean()

        assert self.transformLayer is not None
        need_accept = False

        if self.region.maxradiussq is None:
            # we have been told that radius is currently invalid
            # we need to bootstrap back to a valid state

            # compute radius given current transformLayer
            oldu = self.region.u
            self.region.u = active_u
            self.region_nodes = active_node_ids.copy()
            self.region.set_transformLayer(self.transformLayer)

            _update_region_bootstrap(self.region, nbootstraps, minvol, self.comm if self.use_mpi else None, self.mpi_size)

            # print("made first region, r=%e" % (r))

            # now that we have r, can do clustering
            # but such reclustering would forget the cluster ids

            # instead, track the clusters from before by matching manually
            oldt = self.transformLayer.transform(oldu)
            clusterids = np.zeros(len(active_u), dtype=int)
            nnearby = np.empty(len(self.region.unormed), dtype=int)
            for ci in np.unique(self.transformLayer.clusterids):
                if ci == 0:
                    continue

                # find points from that cluster
                oldti = oldt[self.transformLayer.clusterids == ci]
                # identify which new points are near this cluster
                find_nearby(oldti, self.region.unormed, self.region.maxradiussq, nnearby)
                mask = nnearby != 0
                # assign the nearby ones to this cluster
                # if they have not been set yet
                # if they have, set them to -1
                clusterids[mask] = np.where(clusterids[mask] == 0, ci, -1)

            # clusters we are unsure about (double assignments) go unassigned
            clusterids[clusterids == -1] = 0

            # tell scaling layer the correct cluster information
            self.transformLayer.clusterids = clusterids

            # we want the clustering to repeat to remove remaining zeros
            need_accept = (self.transformLayer.clusterids == 0).any()

            updated = True
            assert len(self.region.u) == len(self.transformLayer.clusterids)

            # verify correctness:
            self.region.create_ellipsoid(minvol=minvol)
            # assert self.region.inside(active_u).all(), self.region.inside(active_u).mean()

        assert len(self.region.u) == len(self.transformLayer.clusterids)
        # rebuild space
        with warnings.catch_warnings(), np.errstate(all='raise'):
            try:
                nextTransformLayer = self.transformLayer.create_new(active_u, self.region.maxradiussq, minvol=minvol)
                assert not (nextTransformLayer.clusterids == 0).any()
                _, cluster_sizes = np.unique(nextTransformLayer.clusterids, return_counts=True)
                smallest_cluster = cluster_sizes.min()
                if self.log and smallest_cluster == 1:
                    self.logger.debug(
                        "clustering found some stray points [need_accept=%s] %s",
                        need_accept,
                        np.unique(nextTransformLayer.clusterids, return_counts=True)
                    )

                nextregion = self.region_class(active_u, nextTransformLayer)
                assert np.isfinite(nextregion.unormed).all()

                if not nextTransformLayer.nclusters < 20:
                    if self.log:
                        self.logger.info(
                            "Found a lot of clusters: %d (%d with >1 members)",
                            nextTransformLayer.nclusters, (cluster_sizes > 1).sum())

                # if self.log:
                #     self.logger.info("computing maxradius...")
                r, f = _update_region_bootstrap(nextregion, nbootstraps, minvol, self.comm if self.use_mpi else None, self.mpi_size)
                # verify correctness:
                nextregion.create_ellipsoid(minvol=minvol)

                # check if live points are numerically colliding or linearly dependent
                self.live_points_healthy = len(active_u) > self.x_dim and \
                    np.all(np.sum(active_u[1:] != active_u[0], axis=0) > self.x_dim) and \
                    np.linalg.matrix_rank(nextregion.ellipsoid_cov) == self.x_dim

                assert (nextregion.u == active_u).all()
                assert np.allclose(nextregion.unormed, nextregion.transformLayer.transform(active_u))
                # assert nextregion.inside(active_u).all(),
                #  ("live points should live in new region, but only %.3f%% do." % (100 * nextregion.inside(active_u).mean()), active_u)
                good_region = nextregion.inside(active_u).all()
                # assert good_region
                if not good_region and self.log:
                    self.logger.warning("Proposed region is inconsistent (maxr=%f,enlarge=%f) and will be skipped.", r, f)

                # avoid cases where every point is its own cluster,
                # and even the largest cluster has fewer than x_dim points
                sensible_clustering = nextTransformLayer.nclusters < len(nextregion.u) \
                    and cluster_sizes.max() >= nextregion.u.shape[1]

                # force shrinkage of volume. avoids reconnecting dying modes
                if good_region and \
                        (need_accept or nextregion.estimate_volume() <= self.region.estimate_volume()) \
                        and sensible_clustering:
                    self.region = nextregion
                    self.transformLayer = self.region.transformLayer
                    self.region_nodes = active_node_ids.copy()
                    updated = True

                    assert not (self.transformLayer.clusterids == 0).any(), (self.transformLayer.clusterids, need_accept, updated)

            except Warning:
                if self.log:
                    self.logger.debug("not updating region", exc_info=True)
            except FloatingPointError:
                if self.log:
                    self.logger.debug("not updating region", exc_info=True)
            except np.linalg.LinAlgError:
                if self.log:
                    self.logger.debug("not updating region", exc_info=True)

        assert len(self.region.u) == len(self.transformLayer.clusterids)

        if active_p is None or not self.build_tregion:
            self.tregion = None
        else:
            try:
                with np.errstate(invalid='raise'):
                    tregion = WrappingEllipsoid(active_p)
                    f = tregion.compute_enlargement(
                        nbootstraps=max(1, nbootstraps // self.mpi_size))
                    if self.use_mpi:
                        recv_enlarge = self.comm.gather(f, root=0)
                        recv_enlarge = self.comm.bcast(recv_enlarge, root=0)
                        f = np.max(recv_enlarge)
                    tregion.enlarge = f
                    tregion.create_ellipsoid()
                    self.tregion = tregion
            except FloatingPointError:
                if self.log:
                    self.logger.debug("not updating t-ellipsoid", exc_info=True)
                    self.tregion = None
            except np.linalg.LinAlgError:
                if self.log:
                    self.logger.debug("not updating t-ellipsoid", exc_info=True)
                    self.tregion = None

        return updated

    def _expand_nodes_before(self, Lmin, nnodes_needed, update_interval_ncall):
        """Expand nodes before `Lmin` to have `nnodes_needed`.

        Returns
        --------
        Llo: float
            lowest parent sampled (-np.inf if sampling from root)
        Lhi: float
            Lmin
        target_min_num_children: int
            number of children that need to be maintained between Llo, Lhi

        """
        self.pointstore.reset()
        parents, weights = find_nodes_before(self.root, Lmin)
        target_min_num_children = self._widen_nodes(parents, weights, nnodes_needed, update_interval_ncall)
        if len(parents) == 0:
            Llo = -np.inf
        else:
            Llo = min(n.value for n in parents)
        Lhi = Lmin
        return Llo, Lhi, target_min_num_children

    def _should_node_be_expanded(
        self, it, Llo, Lhi, minimal_widths_sequence, target_min_num_children,
        node, parallel_values, max_ncalls, max_iters, live_points_healthy
    ):
        """Check if node needs new children.

        Returns
        -------
        expand_node: bool
            True if should sample a new point
            based on this node (above its likelihood value Lmin).

        Parameters
        ----------
        it: int
            current iteration
        node: node
            The node to consider
        parallel_values: array of floats
            loglikelihoods of live points
        max_ncalls: int
            maximum number of likelihood function calls allowed
        max_iters: int
            maximum number of nested sampling iteration allowed
        Llo: float
            lower loglikelihood bound for the strategy
        Lhi: float
            upper loglikelihood bound for the strategy
        minimal_widths_sequence: list
            list of likelihood intervals with minimum number of live points
        target_min_num_children:
            minimum number of live points currently targeted
        live_points_healthy: bool
            indicates whether the live points have become
            linearly dependent (covariance not full rank)
            or have attained the same exact value in some parameter.

        """
        Lmin = node.value
        nlive = len(parallel_values)

        if not (Lmin <= Lhi and Llo <= Lhi):
            return False

        if not live_points_healthy:
            if self.log:
                self.logger.debug("not expanding, because live points are linearly dependent")
            return False

        # some reasons to stop:
        if it > 0:
            if max_ncalls is not None and self.ncall >= max_ncalls:
                # print("not expanding, because above max_ncall")
                return False

            if max_iters is not None and it >= max_iters:
                # print("not expanding, because above max_iters")
                return False

        # in a plateau, only shrink (Fowlie+2020)
        if (Lmin == parallel_values).sum() > 1:
            if self.log:
                self.logger.debug("Plateau detected at L=%e, not replacing live point." % Lmin)
            return False

        expand_node = False
        # we should continue to progress towards Lhi
        while Lmin > minimal_widths_sequence[0][0]:
            minimal_widths_sequence.pop(0)

        # get currently desired width
        if self.region is None:
            minimal_width_clusters = 0
        else:
            # compute number of clusters with more than 1 element
            _, cluster_sizes = np.unique(self.region.transformLayer.clusterids, return_counts=True)
            nclusters = (cluster_sizes > 1).sum()
            minimal_width_clusters = self.cluster_num_live_points * nclusters

        minimal_width = max(minimal_widths_sequence[0][1], minimal_width_clusters)

        # if already has children, no need to expand
        # if we are wider than the width required
        # we do not need to expand this one
        # expand_node = len(node.children) == 0
        # prefer 1 child, or the number required, if specified
        nmin = target_min_num_children.get(node.id, 1) if target_min_num_children else 1
        expand_node = len(node.children) < nmin
        # print("not expanding, because we are quite wide", nlive, minimal_width, minimal_widths_sequence)
        # but we have to expand the first iteration,
        # otherwise the integrator never sets H
        too_wide = nlive > minimal_width and it > 0

        return expand_node and not too_wide

    def run(
            self,
            update_interval_volume_fraction=0.8,
            update_interval_ncall=None,
            log_interval=None,
            show_status=True,
            viz_callback='auto',
            dlogz=0.5,
            dKL=0.5,
            frac_remain=0.01,
            Lepsilon=0.001,
            min_ess=400,
            max_iters=None,
            max_ncalls=None,
            max_num_improvement_loops=-1,
            min_num_live_points=400,
            cluster_num_live_points=40,
            insertion_test_window=10,
            insertion_test_zscore_threshold=4,
            region_class=MLFriends,
    ):
        """Run until target convergence criteria are fulfilled.

        Parameters
        ----------
        update_interval_volume_fraction: float
            Update region when the volume shrunk by this amount.

        update_interval_ncall: int
            Update region after update_interval_ncall likelihood calls (not used).

        log_interval: int
            Update stdout status line every log_interval iterations

        show_status: bool
            show integration progress as a status line.
            If no output desired, set to False.

        viz_callback: function
            callback function when region was rebuilt. Allows to
            show current state of the live points.
            See :func:`nicelogger` or :class:`LivePointsWidget`.
            If no output desired, set to False.

        dlogz: float
            Target evidence uncertainty. This is the std
            between bootstrapped logz integrators.

        dKL: float
            Target posterior uncertainty. This is the
            Kullback-Leibler divergence in nat between bootstrapped integrators.

        frac_remain: float
            Integrate until this fraction of the integral is left in the remainder.
            Set to a low number (1e-2 ... 1e-5) to make sure peaks are discovered.
            Set to a higher number (0.5) if you know the posterior is simple.

        Lepsilon: float
            Terminate when live point likelihoods are all the same,
            within Lepsilon tolerance. Increase this when your likelihood
            function is inaccurate, to avoid unnecessary search.

        min_ess: int
            Target number of effective posterior samples.

        max_iters: int
            maximum number of integration iterations.

        max_ncalls: int
            stop after this many likelihood evaluations.

        max_num_improvement_loops: int
            run() tries to assess iteratively where more samples are needed.
            This number limits the number of improvement loops.

        min_num_live_points: int
            minimum number of live points throughout the run

        cluster_num_live_points: int
            require at least this many live points per detected cluster

        insertion_test_zscore_threshold: float
            z-score used as a threshold for the insertion order test.
            Set to infinity to disable.

        insertion_test_window: int
            Number of iterations after which the insertion order test is reset.

        region_class: MLFriends or RobustEllipsoidRegion
            Whether to use MLFriends+ellipsoidal+tellipsoidal region (better for multi-modal problems)
            or just ellipsoidal sampling (faster for high-dimensional, gaussian-like problems).
        """
        for result in self.run_iter(
            update_interval_volume_fraction=update_interval_volume_fraction,
            update_interval_ncall=update_interval_ncall,
            log_interval=log_interval,
            dlogz=dlogz, dKL=dKL,
            Lepsilon=Lepsilon, frac_remain=frac_remain,
            min_ess=min_ess, max_iters=max_iters,
            max_ncalls=max_ncalls, max_num_improvement_loops=max_num_improvement_loops,
            min_num_live_points=min_num_live_points,
            cluster_num_live_points=cluster_num_live_points,
            show_status=show_status,
            viz_callback=viz_callback,
            insertion_test_window=insertion_test_window,
            insertion_test_zscore_threshold=insertion_test_zscore_threshold,
            region_class=region_class,
        ):
            if self.log:
                self.logger.debug("did a run_iter pass!")
            pass
        if self.log:
            self.logger.info("done iterating.")

        return self.results

    def run_iter(
            self,
            update_interval_volume_fraction=0.8,
            update_interval_ncall=None,
            log_interval=None,
            dlogz=0.5,
            dKL=0.5,
            frac_remain=0.01,
            Lepsilon=0.001,
            min_ess=400,
            max_iters=None,
            max_ncalls=None,
            max_num_improvement_loops=-1,
            min_num_live_points=400,
            cluster_num_live_points=40,
            show_status=True,
            viz_callback='auto',
            insertion_test_window=10000,
            insertion_test_zscore_threshold=2,
            region_class=MLFriends
    ):
        """Iterate towards convergence.

        Use as an iterator like so::

            for result in sampler.run_iter(...):
                print('lnZ = %(logz).2f +- %(logzerr).2f' % result)

        Parameters as described in run() method.

        Yields
        ------
        results: dict
        """
        # frac_remain=1  means 1:1 -> dlogz=log(0.5)
        # frac_remain=0.1 means 1:10 -> dlogz=log(0.1)
        # dlogz_min = log(1./(1 + frac_remain))
        # dlogz_min = -log1p(frac_remain)
        if -np.log1p(frac_remain) > dlogz:
            raise ValueError("To achieve the desired logz accuracy, set frac_remain to a value much smaller than %s (currently: %s)" % (
                exp(-dlogz) - 1, frac_remain))

        # the error is approximately dlogz = sqrt(iterations) / Nlive
        # so we need a minimum, which depends on the number of iterations
        # fewer than 1000 iterations is quite unlikely
        if min_num_live_points < 1000**0.5 / dlogz:
            min_num_live_points = int(np.ceil(1000**0.5 / dlogz))
            if self.log:
                self.logger.info("To achieve the desired logz accuracy, min_num_live_points was increased to %d" % (
                    min_num_live_points))

        if self.log_to_pointstore:
            if len(self.pointstore.stack) > 0:
                self.logger.info("Resuming from %d stored points", len(self.pointstore.stack))
            self.use_point_stack = not self.pointstore.stack_empty
        else:
            self.use_point_stack = False

        assert min_num_live_points >= cluster_num_live_points, \
            ('min_num_live_points(%d) cannot be less than cluster_num_live_points(%d)' %
                (min_num_live_points, cluster_num_live_points))
        self.min_num_live_points = min_num_live_points
        self.cluster_num_live_points = cluster_num_live_points
        self.sampling_slow_warned = False
        self.build_tregion = True
        self.region_class = region_class
        update_interval_volume_log_fraction = log(update_interval_volume_fraction)

        if viz_callback == 'auto':
            viz_callback = get_default_viz_callback()

        self._widen_roots(min_num_live_points)

        Llo, Lhi = -np.inf, np.inf
        Lmax = -np.inf
        strategy_stale = True
        minimal_widths = []
        target_min_num_children = {}
        improvement_it = 0

        assert max_iters is None or max_iters > 0, ("Invalid value for max_iters: %s. Set to None or positive number" % max_iters)
        assert max_ncalls is None or max_ncalls > 0, ("Invalid value for max_ncalls: %s. Set to None or positive number" % max_ncalls)

        if self.log:
            self.logger.debug(
                'run_iter dlogz=%.1f, dKL=%.1f, frac_remain=%.2f, Lepsilon=%.4f, min_ess=%d' % (
                    dlogz, dKL, frac_remain, Lepsilon, min_ess)
            )
            self.logger.debug(
                'max_iters=%d, max_ncalls=%d, max_num_improvement_loops=%d, min_num_live_points=%d, cluster_num_live_points=%d' % (
                    max_iters if max_iters else -1, max_ncalls if max_ncalls else -1,
                    max_num_improvement_loops, min_num_live_points, cluster_num_live_points)
            )

        self.results = None

        while True:
            roots = self.root.children

            nroots = len(roots)

            if update_interval_ncall is None:
                update_interval_ncall = nroots

            if log_interval is None:
                log_interval = max(1, round(0.1 * nroots))
            else:
                log_interval = round(log_interval)
                if log_interval < 1:
                    raise ValueError("log_interval must be >= 1")

            explorer = BreadthFirstIterator(roots)
            # Integrating thing
            main_iterator = MultiCounter(
                nroots=len(roots),
                nbootstraps=max(1, self.num_bootstraps // self.mpi_size),
                random=False, check_insertion_order=False)
            main_iterator.Lmax = max(Lmax, max(n.value for n in roots))
            insertion_test = UniformOrderAccumulator()
            insertion_test_runs = []
            insertion_test_quality = np.inf
            insertion_test_direction = 0

            self.transformLayer = None
            self.region = None
            self.tregion = None
            self.live_points_healthy = True
            it_at_first_region = 0
            self.ib = 0
            self.samples = []
            if self.draw_multiple:
                ndraw = self.ndraw_min
            else:
                ndraw = 40
            self.pointstore.reset()
            if self.log_to_pointstore:
                self.use_point_stack = not self.pointstore.stack_empty
            else:
                self.use_point_stack = False
            if self.use_mpi:
                self.use_point_stack = self.comm.bcast(self.use_point_stack, root=0)

            if self.log and (np.isfinite(Llo) or np.isfinite(Lhi)):
                self.logger.info("Exploring (in particular: L=%.2f..%.2f) ...", Llo, Lhi)
            region_sequence = []
            minimal_widths_sequence = _sequentialize_width_sequence(minimal_widths, self.min_num_live_points)
            if self.log:
                self.logger.debug('minimal_widths_sequence: %s', minimal_widths_sequence)

            saved_nodeids = []
            saved_logl = []
            it = 0
            ncall_at_run_start = self.ncall
            ncall_region_at_run_start = self.ncall_region
            next_update_interval_volume = 1
            last_status = time.time()

            # we go through each live point (regardless of root) by likelihood value
            while True:
                next_node = explorer.next_node()
                if next_node is None:
                    break
                rootid, node, (_, active_rootids, active_values, active_node_ids) = next_node
                assert not isinstance(rootid, float)
                # this is the likelihood level we have to improve upon
                self.Lmin = Lmin = node.value

                # if within suggested range, expand
                if strategy_stale or not (Lmin <= Lhi) or not np.isfinite(Lhi) or (active_values == Lmin).all():
                    # check with advisor if we want to expand this node
                    Llo, Lhi = self._adaptive_strategy_advice(
                        Lmin, active_values, main_iterator,
                        minimal_widths, frac_remain, Lepsilon=Lepsilon)
                    # when we are going to the peak, numerical accuracy
                    # can become an issue. We should try not to get stuck there
                    strategy_stale = Lhi - Llo < max(Lepsilon, 0.01)

                expand_node = self._should_node_be_expanded(
                    it, Llo, Lhi, minimal_widths_sequence,
                    target_min_num_children, node, active_values,
                    max_ncalls, max_iters, self.live_points_healthy)

                region_fresh = False
                if expand_node:
                    # sample a new point above Lmin
                    active_u = self.pointpile.getu(active_node_ids)
                    active_p = self.pointpile.getp(active_node_ids)
                    nlive = len(active_u)
                    # first we check that the region is up-to-date
                    if main_iterator.logVolremaining < next_update_interval_volume:
                        if self.region is None:
                            it_at_first_region = it
                        region_fresh = self._update_region(
                            active_u=active_u, active_p=active_p, active_node_ids=active_node_ids,
                            active_rootids=active_rootids,
                            bootstrap_rootids=main_iterator.rootids[1:,],
                            nbootstraps=self.num_bootstraps,
                            minvol=exp(main_iterator.logVolremaining))

                        if region_fresh and self.stepsampler is not None:
                            self.stepsampler.region_changed(active_values, self.region)

                        _, cluster_sizes = np.unique(self.region.transformLayer.clusterids, return_counts=True)
                        nclusters = (cluster_sizes > 1).sum()
                        region_sequence.append((Lmin, nlive, nclusters, np.max(active_values)))

                        # next_update_interval_ncall = self.ncall + (update_interval_ncall or nlive)
                        next_update_interval_volume = main_iterator.logVolremaining + update_interval_volume_log_fraction

                        # provide nice output to follow what is going on
                        # but skip if we are resuming
                        #  and (self.ncall != ncall_at_run_start and it_at_first_region == it)
                        if self.log and viz_callback:
                            viz_callback(
                                points=dict(u=active_u, p=active_p, logl=active_values),
                                info=dict(
                                    it=it, ncall=self.ncall,
                                    logz=main_iterator.logZ,
                                    logz_remain=main_iterator.logZremain,
                                    logvol=main_iterator.logVolremaining,
                                    paramnames=self.paramnames + self.derivedparamnames,
                                    paramlims=self.transform_limits,
                                    order_test_correlation=insertion_test_quality,
                                    order_test_direction=insertion_test_direction,
                                ),
                                region=self.region, transformLayer=self.transformLayer,
                                region_fresh=region_fresh,
                            )
                        if self.log:
                            self.pointstore.flush()

                    if nlive < cluster_num_live_points * nclusters and improvement_it < max_num_improvement_loops:
                        # make wider here
                        if self.log:
                            self.logger.info(
                                "Found %d clusters, but only have %d live points, want %d.",
                                self.region.transformLayer.nclusters, nlive,
                                cluster_num_live_points * nclusters)
                        break

                    # sample point
                    u, p, L = self._create_point(Lmin=Lmin, ndraw=ndraw, active_u=active_u, active_values=active_values)
                    child = self.pointpile.make_node(L, u, p)
                    main_iterator.Lmax = max(main_iterator.Lmax, L)
                    if np.isfinite(insertion_test_zscore_threshold) and nlive > 1:
                        insertion_test.add((active_values < L).sum(), nlive)
                        if abs(insertion_test.zscore) > insertion_test_zscore_threshold:
                            insertion_test_runs.append(insertion_test.N)
                            insertion_test_quality = insertion_test.N
                            insertion_test_direction = np.sign(insertion_test.zscore)
                            insertion_test.reset()
                        elif insertion_test.N > insertion_test_window:
                            insertion_test_quality = np.inf
                            insertion_test_direction = 0
                            insertion_test.reset()

                    # identify which point is being replaced (from when we built the region)
                    worst = np.where(self.region_nodes == node.id)[0]
                    self.region_nodes[worst] = child.id
                    # if we keep the region informed about the new live points
                    # then the region follows the live points even if maxradius is not updated
                    self.region.u[worst] = u
                    self.region.unormed[worst] = self.region.transformLayer.transform(u)
                    # move also the ellipsoid
                    self.region.ellipsoid_center = np.mean(self.region.u, axis=0)
                    if self.tregion:
                        self.tregion.update_center(np.mean(active_p, axis=0))

                    # if we track the cluster assignment, then in the next round
                    # the ids with the same members are likely to have the same id
                    # this is imperfect
                    # transformLayer.clusterids[worst] = transformLayer.clusterids[father[ib]]
                    # so we just mark the replaced ones as "unassigned"
                    self.transformLayer.clusterids[worst] = 0

                    node.children.append(child)

                    if self.log and (region_fresh or it % log_interval == 0 or time.time() > last_status + 0.1):
                        last_status = time.time()
                        # the number of proposals asked from region
                        ncall_region_here = (self.ncall_region - ncall_region_at_run_start)
                        # the number of proposals returned by the region
                        ncall_here = self.ncall - ncall_at_run_start
                        # the number of likelihood evaluations above threshold
                        it_here = it - it_at_first_region

                        if show_status:
                            if Lmin < -1e8:
                                txt = 'Z=%.1g(%.2f%%) | Like=%.2g..%.2g [%.4g..%.4g]%s| it/evals=%d/%d eff=%.4f%% N=%d \r'
                            elif Llo < -1e8:
                                txt = 'Z=%.1f(%.2f%%) | Like=%.2f..%.2f [%.4g..%.4g]%s| it/evals=%d/%d eff=%.4f%% N=%d \r'
                            else:
                                txt = 'Z=%.1f(%.2f%%) | Like=%.2f..%.2f [%.4f..%.4f]%s| it/evals=%d/%d eff=%.4f%% N=%d \r'
                            sys.stdout.write(txt % (
                                main_iterator.logZ, 100 * (1 - main_iterator.remainder_fraction),
                                Lmin, main_iterator.Lmax, Llo, Lhi, '*' if strategy_stale else ' ', it, self.ncall,
                                np.inf if ncall_here == 0 else it_here * 100 / ncall_here,
                                nlive))
                            sys.stdout.flush()
                        self.logger.debug('iteration=%d, ncalls=%d, regioncalls=%d, ndraw=%d, logz=%.2f, remainder_fraction=%.4f%%, Lmin=%.2f, Lmax=%.2f' % (
                            it, self.ncall, self.ncall_region, ndraw, main_iterator.logZ,
                            100 * main_iterator.remainder_fraction, Lmin, main_iterator.Lmax))

                        # if efficiency becomes low, bulk-process larger arrays
                        if self.draw_multiple:
                            # inefficiency is the number of (region) proposals per successful number of iterations
                            # but improves by parallelism (because we need only the per-process inefficiency)
                            # sampling_inefficiency = (self.ncall - ncall_at_run_start + 1) / (it + 1) / self.mpi_size
                            sampling_inefficiency = (ncall_region_here + 1) / (it_here + 1) / self.mpi_size

                            # smooth update:
                            ndraw_next = 0.04 * sampling_inefficiency + ndraw * 0.96
                            ndraw = max(self.ndraw_min, min(self.ndraw_max, round(ndraw_next), ndraw * 100))

                            if sampling_inefficiency > 100000 and it >= it_at_first_region + 10:
                                # if the efficiency is poor, there are enough samples in each iteration
                                # to estimate the inefficiency
                                ncall_at_run_start = self.ncall
                                it_at_first_region = it
                                ncall_region_at_run_start = self.ncall_region

                else:
                    # we do not want to count iterations without work
                    # otherwise efficiency becomes > 1
                    it_at_first_region += 1

                saved_nodeids.append(node.id)
                saved_logl.append(Lmin)

                # inform iterators (if it is their business) about the arc
                main_iterator.passing_node(rootid, node, active_rootids, active_values)
                if len(node.children) == 0 and self.region is not None:
                    # the region radius needs to increase if nlive decreases
                    # radius is not reliable, so set to inf
                    # (heuristics do not work in practice)
                    self.region.maxradiussq = None
                    # ask for the region to be rebuilt
                    next_update_interval_volume = 1

                it += 1
                explorer.expand_children_of(rootid, node)

            if self.log:
                self.logger.info("Explored until L=%.1g  ", node.value)
            # print_tree(roots[::10])

            self.pointstore.flush()
            self._update_results(main_iterator, saved_logl, saved_nodeids)
            yield self.results

            if max_ncalls is not None and self.ncall >= max_ncalls:
                if self.log:
                    self.logger.info(
                        'Reached maximum number of likelihood calls (%d > %d)...',
                        self.ncall, max_ncalls)
                break

            improvement_it += 1
            if max_num_improvement_loops >= 0 and improvement_it > max_num_improvement_loops:
                if self.log:
                    self.logger.info('Reached maximum number of improvement loops.')
                break

            if ncall_at_run_start == self.ncall and improvement_it > 1:
                if self.log:
                    self.logger.info(
                        'No changes made. '
                        'Probably the strategy was to explore in the remainder, '
                        'but it is irrelevant already; try decreasing frac_remain.')
                break

            Lmax = main_iterator.Lmax
            if len(region_sequence) > 0:
                Lmin, nlive, nclusters, Lhi = region_sequence[-1]
                nnodes_needed = cluster_num_live_points * nclusters
                if nlive < nnodes_needed:
                    Llo, _, target_min_num_children_new = self._expand_nodes_before(Lmin, nnodes_needed, update_interval_ncall or nlive)
                    target_min_num_children.update(target_min_num_children_new)
                    # if self.log:
                    #     print_tree(self.root.children[::10])
                    minimal_widths.append((Llo, Lhi, nnodes_needed))
                    Llo, Lhi = -np.inf, np.inf
                    continue

            if self.log:
                # self.logger.info('  logZ = %.4f +- %.4f (main)' % (main_iterator.logZ, main_iterator.logZerr))
                self.logger.info('  logZ = %.4g +- %.4g', main_iterator.logZ_bs, main_iterator.logZerr_bs)

            saved_logl = np.asarray(saved_logl)
            # reactive nested sampling: see where we have to improve
            dlogz_min_num_live_points, (Llo_KL, Lhi_KL), (Llo_ess, Lhi_ess) = self._find_strategy(
                saved_logl, main_iterator, dlogz=dlogz, dKL=dKL, min_ess=min_ess)
            Llo = min(Llo_ess, Llo_KL)
            Lhi = max(Lhi_ess, Lhi_KL)
            # to avoid numerical issues when all likelihood values are the same
            Lhi = min(Lhi, saved_logl.max() - 0.001)

            if self.use_mpi:
                recv_Llo = self.comm.gather(Llo, root=0)
                recv_Llo = self.comm.bcast(recv_Llo, root=0)
                recv_Lhi = self.comm.gather(Lhi, root=0)
                recv_Lhi = self.comm.bcast(recv_Lhi, root=0)
                recv_dlogz_min_num_live_points = self.comm.gather(dlogz_min_num_live_points, root=0)
                recv_dlogz_min_num_live_points = self.comm.bcast(recv_dlogz_min_num_live_points, root=0)

                Llo = min(recv_Llo)
                Lhi = max(recv_Lhi)
                dlogz_min_num_live_points = max(recv_dlogz_min_num_live_points)

            if dlogz_min_num_live_points > self.min_num_live_points:
                # more live points needed throughout to reach target
                self.min_num_live_points = dlogz_min_num_live_points
                self._widen_roots(self.min_num_live_points)

            elif Llo <= Lhi:
                # if self.log:
                #     print_tree(roots, title="Tree before forking:")
                parents, parent_weights = find_nodes_before(self.root, Llo)
                # double the width / live points:
                _, width = count_tree_between(self.root.children, Llo, Lhi)
                nnodes_needed = width * 2
                if self.log:
                    self.logger.info(
                        'Widening from %d to %d live points before L=%.1g...',
                        len(parents), nnodes_needed, Llo)

                if len(parents) == 0:
                    Llo = -np.inf
                else:
                    Llo = min(n.value for n in parents)
                self.pointstore.reset()
                target_min_num_children.update(self._widen_nodes(parents, parent_weights, nnodes_needed, update_interval_ncall))
                minimal_widths.append((Llo, Lhi, nnodes_needed))
                # if self.log:
                #     print_tree(roots, title="Tree after forking:")
                # print('tree size:', count_tree(roots))
            else:
                break

    def _update_results(self, main_iterator, saved_logl, saved_nodeids):
        if self.log:
            self.logger.info('Likelihood function evaluations: %d', self.ncall)

        results = combine_results(
            saved_logl, saved_nodeids, self.pointpile,
            main_iterator, mpi_comm=self.comm if self.use_mpi else None)

        results['ncall'] = int(self.ncall)
        results['paramnames'] = self.paramnames + self.derivedparamnames
        results['logzerr_single'] = (main_iterator.all_H[0] / self.min_num_live_points)**0.5

        sequence, results2 = logz_sequence(self.root, self.pointpile, random=True, check_insertion_order=True)
        results['insertion_order_MWW_test'] = results2['insertion_order_MWW_test']

        results_simple = dict(results)
        weighted_samples = results_simple.pop('weighted_samples')
        samples = results_simple.pop('samples')
        saved_wt0 = weighted_samples['weights']
        saved_u = weighted_samples['upoints']
        saved_v = weighted_samples['points']

        if self.log_to_disk:
            if self.log:
                self.logger.info("Writing samples and results to disk ...")
            np.savetxt(os.path.join(self.logs['chains'], 'equal_weighted_post.txt'),
                       samples,
                       header=' '.join(self.paramnames + self.derivedparamnames),
                       comments='')
            np.savetxt(os.path.join(self.logs['chains'], 'weighted_post.txt'),
                       np.hstack((saved_wt0.reshape((-1, 1)), np.reshape(saved_logl, (-1, 1)), saved_v)),
                       header=' '.join(['weight', 'logl'] + self.paramnames + self.derivedparamnames),
                       comments='')
            np.savetxt(os.path.join(self.logs['chains'], 'weighted_post_untransformed.txt'),
                       np.hstack((saved_wt0.reshape((-1, 1)), np.reshape(saved_logl, (-1, 1)), saved_u)),
                       header=' '.join(['weight', 'logl'] + self.paramnames + self.derivedparamnames),
                       comments='')

            with open(os.path.join(self.logs['info'], 'results.json'), 'w') as f:
                json.dump(results_simple, f, indent=4)

            np.savetxt(
                os.path.join(self.logs['info'], 'post_summary.csv'),
                [[results['posterior'][k][i] for i in range(self.num_params) for k in ('mean', 'stdev', 'median', 'errlo', 'errup')]],
                header=','.join(['"{0}_mean","{0}_stdev","{0}_median","{0}_errlo","{0}_errup"'.format(k)
                                 for k in self.paramnames + self.derivedparamnames]),
                delimiter=',', comments='',
            )

        if self.log_to_disk:
            keys = 'logz', 'logzerr', 'logvol', 'nlive', 'logl', 'logwt', 'insert_order'
            np.savetxt(os.path.join(self.logs['chains'], 'run.txt'),
                       np.hstack(tuple([np.reshape(sequence[k], (-1, 1)) for k in keys])),
                       header=' '.join(keys),
                       comments='')
            if self.log:
                self.logger.info("Writing samples and results to disk ... done")

        self.results = results
        self.run_sequence = sequence

    def store_tree(self):
        """Store tree to disk (results/tree.hdf5)."""
        if self.log_to_disk:
            dump_tree(os.path.join(self.logs['results'], 'tree.hdf5'),
                      self.root.children, self.pointpile)

    def print_results(self, use_unicode=True):
        """Give summary of marginal likelihood and parameter posteriors.

        Parameters
        ----------
        use_unicode: bool
            Whether to print a unicode plot of the posterior distributions

        """
        if self.log:
            print()
            print('logZ = %(logz).3f +- %(logzerr).3f' % self.results)
            print('  single instance: logZ = %(logz_single).3f +- %(logzerr_single).3f' % self.results)
            print('  bootstrapped   : logZ = %(logz_bs).3f +- %(logzerr_bs).3f' % self.results)
            print('  tail           : logZ = +- %(logzerr_tail).3f' % self.results)
            print('insert order U test : converged: %(converged)s correlation: %(independent_iterations)s iterations' % (
                self.results['insertion_order_MWW_test']))

            print()
            for i, p in enumerate(self.paramnames + self.derivedparamnames):
                v = self.results['samples'][:,i]
                sigma = v.std()
                med = v.mean()
                if sigma == 0:
                    j = 3
                else:
                    j = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % j
                try:
                    if not use_unicode:
                        raise UnicodeEncodeError("")
                    # make fancy terminal visualisation on a best-effort basis
                    ' ▁▂▃▄▅▆▇██'.encode(sys.stdout.encoding)
                    H, edges = np.histogram(v, bins=40)
                    # add a bit of padding, but not outside parameter limits
                    lo, hi = edges[0], edges[-1]
                    step = edges[1] - lo
                    lo = max(min(lo, self.transform_limits[i,0]), lo - 2 * step)
                    hi = min(max(hi, self.transform_limits[i,1]), hi + 2 * step)
                    H, edges = np.histogram(v, bins=np.linspace(lo, hi, 40))
                    lo, hi = edges[0], edges[-1]

                    dist = ''.join([' ▁▂▃▄▅▆▇██'[i] for i in np.ceil(H * 7 / H.max()).astype(int)])
                    print('    %-20s: %-6s│%s│%-6s    %s +- %s' % (p, fmt % lo, dist, fmt % hi, fmt % med, fmt % sigma))
                except Exception:
                    fmts = '    %-20s' + fmt + " +- " + fmt
                    print(fmts % (p, med, sigma))
            print()

    def plot(self):
        """Make corner, run and trace plots.

        calls:

        * plot_corner()
        * plot_run()
        * plot_trace()
        """
        self.plot_corner()
        self.plot_run()
        self.plot_trace()

    def plot_corner(self):
        """Make corner plot.

        Writes corner plot to plots/ directory if log directory was
        specified, otherwise show interactively.

        This does essentially::

            from ultranest.plot import cornerplot
            cornerplot(results)

        """
        from .plot import cornerplot
        import matplotlib.pyplot as plt
        if self.log:
            self.logger.debug('Making corner plot ...')
        cornerplot(self.results, logger=self.logger if self.log else None)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.logs['plots'], 'corner.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.debug('Making corner plot ... done')

    def plot_trace(self):
        """Make trace plot.

        Write parameter trace diagnostic plots to plots/ directory
        if log directory specified, otherwise show interactively.

        This does essentially::

            from ultranest.plot import traceplot
            traceplot(results=results, labels=paramnames + derivedparamnames)

        """
        from .plot import traceplot
        import matplotlib.pyplot as plt
        if self.log:
            self.logger.debug('Making trace plot ... ')
        paramnames = self.paramnames + self.derivedparamnames
        # get dynesty-compatible sequences
        traceplot(results=self.run_sequence, labels=paramnames)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.logs['plots'], 'trace.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.debug('Making trace plot ... done')

    def plot_run(self):
        """Make run plot.

        Write run diagnostic plots to plots/ directory
        if log directory specified, otherwise show interactively.

        This does essentially::

            from ultranest.plot import runplot
            runplot(results=results)

        """
        from .plot import runplot
        import matplotlib.pyplot as plt
        if self.log:
            self.logger.debug('Making run plot ... ')
        # get dynesty-compatible sequences
        runplot(results=self.run_sequence, logplot=True)
        if self.log_to_disk:
            plt.savefig(os.path.join(self.logs['plots'], 'run.pdf'), bbox_inches='tight')
            plt.close()
            self.logger.debug('Making run plot ... done')


def read_file(log_dir, x_dim, num_bootstraps=20, random=True, verbose=False, check_insertion_order=True):
    """
    Read the output HDF5 file of UltraNest.

    Parameters
    ----------
    log_dir: str
        Folder containing results
    x_dim: int
        number of dimensions
    num_bootstraps: int
        number of bootstraps to use for estimating logZ.
    random: bool
        use randomization for volume estimation.
    verbose: bool
        show progress
    check_insertion_order: bool
        whether to perform MWW insertion order test for assessing convergence

    Returns
    ----------
    sequence: dict
        contains arrays storing for each iteration estimates of:

            * logz: log evidence estimate
            * logzerr: log evidence uncertainty estimate
            * logvol: log volume estimate
            * samples_n: number of live points
            * logwt: log weight
            * logl: log likelihood

    final: dict
        same as ReactiveNestedSampler.results and
        ReactiveNestedSampler.run return values

    """
    import h5py
    filepath = os.path.join(log_dir, 'results', 'points.hdf5')
    fileobj = h5py.File(filepath, 'r')
    _, ncols = fileobj['points'].shape
    num_params = ncols - 3 - x_dim

    points = fileobj['points'][:]
    fileobj.close()
    del fileobj
    stack = list(enumerate(points))

    pointpile = PointPile(x_dim, num_params)

    def pop(Lmin):
        """Find matching sample from points file."""
        # look forward to see if there is an exact match
        # if we do not use the exact matches
        #   this causes a shift in the loglikelihoods
        for i, (idx, next_row) in enumerate(stack):
            row_Lmin = next_row[0]
            L = next_row[1]
            if row_Lmin <= Lmin and L > Lmin:
                idx, row = stack.pop(i)
                return idx, row
        return None, None

    roots = []
    while True:
        _, row = pop(-np.inf)
        if row is None:
            break
        logl = row[1]
        u = row[3:3 + x_dim]
        v = row[3 + x_dim:3 + x_dim + num_params]
        roots.append(pointpile.make_node(logl, u, v))

    root = TreeNode(id=-1, value=-np.inf, children=roots)

    def onNode(node, main_iterator):
        """Insert (single) child of node if available."""
        while True:
            _, row = pop(node.value)
            if row is None:
                break
            if row is not None:
                logl = row[1]
                u = row[3:3 + x_dim]
                v = row[3 + x_dim:3 + x_dim + num_params]
                child = pointpile.make_node(logl, u, v)
                assert logl > node.value, (logl, node.value)
                main_iterator.Lmax = max(main_iterator.Lmax, logl)
                node.children.append(child)

    return logz_sequence(root, pointpile, nbootstraps=num_bootstraps,
                         random=random, onNode=onNode, verbose=verbose,
                         check_insertion_order=check_insertion_order)
