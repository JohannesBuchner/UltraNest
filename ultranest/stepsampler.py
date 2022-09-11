"""MCMC-like step sampling within a region.

The classes implemented here are generators that, in each iteration,
only make one likelihood call. This allows keeping a population of
samplers that have the same execution time per call, even if they
do not terminate at the same number of iterations.
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from .utils import listify as _listify


def generate_random_direction(ui, region, scale=1):
    """Draw uniform direction vector in unit cube space of length `scale`.

    Parameters
    -----------
    ui: array
        starting point
    region: MLFriends object
        current region (not used)
    scale: float
        length of direction vector

    Returns
    --------
    v: array
        new direction vector
    """
    del region
    v = np.random.normal(0, 1, size=len(ui))
    v *= scale / (v**2).sum()**0.5
    return v


def generate_cube_oriented_direction(ui, region, scale=1):
    """Draw a unit direction vector in direction of a random unit cube axes.

    Parameters
    -----------
    ui: array
        starting point
    region: MLFriends object
        current region (not used)
    scale: float
        factor to multiple the vector

    Returns
    --------
    v: array
        new direction vector
    """
    del region
    ndim = len(ui)
    # choose axis
    j = np.random.randint(ndim)
    # use doubling procedure to identify left and right maxima borders
    v = np.zeros(ndim)
    v[j] = scale
    return v


def generate_cube_oriented_differential_direction(ui, region, scale=1):
    """Draw a unit direction vector in direction of a random unit cube axes.

    Guess the length from the difference of two points in that axis.

    Parameters
    -----------
    ui: array
        starting point
    region: MLFriends object
        current region
    scale: float
        factor to multiple the vector

    Returns
    --------
    v: array
        new direction vector
    """
    nlive, ndim = region.u.shape
    v = np.zeros(ndim)

    # choose axis
    j = np.random.randint(ndim)
    # choose pair
    while v[j] == 0:
        i = np.random.randint(nlive)
        i2 = np.random.randint(nlive - 1)
        if i2 >= i:
            i2 += 1

        v[j] = (region.u[i,j] - region.u[i2,j]) * scale

    return v


def generate_differential_direction(ui, region, scale=1):
    """Draw a vector using the difference between two points.

    Parameters
    -----------
    ui: array
        starting point
    region: MLFriends object
        current region
    scale: float
        factor to multiple the vector

    Returns
    --------
    v: array
        new direction vector
    """
    nlive, ndim = region.u.shape
    # choose pair
    i = np.random.randint(nlive)
    i2 = np.random.randint(nlive - 1)
    if i2 >= i:
        i2 += 1

    # use doubling procedure to identify left and right maxima borders
    v = (region.u[i,:] - region.u[i2,:]) * scale
    return v


def generate_partial_differential_direction(ui, region, scale=1):
    """Draw a unit direction vector in direction of a random unit cube axes.

    Parameters
    -----------
    ui: array
        starting point
    region: MLFriends object
        current region
    scale: float
        factor to multiple the vector

    Returns
    --------
    v: array
        new direction vector
    """
    nlive, ndim = region.u.shape
    # choose pair
    i = np.random.randint(nlive)
    while True:
        i2 = np.random.randint(nlive - 1)
        if i2 >= i:
            i2 += 1

        v = region.u[i] - region.u[i2]

        mask = np.random.uniform(size=ndim) > 0.1
        # at least one must be on
        mask[np.random.randint(ndim)] = True
        v[mask] = 0
        if (v != 0).any():
            # repeat if live points are identical
            break
    # use doubling procedure to identify left and right maxima borders
    #v = np.zeros(ndim)
    #v[mask] = (region.u[i,mask] - region.u[i2,mask]) * scale
    return v


def generate_region_oriented_direction(ui, region, scale=1):
    """Draw a random direction vector in direction of one of the `region` axes.

    Parameters
    -----------
    ui: array
        starting point
    region: MLFriends object
        current region
    scale: float
        factor to multiple the vector

    Returns
    --------
    v: array
        new direction vector (in u-space)
    """
    # choose axis in transformed space:
    j = np.random.randint(len(ui))
    v = region.transformLayer.axes[j] * scale
    return v


def generate_region_random_direction(ui, region, scale=1):
    """Draw a direction vector in a random direction of the region.

    The vector length is `scale` (in unit cube space).

    Parameters
    -----------
    ui: array
        starting point
    region: MLFriends object
        current region
    scale: float:
        length of direction vector (in t-space)

    Returns
    --------
    v: array
        new direction vector
    """
    # choose axis in transformed space:
    v1 = np.random.normal(0, 1, size=len(ui))
    v1 *= scale / np.linalg.norm(v1)
    v = np.dot(region.transformLayer.axes, v1)
    return v


def generate_mixture_random_direction(ui, region, scale=1):
    """Draw either from a region-direction or a unit axis.

    Parameters
    -----------
    region: MLFriends
        region
    ui: array
        vector of starting point
    scale: float
        length of the vector.

    Returns
    --------
    v: array
        new direction vector
    """
    if np.random.uniform() < 0.5:
        # DE proposal
        return generate_differential_direction(ui, region, scale=scale)
    else:
        # region-oriented random axis proposal
        return generate_region_oriented_direction(ui, region, scale=scale)


def _inside_region(region, unew, uold):
    """Check if `unew` is inside region.

    This is a bit looser than the region, because it adds a
    MLFriends ellipsoid around the old point as well.
    """
    tnew = region.transformLayer.transform(unew)
    told = region.transformLayer.transform(uold)
    mask2 = ((told.reshape((1, -1)) - tnew)**2).sum(axis=1) < region.maxradiussq
    if mask2.all():
        return mask2

    mask = region.inside(unew)
    return np.logical_or(mask, mask2)


def inside_region(region, unew, uold):
    """Check if `unew` is inside region.

    Parameters
    -----------
    region: MLFriends object
        current region
    unew: array
        point to check
    uold: array
        not used

    Returns
    --------
    v: array
        boolean whether point is inside the region
    """
    del uold
    return region.inside(unew)


def adapt_proposal_total_distances(region, history, mean_pair_distance, ndim):
    # compute mean vector of each proposed jump
    # compute total distance of all jumps
    tproposed = region.transformLayer.transform(np.asarray([u for u, _ in history]))
    assert len(tproposed.sum(axis=1)) == len(tproposed)
    d2 = ((((tproposed[0] - tproposed)**2).sum(axis=1))**0.5).sum()
    far_enough = d2 > mean_pair_distance / ndim

    return far_enough, [d2, mean_pair_distance]


def adapt_proposal_total_distances_NN(region, history, mean_pair_distance, ndim):
    # compute mean vector of each proposed jump
    # compute total distance of all jumps
    tproposed = region.transformLayer.transform(np.asarray([u for u, _ in history]))
    assert len(tproposed.sum(axis=1)) == len(tproposed)
    d2 = ((((tproposed[0] - tproposed)**2).sum(axis=1))**0.5).sum()
    far_enough = d2 > region.maxradiussq**0.5

    return far_enough, [d2, region.maxradiussq**0.5]


def adapt_proposal_summed_distances(region, history, mean_pair_distance, ndim):
    # compute sum of distances from each jump
    tproposed = region.transformLayer.transform(np.asarray([u for u, _ in history]))
    d2 = (((tproposed[1:,:] - tproposed[:-1,:])**2).sum(axis=1)**0.5).sum()
    far_enough = d2 > mean_pair_distance / ndim

    return far_enough, [d2, mean_pair_distance]


def adapt_proposal_summed_distances_NN(region, history, mean_pair_distance, ndim):
    # compute sum of distances from each jump
    tproposed = region.transformLayer.transform(np.asarray([u for u, _ in history]))
    d2 = (((tproposed[1:,:] - tproposed[:-1,:])**2).sum(axis=1)**0.5).sum()
    far_enough = d2 > region.maxradiussq**0.5

    return far_enough, [d2, region.maxradiussq**0.5]


def adapt_proposal_move_distances(region, history, mean_pair_distance, ndim):
    # compute distance from start to end
    ustart, _ = history[0]
    ufinal, _ = history[-1]
    tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
    d2 = ((tstart - tfinal)**2).sum()
    far_enough = d2 > region.maxradiussq

    return far_enough, [d2, region.maxradiussq**0.5]


def adapt_proposal_move_distances_midway(region, history, mean_pair_distance, ndim):
    # compute distance from start to end
    ustart, _ = history[0]
    middle = max(1, len(history) // 2)
    ufinal, _ = history[middle]
    tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
    d2 = ((tstart - tfinal)**2).sum()
    far_enough = d2 > region.maxradiussq

    return far_enough, [d2, region.maxradiussq**0.5]


class StepSampler(object):
    """Base class for a simple step sampler, staggering around.

    Scales proposal towards a 50% acceptance rate.
    """

    def __init__(
        self, nsteps, generate_direction,
        scale=1.0, adaptive_nsteps=False, max_nsteps=1000,
        region_filter=False, log=False,
    ):
        """Initialise sampler.

        Parameters
        -----------
        scale: float
            initial proposal size

        nsteps: int
            number of accepted steps until the sample is considered independent.

            To find the right value, run nested sampling several time,
            always doubling nsteps, until Z is stable.

        generate_direction: function
            direction proposal function.

            Available are:

            * :py:func:`generate_cube_oriented_direction` (slice sampling)
            * :py:func:`generate_region_oriented_direction` (slice sampling on the whitened parameter space)
            * :py:class:`SequentialDirectionGenerator` (sequential slice sampling on the whitened parameter space)
            * :py:func:`generate_random_direction` (hit-and-run sampling)
            * :py:func:`generate_region_random_direction` (hit-and-run sampling on the whitened parameter space)
            * :py:func:`generate_cube_oriented_differential_direction` (slice sampling with better proposal scale)
            * :py:func:`generate_differential_direction` (differential evolution slice proposal)
            * :py:func:`generate_partial_differential_direction` (differential evolution slice proposal on only 10% of the parameters)
            * :py:func:`generate_mixture_random_direction` (generate_differential_direction and generate_cube_oriented_differential_direction)

            Additionally, :py:class:`OrthogonalDirectionGenerator` 
            can be applied to a generate_direction.

            When in doubt, try :py:func:`generate_mixture_random_direction`.
            It combines efficient moves along the live point distribution,
            with robustness against collapse to a subspace.
            :py:func:`generate_cube_oriented_direction` works well too.

        adaptive_nsteps: False, 'proposal-distance', 'move-distance'
            Strategy to adapt the number of steps. The strategies
            make sure that:

            * 'move-distance' (recommended): distance between
              start point and final position exceeds the mean distance
              between pairs of live points.
            * 'move-distance-midway': distance between
              start point and position in the middle of the chain
              exceeds the mean distance between pairs of live points.
            * 'proposal-total-distances': mean square distance of
              proposed vectors exceeds the mean distance
              between pairs of live points.
            * 'proposal-total-distances-NN': mean distance
              of chain points from starting point exceeds mean distance
              between pairs of live points.
            * 'proposal-summed-distances-NN': summed distances
              between chain points exceeds mean distance
              between pairs of live points.
            * 'proposal-summed-distances-min-NN': smallest distance
              between chain points exceeds mean distance
              between pairs of live points.

            Adapting can give usable results. However, strictly speaking,
            detailed balance is not maintained, so the results can be biased.
            You can use the logstat property to find out the `nsteps` learned
            from one run (third column), and use the largest value for `nsteps`
            of a fresh run.

        max_nsteps: int
            Maximum number of steps the adaptive_nsteps can reach.

        region_filter: bool
            if True, use region to check if a proposed point can be inside
            before calling likelihood.

        log: file
            log file for sampler statistics, such as acceptance rate,
            proposal scale, number of steps, jump distance and distance
            between live points

        """
        self.history = []
        self.nsteps = nsteps
        self.nrejects = 0
        self.scale = 1.0
        self.max_nsteps = max_nsteps
        self.next_scale = self.scale
        self.nudge = 1.1**(1. / self.nsteps)
        self.nsteps_nudge = 1.01
        self.generate_direction = generate_direction
        adaptive_nsteps_options = {
            False: None,
            'move-distance': adapt_proposal_move_distances,
            'move-distance-midway': adapt_proposal_move_distances_midway,
            'proposal-total-distances': adapt_proposal_total_distances,
            'proposal-total-distances-NN': adapt_proposal_total_distances_NN,
            'proposal-summed-distances': adapt_proposal_summed_distances,
            'proposal-summed-distances-NN': adapt_proposal_summed_distances_NN,
        }

        if adaptive_nsteps not in adaptive_nsteps_options.keys():
            raise ValueError("adaptive_nsteps must be one of: %s, not '%s'" % (adaptive_nsteps_options, adaptive_nsteps))
        self.adaptive_nsteps = adaptive_nsteps
        self.adaptive_nsteps_function = adaptive_nsteps_options[adaptive_nsteps]
        self.adaptive_nsteps_needs_mean_pair_distance = self.adaptive_nsteps in (
            'proposal-total-distances', 'proposal-summed-distances',
        )
        self.mean_pair_distance = np.nan
        self.region_filter = region_filter
        self.log = log

        self.logstat = []
        self.logstat_labels = ['rejection_rate', 'scale', 'steps']
        if adaptive_nsteps:
            self.logstat_labels += ['jump-distance', 'reference-distance']

    def __str__(self):
        if not self.adaptive_nsteps:
            return type(self).__name__ + '(nsteps=%d, generate_direction=%s)' % (self.nsteps, self.generate_direction)
        else:
            return type(self).__name__ + '(adaptive_nsteps=%s, generate_direction=%s)' % (self.adaptive_nsteps, self.generate_direction)

    def plot(self, filename):
        """Plot sampler statistics.

        Parameters
        -----------
        filename: str
            Stores plot into ``filename`` and data into
            ``filename + ".txt.gz"``.
        """
        if len(self.logstat) == 0:
            return

        plt.figure(figsize=(10, 1 + 3 * len(self.logstat_labels)))
        for i, label in enumerate(self.logstat_labels):
            part = [entry[i] for entry in self.logstat]
            plt.subplot(len(self.logstat_labels), 1, 1 + i)
            plt.ylabel(label)
            plt.plot(part)
            x = []
            y = []
            for j in range(0, len(part), 20):
                x.append(j)
                y.append(np.mean(part[j:j + 20]))
            plt.plot(x, y)
            if np.min(part) > 0:
                plt.yscale('log')
        plt.savefig(filename, bbox_inches='tight')
        np.savetxt(filename + '.txt.gz', self.logstat,
                   header=','.join(self.logstat_labels), delimiter=',')
        plt.close()

    def move(self, ui, region, ndraw=1, plot=False):
        """Move around point ``ui``. Stub to be implemented by child classes."""
        raise NotImplementedError()

    def adjust_outside_region(self):
        """Adjust proposal, given that we landed outside region."""
        print("ineffective proposal scale (%g). shrinking..." % self.scale)

        # Usually the region is very generous.
        # Being here means that the scale is very wrong and we are probably stuck.
        # Adjust it and restart the chain
        self.scale /= self.nudge**10
        self.next_scale /= self.nudge**10
        assert self.scale > 0
        assert self.next_scale > 0
        # reset chain
        if self.adaptive_nsteps:
            self.logstat.append([-1.0, self.scale, self.nsteps, np.nan, np.nan])
        else:
            self.logstat.append([-1.0, self.scale, self.nsteps])

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        """Adjust proposal, given that a new point was found after `nc` calls.

        Parameters
        -----------
        accepted: bool
            Whether the most recent proposal was accepted
        unew: array
            new point (in u-space)
        pnew: array
            new point (in p-space)
        Lnew: float
            loglikelihood of new point
        nc: int
            number of likelihood function calls used.
        """
        if accepted:
            self.next_scale *= self.nudge
            self.history.append((unew.copy(), Lnew.copy()))
        else:
            self.next_scale /= self.nudge**10
            self.nrejects += 1
            self.history.append(self.history[-1])
        assert self.next_scale > 0, self.next_scale

    def adapt_nsteps(self, region):
        """
        Adapt the number of steps.

        Parameters
        -----------
        region: MLFriends object
            current region
        """
        if not self.adaptive_nsteps:
            return
        elif len(self.history) < self.nsteps:
            # incomplete or aborted for some reason
            print("not adapting, incomplete history", len(self.history), self.nsteps)
            return

        # assert self.nrejects < len(self.history), (self.nsteps, self.nrejects, len(self.history))
        # assert self.nrejects <= self.nsteps, (self.nsteps, self.nrejects, len(self.history))
        if self.adaptive_nsteps_needs_mean_pair_distance:
            assert np.isfinite(self.mean_pair_distance)
        ndim = region.u.shape[1]
        far_enough, extra_info = self.adaptive_nsteps_function(region, self.history, self.mean_pair_distance, ndim)
        self.logstat[-1] += extra_info

        # adjust nsteps
        if far_enough:
            self.nsteps = min(self.nsteps - 1, int(self.nsteps / self.nsteps_nudge))
        else:
            self.nsteps = max(self.nsteps + 1, int(self.nsteps * self.nsteps_nudge))
        self.nsteps = max(1, min(self.max_nsteps, self.nsteps))

    def finalize_chain(self, region=None, Lmin=None, Ls=None):
        """Store chain statistics and adapt proposal.

        Parameters
        -----------
        region: MLFriends object
            current region
        Lmin: float
            current loglikelihood threshold
        Ls: array
            loglikelihood values of the live points
        """
        self.logstat.append([self.nrejects / self.nsteps, self.scale, self.nsteps])
        if self.log:
            ustart, Lstart = self.history[0]
            ufinal, Lfinal = self.history[-1]
            # mean_pair_distance = region.compute_mean_pair_distance()
            mean_pair_distance = self.mean_pair_distance
            tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
            # L index of start and end
            # Ls_sorted = np.sort(Ls)
            iLstart = np.sum(Ls > Lstart)
            iLfinal = np.sum(Ls > Lfinal)
            # nearest neighbor index of start and end
            itstart = np.argmin((region.unormed - tstart.reshape((1, -1)))**2)
            itfinal = np.argmin((region.unormed - tfinal.reshape((1, -1)))**2)
            np.savetxt(self.log, [_listify(
                [Lmin], ustart, ufinal, tstart, tfinal,
                [self.nsteps, region.maxradiussq**0.5, mean_pair_distance,
                 iLstart, iLfinal, itstart, itfinal])])

        if self.adaptive_nsteps:
            self.adapt_nsteps(region=region)

        if self.next_scale > self.scale * self.nudge**10:
            self.next_scale = self.scale * self.nudge**10
        elif self.next_scale < self.scale / self.nudge**10:
            self.next_scale = self.scale / self.nudge**10
        # print("updating scale: %g -> %g" % (self.scale, self.next_scale))
        self.scale = self.next_scale
        self.history = []
        self.nrejects = 0

    def new_chain(self, region=None):
        """Start a new path, reset statistics."""
        self.history = []
        self.nrejects = 0

    def region_changed(self, Ls, region):
        """React to change of region.

        Parameters
        -----------
        region: MLFriends object
            current region
        Ls: array
            loglikelihood values of the live points
        """
        if self.adaptive_nsteps_needs_mean_pair_distance:
            self.mean_pair_distance = region.compute_mean_pair_distance()
            # print("region changed. new mean_pair_distance: %g" % self.mean_pair_distance)

    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=10, plot=False, tregion=None):
        """Get next point.

        Parameters
        ----------
        region: MLFriends
            region.
        Lmin: float
            loglikelihood threshold
        us: array of vectors
            current live points
        Ls: array of floats
            current live point likelihoods
        transform: function
            transform function
        loglike: function
            loglikelihood function
        ndraw: int
            number of draws to attempt simultaneously.
        plot: bool
            whether to produce debug plots.
        tregion: WrappingEllipsoid
            optional ellipsoid in transformed space for rejecting proposals

        """
        # find most recent point in history conforming to current Lmin
        for j, (uj, Lj) in enumerate(self.history):
            if not Lj > Lmin:
                self.history = self.history[:j]
                # print("wandered out of L constraint; reverting", ui[0])
                break
        if len(self.history) > 0:
            ui, Li = self.history[-1]
        else:
            # select starting point
            self.new_chain(region)
            # choose a new random starting point
            # mask = region.inside(us)
            # assert mask.any(), ("One of the live points does not satisfies the current region!",
            #    region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(len(us))
            self.starti = i
            ui = us[i,:]
            # print("starting at", ui[0])
            # assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[i]
            self.history.append((ui.copy(), Li.copy()))
            del i

        while True:
            unew = self.move(ui, region, ndraw=ndraw, plot=plot)
            # print("proposed: %s -> %s" % (ui, unew))
            if plot:
                plt.plot([ui[0], unew[:,0]], [ui[1], unew[:,1]], '-', color='k', lw=0.5)
                plt.plot(ui[0], ui[1], 'd', color='r', ms=4)
                plt.plot(unew[:,0], unew[:,1], 'x', color='r', ms=4)
            mask = np.logical_and(unew > 0, unew < 1).all(axis=1)
            if not mask.any():
                # print("rejected by unit cube")
                self.adjust_outside_region()
                continue
            unew = unew[mask,:]
            nc = 0
            if self.region_filter:
                mask = inside_region(region, unew, ui)
                if not mask.any():
                    print("rejected by region")
                    self.adjust_outside_region()
                    continue
                unew = unew[mask,:]
                if tregion is not None:
                    pnew = transform(unew)
                    tmask = tregion.inside(pnew)
                    unew = unew[tmask,:]
                    pnew = pnew[tmask,:]

            if len(unew) == 0:
                self.adjust_outside_region()
                continue
            break

        unew = unew[0,:]
        pnew = transform(unew.reshape((1, -1)))
        Lnew = loglike(pnew)[0]
        nc = 1
        if Lnew > Lmin:
            if plot:
                plt.plot(unew[0], unew[1], 'o', color='g', ms=4)
            self.adjust_accept(True, unew, pnew, Lnew, nc)
        else:
            self.adjust_accept(False, unew, pnew, Lnew, nc)

        if len(self.history) > self.nsteps:
            # print("made %d steps" % len(self.history), Lnew, Lmin)
            u, L = self.history[-1]
            p = transform(u.reshape((1, -1)))[0]
            self.finalize_chain(region=region, Lmin=Lmin, Ls=Ls)
            return u, p, L, nc

        # do not have a independent sample yet
        return None, None, None, nc


class MHSampler(StepSampler):
    """Gaussian Random Walk."""

    def move(self, ui, region, ndraw=1, plot=False):
        """Move in u-space with a Gaussian proposal.

        Parameters
        ----------
        ui: array
            current point
        ndraw: int
            number of points to draw.
        region:
            ignored
        plot:
            ignored
        """
        # propose in that direction
        direction = self.generate_direction(ui, region, scale=self.scale)
        jitter = direction * np.random.normal(0, 1, size=(min(10, ndraw), 1))
        unew = ui.reshape((1, -1)) + jitter
        return unew


def CubeMHSampler(*args, **kwargs):
    """Gaussian Metropolis-Hastings sampler, using unit cube."""
    return MHSampler(*args, **kwargs, generate_direction=generate_random_direction)


def RegionMHSampler(*args, **kwargs):
    """Gaussian Metropolis-Hastings sampler, using region."""
    return MHSampler(*args, **kwargs, generate_direction=generate_region_random_direction)


class SliceSampler(StepSampler):
    """Slice sampler, respecting the region."""

    def new_chain(self, region=None):
        """Start a new path, reset slice."""
        self.interval = None
        self.found_left = False
        self.found_right = False
        self.axis_index = 0

        self.history = []
        self.nrejects = 0

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        """see :py:meth:`StepSampler.adjust_accept`"""
        v, left, right, u = self.interval
        if not self.found_left:
            if accepted:
                self.interval = (v, left * 2, right, u)
            else:
                self.found_left = True
        elif not self.found_right:
            if accepted:
                self.interval = (v, left, right * 2, u)
            else:
                self.found_right = True
                # adjust scale
                if -left > self.next_scale or right > self.next_scale:
                    self.next_scale *= 1.1
                else:
                    self.next_scale /= 1.1
                # print("adjusting after accept...", self.next_scale)
        else:
            if accepted:
                # start with a new interval next time
                self.interval = None

                self.history.append((unew.copy(), Lnew.copy()))
            else:
                self.nrejects += 1
                # shrink current interval
                if u == 0:
                    pass
                elif u < 0:
                    left = u
                elif u > 0:
                    right = u

                self.interval = (v, left, right, u)

    def adjust_outside_region(self):
        """Adjust proposal given that we landed outside region."""
        self.adjust_accept(False, unew=None, pnew=None, Lnew=None, nc=0)

    def move(self, ui, region, ndraw=1, plot=False):
        """Advance the slice sampling move. see :py:meth:`StepSampler.move`"""
        if self.interval is None:
            v = self.generate_direction(ui, region)

            # expand direction until it is surely outside
            left = -self.scale
            right = self.scale
            self.found_left = False
            self.found_right = False
            u = 0

            self.interval = (v, left, right, u)

        else:
            v, left, right, u = self.interval

        if plot:
            plt.plot([(ui + v * left)[0], (ui + v * right)[0]],
                     [(ui + v * left)[1], (ui + v * right)[1]],
                     ':o', color='k', lw=2, alpha=0.3)

        # shrink direction if outside
        if not self.found_left:
            xj = ui + v * left

            if not self.region_filter or inside_region(region, xj.reshape((1, -1)), ui):
                return xj.reshape((1, -1))
            else:
                self.found_left = True

        if not self.found_right:
            xj = ui + v * right

            if not self.region_filter or inside_region(region, xj.reshape((1, -1)), ui):
                return xj.reshape((1, -1))
            else:
                self.found_right = True

                # adjust scale
                if -left > self.next_scale or right > self.next_scale:
                    self.next_scale *= 1.1
                else:
                    self.next_scale /= 1.1
                # print("adjusting scale...", self.next_scale)

        while True:
            u = np.random.uniform(left, right)
            xj = ui + v * u

            if not self.region_filter or inside_region(region, xj.reshape((1, -1)), ui):
                self.interval = (v, left, right, u)
                return xj.reshape((1, -1))
            else:
                if u < 0:
                    left = u
                else:
                    right = u
                self.interval = (v, left, right, u)


def CubeSliceSampler(*args, **kwargs):
    """Slice sampler, randomly picking region axes."""
    return SliceSampler(*args, **kwargs, generate_direction=generate_cube_oriented_direction)


def RegionSliceSampler(*args, **kwargs):
    """Slice sampler, randomly picking region axes."""
    return SliceSampler(*args, **kwargs, generate_direction=generate_region_oriented_direction)


def BallSliceSampler(*args, **kwargs):
    """Hit & run sampler. Choose random directions in space."""
    return SliceSampler(*args, **kwargs, generate_direction=generate_random_direction)


def RegionBallSliceSampler(*args, **kwargs):
    """Hit & run sampler. Choose random directions according to region."""
    return SliceSampler(*args, **kwargs, generate_direction=generate_region_random_direction)


class SequentialRegionDirectionGenerator(object):
    def __init__(self):
        """Sequentially proposes one region axes after the next."""
        self.axis_index = 0

    def __call__(self, ui, region, scale=1):
        """Iteratively choose the next axis in t-space.

        Parameters
        -----------
        ui: array
            current point (in u-space)
        region: MLFriends object
            region to use for transformation
        scale: float
            length of direction vector

        Returns
        --------
        v: array
            new direction vector (in u-space)
        """
        ndim = len(ui)
        ti = region.transformLayer.transform(ui)

        # choose axis in transformed space:
        j = self.axis_index % ndim
        self.axis_index = j + 1
        tv = np.zeros(ndim)
        tv[j] = 1.0
        # convert back to unit cube space:
        uj = region.transformLayer.untransform(ti + tv * 1e-3)
        v = uj - ui
        v *= scale / (v**2).sum()**0.5
        return v

    def __str__(self):
        return type(self).__name__ + '()'

def RegionSequentialSliceSampler(*args, **kwargs):
    """Slice sampler, sequentially iterating region axes."""
    return SliceSampler(*args, **kwargs, generate_direction=SequentialRegionDirectionGenerator())


class OrthogonalDirectionGenerator(object):
    def __init__(self, generate_direction):
        """Orthogonalizes proposal vectors.

        Parameters
        -----------
        generate_direction: function
            direction proposal to orthogonalize
        """
        self.axis_index = 0
        self.generate_direction = generate_direction
        self.directions = None
    
    def __str__(self):
        return type(self).__name__ + '(generate_direction=%s)' % self.generate_direction

    def __call__(self, ui, region, scale=1):
        """Iteratively return a orthogonalized vector.

        Parameters
        -----------
        ui: array
            current point (in u-space)
        region: MLFriends object
            region to use for transformation
        scale: float
            length of direction vector

        Returns
        --------
        v: array
            new direction vector (in u-space)
        """
        ndim = len(ui)
        if self.directions is None or self.axis_index >= ndim:
            proposed_directions = np.empty((ndim, ndim))
            for i in range(ndim):
                proposed_directions[i] = self.generate_direction(ui, region, scale=scale)
            q, r = np.linalg.qr(proposed_directions)
            self.directions = np.dot(q, np.diag(np.diag(r)))
            self.axis_index = 0

        v = self.directions[self.axis_index]
        self.axis_index += 1
        return v


class SpeedVariableGenerator(object):
    """Propose directions in region, but only some dimensions at a time, completely user-definable.
    """

    def __init__(self, step_matrix, generate_direction=generate_region_random_direction):
        """Initialise sampler.

        Parameters
        -----------
        step_matrix: matrix or list of slices

            **if a bool matrix of shape (n_steps, n_dims):**

            Each row of the matrix indicates which parameters
            should be updated.

            Example::

                [[True, True], [False, True], [False, True]]

            This would update the first parameter 1/3 times, and the second
            parameters every time. Three steps are made until the point
            is considered independent.

            For a full update in every step, use::

                np.ones((n_steps, n_dims), dtype=bool)

            **if a list of slices:**

            Each entry indicates which parameters should be updated.

            Example::

                [Ellipsis, slice(2,10), slice(5,10)]

            This would update the first parameter 1/3 times, parameters
            2-9 2/3 times and parameter 5-9 in every step.
            Three steps are made until the point is considered independent.

        generate_direction: function
            direction proposal function.
        """
        self.step_matrix = step_matrix
        self.nsteps = len(self.step_matrix)
        self.axis_index = 0
        self.generate_direction = generate_direction

    def __call__(self, ui, region, scale=1):
        """Generate a slice sampling direction, using only some of the axes.

        Parameters
        -----------
        ui: array
            current point (in u-space)
        region: MLFriends object
            region to use for transformation
        scale: float
            length of direction vector

        Returns
        --------
        v: array
            new direction vector
        """
        ndim = len(ui)

        v = self.generate_direction(ui=ui, region=region, scale=scale)
        j = self.axis_index % self.nsteps
        self.axis_index = j + 1
        # only update active dimensions
        active_dims = self.step_matrix[j]
        # project uj onto ui. vary only active dimensions
        uk = np.zeros(ndim)
        uk[active_dims] = v[active_dims]  # if this fails, user passed a faulty step_matrix
        return uk


def SpeedVariableRegionSliceSampler(step_matrix, *args, **kwargs):
    """Slice sampler, in region axes.

    Updates only some dimensions at a time, completely user-definable.
    """
    generate_direction = kwargs.pop('generate_direction', generate_region_random_direction)
    return SliceSampler(
        *args, **kwargs,
        nsteps=kwargs.pop('nsteps', len(step_matrix)),
        generate_direction=SpeedVariableGenerator(
            step_matrix=step_matrix,
            generate_direction=generate_direction
        )
    )


def ellipsoid_bracket(ui, v, ellipsoid_center, ellipsoid_inv_axes, ellipsoid_radius_square):
    """Find line-ellipsoid intersection points.

    For a line from ui in direction v through an ellipsoid
    centered at ellipsoid_center with axes matrix ellipsoid_inv_axes,
    return the lower and upper intersection parameter.

    Parameters
    -----------
    ui: array
        current point (in u-space)
    v: array
        direction vector
    ellipsoid_center: array
        center of the ellipsoid
    ellipsoid_inv_axes: array
        ellipsoid axes matrix, as computed by :class:WrappingEllipsoid
    ellipsoid_radius_square: float
        square of the ellipsoid radius

    Returns
    --------
    left: float
        distance to go until ellipsoid is intersected (non-positive)
    right: float
        distance to go until ellipsoid is intersected (non-negative)
    """
    vell = np.dot(v, ellipsoid_inv_axes)
    # ui in ellipsoid
    xell = np.dot(ui - ellipsoid_center, ellipsoid_inv_axes)
    a = np.dot(vell, vell)
    b = 2 * np.dot(vell, xell)
    c = np.dot(xell, xell) - ellipsoid_radius_square
    assert c <= 0, ("outside ellipsoid", c)
    intersect = b**2 - 4 * a * c
    assert intersect >= 0, ("no intersection", intersect, c)
    d1 = (-b + intersect**0.5) / (2 * a)
    d2 = (-b - intersect**0.5) / (2 * a)
    left = min(0, d1, d2)
    right = max(0, d1, d2)
    return left, right


def crop_bracket_at_unit_cube(ui, v, left, right, epsilon=1e-6):
    """Find line-cube intersection points.

    A line segment from *ui* in direction *v* from t between *left* <= 0 <= *right*
    will be truncated by the unit cube. Returns the bracket and whether cropping was applied.

    Parameters
    -----------
    ui: array
        current point (in u-space)
    v: array
        direction vector
    left: float
        bracket lower end (non-positive)
    right: float
        bracket upper end (non-negative)
    epsilon: float
        small number to allow for numerical effects

    Returns
    --------
    left: float
        new left
    right: float
        new right
    cropped_left: bool
        whether left was changed
    cropped_right: bool
        whether right was changed
    """
    assert (ui > 0).all(), ui
    assert (ui < 1).all(), ui
    leftu = left * v + ui
    rightu = right * v + ui
    # print("crop: current ends:", leftu, rightu)
    cropped_left = False
    leftbelow = leftu <= 0
    if leftbelow.any():
        # choose left so that point is > 0 in all axes
        # 0 = left * v + ui
        del left
        left = (-ui[leftbelow] / v[leftbelow]).max() * (1 - epsilon)
        del leftu
        leftu = left * v + ui
        cropped_left |= True
        assert (leftu >= 0).all(), leftu
    leftabove = leftu >= 1
    if leftabove.any():
        del left
        left = ((1 - ui[leftabove]) / v[leftabove]).max() * (1 - epsilon)
        del leftu
        leftu = left * v + ui
        cropped_left |= True
        assert (leftu <= 1).all(), leftu

    cropped_right = False
    rightabove = rightu >= 1
    if rightabove.any():
        # choose right so that point is < 1 in all axes
        # 1 = left * v + ui
        del right
        right = ((1 - ui[rightabove]) / v[rightabove]).min() * (1 - epsilon)
        del rightu
        rightu = right * v + ui
        cropped_right |= True
        assert (rightu <= 1).all(), rightu

    rightbelow = rightu <= 0
    if rightbelow.any():
        del right
        right = (-ui[rightbelow] / v[rightbelow]).min() * (1 - epsilon)
        del rightu
        rightu = right * v + ui
        cropped_right |= True
        assert (rightu >= 0).all(), rightu

    assert left <= 0 <= right, (left, right)
    return left, right, cropped_left, cropped_right
