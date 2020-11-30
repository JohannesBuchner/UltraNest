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

    Region is not used.
    """
    del region
    v = np.random.normal(0, 1, size=len(ui))
    v *= scale / (v**2).sum()**0.5
    return v


def generate_cube_oriented_direction(ui, region):
    """Draw a unit direction vector in direction of a random unit cube axes.

    `region` is not used.
    """
    del region
    ndim = len(ui)
    # choose axis
    j = np.random.randint(ndim)
    # use doubling procedure to identify left and right maxima borders
    v = np.zeros(ndim)
    v[j] = 1.0
    return v


def generate_region_oriented_direction(ui, region, tscale=1, scale=None):
    """Draw a random direction vector in direction of one of the `region` axes.

    The vector length is `scale` (if given).
    If not, the vector length in transformed space is `tscale`.
    """
    ndim = len(ui)
    ti = region.transformLayer.transform(ui)

    # choose axis in transformed space:
    j = np.random.randint(ndim)
    tv = np.zeros(ndim)
    tv[j] = tscale
    # convert back to unit cube space:
    uj = region.transformLayer.untransform(ti + tv)
    v = uj - ui
    if scale is not None:
        v *= scale / (v**2).sum()**0.5
    return v


def generate_region_random_direction(ui, region, scale=1):
    """Draw a direction vector in a random direction of the region.

    The vector length is `scale` (in unit cube space).
    """
    ti_orig = region.transformLayer.transform(ui)

    # choose axis in transformed space:
    ti = np.random.normal(ti_orig, 1)
    # ti *= scale / (ti**2).sum()**0.5
    # convert back to unit cube space:
    uj = region.transformLayer.untransform(ti)
    v = uj - ui
    v *= scale / (v**2).sum()**0.5
    return v


def generate_mixture_random_direction(ui, region, scale=1, uniform_weight=1e-6):
    """Draw from a mix of a ball proposal and a region-shaped proposal.

    Parameters
    -----------
    region: MLFriends
        region
    uniform_weight: float
        sets the weight for the equal-axis ball contribution
    scale: float
        length of the vector.

    """
    v1 = generate_random_direction(ui, region)
    v1 /= (v1**2).sum()**0.5
    v2 = generate_region_random_direction(ui, region)
    v2 /= (v2**2).sum()**0.5
    v = (v1 * uniform_weight + v2 * (1 - uniform_weight))
    v *= scale * (v2**2).sum()**0.5
    return v


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
    """Check if `unew` is inside region."""
    del uold
    return region.inside(unew)


class StepSampler(object):
    """Base class for a simple step sampler, staggering around.

    Scales proposal towards a 50% acceptance rate.
    """

    def __init__(
        self, nsteps, scale=1.0, adaptive_nsteps=False, max_nsteps=1000,
        region_filter=False, log=False
    ):
        """Initialise sampler.

        Parameters
        -----------
        scale: float
            initial proposal size

        nsteps: int
            number of accepted steps until the sample is considered independent.

        adaptive_nsteps: False, 'proposal-distance', 'move-distance'
            Select a strategy to adapt the number of steps. The strategies
            make sure that:

            * 'move-distance' (recommended): distance between
              start point and final position exceeds the mean distance
              between pairs of live points.
            * 'move-distance-midway': distance between
              start point and position in the middle of the chain
              exceeds the mean distance between pairs of live points.
            * 'proposal-distance': mean square distance of
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
        self.last = None, None
        self.nudge = 1.1**(1. / self.nsteps)
        self.nsteps_nudge = 1.01
        adaptive_nsteps_options = [
            False,
            'proposal-total-distances-NN', 'proposal-summed-distances-NN',
            'proposal-total-distances', 'proposal-summed-distances',
            'move-distance', 'move-distance-midway', 'proposal-summed-distances-min-NN',
            'proposal-variance-min', 'proposal-variance-min-NN'
        ]

        if adaptive_nsteps not in adaptive_nsteps_options:
            raise ValueError("adaptive_nsteps must be one of: %s, not '%s'" % (adaptive_nsteps_options, adaptive_nsteps))
        self.adaptive_nsteps = adaptive_nsteps
        self.adaptive_nsteps_needs_mean_pair_distance = self.adaptive_nsteps in (
            'proposal-total-distances', 'proposal-summed-distances', 'proposal-variance-min'
        )
        self.mean_pair_distance = np.nan
        self.region_filter = region_filter
        self.log = log

        self.logstat = []
        self.logstat_labels = ['rejection_rate', 'scale', 'steps']
        if adaptive_nsteps:
            self.logstat_labels += ['jump-distance', 'reference-distance']

    def __str__(self):
        """Get string representation."""
        if not self.adaptive_nsteps:
            return type(self).__name__ + '(nsteps=%d)' % self.nsteps
        else:
            return type(self).__name__ + '(adaptive_nsteps=%s)' % self.adaptive_nsteps

    def plot(self, filename):
        """Plot sampler statistics."""
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
        """Move around ui. Stub to be implemented."""
        raise NotImplementedError()

    def adjust_outside_region(self):
        """Adjust proposal given that we landed outside region."""
        print("ineffective proposal scale (%g). shrinking..." % self.scale)

        # Usually the region is very generous.
        # Being here means that the scale is very wrong and we are probably stuck.
        # Adjust it and restart the chain
        self.scale /= self.nudge**10
        self.next_scale /= self.nudge**10
        assert self.scale > 0
        assert self.next_scale > 0
        # reset chain
        self.last = None, None
        if self.adaptive_nsteps:
            self.logstat.append([-1.0, self.scale, self.nsteps, np.nan, np.nan])
        else:
            self.logstat.append([-1.0, self.scale, self.nsteps])

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        """Adjust proposal given that we have been `accepted` at a new point after `nc` calls."""
        if accepted:
            self.next_scale *= self.nudge
            self.last = unew, Lnew
            self.history.append((unew.copy(), Lnew.copy()))
        else:
            self.next_scale /= self.nudge**10
            self.nrejects += 1
            self.history.append(self.history[-1])
        assert self.next_scale > 0, self.next_scale

    def adapt_nsteps(self, region):
        """ change nsteps """
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
        if self.adaptive_nsteps == 'proposal-total-distances':
            # compute mean vector of each proposed jump
            # compute total distance of all jumps
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            assert len(tproposed.sum(axis=1)) == len(tproposed)
            d2 = ((((tproposed[0] - tproposed)**2).sum(axis=1))**0.5).sum()
            far_enough = d2 > self.mean_pair_distance / ndim

            self.logstat[-1] = self.logstat[-1] + [d2, self.mean_pair_distance]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, self.mean_pair_distance, d2)
        elif self.adaptive_nsteps == 'proposal-total-distances-NN':
            # compute mean vector of each proposed jump
            # compute total distance of all jumps
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            assert len(tproposed.sum(axis=1)) == len(tproposed)
            d2 = ((((tproposed[0] - tproposed)**2).sum(axis=1))**0.5).sum()
            far_enough = d2 > region.maxradiussq**0.5

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-summed-distances':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = (((tproposed[1:,:] - tproposed[:-1,:])**2).sum(axis=1)**0.5).sum()
            far_enough = d2 > self.mean_pair_distance / ndim
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, self.mean_pair_distance, d2)

            self.logstat[-1] = self.logstat[-1] + [d2, self.mean_pair_distance]
        elif self.adaptive_nsteps == 'proposal-summed-distances-NN':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = (((tproposed[1:,:] - tproposed[:-1,:])**2).sum(axis=1)**0.5).sum()
            far_enough = d2 > region.maxradiussq**0.5

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-summed-distances-min-NN':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = (np.abs(tproposed[1:,:] - tproposed[:-1,:]).sum(axis=1)).min()
            far_enough = d2 > region.maxradiussq**0.5

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-variance-min':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = tproposed.std(axis=0).min()
            far_enough = d2 > self.mean_pair_distance / ndim

            self.logstat[-1] = self.logstat[-1] + [d2, self.mean_pair_distance]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'proposal-variance-min-NN':
            # compute sum of distances from each jump
            tproposed = region.transformLayer.transform(np.asarray([u for u, _ in self.history]))
            d2 = tproposed.std(axis=0).min()
            far_enough = d2 > region.maxradiussq**0.5

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'move-distance':
            # compute distance from start to end
            ustart, _ = self.history[0]
            ufinal, _ = self.history[-1]
            tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
            d2 = ((tstart - tfinal)**2).sum()
            far_enough = d2 > region.maxradiussq

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        elif self.adaptive_nsteps == 'move-distance-midway':
            # compute distance from start to end
            ustart, _ = self.history[0]
            middle = max(1, len(self.history) // 2)
            ufinal, _ = self.history[middle]
            tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
            d2 = ((tstart - tfinal)**2).sum()
            far_enough = d2 > region.maxradiussq

            self.logstat[-1] = self.logstat[-1] + [d2, region.maxradiussq**0.5]
            # print(self.adaptive_nsteps, self.nsteps, self.nrejects, far_enough, region.maxradiussq**0.5, d2)
        else:
            assert False, self.adaptive_nsteps

        # adjust nsteps
        if far_enough:
            self.nsteps = min(self.nsteps - 1, int(self.nsteps / self.nsteps_nudge))
        else:
            self.nsteps = max(self.nsteps + 1, int(self.nsteps * self.nsteps_nudge))
        self.nsteps = max(1, min(self.max_nsteps, self.nsteps))

    def finalize_chain(self, region=None, Lmin=None, Ls=None):
        """Store chain statistics and adapt proposal."""
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
        self.last = None, None
        self.history = []
        self.nrejects = 0

    def new_chain(self, region=None):
        """Starts a new path, reset statistics."""
        self.history = []
        self.nrejects = 0

    def region_changed(self, Ls, region):
        """React to change of region. """

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
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            print("wandered out of L constraint; resetting", ui[0])
            del ui, Li
            ui, Li = None, None

        if Li is None and self.history:
            # try to resume from a previous point above the current contour
            for j, (uj, Lj) in enumerate(self.history[::-1]):
                if Lj > Lmin and region.inside(uj.reshape((1,-1))) and (tregion is None or tregion.inside(transform(uj.reshape((1, -1))))):
                    del ui, Li
                    ui, Li = uj, Lj
                    # print("recovered off-track walk from point %d/%d " % (j+1, len(self.history)))
                    self.last = ui, Li

                    # pj = transform(uj.reshape((1, -1)))
                    # Lj2 = loglike(pj)[0]
                    # assert Lj2 > Lmin, (Lj2, Lj, uj, pj)

                    break
            pass

        # select starting point
        if Li is None:
            self.new_chain(region)
            # choose a new random starting point
            # mask = region.inside(us)
            # assert mask.any(), ("One of the live points does not satisfies the current region!",
            #    region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(len(us))
            self.starti = i
            del Li, ui
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


class CubeMHSampler(StepSampler):
    """Simple step sampler, staggering around in cube space."""

    def move(self, ui, region, ndraw=1, plot=False):
        """Move in cube space."""
        # propose in that direction
        jitter = np.random.normal(0, 1, size=(min(10, ndraw), len(ui))) * self.scale
        unew = ui.reshape((1, -1)) + jitter
        return unew


class RegionMHSampler(StepSampler):
    """Simple step sampler, staggering around in transformLayer space."""

    def move(self, ui, region, ndraw=1, plot=False):
        """Move in transformLayer space."""
        ti = region.transformLayer.transform(ui)
        jitter = np.random.normal(0, 1, size=(min(10, ndraw), len(ui))) * self.scale
        tnew = ti.reshape((1, -1)) + jitter
        unew = region.transformLayer.untransform(tnew)
        return unew


class CubeSliceSampler(StepSampler):
    """Slice sampler, respecting the region."""

    def new_chain(self, region=None):
        """Starts a new path, reset slice."""
        self.interval = None
        self.found_left = False
        self.found_right = False
        self.axis_index = 0

        self.history = []
        self.last = None, None
        self.nrejects = 0

    def generate_direction(self, ui, region):
        """Start in a new direction, by choosing a random parameter."""
        return generate_cube_oriented_direction(ui, region)

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        """Adjust proposal given that we have been `accepted` at a new point after `nc` calls."""
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

                self.last = unew, Lnew
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
        """Advance the slice sampling move."""
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


class RegionSliceSampler(CubeSliceSampler):
    """Slice sampler, randomly picking region axes."""

    def generate_direction(self, ui, region):
        """Choose a random axis from region.transformLayer."""
        return generate_region_oriented_direction(ui, region, tscale=self.scale, scale=None)


class RegionSequentialSliceSampler(CubeSliceSampler):
    """Slice sampler, sequentially iterating region axes."""

    def generate_direction(self, ui, region, scale=1):
        """Choose from region.transformLayer the next axis iteratively."""
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


class BallSliceSampler(CubeSliceSampler):
    """Hit & run sampler. Choose random directions in space."""

    def generate_direction(self, ui, region):
        """Choose a isotropically random direction from region.transformLayer."""
        return generate_random_direction(ui, region)


class RegionBallSliceSampler(CubeSliceSampler):
    """Hit & run sampler. Choose random directions according to region."""

    def generate_direction(self, ui, region):
        """Choose a isotropically random direction from region.transformLayer."""
        return generate_region_random_direction(ui, region)


class SpeedVariableRegionSliceSampler(CubeSliceSampler):
    """Slice sampler, in region axes.

    Updates only some dimensions at a time, completely user-definable.
    """

    def __init__(self, step_matrix):
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

        """
        nsteps = len(step_matrix)

        StepSampler.__init__(self, nsteps=nsteps)
        self.reset()
        self.step_matrix = step_matrix

    def generate_direction(self, ui, region, scale=1):
        """Generate a slice sampling direction, using only some of the axes."""
        ndim = len(ui)
        ti = region.transformLayer.transform(ui)

        # choose random axis in transformed space
        j = np.random.randint(ndim)
        tv = np.zeros(ndim)
        tv[j] = 1.0
        # convert back to unit cube space:
        uj = region.transformLayer.untransform(ti + tv * 1e-3)
        uk = ui.copy()

        j = self.axis_index % self.nsteps
        self.axis_index = j + 1
        # only update active dimensions
        active_dims = self.step_matrix[j]
        # project uj onto ui. vary only active dimensions
        uk[active_dims] = uj[active_dims]

        v = uk - ui
        v *= scale / (v**2).sum()**0.5
        return v


def ellipsoid_bracket(ui, v, ellipsoid_center, ellipsoid_inv_axes, ellipsoid_radius_square):
    """ For a line from ui in direction v through an ellipsoid
    centered at ellipsoid_center with axes matrix ellipsoid_inv_axes,
    return the lower and upper intersection parameter."""
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
    """A line segment from *ui* in direction *v* from t between *left* <= 0 <= *right*
    will be truncated by the unit cube. Returns newleft, newright, cropped_left, cropped_right,
    i.e., the new end parameters and whether cropping was applied.
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


class AHARMSampler(StepSampler):
    """Accelerated hit-and-run/slice sampler, vectorised.

    Uses region ellipsoid to propose a sequence of points
    on a randomly drawn line.
    """

    def __init__(
        self, nsteps, adaptive_nsteps=False, max_nsteps=1000,
        region_filter=False, log=False, direction=generate_region_random_direction,
        orthogonalise=True,
    ):
        """Initialise vectorised hit-and-run/slice sampler.

        Parameters
        -----------
        nsteps: int
            number of accepted steps until the sample is considered independent.

        adaptive_nsteps: False, 'proposal-distance', 'move-distance'
            Select a strategy to adapt the number of steps. The strategies
            make sure that:

            * 'move-distance' (recommended): distance between
              start point and final position exceeds the mean distance
              between pairs of live points.
            * 'move-distance-midway': distance between
              start point and position in the middle of the chain
              exceeds the mean distance between pairs of live points.

        max_nsteps: int
            Maximum number of steps the adaptive_nsteps can reach.

        region_filter: bool
            if True, use region to check if a proposed point can be inside
            before calling likelihood.

        direction: function
            function that draws slice direction given a point and
            the current region.

        orthogonalise: bool
            If true, make subsequent proposed directions orthogonal
            to each other.

        log: file
            log file for sampler statistics, such as acceptance rate,
            proposal scale, number of steps, jump distance and distance
            between live points

        """
        self.history = []
        self.nsteps = nsteps
        self.nrejects = 0
        self.max_nsteps = max_nsteps
        self.last = None, None
        self.generate_direction = direction
        adaptive_nsteps_options = [
            False,
            'move-distance', 'move-distance-midway',
        ]

        if adaptive_nsteps not in adaptive_nsteps_options:
            raise ValueError("adaptive_nsteps must be one of: %s, not '%s'" % (adaptive_nsteps_options, adaptive_nsteps))
        self.adaptive_nsteps = adaptive_nsteps
        self.region_filter = region_filter
        self.log = log
        self.adaptive_nsteps_needs_mean_pair_distance = False
        self.nsteps_nudge = 1.01
        self.orthogonalise = orthogonalise

        self.logstat = []
        self.logstat_labels = ['rejection_rate', 'steps']
        if adaptive_nsteps:
            self.logstat_labels += ['jump-distance', 'reference-distance']

    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=1024, plot=False, tregion=None, verbose=False):
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
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            print("wandered out of L constraint; resetting", ui[0])
            ui, Li = None, None

        if ui is not None and not region.inside_ellipsoid(ui.reshape((1, -1))):
            print("wandered out of ellipsoid; resetting", ui[0])
            ui, Li = None, None

        if Li is None and self.history:
            # try to resume from a previous point above the current contour
            for j, (uj, Lj) in enumerate(self.history[::-1]):
                if Lj > Lmin and region.inside(uj.reshape((1,-1))) and (tregion is None or tregion.inside(transform(uj.reshape((1, -1))))):
                    ui, Li = uj, Lj
                    # print("recovering at point %d/%d " % (j+1, len(self.history)))
                    self.last = ui, Li

                    # pj = transform(uj.reshape((1, -1)))
                    # Lj2 = loglike(pj)[0]
                    # assert Lj2 > Lmin, (Lj2, Lj, uj, pj)
                    assert region.inside_ellipsoid(ui.reshape((1, -1)))

                    break
            pass

        # select starting point
        ndim = us.shape[1]
        if Li is None:
            self.directions = None

            self.history = []
            self.last = None, None
            self.nrejects = 0

            # choose a new random starting point
            i = np.random.randint(len(us))
            self.starti = i
            ui = us[i,:]
            assert region.inside_ellipsoid(ui.reshape((1, -1)))
            assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[i]
            self.history.append((ui.copy(), Li.copy()))
            del i
            # set initially nleft = nsteps
            self.nsteps_done = 0

            # generate nsteps directions
            self.directions = []
            for i in range(self.nsteps):
                v = self.generate_direction(ui, region)
                self.directions.append(v)
            self.directions = np.array(self.directions)

            if verbose:
                print("directions:", self.directions)
            if self.orthogonalise:
                # orthogonalise relative to this previous direction
                for i in range(self.nsteps // ndim):
                    # go back only ndim steps, then start fresh
                    self.directions[i * ndim:(i + 1) * ndim], _ = np.linalg.qr(self.directions[i * ndim:(i + 1) * ndim])

            assert (ui >= 0).all(), ui
            assert (ui <= 1).all(), ui
            self.current_interval = ui, None, None
            if self.region_filter:
                assert region.inside(ui.reshape((1, ndim))), ('cannot start from outside region!', region.inside(ui.reshape((1, ndim))))

        del ui
        nc = 0
        while True:
            # prepare a sequence of points until nsteps are reached
            point_sequence = []
            point_expectation = []
            intervals = []
            nsteps_prepared = 0
            while nsteps_prepared + self.nsteps_done < self.nsteps and len(point_sequence) < ndraw:
                v = self.directions[self.nsteps_done + nsteps_prepared]
                if len(point_sequence) == 0:
                    ucurrent, left, right = self.current_interval
                    assert (ucurrent >= 0).all(), ucurrent
                    assert (ucurrent <= 1).all(), ucurrent
                    assert region.inside_ellipsoid(ucurrent.reshape((1, ndim))), (
                        'cannot start from outside ellipsoid!', region.inside_ellipsoid(ucurrent.reshape((1, ndim))))
                    if self.region_filter:
                        assert region.inside(ucurrent.reshape((1, ndim))), (
                            'cannot start from outside region!', region.inside(ucurrent.reshape((1, ndim))))
                    assert loglike(transform(ucurrent.reshape((1, ndim)))) >= Lmin, (
                        'cannot start from outside!', loglike(transform(ucurrent.reshape((1, ndim)))), Lmin)
                else:
                    left, right = None, None
                assert (ucurrent >= 0).all(), ucurrent
                assert (ucurrent <= 1).all(), ucurrent
                if verbose:
                    print("preparing step: %d from %s" % (nsteps_prepared + self.nsteps_done, ucurrent))

                if left is None or right is None:
                    # in each, find the end points using the expanded ellipsoid
                    assert region.inside_ellipsoid(ucurrent.reshape((1, ndim))), ('current point outside ellipsoid!')
                    left, right = ellipsoid_bracket(ucurrent, v, region.ellipsoid_center, region.ellipsoid_inv_axes, region.enlarge)
                    left, right, _, _ = crop_bracket_at_unit_cube(ucurrent, v, left, right)
                    assert (ucurrent + v * left <= 1).all(), (
                        ucurrent, v, region.ellipsoid_center, region.ellipsoid_inv_axes, region.ellipsoid_invcov, region.enlarge)
                    assert (ucurrent + v * right <= 1).all(), (
                        ucurrent, v, region.ellipsoid_center, region.ellipsoid_inv_axes, region.ellipsoid_invcov, region.enlarge)
                    assert (ucurrent + v * left >= 0).all(), (
                        ucurrent, v, region.ellipsoid_center, region.ellipsoid_inv_axes, region.ellipsoid_invcov, region.enlarge)
                    assert (ucurrent + v * right >= 0).all(), (
                        ucurrent, v, region.ellipsoid_center, region.ellipsoid_inv_axes, region.ellipsoid_invcov, region.enlarge)

                    assert left <= 0 <= right, (left, right)
                    if verbose:
                        print("   ellipsoid bracket found:", left, right)

                while True:
                    # sample in each a point until presumed success:
                    assert region.inside_ellipsoid(ucurrent.reshape((1, ndim))), ('current point outside ellipsoid!')
                    t = np.random.uniform(left, right)
                    unext = ucurrent + v * t
                    assert (unext >= 0).all(), unext
                    assert (unext <= 1).all(), unext
                    assert region.inside_ellipsoid(unext.reshape((1, ndim))), ('proposal landed outside ellipsoid!', t, left, right)

                    # compute distance vector to center
                    d = unext - region.ellipsoid_center
                    # distance in normalised coordates: vector . matrix . vector
                    # where the matrix is the ellipsoid inverse covariance
                    r = np.einsum('j,jk,k->', d, region.ellipsoid_invcov, d)

                    likely_inside = r <= 1
                    if not likely_inside and r <= region.enlarge:
                        # The exception is, when a point is between projected ellipsoid center and current point
                        # then it is also likely inside (if still inside the ellipsoid)

                        # project ellipsoid center onto line
                        # region.ellipsoid_center = ucurrent + tc * v
                        tc = np.dot(region.ellipsoid_center - ucurrent, v)
                        # current point is at 0 by definition
                        if 0 < t < tc or tc < t < 0:
                            if verbose:
                                print("   proposed point is further inside than current point")
                            likely_inside = True
                        #    print("   proposed point %.3f is going towards center %.3f" % (t, tc))
                        # else:
                        #    print("   proposed point %.3f is going away from center %.3f" % (t, tc))
                        else:
                            # another exception is that points very close to the current point
                            # are very likely also inside
                            # to find that out, project all live points on the line
                            tall = np.einsum('ij,j->i', region.u - ucurrent, v)
                            # find the range and identify a small part of it
                            epsilon_nearby = 1e-6
                            if tc < (tall.max() - tall.min()) * epsilon_nearby:
                                likely_inside = True
                                if verbose:
                                    print("   proposed point is very nearby")

                    if verbose:
                        print("   proposed point %s (%f) is likely %s (r=%f)" % (unext, t, 'inside' if likely_inside else 'outside', r))
                    intervals.append((nsteps_prepared, ucurrent, v, left, right, t))
                    point_sequence.append(unext)
                    point_expectation.append(likely_inside)
                    #   If point radius in ellipsoid is <1, presume that it will be successful
                    if likely_inside:
                        nsteps_prepared += 1
                        ucurrent = unext
                        assert region.inside_ellipsoid(ucurrent.reshape((1, ndim))), ('current point outside ellipsoid!')
                        break

                    #   Else, presume it will be unsuccessful, and sample another point
                    #   shrink interval
                    if t > 0:
                        right = t
                    else:
                        left = t

                # the above defines a sequence of points (u)

            assert len(point_sequence) > 0, (len(point_sequence), ndraw, nsteps_prepared, self.nsteps_done, self.nsteps)
            point_sequence = np.array(point_sequence, dtype=float)
            point_expectation = np.array(point_expectation, dtype=bool)
            if verbose:
                print("proposed sequence:", point_sequence)
                print("expectations:", point_expectation)
            truncated = False
            # region-filter, transform, tregion-filter, and evaluate the likelihood
            if self.region_filter:
                mask_inside = region.inside(point_sequence)
                # identify first point that was expected to be inside, but was marked outside-of-region
                i = np.where(np.logical_and(point_expectation, ~mask_inside))[0]
                if verbose:
                    print("region filter says:", mask_inside, i)
                if len(i) > 0:
                    imax = i[0] + 1
                    # truncate there
                    point_sequence = point_sequence[:imax]
                    point_expectation = point_expectation[:imax]
                    mask_inside = mask_inside[:imax]
                    truncated |= True
                    del imax
                if not mask_inside.any():
                    continue

            t_point_sequence = transform(point_sequence)
            if self.region_filter and tregion is not None:
                tmask = tregion.inside(t_point_sequence)
                # identify first point that was expected to be inside, but was marked outside-of-region
                i = np.where(np.logical_and(point_expectation, ~tmask))[0]
                if verbose:
                    print("tregion filter says:", tmask, i)
                mask_inside[~tmask] = False
                del tmask
                if len(i) > 0:
                    imax = i[0] + 1
                    # truncate there
                    point_sequence = point_sequence[:imax]
                    point_expectation = point_expectation[:imax]
                    t_point_sequence = t_point_sequence[:imax]
                    mask_inside = mask_inside[:imax]
                    truncated |= True
                    del imax
                if not mask_inside.any():
                    continue

            # we expect the last point to be an accept, otherwise we would not terminate the sequence
            assert point_expectation[-1]
            if self.region_filter:
                # set filtered ones to -np.inf
                L = np.ones(len(t_point_sequence)) * -np.inf
                nc += mask_inside.sum()
                L[mask_inside] = loglike(t_point_sequence[mask_inside,:])
            else:
                nc += len(point_sequence)
                L = loglike(t_point_sequence)
            Lmask = L > Lmin
            i = np.where(point_expectation != Lmask)[0]
            self.nrejects += (~Lmask).sum()
            if verbose:
                print("reality:", Lmask)
                print("difference:", point_expectation == Lmask)
            print("calling likelihood with %5d prepared points, accepted:" % (
                len(point_sequence)), '=' * (i[0] + Lmask[i[0]] * 1 if len(i) > 0 else len(Lmask)))
            # identify first point that was unexpected
            if len(i) == 0 and nsteps_prepared + self.nsteps_done == self.nsteps:
                # everything according to prediction.
                if verbose:
                    print("everything according to prediction and done")
                # done, return last point
                for ui, Li in zip(point_sequence[Lmask], L[Lmask]):
                    self.history.append((ui, Li))
                self.finalize_chain(region=region, Lmin=Lmin, Ls=Ls)
                return point_sequence[-1], t_point_sequence[-1], L[-1], nc
            elif len(i) == 0:
                # everything according to prediction.
                if verbose:
                    print("everything according to prediction")
                # continue from last point
                for ui, Li in zip(point_sequence[Lmask], L[Lmask]):
                    self.history.append((ui, Li))
                self.nsteps_done += nsteps_prepared
                assert self.nsteps_done == len(self.history), (self.nsteps_done, len(self.history))
                nsteps_prepared, ucurrent, v, left, right, t = intervals[-1]
                assert (ucurrent >= 0).all(), ucurrent
                assert (ucurrent <= 1).all(), ucurrent
                self.current_interval = ucurrent, None, None
                if self.region_filter:
                    assert region.inside(ucurrent.reshape((1, ndim))), ('suggested point outside region!', region.inside(ucurrent.reshape((1, ndim))))
            else:
                # point i unexpectedly inside or outside
                imax = i[0]
                for ui, Li in zip(point_sequence[:imax][Lmask[:imax]], L[:imax][Lmask[:imax]]):
                    self.history.append((ui, Li))
                nsteps_prepared, ucurrent, v, left, right, t = intervals[imax]
                if self.region_filter:
                    assert region.inside(ucurrent.reshape((1, ndim))), ('suggested point outside region!', region.inside(ucurrent.reshape((1, ndim))))
                assert (ucurrent >= 0).all(), ucurrent
                assert (ucurrent <= 1).all(), ucurrent
                if point_expectation[imax]:
                    if verbose:
                        print("following prediction until %d, which was unexpectedly rejected" % imax)
                    # expected point to lie inside, but rejected
                    # need to repair interval
                    self.nsteps_done += nsteps_prepared
                    assert self.nsteps_done + 1 == len(self.history), (self.nsteps_done, len(self.history))
                    if t > 0:
                        right = t
                    else:
                        left = t
                    if verbose:
                        print("%d steps done, continuing from unexpected outside point" % self.nsteps_done, imax, point_sequence[imax], "interval:", t)
                    self.current_interval = ucurrent, left, right
                else:
                    if verbose:
                        print("following prediction until %d, which was unexpectedly accepted" % imax)
                    if imax == len(point_sequence) - 1 and truncated:
                        assert False
                    ucurrent = point_sequence[imax]
                    if self.region_filter:
                        assert region.inside(ucurrent.reshape((1, ndim))), ('accepted point outside region!', region.inside(ucurrent.reshape((1, ndim))))
                    # expected point to lie outside, but actually inside
                    # adopt as point and continue
                    # print(len(self.history), self.nsteps_done, nsteps_prepared, Lmask[:imax].sum())
                    self.nsteps_done += nsteps_prepared + 1
                    self.history.append((ucurrent.copy(), L[imax]))
                    assert self.nsteps_done + 1 == len(self.history), (self.nsteps_done, len(self.history))
                    self.current_interval = ucurrent, None, None
                    if self.nsteps_done == self.nsteps:
                        # last point was inside, so we are actually done there
                        self.finalize_chain(region=region, Lmin=Lmin, Ls=Ls)
                        return point_sequence[-1], t_point_sequence[-1], L[-1], nc
                    else:
                        if verbose:
                            print("%d steps done, continuing from unexpected inside point" % self.nsteps_done, imax, point_sequence[imax])

            # need to exit here to only do one likelihood evaluation
            # per function call
            if verbose:
                print("breaking")
            break

        # do not have a independent sample yet
        return None, None, None, nc

    def region_changed(self, Ls, region):
        assert region.inside_ellipsoid(region.u).all()
        ui, Li = self.last
        if ui is not None and not region.inside(ui.reshape((1, -1))):
            print("wandered out of ellipsoid; resetting", ui[0])
            self.last = None, None

    def finalize_chain(self, region=None, Lmin=None, Ls=None):
        """Store chain statistics and adapt proposal."""
        self.logstat.append([self.nrejects / self.nsteps, self.nsteps])
        if self.log:
            ustart, Lstart = self.history[0]
            ufinal, Lfinal = self.history[-1]
            # mean_pair_distance = region.compute_mean_pair_distance()
            mean_pair_distance = np.nan
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

        self.last = None, None
        self.history = []
        self.nrejects = 0

    def generate_new_interval(self, ui, region):
        v = self.generate_direction(ui, region)
        assert region.inside_ellipsoid(ui.reshape((1, -1)))
        assert (ui > 0).all(), ui
        assert (ui < 1).all(), ui

        # use region ellipsoid to identify limits
        # rotate line so that ellipsoid is a sphere
        left, right = ellipsoid_bracket(ui, v, region.ellipsoid_center, region.ellipsoid_inv_axes, region.enlarge)
        left, right, _, _ = crop_bracket_at_unit_cube(ui, v, left, right)
        self.interval = (v, left, right, 0)
