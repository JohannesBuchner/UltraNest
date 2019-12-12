"""MCMC-like step sampling within a region.

The classes implemented here are generators that, in each iteration,
only make one likelihood call. This allows keeping a population of
samplers that have the same execution time per call, even if they
do not terminate at the same number of iterations.
"""

import numpy as np

import matplotlib.pyplot as plt


def generate_random_direction(ui, region, scale=1):
    """Draw uniform direction vector in unit cube space of length `scale`.

    Region is not used.
    """
    v = np.random.normal(0, 1, size=len(ui))
    v *= scale / (v**2).sum()**0.5
    return v


def generate_cube_oriented_direction(ui, region):
    """Draw a unit direction vector in direction of a random unit cube axes.

    `region` is not used.
    """
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
    ti = region.transformLayer.transform(ui)

    # choose axis in transformed space:
    ti = np.random.normal(ti, 1)
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


def inside_region(region, unew, uold):
    """Check if `unew` is inside region.

    always returns True at the moment.
    """
    return np.ones(len(unew), dtype=bool)

    tnew = region.transformLayer.transform(unew)
    told = region.transformLayer.transform(uold)
    mask2 = ((told.reshape((1, -1)) - tnew)**2).sum(axis=1) < region.maxradiussq
    if mask2.all():
        return mask2

    mask = region.inside(unew)
    return np.logical_or(mask, mask2)


class StepSampler(object):
    """Base class for a simple step sampler, staggering around.

    Scales proposal towards a 50% acceptance rate.
    """

    def __init__(self, nsteps, adapt_nsteps=False):
        """Initialise sampler.

        Parameters
        -----------
        nsteps: int
            number of accepted steps until the sample is considered independent.
        
        adapt_nsteps: False, 'proposal-distance', 'move-distance'
            if not false, allow earlier termination than nsteps.
            The 'proposal-distance' strategy stops when the sum of 
            all proposed vectors exceeds the mean distance
            between pairs of live points.
            As distance, the Mahalanobis distance is used.
            The 'move-distance' strategy stops when the distance between
            start point and current position exceeds the mean distance
            between pairs of live points.

        """
        self.history = []
        self.nsteps = nsteps
        self.nrejects = 0
        self.scale = 1.0
        self.last = None, None
        self.nudge = 1.1**(1. / self.nsteps)
        
        assert adapt_nsteps in (False, 'proposal-distance', 'move-distance'), \
            "adapt_nsteps must be either False, 'proposal-distance' or 'move-distance', not '%s'" % adapt_nsteps
        self.adapt_nsteps = adapt_nsteps
        self.logstat = []
        self.logstat_labels = ['accepted', 'scale']

    def __str__(self):
        """Get string representation."""
        return type(self).__name__ + '(nsteps=%d)' % self.nsteps

    def plot(self, filename):
        """Plot sampler statistics."""
        if len(self.logstat) == 0:
            return

        parts = np.transpose(self.logstat)
        plt.figure(figsize=(10, 1 + 3 * len(parts)))
        for i, (label, part) in enumerate(zip(self.logstat_labels, parts)):
            plt.subplot(len(parts), 1, 1 + i)
            plt.ylabel(label)
            plt.plot(part)
            x = []
            y = []
            for j in range(0, len(part), 20):
                x.append(j)
                y.append(part[j:j + 20].mean())
            plt.plot(x, y)
            if np.min(part) > 0:
                plt.yscale('log')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def move(self, ui, region, ndraw=1, plot=False):
        """Move around ui. Stub to be implemented."""
        raise NotImplementedError()

    def adjust_outside_region(self):
        """Adjust proposal given that we landed outside region."""
        # print("ineffective proposal scale (%e). shrinking..." % self.scale)
        self.scale /= self.nudge**10
        assert self.scale > 0
        self.last = None, None
        self.logstat.append([False, self.scale])

    def adjust_accept(self, accepted, unew, pnew, Lnew, nc):
        """Adjust proposal given that we have been `accepted` at a new point after `nc` calls."""
        if accepted:
            # if self.scale < 1:
            self.scale *= self.nudge
            self.last = unew, Lnew
            self.history.append((unew, Lnew))
        else:
            self.scale /= self.nudge**10
            self.nrejects += 1
        self.logstat.append([accepted, self.scale])

    def reset(self):
        """Reset current path statistic."""
        self.nrejects = 0
    
    def region_changed(self, Ls, region):
        """React to change of region. Stub to be implemented."""
        
        if self.adapt_nsteps:
            self.pair_mean_distance = region.compute_pair_mean_distance()

    def is_converged(self, region, u):
        # check if length exceeds expected maximum
        if len(self.history) >= self.nsteps:
            return True
        
        # if we do not adapt, we are not done yet
        if not self.adapt_nsteps:
            return False
        elif self.adapt_nsteps == 'proposal-distance':
            d2 = self.proposal_distance
            far_enough = d2 > self.pair_mean_distance
            return far_enough
        elif self.adapt_nsteps == 'move-distance':
            d2 = ((region.transform(self.startu) - region.transform(u))**2).sum()
            far_enough = d2 > self.pair_mean_distance
            return far_enough
        assert False
        
    def __next__(self, region, Lmin, us, Ls, transform, loglike, ndraw=40, plot=False):
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

        """
        # find most recent point in history conforming to current Lmin
        ui, Li = self.last
        if Li is not None and not Li >= Lmin:
            print("wandered out of L constraint; resetting", ui[0])
            ui, Li = None, None

        # if Li is not None and not region.inside(ui.reshape((1,-1))):
        #     # region was updated and we are not inside anymore
        #     # so reset
        #     print("wandered out of region; resetting", ui[0])
        #     ui, Li = None, None

        if Li is None and self.history:
            # try to resume from a previous point above the current contour
            for uj, Lj in self.history[::-1]:
                if Lj > Lmin and region.inside(uj.reshape((1,-1))):
                    ui, Li = uj, Lj
                    break
            pass

        # select starting point
        if Li is None:
            # choose a new random starting point
            mask = region.inside(us)
            assert mask.any(), ("One of the live points does not satisfies the current region!",
                region.maxradiussq, region.u, region.unormed, us)
            i = np.random.randint(mask.sum())
            self.starti = i
            ui = us[mask,:][i]
            # print("starting at", ui[0])
            assert np.logical_and(ui > 0, ui < 1).all(), ui
            Li = Ls[mask][i]
            self.reset()
            self.history.append((ui, Li))

        unew = self.move(ui, region, ndraw=ndraw, plot=plot)
        # print("proposed", unew)
        if plot:
            plt.plot([ui[0], unew[:,0]], [ui[1], unew[:,1]], '-', color='k', lw=0.5)
            plt.plot(ui[0], ui[1], 'd', color='r', ms=4)
            plt.plot(unew[:,0], unew[:,1], 'x', color='r', ms=4)
        mask = np.logical_and(unew > 0, unew < 1).all(axis=1)
        unew = unew[mask,:]
        mask = inside_region(region, unew, ui)
        nc = 0

        if mask.any():
            i = np.where(mask)[0][0]
            unew = unew[i,:]
            pnew = transform(unew.reshape((1, -1)))
            Lnew = loglike(pnew)[0]
            nc = 1
            if Lnew > Lmin:
                if plot:
                    plt.plot(unew[0], unew[1], 'o', color='g', ms=4)
                self.adjust_accept(True, unew, pnew, Lnew, nc)
                if self.is_converged(region=region, u=unew):
                    # print("made %d steps" % len(self.history), Lnew, Lmin)
                    self.history = []
                    self.last = None, None
                    return unew, pnew, Lnew, nc
            else:
                self.adjust_accept(False, unew, pnew, Lnew, nc)
        else:
            self.adjust_outside_region()

        # do not have a independent sample yet
        return None, None, None, nc


class CubeMHSampler(StepSampler):
    """Simple step sampler, staggering around in cube space."""

    def move(self, ui, region, ndraw=1, plot=False):
        """Move in cube space."""
        # propose in that direction
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        unew = ui.reshape((1, -1)) + jitter
        return unew


class RegionMHSampler(StepSampler):
    """Simple step sampler, staggering around in transformLayer space."""

    def move(self, ui, region, ndraw=1, plot=False):
        """Move in transformLayer space."""
        ti = region.transformLayer.transform(ui)
        jitter = np.random.normal(0, 1, size=(ndraw, len(ui))) * self.scale
        tnew = ti.reshape((1, -1)) + jitter
        unew = region.transformLayer.untransform(tnew)
        return unew


class CubeSliceSampler(StepSampler):
    """Slice sampler, respecting the region."""

    def __init__(self, nsteps):
        """Initialise sampler.

        Parameters
        -----------
        nsteps: int
            number of accepted steps until the sample is considered independent.

        """
        StepSampler.__init__(self, nsteps=nsteps)
        self.reset()

    def reset(self):
        """Reset current slice."""
        self.interval = None
        self.found_left = False
        self.found_right = False
        self.axis_index = 0

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
                if -left > self.scale or right > self.scale:
                    self.scale *= 1.1
                else:
                    self.scale /= 1.1
        else:
            if accepted:
                # start with a new interval next time
                self.interval = None

                self.last = unew, Lnew
                self.history.append((unew, Lnew))
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
        self.logstat.append([accepted, self.scale])

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

            if inside_region(region, xj.reshape((1, -1)), ui):
                return xj.reshape((1, -1))
            else:
                self.found_left = True

        if not self.found_right:
            xj = ui + v * right

            if inside_region(region, xj.reshape((1, -1)), ui):
                return xj.reshape((1, -1))
            else:
                self.found_right = True

        while True:
            u = np.random.uniform(left, right)
            xj = ui + v * u

            if inside_region(region, xj.reshape((1, -1)), ui):
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

