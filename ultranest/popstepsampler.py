# noqa: D400 D205
"""
Vectorized step samplers
------------------------

Likelihood based on GPUs (model emulators based on neural networks,
or JAX implementations) can evaluate hundreds of points as efficiently
as one point. The implementations in this module leverage this power,
by providing random walks of populations of walkers.
"""

import numpy as np
import scipy.stats

from ultranest.stepfuncs import (evolve, generate_cube_oriented_direction,
                                 generate_cube_oriented_direction_scaled,
                                 generate_differential_direction,
                                 generate_mixture_random_direction,
                                 generate_random_direction,
                                 generate_region_oriented_direction,
                                 generate_region_random_direction, int_dtype,
                                 step_back, update_vectorised_slice_sampler)
from ultranest.utils import submasks


def unitcube_line_intersection(ray_origin, ray_direction):
    r"""Compute intersection of a line (ray) and a unit box (0:1 in all axes).

    Based on
    http://www.iquilezles.org/www/articles/intersectors/intersectors.htm

    Parameters
    -----------
    ray_origin: array of vectors
        starting point of line
    ray_direction: vector
        line direction vector

    Returns
    --------
    tleft: array
        negative intersection point distance from ray\_origin in units in ray\_direction
    tright: array
        positive intersection point distance from ray\_origin in units in ray\_direction

    """
    # make sure ray starts inside the box
    assert (ray_origin >= 0).all(), ray_origin
    assert (ray_origin <= 1).all(), ray_origin
    assert ((ray_direction**2).sum()**0.5 > 1e-200).all(), ray_direction

    # step size
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1. / ray_direction
        n = m * (ray_origin - 0.5)
        k = np.abs(m) * 0.5
        # line coordinates of intersection
        # find first intersecting coordinate
        t1 = -n - k
        t2 = -n + k
        return np.nanmax(t1, axis=1), np.nanmin(t2, axis=1)


def diagnose_move_distances(region, ustart, ufinal):
    """Compare random walk travel distance to MLFriends radius.

    Compares in whitened space (t-space), the L2 norm between final
    point and starting point to the MLFriends bootstrapped radius.

    Parameters
    ----------
    region: MLFriends
        built region
    ustart: array
        starting positions
    ufinal: array
        final positions

    Returns
    -------
    far_enough: bool
        whether the distance is larger than the radius
    move_distance: float
        distance between start and final point in whitened space
    reference_distance: float
        MLFriends radius
    """
    assert ustart.shape == ufinal.shape, (ustart.shape, ufinal.shape)
    tstart = region.transformLayer.transform(ustart)
    tfinal = region.transformLayer.transform(ufinal)
    d2 = ((tstart - tfinal)**2).sum(axis=1)
    far_enough = d2 > region.maxradiussq

    return far_enough, [d2**0.5, region.maxradiussq**0.5]


class GenericPopulationSampler():
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

        import matplotlib.pyplot as plt
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

    @property
    def mean_jump_distance(self):
        """Geometric mean jump distance."""
        if len(self.logstat) == 0:
            return np.nan
        return np.exp(np.average(
            np.log([entry[-1] + 1e-10 for entry in self.logstat]),
            weights=([entry[0] for entry in self.logstat])
        ))

    @property
    def far_enough_fraction(self):
        """Fraction of jumps exceeding reference distance."""
        if len(self.logstat) == 0:
            return np.nan
        return np.average(
            [entry[-2] for entry in self.logstat],
            weights=([entry[0] for entry in self.logstat])
        )

    def get_info_dict(self):
        return dict(
            num_logs=len(self.logstat),
            rejection_rate=1 - np.nanmean([entry[0] for entry in self.logstat]) if len(self.logstat) > 0 else np.nan,
            mean_scale=np.nanmean([entry[1] for entry in self.logstat]) if len(self.logstat) > 0 else np.nan,
            mean_nsteps=np.nanmean([entry[2] for entry in self.logstat]) if len(self.logstat) > 0 else np.nan,
            mean_distance=self.mean_jump_distance,
            frac_far_enough=self.far_enough_fraction,
            last_logstat=dict(zip(self.logstat_labels, self.logstat[-1] if len(self.logstat) > 1 else [np.nan] * len(self.logstat_labels)))
        )

    def print_diagnostic(self):
        """Print diagnostic of step sampler performance."""
        if len(self.logstat) == 0:
            print("diagnostic unavailable, no recorded steps found")
            return
        frac_farenough = self.far_enough_fraction
        average_distance = self.mean_jump_distance
        if frac_farenough < 0.5:
            advice = ': very fishy. Double nsteps and see if fraction and lnZ change)'
        elif frac_farenough < 0.66:
            advice = ': fishy. Double nsteps and see if fraction and lnZ change)'
        else:
            advice = ' (should be >50%)'
        print('step sampler diagnostic: jump distance %.2f (should be >1), far enough fraction: %.2f%% %s' % (
            average_distance, frac_farenough * 100, advice))

    def plot_jump_diagnostic_histogram(self, filename, **kwargs):
        """Plot jump diagnostic histogram."""
        if len(self.logstat) == 0:
            return
        import matplotlib.pyplot as plt
        plt.hist(np.log10([entry[-1] for entry in self.logstat]), **kwargs)
        ylo, yhi = plt.ylim()
        plt.vlines(self.mean_jump_distance, ylo, yhi)
        plt.ylim(ylo, yhi)
        plt.xlabel('log(relative step distance)')
        plt.ylabel('Frequency')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


class PopulationRandomWalkSampler(GenericPopulationSampler):
    """Vectorized Gaussian Random Walk sampler."""

    def __init__(
        self, popsize, nsteps, generate_direction, scale,
        scale_adapt_factor=0.9, scale_min=1e-20, scale_max=20, log=False, logfile=None
    ):
        """Initialise.

        Parameters
        ----------
        popsize: int
            number of walkers to maintain.
            this should be fairly large (~100), if too large you probably get memory issues
            Also, some results have to be discarded as the likelihood threshold increases.
            Observe the nested sampling efficiency.
        nsteps: int
            number of steps to take until the found point is accepted as independent.
            To find the right value, see :py:class:`ultranest.calibrator.ReactiveNestedCalibrator`
        generate_direction: function
            Function that gives proposal kernel shape, one of:
            :py:func:`ultranest.popstepsampler.generate_cube_oriented_direction`
            :py:func:`ultranest.popstepsampler.generate_cube_oriented_direction_scaled`
            :py:func:`ultranest.popstepsampler.generate_random_direction`
            :py:func:`ultranest.popstepsampler.generate_region_oriented_direction`
            :py:func:`ultranest.popstepsampler.generate_region_random_direction`
        scale: float
            initial guess for the proposal scaling factor
        scale_adapt_factor: float
            if 1, no adapting is done.
            if <1, the scale is increased if the acceptance rate is below 23.4%,
            or decreased if it is above, by *scale_adapt_factor*.
        scale_min: float
            lowest value allowed for scale, do not adapt down further
        scale_max: float
            highest value allowed for scale, do not adapt up further
        logfile: file
            where to print the current scaling factor and acceptance rate

        """
        self.nsteps = nsteps
        self.nrejects = 0
        self.scale = scale
        self.ncalls = 0
        assert scale_adapt_factor <= 1
        self.scale_adapt_factor = scale_adapt_factor
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.log = log
        self.logfile = logfile
        self.logstat = []
        self.logstat_labels = ['accept_rate', 'efficiency', 'scale', 'far_enough', 'mean_rel_jump']
        self.prepared_samples = []

        self.popsize = popsize
        self.generate_direction = generate_direction

    def __str__(self):
        """Return string representation."""
        return 'PopulationRandomWalkSampler(popsize=%d, nsteps=%d, generate_direction=%s, scale=%.g)' % (
            self.popsize, self.nsteps, self.generate_direction, self.scale)

    def region_changed(self, Ls, region):
        """Act upon region changed. Currently unused."""
        pass

    def __next__(
        self, region, Lmin, us, Ls, transform, loglike, ndraw=10,
        plot=False, tregion=None, log=False
    ):
        """Sample a new live point.

        Parameters
        ----------
        region: MLFriends object
            Region
        Lmin: float
            current log-likelihood threshold
        us: np.array((nlive, ndim))
            live points
        Ls: np.array(nlive)
            loglikelihoods live points
        transform: function
            prior transform function
        loglike: function
            loglikelihood function
        ndraw: int
            not used
        plot: bool
            not used
        tregion: bool
            not used
        log: bool
            not used

        Returns
        -------
        u: np.array(ndim) or None
            new point coordinates (None if not yet available)
        p: np.array(nparams) or None
            new point transformed coordinates (None if not yet available)
        L: float or None
            new point likelihood (None if not yet available)
        nc: int

        """
        nlive, ndim = us.shape

        # fill if empty:
        if len(self.prepared_samples) == 0:
            # choose live points
            ilive = np.random.randint(0, nlive, size=self.popsize)
            allu = us[ilive,:]
            allp = None
            allL = Ls[ilive]
            nc = self.nsteps * self.popsize
            nrejects_expected = self.nrejects + self.nsteps * self.popsize * (1 - 0.234)

            for _i in range(self.nsteps):
                # perturb walker population
                v = self.generate_direction(allu, region, self.scale)
                # compute intersection of u + t * v with unit cube
                tleft, tright = unitcube_line_intersection(allu, v)
                proposed_t = scipy.stats.truncnorm.rvs(tleft, tright, loc=0, scale=1).reshape((-1, 1))

                proposed_u = allu + v * proposed_t
                mask_outside = ~np.logical_and(proposed_u > 0, proposed_u < 1).all(axis=1)
                assert not mask_outside.any(), proposed_u[mask_outside, :]

                proposed_p = transform(proposed_u)
                # accept if likelihood threshold exceeded
                proposed_L = loglike(proposed_p)
                mask_accept = proposed_L > Lmin
                self.nrejects += (~mask_accept).sum()
                allu[mask_accept,:] = proposed_u[mask_accept,:]
                if allp is None:
                    del allp
                    allp = proposed_p * np.nan
                allp[mask_accept,:] = proposed_p[mask_accept,:]
                allL[mask_accept] = proposed_L[mask_accept]
            assert np.isfinite(allp).all(), 'some walkers never moved! Double nsteps of PopulationRandomWalkSampler.'
            far_enough, (move_distance, reference_distance) = diagnose_move_distances(region, us[ilive[mask_accept],:], allu[mask_accept,:])
            self.prepared_samples = list(zip(allu, allp, allL))

            self.logstat.append([
                mask_accept.mean(),
                1 - (self.nrejects - (nrejects_expected - self.nsteps * self.popsize * (1 - 0.234))) / (self.nsteps * self.popsize),
                self.scale,
                self.nsteps,
                np.mean(far_enough),
                np.exp(np.mean(np.log(move_distance / reference_distance + 1e-10)))
            ])
            if self.logfile:
                self.logfile.write("rescale\t%.4f\t%.4f\t%g\t%.4f%g\n" % self.logstat[-1])

            # adapt slightly
            if self.nrejects > nrejects_expected and self.scale > self.scale_min:
                # lots of rejects, decrease scale
                self.scale *= self.scale_adapt_factor
            elif self.nrejects < nrejects_expected and self.scale < self.scale_max:
                self.scale /= self.scale_adapt_factor
        else:
            nc = 0

        u, p, L = self.prepared_samples.pop(0)
        return u, p, L, nc


class PopulationSliceSampler(GenericPopulationSampler):
    """Vectorized slice/HARM sampler.

    Can revert until all previous steps have likelihoods allL above Lmin.
    Updates currentt, generation and allL, in-place.
    """

    def __init__(
        self, popsize, nsteps, generate_direction, scale=1.0,
        scale_adapt_factor=0.9, log=False, logfile=None
    ):
        """Initialise.

        Parameters
        ----------
        popsize: int
            number of walkers to maintain
        nsteps: int
            number of steps to take until the found point is accepted as independent.
            To find the right value, see :py:class:`ultranest.calibrator.ReactiveNestedCalibrator`
        generate_direction: function `(u, region, scale) -> v`
            function such as `generate_unit_directions`, which
            generates a random slice direction.
        scale: float
            initial guess scale for the length of the slice
        scale_adapt_factor: float
            smoothing factor for updating scale.
            if near 1, scale is barely updating, if near 0,
            the last slice length is used as a initial guess for the next.

        """
        self.nsteps = nsteps
        self.nrejects = 0
        self.scale = scale
        self.scale_adapt_factor = scale_adapt_factor
        self.allu = []
        self.allL = []
        self.currentt = []
        self.currentv = []
        self.currentp = []
        self.generation = []
        self.current_left = []
        self.current_right = []
        self.searching_left = []
        self.searching_right = []
        self.ringindex = 0

        self.log = log
        self.logfile = logfile
        self.logstat = []
        self.logstat_labels = ['accept_rate', 'efficiency', 'scale', 'far_enough', 'mean_rel_jump']

        self.popsize = popsize
        self.generate_direction = generate_direction

    def __str__(self):
        """Return string representation."""
        return 'PopulationSliceSampler(popsize=%d, nsteps=%d, generate_direction=%s, scale=%.g)' % (
            self.popsize, self.nsteps, self.generate_direction, self.scale)

    def region_changed(self, Ls, region):
        """Act upon region changed. Currently unused."""
        # self.scale = region.us.std(axis=1).mean()
        if self.logfile:
            self.logfile.write("region-update\t%g\t%g\n" % (self.scale, region.us.std(axis=1).mean()))

    def _setup(self, ndim):
        """Allocate arrays."""
        self.allu = np.zeros((self.popsize, self.nsteps + 1, ndim)) + np.nan
        self.allL = np.zeros((self.popsize, self.nsteps + 1)) + np.nan
        self.currentt = np.zeros(self.popsize) + np.nan
        self.currentv = np.zeros((self.popsize, ndim)) + np.nan
        self.generation = np.zeros(self.popsize, dtype=int_dtype) - 1
        self.current_left = np.zeros(self.popsize)
        self.current_right = np.zeros(self.popsize)
        self.searching_left = np.zeros(self.popsize, dtype=bool)
        self.searching_right = np.zeros(self.popsize, dtype=bool)

    def setup_start(self, us, Ls, starting):
        """Initialize walker starting points.

        For iteration zero, randomly selects a live point as starting point.

        Parameters
        ----------
        us: np.array((nlive, ndim))
            live points
        Ls: np.array(nlive)
            loglikelihoods live points
        starting: np.array(nwalkers, dtype=bool)
            which walkers to initialize.

        """
        if self.log:
            print("setting up:", starting)
        nlive = len(us)
        i = np.random.randint(nlive, size=starting.sum())

        if not starting.all():
            while starting[self.ringindex]:
                # if the one we are waiting for is being restarted,
                # we may as well pick the next one to wait for
                # because every other one is started from a random point
                # as well
                self.shift()

        self.allu[starting,0] = us[i]
        self.allL[starting,0] = Ls[i]
        self.generation[starting] = 0

    @property
    def status(self):
        """Return compact string representation of the current status."""
        s1 = ('G:' + ''.join(['%d' % g if g >= 0 else '_' for g in self.generation]))
        s2 = ('S:' + ''.join([
            'S' if not np.isfinite(self.currentt[i]) else 'L' if self.searching_left[i] else 'R' if self.searching_right[i] else 'B'
            for i in range(self.popsize)]))
        return s1 + '  ' + s2

    def setup_brackets(self, mask_starting, region):
        """Pick starting direction and range for slice.

        Parameters
        ----------
        mask_starting: np.array(nwalkers, dtype=bool)
            which walkers to set up.
        region: MLFriends object
            Region

        """
        if self.log:
            print("starting brackets:", mask_starting)
        i_starting, = np.where(mask_starting)
        self.current_left[i_starting] = -self.scale
        self.current_right[i_starting] = self.scale
        self.searching_left[i_starting] = True
        self.searching_right[i_starting] = True
        self.currentt[i_starting] = 0
        # choose direction for new slice
        self.currentv[i_starting,:] = self.generate_direction(
            self.allu[i_starting, self.generation[i_starting]],
            region)

    def _setup_currentp(self, nparams):
        if self.log:
            print("setting currentp")
        self.currentp = np.zeros((self.popsize, nparams)) + np.nan

    def advance(self, transform, loglike, Lmin, region):
        """Advance the walker population.

        Parameters
        ----------
        transform: function
            prior transform function
        loglike: function
            loglikelihood function
        Lmin: float
            current log-likelihood threshold
        region: MLFriends object
            Region

        Returns
        -------
        nc: int
            Number of likelihood function calls
        """
        movable = self.generation < self.nsteps
        all_movable = movable.all()
        # print("moving ", movable.sum(), self.popsize)
        if all_movable:
            i = np.arange(self.popsize)
            args = [
                self.allu[i, self.generation],
                self.allL[i, self.generation],
                # pass values directly
                self.currentt,
                self.currentv,
                self.current_left,
                self.current_right,
                self.searching_left,
                self.searching_right
            ]
            del i
        else:
            args = [
                self.allu[movable, self.generation[movable]],
                self.allL[movable, self.generation[movable]],
                # this makes copies
                self.currentt[movable],
                self.currentv[movable],
                self.current_left[movable],
                self.current_right[movable],
                self.searching_left[movable],
                self.searching_right[movable]
            ]
        if self.log:
            print("evolve will advance:", movable)

        uorig = args[0].copy()
        (
            (
                currentt, currentv,
                current_left, current_right, searching_left, searching_right
            ),
            (success, unew, pnew, Lnew),
            nc
        ) = evolve(transform, loglike, Lmin, *args)

        if success.any():
            far_enough, (move_distance, reference_distance) = diagnose_move_distances(region, uorig[success,:], unew)
            self.logstat.append([
                success.mean(),
                self.scale,
                self.nsteps,
                np.mean(far_enough) if len(far_enough) > 0 else 0,
                np.exp(np.mean(np.log(move_distance / reference_distance + 1e-10))) if len(far_enough) > 0 else 0
            ])
            if self.logfile:
                self.logfile.write("rescale\t%.4f\t%.4f\t%g\t%.4f%g\n" % self.logstat[-1])

        if self.log:
            print("movable", movable.shape, movable.sum(), success.shape)
        moved = submasks(movable, success)
        if self.log:
            print("evolve moved:", moved)
        self.generation[moved] += 1
        if len(pnew) > 0:
            if len(self.currentp) == 0:
                self._setup_currentp(nparams=pnew.shape[1])

            if self.log:
                print("currentp", self.currentp[moved,:].shape, pnew.shape)
            self.currentp[moved,:] = pnew

        # update with what we learned
        # print(currentu.shape, currentL.shape, success.shape, self.generation[movable])
        self.allu[moved, self.generation[moved]] = unew
        self.allL[moved, self.generation[moved]] = Lnew
        if all_movable:
            # in this case, the values were directly overwritten
            pass
        else:
            self.currentt[movable] = currentt
            self.currentv[movable] = currentv
            self.current_left[movable] = current_left
            self.current_right[movable] = current_right
            self.searching_left[movable] = searching_left
            self.searching_right[movable] = searching_right
        return nc

    def shift(self):
        """Update walker from which to pick next."""
        # this is a ring buffer
        # shift index forward, wrapping around
        # this is better than copying memory around when a element is removed
        self.ringindex = (self.ringindex + 1) % self.popsize

    def __next__(
        self, region, Lmin, us, Ls, transform, loglike, ndraw=10,
        plot=False, tregion=None, log=False
    ):
        """Sample a new live point.

        Parameters
        ----------
        region: MLFriends object
            Region
        Lmin: float
            current log-likelihood threshold
        us: np.array((nlive, ndim))
            live points
        Ls: np.array(nlive)
            loglikelihoods live points
        transform: function
            prior transform function
        loglike: function
            loglikelihood function
        ndraw: int
            not used
        plot: bool
            not used
        tregion: bool
            not used
        log: bool
            not used

        Returns
        -------
        u: np.array(ndim) or None
            new point coordinates (None if not yet available)
        p: np.array(nparams) or None
            new point transformed coordinates (None if not yet available)
        L: float or None
            new point likelihood (None if not yet available)
        nc: int

        """
        nlive, ndim = us.shape
        # initialize
        if len(self.allu) == 0:
            self._setup(ndim)

        step_back(Lmin, self.allL, self.generation, self.currentt)

        starting = self.generation < 0
        if starting.any():
            self.setup_start(us[Ls > Lmin], Ls[Ls > Lmin], starting)
        assert (self.generation >= 0).all(), self.generation

        # find those where bracket is undefined:
        mask_starting = ~np.isfinite(self.currentt)
        if mask_starting.any():
            self.setup_brackets(mask_starting, region)

        if self.log:
            print(str(self), "(before)")
        nc = self.advance(transform, loglike, Lmin, region)
        if self.log:
            print(str(self), "(after)")

        # harvest top individual if possible
        if self.generation[self.ringindex] == self.nsteps:
            if self.log:
                print("have a candidate")
            u, p, L = self.allu[self.ringindex, self.nsteps, :].copy(), self.currentp[self.ringindex, :].copy(), self.allL[self.ringindex, self.nsteps].copy()
            assert np.isfinite(u).all(), u
            assert np.isfinite(p).all(), p
            self.generation[self.ringindex] = -1
            self.currentt[self.ringindex] = np.nan
            self.allu[self.ringindex,:,:] = np.nan
            self.allL[self.ringindex,:] = np.nan

            # adjust guess length
            newscale = (self.current_right[self.ringindex] - self.current_left[self.ringindex]) / 2
            self.scale = self.scale * 0.9 + 0.1 * newscale

            self.shift()
            return u, p, L, nc
        else:
            return None, None, None, nc


def slice_limit_to_unitcube(tleft, tright):
    """
    Return the slice limits as of the intersection between the slice and the unit cube boundaries.

    Parameters
    ----------
    tleft: float
        Intersection of the unit cube with the slice in the negative direction
    tright: float
        Intersection of the unit cube with the slice in the positive direction

    Returns
    -------
    tnew: tuple
        Positive and negative slice limits, `(tleft_new, tright_new) = tnew`
    """
    tleft_new, tright_new = tleft.copy(), tright.copy()

    return tleft_new, tright_new


def slice_limit_to_scale(tleft, tright):
    """Return -1..+1 or the intersection between slice and unit cube if that is shorter.

    Parameters
    ----------
    tleft: float
        Intersection of the unit cube with the slice in the negative direction
    tright: float
        Intersection of the unit cube with the slice in the positive direction

    Returns
    -------
    tnew: tuple
        Positive and negative slice limits, `(tleft_new, tright_new) = tnew`
    """
    tleft_new = np.fmax(tleft, -1. + np.zeros_like(tleft))
    tright_new = np.fmin(tright, 1. + np.zeros_like(tright))

    return tleft_new, tright_new


class PopulationSimpleSliceSampler(GenericPopulationSampler):
    """Vectorized Slice sampler without stepping out procedure for quick look fits.

    Unlike `:py:class:PopulationSliceSampler`, in `:py:class:PopulationSimpleSliceSampler`,
    the likelihood is always called with the same number of points.

    Sliced are defined by the `:py:func:generate_direction` function on a interval defined
    around the current point. The centred interval has the width of the scale parameter,
    i.e, there is no stepping out procedure as in `:py:class:PopulationSliceSampler`.
    Slices are then shrink towards the current point until a point is found with a
    likelihood above the threshold.

    In the default case, i.e. `scale=None`, the slice width is defined as the
    intersection between itself and the unit cube. To improve the efficiency of the sampler,
    the slice can be reduced to an interval of size `2*scale` centred on the point. `scale`
    can be adapted with the `scale_adapt_factor` parameter based on the median distance
    between the current and the next point in a chains among all the chains. If the median
    distance is above `scale/adapt_slice_scale_target`, the scale is increased by `scale_adapt_factor`,
    and decreased otherwise. The `scale` parameter can also be jittered by a user supplied
    function `:py:func:scale_jitter_func` to counter balance the effect of a strong adaptation.

    In the case `scale!=None`, the detailed balance is not guaranteed, so this sampler should
    be use with caution.

    Multiple (`popsize`) slice sampling chains are run independently and in parallel.
    In that case, we read points as if they were the next selected each after the other.
    For a points to update the slice, it needs to be still in the part of the slices
    searched after the first point have been read. In that case, we update as normal,
    otherwise we discard the point.
    """

    def __init__(
        self, popsize, nsteps, generate_direction,
        scale_adapt_factor=1.0, adapt_slice_scale_target=2.0,
        scale=1.0, scale_jitter_func=None, slice_limit=slice_limit_to_unitcube,
        max_it=100, shrink_factor=1.0
    ):
        """Initialise.

        Parameters
        ----------
        popsize: int
            number of walkers to maintain.
        nsteps: int
            number of steps to take until the found point is accepted as independent.
            To calibrate, try several runs with increasing nsteps (doubling).
            The ln(Z) should become stable at some value.
        generate_direction: function
            Function that gives proposal kernel shape, one of:
            :py:func:`ultranest.popstepsampler.generate_random_direction`
            :py:func:`ultranest.popstepsampler.generate_region_oriented_direction`
            :py:func:`ultranest.popstepsampler.generate_region_random_direction`
            :py:func:`ultranest.popstepsampler.generate_differential_direction`
            :py:func:`ultranest.popstepsampler.generate_mixture_random_direction`
            :py:func:`ultranest.popstepsampler.generate_cube_oriented_direction` -> no adaptation in that case
            :py:func:`ultranest.popstepsampler.generate_cube_oriented_direction_scaled` -> no adaptation in that case
        scale: float
            initial guess for the slice width.
        scale_jitter_func: function
            User supplied function to multiply the `scale` by a random factor. For example,
            :py:func:`lambda : scipy.stats.truncnorm.rvs(-0.5, 5., loc=0, scale=1)+1.`
        scale_adapt_factor: float
            adaptation of `scale`. If 1: no adaptation. if <1, the scale is increased/decreased by this factor if the
            final slice length is shorter/longer than the `adapt_slice_scale_target*scale`.
        adapt_slice_scale_target: float
            Targeted ratio of the median distance between slice mid and final point among all chains of `scale`.
            Default: 2.0. Higher values are more conservative, lower values are faster.
        slice_limit: function
            Function setting the initial slice upper and lower bound. The default is `:py:func:slice_limit_to_unitcube`
            which defines  the slice limit as the intersection between the slice and the unit cube. An alternative
            when the `scale` is used is `:py:func:slice_limit_to_scale` which defines the slice limit as an interval
            of size `2*scale`. This function should either return a copy of the `tleft` and `tright` arguments or
            new arrays of the same shape.
        max_it: int
            maximum number of iterations to find a point on the slice. If the maximum number of iterations is reached,
            the current point is returned as the next one.
        shrink_factor: float
            For standard slice sampling shrinking, `shrink_factor=1`, the slice bound is updated to the last
            rejected point. Setting `shrink_factor>1` aggressively accelerates the shrinkage, by updating the
            new slice bound to `1/shrink_factor` of the distance between the current point and rejected point.
        """
        self.nsteps = nsteps

        self.max_it = max_it
        self.nrejects = 0
        self.generate_direction = generate_direction
        self.scale_adapt_factor = scale_adapt_factor
        self.ncalls = 0
        self.discarded = 0
        self.shrink_factor = shrink_factor
        assert shrink_factor >= 1.0, "The shrink factor should be greater than 1.0 to be efficient"

        self.scale = float(scale)

        self.adapt_slice_scale_target = adapt_slice_scale_target

        if scale_jitter_func is None:
            self.scale_jitter_func = lambda: 1.
        else:
            self.scale_jitter_func = scale_jitter_func
        self.prepared_samples = []
        self.popsize = popsize

        self.slice_limit = slice_limit

        self.logstat = []
        self.logstat_labels = ['accept_rate', 'efficiency', 'scale', 'far_enough', 'mean_rel_jump']

    def __str__(self):
        """Return string representation."""
        return 'PopulationSimpleSliceSampler(popsize=%d, nsteps=%d, generate_direction=%s, scale=%.g)' % (
            self.popsize, self.nsteps, self.generate_direction, self.scale)

    def region_changed(self, Ls, region):
        """Act upon region changed. Currently unused."""
        pass

    def __next__(
        self, region, Lmin, us, Ls, transform, loglike, ndraw=10,
        plot=False, tregion=None, log=False, test=False
    ):
        """Sample a new live point.

        Parameters
        ----------
        region: MLFriends object
            Region
        Lmin: float
            current log-likelihood threshold
        us: np.array((nlive, ndim))
            live points
        Ls: np.array(nlive)
            loglikelihoods live points
        transform: function
            prior transform function
        loglike: function
            loglikelihood function
        ndraw: int
            not used
        plot: bool
            not used
        tregion: bool
            not used
        log: bool
            not used
        test: bool
            In case of test of the reversibility of the sampler, the points drawn
            from the live points needs to be deterministic. This parameters is
            ensuring that.

        Returns
        -------
        u: np.array(ndim) or None
            new point coordinates (None if not yet available)
        p: np.array(nparams) or None
            new point transformed coordinates (None if not yet available)
        L: float or None
            new point likelihood (None if not yet available)
        nc: int

        """
        nlive, ndim = us.shape

        # fill if empty:
        if len(self.prepared_samples) == 0:
            # choose live points
            ilive = np.random.randint(0, nlive, size=self.popsize)
            allu = np.array(us[ilive,:]) if not test else np.array(us)
            allp = np.zeros((self.popsize, ndim)) * np.nan
            allL = np.array(Ls[ilive])
            nc = 0
            n_discarded = 0

            interval_final = 0.

            for _k in range(self.nsteps):
                # Defining scale jitter
                factor_scale = self.scale_jitter_func()
                # Defining slice direction
                v = self.generate_direction(allu, region, scale=1.0) * self.scale * factor_scale

                # limite of the slice based on the unit cube boundaries
                tleft_unitcube, tright_unitcube = unitcube_line_intersection(allu, v)

                # Defining bound of the slice
                # Bounds for each points and likelihood calls are identical initially

                # Slice bounds for each likelihood call
                tleft_worker, tright_worker = self.slice_limit(tleft_unitcube,tright_unitcube)

                # Slice bounds for each points
                tleft, tright = self.slice_limit(tleft_unitcube,tright_unitcube)
                # Index of the workers working concurrently
                worker_running = np.arange(self.popsize, dtype=int_dtype)
                # Status indicating if a points has already find its next position
                status = np.zeros(self.popsize, dtype=int_dtype)  # one for success, zero for running

                # Loop until each points has found its next position or we reached 100 iterations
                for _it in range(self.max_it):
                    # Sampling points on the slices
                    slice_position = np.random.uniform(size=(self.popsize,))
                    t = tleft_worker + (tright_worker - tleft_worker) * slice_position

                    points = allu[worker_running, :]
                    v_worker = v[worker_running, :]
                    proposed_u = points + t.reshape((-1,1)) * v_worker

                    proposed_p = transform(proposed_u)
                    proposed_L = loglike(proposed_p)
                    nc += self.popsize

                    # Updating the pool of points based on the newly sampled points
                    tleft, tright, worker_running, status, allu, allL, allp, n_discarded_it = update_vectorised_slice_sampler(
                        t, tleft, tright, proposed_L, proposed_u, proposed_p, worker_running, status, Lmin, self.shrink_factor,
                        allu, allL, allp, self.popsize)
                    n_discarded += n_discarded_it

                    # Update of the limits of the slices
                    tleft_worker = tleft[worker_running]
                    tright_worker = tright[worker_running]

                    if not np.any(status == 0):
                        break

                # Record of the final interval on theta for scale adaptation
                interval_final += np.median(tright - tleft)

            interval_final = interval_final / self.nsteps
            self.discarded += n_discarded
            self.ncalls += nc

            assert np.isfinite(allp).all(), 'some walkers never moved! Double nsteps of PopulationSimpleSliceSampler.'
            far_enough, (move_distance, reference_distance) = diagnose_move_distances(region, us[ilive,:], allu)
            self.prepared_samples = list(zip(allu, allp, allL))

            self.logstat.append([
                self.popsize / nc,
                self.scale,  # will always be 1. in the default case
                self.nsteps,
                np.mean(far_enough) if len(far_enough) > 0 else 0,
                np.exp(np.mean(np.log(move_distance / reference_distance + 1e-10))) if len(far_enough) > 0 else 0
            ])

            # Scale adaptation such that the final interval is
            # half the scale. There may be better things to do
            # here, but it seems to work.
            if interval_final >= 1. / self.adapt_slice_scale_target:
                self.scale *= 1. / self.scale_adapt_factor
            else:
                self.scale *= self.scale_adapt_factor
            # print("percentage of throws %.3f\n\n"%((self.throwed/self.ncalls)*100.))

        else:
            nc = 0

        u, p, L = self.prepared_samples.pop(0)
        return u, p, L, nc


__all__ = [
    "generate_cube_oriented_direction", "generate_cube_oriented_direction_scaled",
    "generate_random_direction", "generate_region_oriented_direction", "generate_region_random_direction",
    "PopulationRandomWalkSampler", "PopulationSliceSampler","PopulationSimpleSliceSampler"]
