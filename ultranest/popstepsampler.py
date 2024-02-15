"""
Vectorized step samplers
------------------------

Likelihood based on GPUs (model emulators based on neural networks,
or JAX implementations) can evaluate hundreds of points as efficiently
as one point. The implementations in this module leverage this power,
by providing random walks of populations of walkers.
"""

import numpy as np
from ultranest.utils import submasks
from ultranest.stepfuncs import evolve, step_back
from ultranest.stepfuncs import generate_cube_oriented_direction, generate_cube_oriented_direction_scaled
from ultranest.stepfuncs import generate_random_direction, generate_region_oriented_direction, generate_region_random_direction
from ultranest.stepfuncs import generate_differential_direction, generate_mixture_random_direction
import scipy.stats


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
    """Compares random walk travel distance to MLFriends radius.

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
        return np.exp(np.nanmean(np.log([entry[-1] for entry in self.logstat])))

    @property
    def far_enough_fraction(self):
        """Fraction of jumps exceeding reference distance."""
        if len(self.logstat) == 0:
            return np.nan
        return np.nanmean([entry[-2] for entry in self.logstat])

    def get_info_dict(self):
        return dict(
            num_logs=len(self.logstat),
            rejection_rate=np.nanmean([entry[0] for entry in self.logstat]),
            mean_scale=np.nanmean([entry[1] for entry in self.logstat]),
            mean_nsteps=np.nanmean([entry[2] for entry in self.logstat]),
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

            for i in range(self.nsteps):
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
        self.generation = np.zeros(self.popsize, dtype=int) - 1
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
        region: MLFriends object
            Region
        mask_starting: np.array(nwalkers, dtype=bool)
            which walkers to set up.

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

        (
            (
                currentt, currentv,
                current_left, current_right, searching_left, searching_right
            ),
            (success, unew, pnew, Lnew),
            nc
        ) = evolve(transform, loglike, Lmin, *args)
        
        far_enough, (move_distance, reference_distance) = diagnose_move_distances(region, args[0][success,:], unew)
        self.logstat.append([
            success.mean(),
            self.scale,
            self.nsteps,
            np.mean(far_enough),
            np.exp(np.mean(np.log(move_distance / reference_distance + 1e-10)))
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


__all__ = [
    "generate_cube_oriented_direction", "generate_cube_oriented_direction_scaled",
    "generate_random_direction", "generate_region_oriented_direction", "generate_region_random_direction",
    "PopulationRandomWalkSampler", "PopulationSliceSampler"]
