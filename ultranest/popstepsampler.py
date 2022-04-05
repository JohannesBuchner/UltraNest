#!/usr/bin/env python
# coding: utf-8

import numpy as np
from ultranest.utils import submasks
from ultranest.stepfuncs import evolve, step_back
from ultranest.stepfuncs import generate_cube_oriented_direction, \
   generate_random_direction, generate_region_oriented_direction, generate_region_random_direction


class PopulationSliceSampler():
    def __init__(
        self, popsize, nsteps, generate_direction, scale=1.0, 
        scale_adapt_factor=0.9, log=False, logfile=None
    ):
        """
        Vectorized slice/HARM sampler.

        Revert until all previous steps have likelihoods allL above Lmin.
        Updates currentt, generation and allL, in-place.

        Parameters
        ----------
        popsize: int
            number of walkers to maintain
        nsteps: int
            number of steps to take until the found point is accepted as independent.
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

        self.popsize = popsize
        self.generate_direction = generate_direction

    def __str__(self):
        return 'PopulationSliceSampler(popsize=%d, nsteps=%d, generate_direction=%s, scale=%.g)' % (
                self.popsize, self.nsteps, self.generate_direction, self.scale)

    def region_changed(self, Ls, region):
        """notification that the region changed. Currently not used."""
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

    def step_back(self, Lmin):
        """see `:func:ultranest.stepfuncs.step_back` :func:ultranest.stepfuncs.step_back."""
        step_back(Lmin, self.allL, self.generation, self.currentt)

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
        if self.log: print("setting up:", starting)
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
        s1 = ('G:' + ''.join(['%d' % g if g >= 0 else '_' for g in self.generation]))
        s2 = ('S:' + ''.join(['S' if not np.isfinite(self.currentt[i]) else 'L' if self.searching_left[i] else 'R' if self.searching_right[i] else 'B'
            for i in range(self.popsize)]))
        return s1 + '  ' + s2

    def setup_brackets(self, mask_starting, region):
        """Pick starting direction and range for slice

        Parameters
        ----------
        region: MLFriends object
            Region
        mask_starting: np.array(nwalkers, dtype=bool)
            which walkers to set up.

        """
        if self.log: print("starting brackets:", mask_starting)
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
        if self.log: print("setting currentp")
        self.currentp = np.zeros((self.popsize, nparams)) + np.nan

    def advance(self, transform, loglike, Lmin):
        """Advance the walker population

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
        if self.log: print("evolve will advance:", movable)

        (
            (
            currentt, currentv,
            current_left, current_right, searching_left, searching_right),
            (success, unew, pnew, Lnew),
            nc
        ) = evolve(transform, loglike, Lmin, *args, log=self.log)

        if self.log: print("movable", movable.shape, movable.sum(), success.shape)
        moved = submasks(movable, success)
        if self.log: print("evolve moved:", moved)
        self.generation[moved] += 1
        if len(pnew) > 0:
            if len(self.currentp) == 0:
                self._setup_currentp(nparams=pnew.shape[1])

            if self.log: print("currentp", self.currentp[moved,:].shape, pnew.shape)
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

        #print(str(self), "(start)")
        self.step_back(Lmin)

        starting = self.generation < 0
        if starting.any():
            self.setup_start(us[Ls > Lmin], Ls[Ls > Lmin], starting)
        assert (self.generation >= 0).all(), self.generation

        #if self.log: print("generation:", self.generation)

        # find those where bracket is undefined:
        mask_starting = ~np.isfinite(self.currentt)
        if mask_starting.any():
            self.setup_brackets(mask_starting, region)

        if self.log: print(str(self), "(before)")
        nc = self.advance(transform, loglike, Lmin)
        if self.log: print(str(self), "(after)")

        # harvest top individual if possible
        if self.generation[self.ringindex] == self.nsteps:
            if self.log: print("have a candidate")
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
