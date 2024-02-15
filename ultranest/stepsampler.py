"""
MCMC-like step sampling
-----------------------

The classes implemented here are generators that, in each iteration,
only make one likelihood call. This allows running in parallel a
population of samplers that have the same execution time per call,
even if they do not terminate at the same number of iterations.
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from .utils import listify as _listify


def generate_random_direction(ui, region, scale=1):
    """Sample uniform direction vector in unit cube space of length `scale`.

    Samples a direction from a unit multi-variate Gaussian.

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
    """Sample a unit direction vector in direction of a random unit cube axes.

    Chooses one parameter, randomly uniformly, upon which the slice will be defined.

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
    """Sample a direction vector on a randomly chose parameter based on two randomly selected live points.

    Chooses one parameter, randomly uniformly, upon which the slice will be defined.
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
    """Sample a vector using the difference between two randomly selected live points.

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
    """Sample a vector using the difference between two randomly selected live points.

    Only 10% of parameters are allowed to vary at a time.

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

        # choose which parameters to be off
        mask = np.random.uniform(size=ndim) > 0.1
        # at least one must be free to vary
        mask[np.random.randint(ndim)] = False
        v[mask] = 0
        if (v != 0).any():
            # repeat if live points are identical
            break
    # use doubling procedure to identify left and right maxima borders
    # v = np.zeros(ndim)
    # v[mask] = (region.u[i,mask] - region.u[i2,mask]) * scale
    return v


def generate_region_oriented_direction(ui, region, scale=1):
    """Sample a vector along one `region` principle axes, chosen at random.

    The region transformLayer axes are considered (:py:class:`AffineLayer` or :py:class:`ScalingLayer`).
    One axis is chosen at random.

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
    """Sample a direction vector based on the region covariance.

    The region transformLayer axes are considered (:py:class:`AffineLayer` or :py:class:`ScalingLayer`).
    With this covariance matrix, a random direction is generated.
    Generating proceeds by transforming a unit multi-variate Gaussian.

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
    """Sample randomly uniformly from two proposals.

    Randomly applies either :py:func:`generate_differential_direction`,
    which transports far, or :py:func:`generate_region_oriented_direction`,
    which is stiffer.

    Best method according to https://arxiv.org/abs/2211.09426

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


def generate_region_sample_direction(ui, region, scale=1):
    """Sample a point directly from the region, and return the difference vector to the current point.

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
    while True:
        upoints = region.sample(nsamples=200)
        if len(upoints) != 0:
            break
    # we only need the first one
    u = upoints[0,:]
    return (u - ui) * scale


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
    """Compares random walk travel distance to MLFriends radius.

    Compares in whitened space (t-space), the L2 norm between final
    point and starting point to the MLFriends bootstrapped radius.

    Parameters
    ----------
    region: MLFriends
        built region
    history: list
        list of tuples, containing visited point and likelihood.
    mean_pair_distance: float
        not used
    ndim: int
        dimensionality

    Returns
    -------
    far_enough: bool
        whether the distance is larger than the radius
    info: tuple
        distance and radius (both float)
    """
    # compute distance from start to end
    ustart, _ = history[0]
    ufinal, _ = history[-1]
    tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
    d2 = ((tstart - tfinal)**2).sum()
    far_enough = d2 > region.maxradiussq

    return far_enough, [d2**0.5, region.maxradiussq**0.5]


def adapt_proposal_move_distances_midway(region, history, mean_pair_distance, ndim):
    """Compares first half of the travel distance to MLFriends radius.

    Compares in whitened space (t-space), the L2 norm between the
    middle point of the walk and the starting point,
    to the MLFriends bootstrapped radius.

    Parameters
    ----------
    region: MLFriends
        built region
    history: list
        list of tuples, containing visited point and likelihood.
    mean_pair_distance: float
        not used
    ndim: int
        dimensionality

    Returns
    -------
    far_enough: bool
        whether the distance is larger than the radius
    info: tuple
        distance and radius (both float)
    """
    # compute distance from start to end
    ustart, _ = history[0]
    middle = max(1, len(history) // 2)
    ufinal, _ = history[middle]
    tstart, tfinal = region.transformLayer.transform(np.vstack((ustart, ufinal)))
    d2 = ((tstart - tfinal)**2).sum()
    far_enough = d2 > region.maxradiussq

    return far_enough, [d2**0.5, region.maxradiussq**0.5]


def select_random_livepoint(us, Ls, Lmin):
    """Select random live point as chain starting point.

    Parameters
    -----------
    us: array
        positions of live points
    Ls: array
        likelihood of live points
    Lmin: float
        current log-likelihood threshold

    Returns
    -------
    i: int
        index of live point selected
    """
    return np.random.randint(len(Ls))


class IslandPopulationRandomLivepointSelector(object):
    def __init__(self, island_size, exchange_probability=0):
        """Set up multiple isolated islands.

        To replace dead points, chains are only started from the same
        island as the dead point. Island refers to chunks of
        live point indices (0,1,2,3 as stored, not sorted).
        Each chunk has size ´island_size´.

        If ´island_size´ is large, for example, the total number of live points,
        then clumping can occur more easily. This is the observed behaviour
        that a limited random walk is run from one live point, giving
        two similar points, then the next dead point replacement is
        likely run again from these, giving more and more similar live points.
        This gives a run-away process leading to clumps of similar,
        highly correlated points.

        If ´island_size´ is small, for example, 1, then each dead point
        is replaced by a chain started from it. This is a problem because
        modes can never die out. Nested sampling can then not complete.

        In a multi-modal run, within a given number of live points,
        the number of live points per mode is proportional to the mode's
        prior volume, but can fluctuate. If the number of live points
        is small, a fluctuation can lead to mode die-out, which cannot
        be reversed. Therefore, the number of island members should be
        large enough to represent each mode.

        Parameters
        -----------
        island_size: int
            maximum number of members on each isolated live point
            population.

        exchange_probability: float
            Probability that a member from a random island will be picked.

        """
        assert island_size > 0
        self.island_size = island_size
        assert 0 <= exchange_probability <= 1
        self.exchange_probability = exchange_probability

    def __call__(self, us, Ls, Lmin):
        """Select live point as chain starting point.

        Parameters
        -----------
        us: array
            positions of live points
        Ls: array
            likelihood of live points
        Lmin: float
            current log-likelihood threshold

        Returns
        -------
        i: int
            index of live point selected
        """
        mask_deadpoints = Lmin == Ls
        if not mask_deadpoints.any() or (self.exchange_probability > 0 and np.random.uniform() < self.exchange_probability):
            return np.random.randint(len(Ls))

        # find the dead point we should replace
        j = np.where(mask_deadpoints)[0][0]
        # start in the same island
        island = j // self.island_size
        # pick a random member from the island
        return np.random.randint(
            island * self.island_size,
            min(len(Ls), (island + 1) * self.island_size))


class StepSampler(object):
    """Base class for a simple step sampler, staggering around.

    Scales proposal towards a 50% acceptance rate.
    """

    def __init__(
        self, nsteps, generate_direction,
        scale=1.0, check_nsteps='move-distance', adaptive_nsteps=False, max_nsteps=1000,
        region_filter=False, log=False,
        starting_point_selector=select_random_livepoint,
    ):
        """Initialise sampler.

        Parameters
        -----------
        scale: float
            initial proposal size

        nsteps: int
            number of accepted steps until the sample is considered independent.

            To find the right value, see :py:class:`ultranest.calibrator.ReactiveNestedCalibrator`

        generate_direction: function
            direction proposal function.

            Available are:

            * :py:func:`generate_cube_oriented_direction` (slice sampling, picking one random parameter to vary)
            * :py:func:`generate_random_direction` (hit-and-run sampling, picking a random direction varying all parameters)
            * :py:func:`generate_differential_direction` (differential evolution direction proposal)
            * :py:func:`generate_region_oriented_direction` (slice sampling, but in the whitened parameter space)
            * :py:func:`generate_region_random_direction` (hit-and-run sampling, but in the whitened parameter space)
            * :py:class:`SequentialDirectionGenerator` (sequential slice sampling, i.e., iterate deterministically through the parameters)
            * :py:class:`SequentialRegionDirectionGenerator` (sequential slice sampling in the whitened parameter space, i.e., iterate deterministically through the principle axes)
            * :py:func:`generate_cube_oriented_differential_direction` (like generate_differential_direction, but along only one randomly chosen parameter)
            * :py:func:`generate_partial_differential_direction` (differential evolution slice proposal on only 10% of the parameters)
            * :py:func:`generate_mixture_random_direction` (combined proposal)

            Additionally, :py:class:`OrthogonalDirectionGenerator`
            can be applied to any generate_direction function.

            When in doubt, try :py:func:`generate_mixture_random_direction`.
            It combines efficient moves along the live point distribution,
            with robustness against collapse to a subspace.
            :py:func:`generate_cube_oriented_direction` works well too.

        adaptive_nsteps: False or str
            Strategy to adapt the number of steps. 
            The possible values are the same as for `check_nsteps`.

            Adapting can give usable results. However, strictly speaking,
            detailed balance is not maintained, so the results can be biased.
            You can use the stepsampler.logstat property to find out the `nsteps` learned
            from one run (third column), and use the largest value for `nsteps`
            for a fresh run.
            The forth column is the jump distance, the fifth column is the reference distance.

        check_nsteps: False or str
            Method to diagnose the step sampler walks. The options are:

            * False: no checking
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

            Each step sampler walk adds one row to stepsampler.logstat.
            The jump distance (forth column) should be compared to 
            the reference distance (fifth column).

        max_nsteps: int
            Maximum number of steps the adaptive_nsteps can reach.

        region_filter: bool
            if True, use region to check if a proposed point can be inside
            before calling likelihood.

        log: file
            log file for sampler statistics, such as acceptance rate,
            proposal scale, number of steps, jump distance and distance
            between live points

        starting_point_selector: func
            function which given the live point positions us,
            their log-likelihoods Ls and the current log-likelihood
            threshold Lmin, returns the index i of the selected live
            point to start a new chain from.
            Examples: :py:func:`select_random_livepoint`, which has
            always been the default behaviour,
            or an instance of :py:class:`IslandPopulationRandomLivepointSelector`.

        """
        self.history = []
        self.nsteps = nsteps
        self.nrejects = 0
        self.scale = scale
        self.max_nsteps = max_nsteps
        self.next_scale = self.scale
        self.nudge = 1.1**(1. / self.nsteps)
        self.nsteps_nudge = 1.01
        self.generate_direction = generate_direction
        check_nsteps_options = {
            False: None,
            'move-distance': adapt_proposal_move_distances,
            'move-distance-midway': adapt_proposal_move_distances_midway,
            'proposal-total-distances': adapt_proposal_total_distances,
            'proposal-total-distances-NN': adapt_proposal_total_distances_NN,
            'proposal-summed-distances': adapt_proposal_summed_distances,
            'proposal-summed-distances-NN': adapt_proposal_summed_distances_NN,
        }
        adaptive_nsteps_options = dict(check_nsteps_options)

        if adaptive_nsteps not in adaptive_nsteps_options.keys():
            raise ValueError("adaptive_nsteps must be one of: %s, not '%s'" % (adaptive_nsteps_options, adaptive_nsteps))
        if check_nsteps not in check_nsteps_options.keys():
            raise ValueError("check_nsteps must be one of: %s, not '%s'" % (adaptive_nsteps_options, adaptive_nsteps))
        self.adaptive_nsteps = adaptive_nsteps
        if self.adaptive_nsteps:
            assert nsteps <= max_nsteps, 'Invalid adapting configuration: provided nsteps=%d exceeds provided max_nsteps=%d' % (nsteps, max_nsteps)
        self.adaptive_nsteps_function = adaptive_nsteps_options[adaptive_nsteps]
        self.check_nsteps = check_nsteps
        self.check_nsteps_function = check_nsteps_options[check_nsteps]
        self.adaptive_nsteps_needs_mean_pair_distance = self.adaptive_nsteps in (
            'proposal-total-distances', 'proposal-summed-distances',
        ) or self.check_nsteps in (
            'proposal-total-distances', 'proposal-summed-distances',
        )
        self.starting_point_selector = starting_point_selector
        self.mean_pair_distance = np.nan
        self.region_filter = region_filter
        if log:
            assert hasattr(log, 'write'), 'log argument should be a file, use log=open(filename, "w") or similar' 
        self.log = log

        self.logstat = []
        self.logstat_labels = ['rejection_rate', 'scale', 'steps']
        if adaptive_nsteps or check_nsteps:
            self.logstat_labels += ['jump-distance', 'reference-distance']

    def __str__(self):
        """Return string representation."""
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

    @property
    def mean_jump_distance(self):
        """Geometric mean jump distance."""
        if len(self.logstat) == 0:
            return np.nan
        if 'jump-distance' not in self.logstat_labels or 'reference-distance' not in self.logstat_labels:
            return np.nan
        i = self.logstat_labels.index('jump-distance')
        j = self.logstat_labels.index('reference-distance')
        jump_distances = np.array([entry[i] for entry in self.logstat])
        reference_distances = np.array([entry[j] for entry in self.logstat])
        return np.exp(np.nanmean(np.log(jump_distances / reference_distances + 1e-10)))

    @property
    def far_enough_fraction(self):
        """Fraction of jumps exceeding reference distance."""
        if len(self.logstat) == 0:
            return np.nan
        if 'jump-distance' not in self.logstat_labels or 'reference-distance' not in self.logstat_labels:
            return np.nan
        i = self.logstat_labels.index('jump-distance')
        j = self.logstat_labels.index('reference-distance')
        jump_distances = np.array([entry[i] for entry in self.logstat])
        reference_distances = np.array([entry[j] for entry in self.logstat])
        return np.nanmean(jump_distances > reference_distances)

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
        if 'jump-distance' not in self.logstat_labels or 'reference-distance' not in self.logstat_labels:
            print("turn on check_nsteps in the step sampler for diagnostics")
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
        if 'jump-distance' not in self.logstat_labels:
            return 
        if 'reference-distance' not in self.logstat_labels:
            return 
        i = self.logstat_labels.index('jump-distance')
        j = self.logstat_labels.index('reference-distance')
        jump_distances = np.array([entry[i] for entry in self.logstat])
        reference_distances = np.array([entry[j] for entry in self.logstat])
        plt.hist(np.log10(jump_distances / reference_distances + 1e-10), **kwargs)
        ylo, yhi = plt.ylim()
        plt.vlines(np.log10(self.mean_jump_distance), ylo, yhi)
        plt.ylim(ylo, yhi)
        plt.title(self.check_nsteps or self.adaptive_nsteps)
        plt.xlabel('log(relative step distance)')
        plt.ylabel('Frequency')
        plt.savefig(filename, bbox_inches='tight')
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
        if self.adaptive_nsteps or self.check_nsteps:
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
        if not (self.adaptive_nsteps or self.check_nsteps):
            return
        if len(self.history) < self.nsteps:
            # incomplete or aborted for some reason
            print("not adapting/checking nsteps, incomplete history", len(self.history), self.nsteps)
            return

        if self.adaptive_nsteps_needs_mean_pair_distance:
            assert np.isfinite(self.mean_pair_distance)
        ndim = region.u.shape[1]
        if self.check_nsteps:
            far_enough, extra_info = self.check_nsteps_function(region, self.history, self.mean_pair_distance, ndim)
            self.logstat[-1] += extra_info
        if not self.adaptive_nsteps:
            return

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
            self.log.flush()

        if self.adaptive_nsteps or self.check_nsteps:
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
        tregion: :py:class:`WrappingEllipsoid`
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
            i = self.starting_point_selector(us, Ls, Lmin)
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
        """See :py:meth:`StepSampler.adjust_accept`."""
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
        """Advance the slice sampling move. see :py:meth:`StepSampler.move`."""
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

                # adjust scale to final slice length
                if -left > self.next_scale or right > self.next_scale:
                #if right - left > self.next_scale:
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
    return SliceSampler(*args, **kwargs, generate_direction=SequentialDirectionGenerator())


def RegionSliceSampler(*args, **kwargs):
    """Slice sampler, randomly picking region axes."""
    return SliceSampler(*args, **kwargs, generate_direction=generate_region_oriented_direction)


def BallSliceSampler(*args, **kwargs):
    """Hit & run sampler. Choose random directions in space."""
    return SliceSampler(*args, **kwargs, generate_direction=generate_random_direction)


def RegionBallSliceSampler(*args, **kwargs):
    """Hit & run sampler. Choose random directions according to region."""
    return SliceSampler(*args, **kwargs, generate_direction=generate_region_random_direction)


class SequentialDirectionGenerator(object):
    """Sequentially proposes one parameter after the next."""
    def __init__(self):
        """Initialise."""
        self.axis_index = 0

    def __call__(self, ui, region, scale=1):
        """Choose the next axis in u-space.

        Parameters
        -----------
        ui: array
            current point (in u-space)
        region: MLFriends object
            pick random two live points for length along axis
        scale: float
            length of direction vector

        Returns
        --------
        v: array
            new direction vector (in u-space)
        """
        nlive, ndim = region.u.shape
        j = self.axis_index % ndim
        self.axis_index = j + 1

        v = np.zeros(ndim)
        # choose pair of live points
        while v[j] == 0:
            i = np.random.randint(nlive)
            i2 = np.random.randint(nlive - 1)
            if i2 >= i:
                i2 += 1

            v[j] = (region.u[i,j] - region.u[i2,j]) * scale

        return v

    def __str__(self):
        return type(self).__name__ + '()'


class SequentialRegionDirectionGenerator(object):
    """Sequentially proposes one region axes after the next."""
    def __init__(self):
        """Initialise."""
        self.axis_index = 0

    def __call__(self, ui, region, scale=1):
        """Choose the next axis in t-space.

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
    """Orthogonalizes proposal vectors.

    Samples N proposed vectors by a provided method, then orthogonalizes
    them with Gram-Schmidt (QR decomposition).
    """

    def __init__(self, generate_direction):
        """Initialise.

        Parameters
        -----------
        generate_direction: function
            direction proposal to orthogonalize
        """
        self.axis_index = 0
        self.generate_direction = generate_direction
        self.directions = None

    def __str__(self):
        """Return string representation."""
        return type(self).__name__ + '(generate_direction=%s)' % self.generate_direction

    def __call__(self, ui, region, scale=1):
        """Return next orthogonalized vector.

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
    """Propose directions with only some parameters variable.

    Propose in region direction, but only include some dimensions at a time.
    Completely configurable.
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
        ellipsoid axes matrix, as computed by :py:class:`WrappingEllipsoid`
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
