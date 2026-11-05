"""Integration tests for the SnowballingNestedSampler.

The three runs vary the step sampler, region class and termination
budgets to point out crashes. The gauss test compares against
ReactiveNestedSampler; the rosenbrock runs do not assert evidence
correctness -- the ncall- and time-limited results depend on the load
of the machine.
"""
import numpy as np

from ultranest import ReactiveNestedSampler
from ultranest.mlfriends import RobustEllipsoidRegion
from ultranest.perfectns import NonIsotropicGaussianLRPS
from ultranest.snowballingns import SnowballingNestedSampler
from ultranest.snowballingns.utils import CubeRegion
from ultranest.stepfuncs import generate_mixture_random_direction


def make_rosenbrock2d():
    """2d Rosenbrock function on [-10, 10]^2 with a flat prior."""

    def loglike(theta):
        a = theta[:, :-1]
        b = theta[:, 1:]
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum(axis=1)

    def transform(u):
        return u * 20 - 10

    paramnames = ['param%d' % (i + 1) for i in range(2)]
    return paramnames, transform, loglike


def test_gauss_2d_one_rotation_matches_reactive(tmp_path):
    """One snowball rotation with K=400 agrees with ReactiveNestedSampler.

    Uses the default arguments: the quadratic slice direction on the
    CovarianceRegion.
    """
    problem = NonIsotropicGaussianLRPS(2)

    sampler = SnowballingNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True, num_live_points_base=400, num_steps=1,
        integrator='custom', log_dir=str(tmp_path / 'snowball'))
    result = sampler.run(frac_remain=0.5, max_num_live_points=400,
                         resume=False)
    assert np.isfinite(result['logz'])
    # the live-points limit stops the run after one rotation
    assert sampler.nrotations == 1
    # a single rotation cannot fulfil the convergence criterion
    assert result['converged'] is False
    sstate = sampler.state
    sstate.close()

    reactive = ReactiveNestedSampler(
        problem.paramnames, problem.loglike, transform=problem.transform,
        vectorized=True)
    rresult = reactive.run(min_num_live_points=400)

    # both estimate the same evidence; one rotation is not converged, so
    # the tolerance is loose
    sigma = max(result['logzerr'], rresult['logzerr'], 0.05)
    assert abs(result['logz'] - rresult['logz']) < 6 * sigma


def test_rosenbrock_2d_ncall_budget(tmp_path):
    """A full rosenbrock run terminated by the model-evaluation budget.

    Uses the mixture direction proposal on the MLFriends region
    (RobustEllipsoidRegion).
    """
    paramnames, transform, loglike = make_rosenbrock2d()
    sampler = SnowballingNestedSampler(
        paramnames, loglike, transform=transform, vectorized=True,
        num_live_points_base=50, num_steps=1,
        stepsampler=generate_mixture_random_direction,
        region_class=RobustEllipsoidRegion, integrator='custom',
        log_dir=str(tmp_path / 'snowball'))
    result = sampler.run(frac_remain=0.5, max_ncalls=10000, resume=False)
    # the budget check fires at the rotation boundary; the overshoot is
    # bounded by the rotation that crossed it
    assert sampler.ncall < 100000
    assert np.isfinite(result['logz'])
    assert isinstance(result['converged'], bool)
    sampler.state.close()


def test_rosenbrock_2d_time_budget(tmp_path):
    """A full rosenbrock run terminated by the wall-time budget.

    Uses the quadratic slice direction on a CubeRegion with two accepted
    steps per walk. No evidence check: the result depends on the load of
    the machine.
    """
    paramnames, transform, loglike = make_rosenbrock2d()
    sampler = SnowballingNestedSampler(
        paramnames, loglike, transform=transform, vectorized=True,
        num_live_points_base=50, num_steps=2, region_class=CubeRegion,
        integrator='custom', log_dir=str(tmp_path / 'snowball'))
    result = sampler.run(frac_remain=0.5, max_execution_time_in_s=10.0,
                         resume=False)
    # the budget fires at the rotation boundary; the overshoot is
    # bounded by one rotation
    assert result['wall_time'] < 120
    assert np.isfinite(result['logz'])
    sampler.state.close()
