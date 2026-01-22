import numpy as np
import ultranest
import ultranest.stepsampler
import ultranest.calibrator
import scipy.stats


def test_calibrator(plot=False):
    # velocity dispersions of dwarf galaxies by van Dokkum et al., Nature, 555, 629 https://arxiv.org/abs/1803.10237v1

    values = np.array([15, 4, 2, 11, 1, -2, -1, -14, -39, -3])
    values_lo = np.array([7, 16, 6, 3, 6, 5, 10, 6, 11, 13])
    values_hi = np.array([7, 15, 8, 3, 6, 6, 10, 7, 14, 14])

    n_data = len(values)

    np.random.seed(42)

    samples = []

    for i in range(n_data):
        # draw normal random points
        u = np.random.normal(size=400)
        v = values[i] + np.where(u < 0, u * values_lo[i], u * values_hi[i])

        samples.append(v)

    samples = np.array(samples)

    # Define functions inside test_calibrator to access samples and n_data
    def prior_transform(cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        # let slope go from -3 to +3
        lo = -100
        hi = +100
        params[0] = cube[0] * (hi - lo) + lo
        # let scatter go from 1 to 1000
        lo = np.log10(1)
        hi = np.log10(1000)
        params[1] = 10**(cube[1] * (hi - lo) + lo)
        return params

    def log_likelihood(params):
        # unpack the current parameters:
        mean, scatter = params

        # compute the probability of each sample
        probs_samples = scipy.stats.norm(mean, scatter).pdf(samples)
        # average over each galaxy, because we assume one of the points is the correct one (logical OR)
        probs_objects = probs_samples.mean(axis=1)
        assert len(probs_objects) == n_data
        # multiply over the galaxies, because we assume our model holds true for all objects (logical AND)
        # for numerical stability, we work in log and avoid zeros
        loglike = np.log(probs_objects + 1e-100).sum()
        return loglike

    parameters = ['mean', 'scatter']

    sampler = ultranest.calibrator.ReactiveNestedCalibrator(
        parameters, log_likelihood, prior_transform, log_dir="logs"
    )

    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=len(parameters),  
        generate_direction=ultranest.stepsampler.generate_region_oriented_direction
    )

    sampler.run(min_num_live_points=400)

    if plot:
        sampler.plot()

if __name__ == '__main__':
    test_calibrator(plot=True)