import numpy as np
from ultranest import ReactiveNestedSampler
from ultranest.popstepsampler import PopulationSliceSampler, generate_unit_directions

def loglike_vectorized(z):
    a = np.array([-0.5 * sum([((xi - 0.7 + i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    b = np.array([-0.5 * sum([((xi - 0.3 - i*0.001)/0.1)**2 for i, xi in enumerate(x)]) for x in z])
    return np.logaddexp(a, b)

def loglike(x):
    a = -0.5 * sum([((xi - 0.7 + i*0.001)/0.1)**2 for i, xi in enumerate(x)])
    b = -0.5 * sum([((xi - 0.3 - i*0.001)/0.1)**2 for i, xi in enumerate(x)])
    return np.logaddexp(a, b)

def transform(x):
    return x # * 10. - 5.

paramnames = ['param%d' % i for i in range(3)]

def test_stepsampler_cubeslice(plot=False):
    np.random.seed(3)
    nsteps = np.random.randint(10, 50)
    popsize = np.random.randint(1, 20)
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)

    sampler.stepsampler = PopulationSliceSampler(
        popsize=popsize, nsteps=nsteps, 
        generate_direction=generate_unit_directions,
    )
    r = sampler.run(viz_callback=None, log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

if __name__ == '__main__':
    test_stepsampler_cubeslice()
