import numpy as np

def test_reactive_run():
    from mininest import ReactiveNestedSampler

    def loglike(z):
        a = np.array([-0.5 * sum([((xi - 0.83456 + i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        b = np.array([-0.5 * sum([((xi - 0.43456 - i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        return np.logaddexp(a, b)

    def transform(x):
        return 10. * x - 5.
    
    paramnames = ['Hinz', 'Kunz']

    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, min_num_live_points=400)
    sampler.run(log_interval=50)
    sampler.plot()

def test_run():
    from mininest import NestedSampler

    def loglike(z):
        a = np.array([-0.5 * sum([((xi - 0.83456 + i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        b = np.array([-0.5 * sum([((xi - 0.43456 - i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        return np.logaddexp(a, b)

    def transform(x):
        return 10. * x - 5.
    
    paramnames = ['Hinz', 'Kunz']

    sampler = NestedSampler(paramnames, loglike, transform=transform, num_live_points=400)
    sampler.run(log_interval=50)
    sampler.plot()



if __name__ == '__main__':
    test_reactive_run()
    test_run()
