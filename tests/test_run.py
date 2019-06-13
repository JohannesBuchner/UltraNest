import numpy as np
import shutil
import tempfile
import pytest

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
    r = sampler.run(log_interval=50)
    open('nestedsampling_results.txt', 'a').write("%.3f\n" % r['logz'])
    sampler.plot()

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
    r = sampler.run(log_interval=50)
    open('nestedsampling_reactive_results.txt', 'a').write("%.3f\n" % r['logz'])
    sampler.plot()

@pytest.mark.parametrize("dlogz", [0.5, 0.1, 0.01])
def test_run_resume(dlogz):
    from mininest import NestedSampler
    sigma = 0.01
    ndim = 1

    def loglike(theta):
        like = -0.5 * (((theta - 0.5)/sigma)**2).sum(axis=1) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
        return like

    def transform(x):
        return x
    
    paramnames = ['a']
    def myadd(row):
        assert False, (row, 'should not need to add more points in resume')

    last_results = None
    #for dlogz in 0.5, 0.1, 0.01:
    np.random.seed(int(dlogz*100))
    folder = tempfile.mkdtemp()
    try:
        for i in range(2):
            sampler = NestedSampler(paramnames, loglike, transform=transform, 
                num_live_points=400, log_dir=folder, 
                append_run_num=False)
            r = sampler.run(log_interval=50, dlogz=dlogz)
            sampler.print_results()
            sampler.pointstore.close()
            if i == 1:
                sampler.pointstore.add = myadd
            del r['weighted_samples']
            del r['samples']
            if last_results is not None:
                print("ran with dlogz:", dlogz)
                print("first run gave:", last_results)
                print("second run gave:", r)
                assert last_results['logzerr'] < 1.0
                assert r['logzerr'] < 1.0
                assert np.isclose(last_results['logz'], r['logz'], atol=0.5)
            last_results = r
    finally:
        shutil.rmtree(folder, ignore_errors=True)

"""
def test_reactive_run_resume_eggbox():
    from mininest import ReactiveNestedSampler
    ndim = 2

    def loglike(z):
        chi = (np.cos(z / 2.)).prod(axis=1)
        return (2. + chi)**5

    def transform(x):
        return x * 10 * np.pi

    paramnames = ['a', 'b']
    
    last_results = None
    folder = tempfile.mkdtemp()
    np.random.seed(1)
    try:
        for i in range(2):
            sampler = ReactiveNestedSampler(paramnames, loglike, 
                min_num_live_points=100, 
                log_dir=folder, 
                cluster_num_live_points=0,
                append_run_num=False, 
                )
            r = sampler.run(log_interval=1000, max_iters=5600)
            sampler.print_results()
            sampler.pointstore.close()
            last_results = r
    finally:
        shutil.rmtree(folder, ignore_errors=True)
"""

def test_run_compat():
    from mininest.solvecompat import pymultinest_solve_compat as solve
    
    ndim = 2
    sigma = 0.01
    centers = 0.5
    paramnames = ['a', 'b']

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2).sum() - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
        return like

    def transform(x):
        return 10 * x - 5.

    result = solve(LogLikelihood=loglike, Prior=transform, 
        n_dims=ndim, outputfiles_basename=None,
        verbose=True, resume=True, importance_nested_sampling=False)
    
    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(paramnames, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    

if __name__ == '__main__':
    test_run_compat()
    #test_run_resume(dlogz=0.5)
    #test_reactive_run_resume(dlogz=0.5, min_ess=1000)
    #test_reactive_run()
    #test_run()
