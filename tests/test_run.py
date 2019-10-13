import numpy as np
import shutil
import tempfile
import pytest

def test_run():
    from ultranest import NestedSampler

    def loglike(z):
        a = np.array([-0.5 * sum([((xi - 0.83456 + i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        b = np.array([-0.5 * sum([((xi - 0.43456 - i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        loglike.ncalls += len(a)
        return np.logaddexp(a, b)
    loglike.ncalls = 0

    def transform(x):
        return 10. * x - 5.
    
    paramnames = ['Hinz', 'Kunz']

    sampler = NestedSampler(paramnames, loglike, transform=transform, num_live_points=400, vectorized=True)
    r = sampler.run(log_interval=50)
    
    ncalls = loglike.ncalls
    if sampler.mpi_size > 1:
        ncalls = sampler.comm.gather(ncalls, root=0)
        if sampler.mpi_rank == 0:
            print("ncalls on the different MPI ranks:", ncalls)
        ncalls = sum(sampler.comm.bcast(ncalls, root=0))
    assert abs(r['ncall'] - ncalls) <= 2 * sampler.mpi_size, (r['ncall'], ncalls)
    open('nestedsampling_results.txt', 'a').write("%.3f\n" % r['logz'])
    sampler.plot()


def test_reactive_run():
    from ultranest import ReactiveNestedSampler
    np.random.seed(1)
    evals = set()

    def loglike(z):
        #print(loglike.ncalls, z[0,0])
        [evals.add(str(x[0])) for x in z]
        a = np.array([-0.5 * sum([((xi - 0.83456 + i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        b = np.array([-0.5 * sum([((xi - 0.43456 - i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        loglike.ncalls += len(a)
        return np.logaddexp(a, b)
    loglike.ncalls = 0

    def transform(x):
        return 10. * x - 5.
    
    paramnames = ['Hinz', 'Kunz']

    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
        draw_multiple=False, vectorized=True)
    r = sampler.run(log_interval=50, min_num_live_points=400)
    ncalls = loglike.ncalls
    nunique = len(evals)
    if sampler.mpi_size > 1:
        ncalls = sampler.comm.gather(ncalls, root=0)
        if sampler.mpi_rank == 0:
            print("ncalls on the different MPI ranks:", ncalls)
        ncalls = sum(sampler.comm.bcast(ncalls, root=0))

        allevals = sampler.comm.gather(evals, root=0)
        if sampler.mpi_rank == 0:
            print("evals on the different MPI ranks:", [len(e) for e in allevals])
            allevals = len(set.union(*allevals))
        else:
            allevals = None
        nunique = sampler.comm.bcast(allevals, root=0)
    
    if sampler.mpi_rank == 0:
        print('ncalls:', ncalls, 'nunique:', nunique)
    
    assert abs(r['ncall'] - ncalls) <= 2 * sampler.mpi_size, (r['ncall'], ncalls)
    assert ncalls == nunique, (ncalls, nunique)
    if sampler.mpi_rank == 0:
        open('nestedsampling_reactive_results.txt', 'a').write("%.3f\n" % r['logz'])
    sampler.plot()

@pytest.mark.parametrize("dlogz", [2.0, 0.5, 0.1])
def test_run_resume(dlogz):
    from ultranest import ReactiveNestedSampler
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
            sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
                log_dir=folder, resume=True, vectorized=True)
            r = sampler.run(log_interval=50, dlogz=dlogz, min_num_live_points=400)
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

def test_reactive_run_resume_eggbox():
    from ultranest import ReactiveNestedSampler

    def loglike(z):
        chi = (np.cos(z / 2.)).prod(axis=1)
        loglike.ncalls += len(z)
        return (2. + chi)**5
    loglike.ncalls = 0
    
    def transform(x):
        return x * 10 * np.pi

    paramnames = ['a', 'b']
    
    #last_results = None
    folder = tempfile.mkdtemp()
    np.random.seed(1)
    try:
        for i in range(2):
            print()
            print("====== Running Eggbox problem [%d] =====" % (i+1))
            print()
            sampler = ReactiveNestedSampler(paramnames, 
                loglike, transform=transform,
                log_dir=folder, resume=True, vectorized=True)
            initial_ncalls = int(sampler.ncall)
            loglike.ncalls = 0
            r = sampler.run(max_iters=200 + i*200, 
                max_num_improvement_loops=1, 
                min_num_live_points=100, 
                cluster_num_live_points=0)
            sampler.print_results()
            sampler.pointstore.close()
            print(loglike.ncalls, r['ncall'], initial_ncalls)

            ncalls = loglike.ncalls
            if sampler.mpi_size > 1:
                ncalls = sampler.comm.gather(ncalls, root=0)
                if sampler.mpi_rank == 0:
                    print("ncalls on the different MPI ranks:", ncalls)
                ncalls = sum(sampler.comm.bcast(ncalls, root=0))
            ncalls = ncalls + initial_ncalls
            assert abs(r['ncall'] - ncalls) <= 2 * sampler.mpi_size, (i, r['ncall'], ncalls, r['ncall'] - ncalls)
            #last_results = r
    finally:
        shutil.rmtree(folder, ignore_errors=True)

def test_run_compat():
    from ultranest.solvecompat import pymultinest_solve_compat as solve
    
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
        verbose=True, resume='resume', importance_nested_sampling=False)
    
    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(paramnames, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    
if __name__ == '__main__':
    #test_run_compat()
    #test_run_resume(dlogz=0.5)
    #test_reactive_run_resume(dlogz=0.5, min_ess=1000)
    test_reactive_run()
    test_run()
