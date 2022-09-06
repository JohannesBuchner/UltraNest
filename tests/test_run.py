import os
import numpy as np
import shutil
import tempfile
import pytest
import json
import pandas
from ultranest import NestedSampler, ReactiveNestedSampler, read_file
from ultranest.integrator import warmstart_from_similar_file
import ultranest.mlfriends
from numpy.testing import assert_allclose

def test_run():
    def loglike(y):
        z = np.log10(y)
        a = np.array([-0.5 * sum([((xi - 0.83456 + i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        b = np.array([-0.5 * sum([((xi - 0.43456 - i*0.1)/0.5)**2 for i, xi in enumerate(x)]) for x in z])
        loglike.ncalls += len(a)
        return np.logaddexp(a, b)
    loglike.ncalls = 0

    def transform(x):
        return 10**(10. * x - 5.)

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


def test_dlogz_reactive_run_SLOW():
    def loglike(y):
        return -0.5 * np.sum(((y - 0.5)/0.001)**2, axis=1)

    paramnames = ['Hinz', 'Kunz']

    sampler = ReactiveNestedSampler(paramnames, loglike, vectorized=True)
    print("running for ess")
    firstresults = sampler.run(min_num_live_points=50, cluster_num_live_points=0, max_num_improvement_loops=3, min_ess=10000, viz_callback=None)
    print()
    print({k:v for k, v in firstresults.items() if 'logzerr' in k})
    print()
    assert firstresults['logzerr'] > 0.1 * 2
    print("running again for logz")
    for niter, results in enumerate(sampler.run_iter(min_num_live_points=1, cluster_num_live_points=0, max_num_improvement_loops=10, dlogz=0.1, viz_callback=None, region_class=ultranest.mlfriends.RobustEllipsoidRegion)):
        print("logzerr in iteration %d" % niter, results['logzerr'])
    print()
    print({k:v for k, v in results.items() if 'logzerr' in k})
    assert results['logzerr'] < 0.1 * 2

def test_reactive_run():
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

    # test that the number of likelihood calls is correct

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

    print(r)
    assert r['niter'] > 100
    assert -10 < r['logz'] < 10
    assert 0.01 < r['logzerr'] < 0.5
    assert 1 < r['ess'] < 10000


    sampler.plot()


def test_reactive_run_extraparams():
    np.random.seed(1)

    def loglike(z):
        return -0.5 * z[-1].sum()
    loglike.ncalls = 0

    def transform(x):
        z = 10. * x - 5.
        return np.append(z, np.abs(z).sum())

    paramnames = ['Hinz', 'Kunz']

    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform,
        derived_param_names=['ctr_distance'])
    r = sampler.run()
    assert r['samples'].shape[1] == 3
    sampler.plot()

def test_return_summary():
    sigma = np.array([0.1, 0.01])
    centers = np.array([0.5, 0.75])
    paramnames = ['a', 'b']
    ndim = len(paramnames)

    def loglike(theta):
        like = -0.5 * (((theta - centers)/sigma)**2) - 0.5 * np.log(2 * np.pi * sigma**2) * ndim
        return like.sum()

    def transform(x):
        return x

    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform)
    r = sampler.run()

    print(r)
    assert r['paramnames'] == paramnames
    assert r['niter'] > 100
    assert -10 < r['logz'] < 10
    assert 0.01 < r['logzerr'] < 0.5
    assert 1 < r['ess'] < 10000
    assert 0.4 < r['posterior']['mean'][0] < 0.6
    assert 0.74 < r['posterior']['mean'][1] < 0.76
    assert 0.4 < r['posterior']['median'][0] < 0.6
    assert 0.74 < r['posterior']['median'][1] < 0.76
    assert 0.05 < r['posterior']['stdev'][0] < 0.2
    assert 0.005 < r['posterior']['stdev'][1] < 0.02

    assert 0.35 < r['posterior']['errlo'][0] < 0.45
    assert 0.72 < r['posterior']['errlo'][1] < 0.75
    assert 0.55 < r['posterior']['errup'][0] < 0.65
    assert 0.75 < r['posterior']['errup'][1] < 0.78

    N, ndim2 = r['samples'].shape
    assert ndim2 == ndim
    assert N > 10
    N, ndim2 = r['weighted_samples']['points'].shape
    assert ndim2 == ndim
    assert N > 10

    assert r['weighted_samples']['logw'].shape == (N,)
    assert r['weighted_samples']['weights'].shape == (N,)
    assert r['weighted_samples']['bootstrapped_weights'].shape[0] == N
    assert r['weighted_samples']['logl'].shape == (N,)

@pytest.mark.parametrize("dlogz", [2.0, 0.5, 0.1])
def test_run_resume(dlogz):
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

@pytest.mark.parametrize("storage_backend", ['hdf5', 'tsv', 'csv'])
def test_reactive_run_resume_eggbox(storage_backend):
    def loglike(z):
        chi = (np.cos(z / 2.)).prod(axis=1)
        loglike.ncalls += len(z)
        return (2. + chi)**5
    loglike.ncalls = 0

    def transform(x):
        return x * 10 * np.pi

    paramnames = ['a', 'b']
    ndim = len(paramnames)

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
                log_dir=folder, resume=True, vectorized=True, draw_multiple=False,
                storage_backend=storage_backend)
            initial_ncalls = int(sampler.ncall)
            num_live_points = 100
            loglike.ncalls = 0
            r = sampler.run(max_iters=200 + i*200,
                max_num_improvement_loops=0,
                min_num_live_points=num_live_points,
                cluster_num_live_points=0)
            sampler.print_results()
            if storage_backend == 'hdf5':
                print("pointstore:", sampler.pointstore.fileobj['points'].shape)
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
            assert paramnames == r['paramnames'], 'paramnames should be in results'

            results2 = json.load(open(folder + '/info/results.json'))
            print('CSV content:')
            print(open(folder + '/info/post_summary.csv').read())
            post_summary = pandas.read_csv(folder + '/info/post_summary.csv')
            print(post_summary, post_summary.columns)
            for k, v in r.items():
                if k in results2:
                    print("checking results[%s] ..." % k)
                    assert results2[k] == r[k], (k, results2[k], r[k])
            
            assert r['paramnames'] == paramnames
            samples = np.loadtxt(folder + '/chains/equal_weighted_post.txt', skiprows=1)
            data = np.loadtxt(folder + '/chains/weighted_post.txt', skiprows=1)
            data_u = np.loadtxt(folder + '/chains/weighted_post_untransformed.txt', skiprows=1)
            assert (data[:,:2] == data_u[:,:2]).all()
            
            assert_allclose(samples.mean(axis=0), r['posterior']['mean'])
            assert_allclose(np.median(samples, axis=0), r['posterior']['median'])
            assert_allclose(np.std(samples, axis=0), r['posterior']['stdev'])
            for k, v in r.items():
                if k == 'posterior':
                    for k1, v1 in v.items():
                        if k1 == 'information_gain_bits':
                            continue
                        for param, value in zip(paramnames, v[k1]):
                            print("checking %s of parameter '%s':" % (k1, param), value)
                            assert np.isclose(post_summary[param + '_' + k1].values, value), (param, k1, post_summary[param + '_' + k1].values, value)
                elif k == 'samples':
                    assert_allclose(samples, r['samples'])
                elif k == 'paramnames':
                    assert v == paramnames
                elif k == 'weighted_samples':
                    print(k, v.keys())
                    assert_allclose(data[:,0], v['weights'])
                    assert_allclose(data[:,1], v['logl'])
                    assert_allclose(data[:,2:], v['points'])
                    assert_allclose(data_u[:,2:], v['upoints'])
                elif k == 'maximum_likelihood':
                    print(k, v.keys())
                    assert_allclose(data[-1,1], v['logl'])
                    assert_allclose(data[-1,2:], v['point'])
                    assert_allclose(data_u[-1,2:], v['point_untransformed'])
                    
                elif k.startswith('logzerr') or '_bs' in k or 'Herr' in k:
                    print("   skipping", k, np.shape(v))
                    #assert_allclose(r[k], v, atol=0.5)
                elif k == 'insertion_order_MWW_test':
                    print('insertion_order_MWW_test:', r[k], v)
                    assert r[k] == v, (r[k], v)
                else:
                    print("  ", k, np.shape(v))
                    assert_allclose(r[k], v)

            logw = r['weighted_samples']['logw']
            v = r['weighted_samples']['points']
            L = r['weighted_samples']['logl']

            assert results2['niter'] == len(r['samples'])

        # the results are not exactly the same, because the sampling adds
        #ncalls = loglike.ncalls
        #sampler = ReactiveNestedSampler(paramnames,
        #    loglike, transform=transform,
        #    log_dir=folder, resume=True, vectorized=True, num_test_samples=0)
        #print("pointstore:", sampler.pointstore.fileobj['points'].shape)
        #assert ncalls == loglike.ncalls, (ncalls, loglike.ncalls)
        if storage_backend == 'hdf5':
            sequence, results = read_file(folder, ndim, random=False, num_bootstraps=0)

            print("sampler results: ********************")
            print({k:v for k, v in r.items() if np.asarray(v).size < 20 and k != 'weighted_samples'})
            print("reader results: ********************")
            print({k:v for k, v in results.items() if np.asarray(v).size < 20 and k != 'weighted_samples'})
            for k, v in results.items():
                if k == 'posterior' or k == 'samples':
                    pass
                elif k == 'weighted_samples' or k == 'maximum_likelihood':
                    for k2, v2 in results[k].items():
                        if k2 == 'bootstrapped_weights': continue
                        print("  ", k, "::", k2, np.shape(v2))
                        assert_allclose(r[k][k2], v2)
                elif k.startswith('logzerr') or '_bs' in k or 'Herr' in k:
                    print("   skipping", k, np.shape(v))
                    #assert_allclose(r[k], v, atol=0.5)
                elif k == 'insertion_order_MWW_test':
                    print('insertion_order_MWW_test:', r[k], v)
                    assert r[k] == v, (r[k], v)
                else:
                    print("  ", k, np.shape(v))
                    assert_allclose(r[k], v)

            logw = r['weighted_samples']['logw']
            v = r['weighted_samples']['points']
            L = r['weighted_samples']['logl']

            assert sequence['logz'][-1] - r['logz'] < 0.5, (results['logz'][-1], r['logz'])
            assert sequence['logzerr'][-1] <= r['logzerr_single'], (results['logzerr'][-1], r['logzerr'])
            #assert_allclose(sequence['logz_final'], r['logz_single'], atol=0.3)
            #assert_allclose(sequence['logzerr_final'], r['logzerr_single'], atol=0.1)
            assert r['niter'] <= sequence['niter'] <= r['niter'], (sequence['niter'], r['niter'])
            assert results['niter'] == len(sequence['logz']) == len(sequence['logzerr']) == len(sequence['logvol']) == len(sequence['logwt'])
            assert results['niter'] == len(results['samples'])
            data = np.loadtxt(folder + '/chains/weighted_post.txt', skiprows=1)
            assert_allclose(data[:,0], results['weighted_samples']['weights'])
            assert_allclose(data[:,1], results['weighted_samples']['logl'])
            assert_allclose(v, results['weighted_samples']['points'])
            assert_allclose(logw, results['weighted_samples']['logw'])
            assert_allclose(L, results['weighted_samples']['logl'])

            assert_allclose(L, sequence['logl'])
            #assert_allclose(logw + L, sequence['logwt'])
            assert sequence['logvol'].shape == logw.shape == (len(L),), (sequence['logvol'].shape, logw.shape)
            assert sequence['logwt'].shape == logw.shape == (len(L),), (sequence['logwt'].shape, logw.shape)
            #assert_allclose(logw, sequence['logvols'])
            #assert results['samples_untransformed'].shape == v.shape == (len(L), ndim), (results['samples_untransformed'].shape, v.shape)

    finally:
        shutil.rmtree(folder, ignore_errors=True)

def test_reactive_run_warmstart_gauss():
    center = 0

    def loglike(z):
        chi2 = (((z - center)/0.001)**2).sum(axis=1)
        loglike.ncalls += len(z)
        return -0.5 * chi2
    loglike.ncalls = 0

    def transform(x):
        return x * 20000 - 10000

    paramnames = ['a']

    folder = tempfile.mkdtemp()
    np.random.seed(1)
    first_ncalls = None
    resume_ncalls = None
    try:
        for i, resume in enumerate(['overwrite', 'resume', 'resume-similar']):
            print()
            print("====== Running Gauss problem [%d] =====" % (i+1))
            print()
            center = (i+1) * 1e-4
            try:
                sampler = ReactiveNestedSampler(paramnames,
                    loglike, transform=transform,
                    log_dir=folder, resume=resume, vectorized=True, draw_multiple=False,
                    warmstart_max_tau=0.5)
            except Exception as e:
                # we expect an error for resuming with a changed likelihood
                if resume != 'resume':
                    raise e
                else:
                    assert 'loglikelihood function changed' in str(e), e
                    print("Exception as expected:", e)
                    continue
            initial_ncalls = int(sampler.ncall)
            if i == 0:
                assert initial_ncalls == 0
            num_live_points = 100
            loglike.ncalls = 0
            r = sampler.run(
                max_num_improvement_loops=0,
                min_num_live_points=num_live_points,
                cluster_num_live_points=0, viz_callback=None, frac_remain=
                0.5)
            sampler.print_results()
            print("pointstore:", sampler.pointstore.fileobj['points'].shape)
            sampler.pointstore.close()
            print(loglike.ncalls, r['ncall'], initial_ncalls)

            ncalls = loglike.ncalls
            if sampler.mpi_size > 1:
                ncalls = sampler.comm.gather(ncalls, root=0)
                if sampler.mpi_rank == 0:
                    print("ncalls on the different MPI ranks:", ncalls)
                ncalls = sum(sampler.comm.bcast(ncalls, root=0))
            ncalls = ncalls + initial_ncalls
            if i == 0:
                first_ncalls = ncalls
            if i == 2:
                resume_ncalls = loglike.ncalls
            assert abs(r['ncall'] - ncalls) <= 2 * sampler.mpi_size, (i, r['ncall'], ncalls, r['ncall'] - ncalls)
            assert paramnames == r['paramnames'], 'paramnames should be in results'

    finally:
        shutil.rmtree(folder, ignore_errors=True)
    
    # make sure warm start is much faster
    assert resume_ncalls < first_ncalls - 800, (resume_ncalls, first_ncalls)

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
        params = x.copy()
        params[0] = 10 * x[0] - 5.
        params[1] = 10**(x[1] - 1)
        return params

    result = solve(LogLikelihood=loglike, Prior=transform,
        n_dims=ndim, outputfiles_basename=None,
        verbose=True, resume='resume', importance_nested_sampling=False)

    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(paramnames, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))


def test_run_warmstart_gauss_SLOW():
    center = None
    stdev = 0.001

    def loglike(z):
        chi2 = (((z - center) / stdev)**2).sum(axis=1)
        loglike.ncalls += len(z)
        return -0.5 * chi2
    loglike.ncalls = 0

    def transform(x):
        return x * 20000 - 10000

    paramnames = ['a']

    folder = tempfile.mkdtemp()
    np.random.seed(1)
    ncalls = []
    try:
        for i, resume in enumerate(['overwrite', 'resume-hot', 'resume-hot', 'resume-hot']):
            print()
            print("====== Running Gauss problem [%d] =====" % (i+1))
            print()
            center = [0, 0, stdev, 1][i]
            print("center:", center, "folder:", folder)
            if i == 0:
                sampler = ReactiveNestedSampler(paramnames,
                    loglike, transform=transform,
                    log_dir=folder, resume=resume, vectorized=True)
            else:
                aux_param_names, aux_loglike, aux_transform, vectorized = warmstart_from_similar_file(
                    os.path.join(folder, 'chains', 'weighted_post_untransformed.txt'),
                    paramnames, loglike=loglike, transform=transform, vectorized=True,
                )
                sampler = ReactiveNestedSampler(aux_param_names,
                    aux_loglike, transform=aux_transform, vectorized=True)

            sampler.run(viz_callback=None)
            sampler.print_results()
            print("expected posterior:", center, '+-', stdev)
            print(sampler.results.keys())
            print(sampler.results['posterior'].keys())
            print(sampler.results['posterior']['mean'], sampler.results['posterior']['stdev'])
            print(sampler.results['weighted_samples']['upoints'], sampler.results['weighted_samples']['weights'])
            assert center - stdev < sampler.results['posterior']['mean'][0] < center + stdev, (center, sampler.results['posterior'])
            assert stdev * 0.8 < sampler.results['posterior']['stdev'][0] < stdev * 1.2, (center, sampler.results['posterior'])
            ncalls.append(sampler.ncall)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print(ncalls)
    
    # make sure hot start is much faster
    assert ncalls[1] < ncalls[0] - 800, (ncalls)
    assert ncalls[2] < ncalls[0] - 800, (ncalls)

if __name__ == '__main__':
    #test_run_compat()
    #test_run_resume(dlogz=0.5)
    #test_reactive_run_resume(dlogz=0.5, min_ess=1000)
    #test_reactive_run()
    #test_run()
    #test_reactive_run_warmstart_gauss()
    #test_reactive_run_extraparams()
    test_reactive_run_resume_eggbox('hdf5')
    #test_dlogz_reactive_run()
