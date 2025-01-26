import os
import tempfile
import numpy as np

from ultranest import ReactiveNestedSampler
from ultranest.mlfriends import AffineLayer, ScalingLayer, MLFriends, RobustEllipsoidRegion, SimpleRegion
from ultranest.popstepsampler import PopulationSliceSampler, PopulationRandomWalkSampler, PopulationSimpleSliceSampler
from ultranest.popstepsampler import generate_cube_oriented_direction, generate_random_direction, generate_cube_oriented_direction_scaled
from ultranest.popstepsampler import generate_region_oriented_direction, generate_region_random_direction
from ultranest.popstepsampler import slice_limit_to_unitcube,slice_limit_to_scale
from ultranest.popstepsampler import int_dtype

def make_region(ndim, us=None, nlive=400):
    if us is None:
        us = np.random.uniform(size=(nlive, ndim))
    
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=1.0)
    return region

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
        generate_direction=generate_cube_oriented_direction,
        log=True,
    )
    r = sampler.run(viz_callback=None, log_interval=50)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

    with tempfile.TemporaryDirectory() as tempdir:
        prefix = os.path.join(tempdir, 'test-stepsampler')
        sampler.stepsampler.plot(prefix + '-plot.pdf')
        assert os.path.exists(prefix + '-plot.pdf')
        sampler.stepsampler.plot_jump_diagnostic_histogram(prefix + '-plot-jumps.pdf')
        assert os.path.exists(prefix + '-plot-jumps.pdf')
        sampler.stepsampler.print_diagnostic()
        print(sampler.stepsampler)
        print(sampler.stepsampler.status)

def test_stepsampler_cubegausswalk(plot=False):
    np.random.seed(2)
    nsteps = np.random.randint(10, 50)
    popsize = np.random.randint(1, 20)
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)

    sampler.stepsampler = PopulationRandomWalkSampler(
        popsize=popsize, nsteps=nsteps, 
        generate_direction=generate_cube_oriented_direction,
        scale=0.1, log=True,
    )
    r = sampler.run(viz_callback=None, log_interval=50, max_iters=200, max_num_improvement_loops=0)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1

def test_stepsampler_randomSimSlice(plot=False):
    np.random.seed(4)
    nsteps = np.random.randint(10, 50)
    popsize = np.random.randint(1, 20)
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)

    sampler.stepsampler = PopulationSimpleSliceSampler(
        popsize=popsize, nsteps=nsteps, 
        generate_direction=generate_random_direction,
    )
    r = sampler.run(viz_callback=None, log_interval=50, max_iters=200, max_num_improvement_loops=0)
    sampler.print_results()
    a = (np.abs(r['samples'] - 0.7) < 0.1).all(axis=1)
    b = (np.abs(r['samples'] - 0.3) < 0.1).all(axis=1)
    assert a.sum() > 1
    assert b.sum() > 1



    with tempfile.TemporaryDirectory() as tempdir:
        prefix = os.path.join(tempdir, 'test-stepsampler')
        sampler.stepsampler.plot(prefix + '-plot.pdf')
        assert os.path.exists(prefix + '-plot.pdf')
        sampler.stepsampler.plot_jump_diagnostic_histogram(prefix + '-plot-jumps.pdf')
        assert os.path.exists(prefix + '-plot-jumps.pdf')
        sampler.stepsampler.print_diagnostic()
        print(sampler.stepsampler)

def test_direction_proposals():
    proposals = [generate_cube_oriented_direction, generate_random_direction, 
        generate_region_oriented_direction, generate_region_random_direction]

    points = np.random.uniform(size=(100, 10))
    minvol = 1.0

    scale = 1. # np.random.uniform()
    for layer in AffineLayer, ScalingLayer:
        transformLayer = layer()
        transformLayer.optimize(points, points)
        for region_class in MLFriends, RobustEllipsoidRegion, SimpleRegion:
            region = region_class(points, transformLayer)
            r, f = region.compute_enlargement(minvol=minvol, nbootstraps=30)
            region.maxradiussq = r
            region.enlarge = f
            region.create_ellipsoid(minvol=minvol)

            for prop in proposals:
                print("test of proposal:", prop, "with region:", region_class, "layer:", layer)
                directions = prop(points, region, scale=scale)
                assert directions.shape == points.shape, (directions.shape, points.shape)
                #assert np.allclose(norms, scale), (norms, scale)


def test_slice_limit():

    slice_limit_func = [slice_limit_to_unitcube, slice_limit_to_scale]
    fake_tleft = [-0.5, -0.2, -1.5]
    fake_tright = [0.2, 2.4, 0.2]

    fake_tleft_scale = [-0.5, -0.2, -1.]
    fake_tright_scale = [0.2, 1.0, 0.2]

    true_tleft = [fake_tleft, fake_tleft_scale]
    true_tright = [fake_tright, fake_tright_scale]

    for i,func in enumerate(slice_limit_func):
        tleft, tright = func(fake_tleft, fake_tright)
        assert np.allclose(tleft, true_tleft[i]), (tleft, true_tleft[i])
        assert np.allclose(tright, true_tright[i]), (tright, true_tright[i])


from ultranest.stepfuncs import update_vectorised_slice_sampler

def test_update_slice_sampler():
    """
    Test goal: Testing the update in each different typical cases.
    
    There are 3 points searched with 4 points sampled on their slices:
        - In the first case, no point is satisfying the Lmin condition. 
    The functions should just update the slice limits and keep the status
    unchanged.
        - In the second case, one point is satisfying the Lmin condition.
    But it will be discarded as it will be outside the slice limits. The
    function should update the slice limits and keep the same status.
        - In the third case, one point is satisfying the Lmin condition and
    the slice limits. The function should update the slice limits and change
    the status.

    The workers should be split among the 2 unfinished points at the end.
    """
    
    worker_running = np.array([0,0,0,0,1,1,1,1,2,2,2,2], dtype=int_dtype)
    popsize = 12
    status = np.zeros(12, dtype=int_dtype)
    status[3:] = 1
    Lmin = 1.
    shrink = 1.0 
    proposed_L = np.array([-12.,0.5,0.09,-2.,0.4,-5,2.4,0.3,-3.4,1.2,0.1,0.5])
    tleft = -np.ones(12)
    tright = np.ones(12)
    t = np.array([-0.8,-0.2,0.4,-0.5,-0.3,0.9,-0.7,0.2,-0.8,0.5,-0.4,0.6])
    proposed_u = np.array([[0.,0.,0.,0.,1.,1.,1.,1.,2.,2.5,2.,2.]]).T
    proposed_p = np.array([[0.,0.,0.,0.,1.,1.,1.,1.,2.,2.5,2.,2.]]).T
    allL = np.zeros(12)
    allu = np.zeros((12,1))
    allp = np.zeros((12,1))
   
    
    tleft, tright, worker_running, status, allu, allL, allp,discarded= update_vectorised_slice_sampler(
        t, tleft,tright,proposed_L,proposed_u,proposed_p,worker_running,status,Lmin,shrink,allu,allL,allp,popsize)

    true_worker= np.array([0,1,0,1,0,1,0,1,0,1,0,1])
    true_status = np.array([0,0,1,1,1,1,1,1,1,1,1,1])
    true_allL = np.array([0.,0.,1.2,0,0,0,0,0,0,0,0,0])
    true_allu = np.array([[0.,0.,2.5,0,0,0,0,0,0,0,0,0]]).T
    true_allp = np.array([[0.,0.,2.5,0,0,0,0,0,0,0,0,0]]).T
    true_discarded = 1
    true_tleft = np.array([-0.2,-.3,-0.4,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    true_tright = np.array([0.4,0.2,0.5,1,1,1,1,1,1,1,1,1])

    assert np.allclose(worker_running, true_worker), (worker_running, true_worker)
    assert np.allclose(status, true_status), (status, true_status)
    assert np.allclose(allL, true_allL), (allL, true_allL)
    assert np.allclose(allu, true_allu), (allu, true_allu)
    assert np.allclose(allp, true_allp), (allp, true_allp)
    assert np.allclose(discarded, true_discarded), (discarded, true_discarded)
    assert np.allclose(tleft, true_tleft), (tleft, true_tleft)
    assert np.allclose(tright, true_tright), (tright, true_tright)


# aim at checking the sanity of the results of 
# one iteration of the slice sampler. 
def test_SimpleSliceSampler_SLOW(seed=4):
    np.random.seed(seed)
    nsteps = 1 
    popsize = 100
    ndim = 10
    sampler = ReactiveNestedSampler(paramnames, loglike_vectorized, transform=transform, vectorized=True)

    sampler.stepsampler = PopulationSimpleSliceSampler(
        popsize=popsize, nsteps=nsteps, 
        generate_direction=generate_random_direction,
    )
    stepsampler = sampler.stepsampler
    # start with a random point in the unit cube
    us = (np.random.uniform(size=(popsize, ndim))-0.5)*0.9+0.5
    Ls = loglike_vectorized(us)
    Lmin = np.min(Ls)

    u,L=np.zeros((popsize,ndim)),np.zeros(popsize)

    # initialising a region
    #print(us) 
    region= RobustEllipsoidRegion(us, AffineLayer())
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=1.0)

    # resetting the seed to check the slice axes
    np.random.seed(seed)
    for i in range(popsize):
        u[i],_,L[i],_= stepsampler.__next__(region, Lmin, us.copy(), Ls.copy(), transform, loglike_vectorized, test=True)

    # Basic check
    assert (L>Lmin).all(), (L,Lmin) # Lmin check
    assert (u>0).all() and (u<1).all(), u # u in the unit cube check
    
    np.random.seed(seed)
    # resetting the random generation inside the sampler
    np.random.randint(0, us.shape[0], size=stepsampler.popsize)
    stepsampler.scale_jitter_func()

    # Getting the slice axes
    slice_axes =  stepsampler.generate_direction(us.copy(), region,scale= 1.0)
    for i in range(popsize):
        v = (u[i,:] - us[i,:]) / slice_axes[i, :]
        mean_v = np.mean(v)
        assert np.allclose(mean_v, v, atol=1e-10), (mean_v, v)



    
def test_direction_proposal_values():
    ndim = 10
    np.random.seed(12)
    region = make_region(ndim, nlive=400)
    ui = region.u[::2]
    
    scale = np.random.uniform()
    vcube = generate_cube_oriented_direction(ui, region, scale)
    assert vcube.shape == ui.shape
    assert vcube.sum(axis=1).shape == (len(ui),)
    assert ((vcube != 0).sum(axis=1) == 1).all(), vcube
    assert np.allclose(np.linalg.norm(vcube, axis=1), scale), (vcube, np.linalg.norm(vcube, axis=1), scale)

    vharm = generate_random_direction(ui, region, scale)
    assert (vharm != 0).all(), vharm
    vregionslice = generate_region_oriented_direction(ui, region, scale)
    assert (vregionslice != 0).all(), vregionslice
    vregionharm = generate_region_random_direction(ui, region, scale)
    assert (vregionharm != 0).all(), vregionharm
    vcubestd = generate_cube_oriented_direction_scaled(ui, region, scale)
    assert vcubestd.shape == ui.shape
    assert vcubestd.sum(axis=1).shape == (len(ui),)
    assert ((vcubestd != 0).sum(axis=1) == 1).all(), vcubestd


if __name__ == '__main__':
    #test_stepsampler_cubegausswalk()
    #test_stepsampler_randomSimSlice()
    #test_direction_proposals()
    test_slice_limit()
    #test_update_slice_sampler()
    #Test_SimpleSliceSampler(4)
    
    
