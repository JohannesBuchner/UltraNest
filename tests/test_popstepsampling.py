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
    fake_tleft = np.array([-0.5, -0.2, -1.5])
    fake_tright = np.array([0.2, 2.4, 0.2])

    fake_tleft_scale = np.array([-0.5, -0.2, -1.])
    fake_tright_scale = np.array([0.2, 1.0, 0.2])

    true_tleft = [fake_tleft, fake_tleft_scale]
    true_tright = [fake_tright, fake_tright_scale]

    for i,func in enumerate(slice_limit_func):
        tleft, tright = func(fake_tleft, fake_tright)
        assert np.allclose(tleft, true_tleft[i]), (tleft, true_tleft[i])
        assert np.allclose(tright, true_tright[i]), (tright, true_tright[i])


from ultranest.stepfuncs import update_vectorised_slice_sampler,QuantileDistribution,LinearInterpolator

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


def simple_data():
    """Simple linear test data"""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    return x, y
    
    
def nonlinear_data():
    """Non-linear test data"""
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    return x, y
    
def test_linear_interpolation_simple():
    """Test linear interpolator on simple linear data"""
    
    
    x, y = simple_data()
    interp = LinearInterpolator(x, y, fill_left=0.0, fill_right=8.0)
    
    # Test exact points
    for xi, yi in zip(x, y):
        assert np.isclose(interp(xi), yi, atol=1e-10)
    
    # Test midpoints (should be exact for linear data)
    assert np.isclose(interp(0.5), 1.0, atol=1e-10)
    assert np.isclose(interp(1.5), 3.0, atol=1e-10)
    assert np.isclose(interp(2.5), 5.0, atol=1e-10)
    
def test_linear_vs_scipy():
    """Compare LinearInterpolator against scipy.interpolate.interp1d"""
    
    from scipy.interpolate import interp1d 
    x, y = nonlinear_data()
    
    # Create both interpolators
    cython_interp = LinearInterpolator(x, y, fill_left=y[0], fill_right=y[-1])
    scipy_interp = interp1d(x, y, kind='linear', bounds_error=False, 
                   fill_value=(y[0], y[-1]))
    
    # Test at various points
    test_points = np.linspace(x[0], x[-1], 50)
    cython_results = np.array([cython_interp(xi) for xi in test_points])
    scipy_results = scipy_interp(test_points)
    
    np.testing.assert_allclose(cython_results, scipy_results, rtol=1e-10)
    
def test_boundary_conditions():
    """Test fill_left and fill_right behavior"""
    
    
    x, y = simple_data()
    interp = LinearInterpolator(x, y, fill_left=-999.0, fill_right=999.0)
    
    # Test left boundary
    assert interp(-10.0) == -999.0,f"Expected -999.0, got {interp(-10.0)}"
    assert interp(0.0) == 0.0,f"Expected 0.0, got {interp(0.01)}"
    
    # Test right boundary
    assert interp(10.0) == 999.0,f"Expected 999.0, got {interp(10.0)}"
    assert interp(4.0) == 8.0,f"Expected 8.0, got {interp(4.0)}"
    
def test_dense_interpolation():
    """Test interpolation at many dense points"""
    
    from scipy.interpolate import interp1d 
    x = np.linspace(0, 100, 1000)
    y = np.cos(x) * np.exp(-x / 50)
    
    interp = LinearInterpolator(x, y, fill_left=y[0], fill_right=y[-1])
    scipy_interp = interp1d(x, y, kind='linear', bounds_error=False,
                   fill_value=(y[0], y[-1]))
    
    # Test at 500 random points
    test_points = np.random.uniform(x[0], x[-1], 500)
    cython_results = np.array([interp(xi) for xi in test_points])
    scipy_results = scipy_interp(test_points)
    
    np.testing.assert_allclose(cython_results, scipy_results, rtol=1e-10)
 
 
def normal_quantiles():
    """Generate quantiles from normal distribution"""
    np.random.seed(42)
    popsize = 5
    n_quantiles = 10
    
    # Create synthetic "live points" from normal distribution
    projected_live_points = np.random.normal(0, 1, size=(100, popsize))
    
    
    
    # Compute quantiles
    quantiles_points = np.linspace(0, 100, n_quantiles)
    quantiles = np.percentile(projected_live_points, quantiles_points, axis=0)
    
    return quantiles, quantiles_points, popsize
    

def uniform_quantiles():
    """Generate quantiles from uniform distribution"""
    np.random.seed(42)
    popsize = 3
    n_quantiles = 8
    
    # Uniform samples
    samples = np.random.uniform(0, 1, size=(200, popsize))
    
    quantiles_points = np.linspace(0, 100, n_quantiles)
    quantiles = np.percentile(samples, quantiles_points, axis=0)
    
    return quantiles, quantiles_points, popsize
    
def test_cdf_ppf_inverse_property():
    """Test that CDF and PPF are exact inverses"""
    
    
    quantiles, quantiles_points, popsize = normal_quantiles()
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    
    # Create test points uniformly distributed in probability space
    test_probs = np.linspace(0.01, 0.99, 50)
    worker_indices = np.zeros(len(test_probs), dtype=int)
    
    # PPF then CDF should return original probabilities
    ppf_vals = dist.ppf(test_probs, worker_indices)
    cdf_vals = dist.cdf(ppf_vals, worker_indices)
    
    np.testing.assert_allclose(cdf_vals, test_probs, rtol=1e-6, atol=1e-8)
    
    # Test for each worker
    for w in range(popsize):
        worker_indices = np.full(len(test_probs), w, dtype=int)
        ppf_vals = dist.ppf(test_probs, worker_indices)
        cdf_vals = dist.cdf(ppf_vals, worker_indices)
        np.testing.assert_allclose(cdf_vals, test_probs, rtol=1e-6, atol=1e-8)
    
def test_cdf_ppf_round_trip_data_space( ):
    """Test CDF->PPF->CDF round-trip in data space"""
    
    
    quantiles, quantiles_points, popsize = normal_quantiles()
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    
    # Start with data space values
    test_values = np.linspace(-1,1 , 40)
    worker_indices = np.zeros(len(test_values), dtype=int)
    # CDF to probability space, then PPF back to data space
    probs = dist.cdf(test_values, worker_indices)
    recovered_values = dist.ppf(probs, worker_indices)
    
    assert np.allclose(recovered_values, test_values),f"{recovered_values} != {test_values}"
    
def test_cdf_bounds( ):
    """Test that CDF values are always in [0, 1]"""
    
    
    quantiles, quantiles_points, popsize = normal_quantiles()
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    
    # Test values far outside the quantile range
    test_values = np.concatenate([
        np.linspace(quantiles.min() - 100, quantiles.min() - 1, 10),
        np.linspace(quantiles.max() + 1, quantiles.max() + 100, 10)
    ])
    worker_indices = np.full(len(test_values), 0, dtype=int)
    
    cdf_vals = dist.cdf(test_values, worker_indices)
    
    # Should respect fill values
    assert np.all(cdf_vals[test_values < quantiles[:, 0].min()] >= 0)
    assert np.all(cdf_vals[test_values > quantiles[:, 0].max()] <= 1)
    
def test_ppf_bounds( ):
    """Test that PPF returns values within quantile range"""
    
    
    quantiles, quantiles_points, popsize = normal_quantiles()
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    
    # Test edge probabilities
    test_probs = np.array([0.0, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0])
    
    for w in range(popsize):
        worker_indices = np.full(len(test_probs), w, dtype=int)
        ppf_vals = dist.ppf(test_probs, worker_indices)
        
        # PPF values should be within or very close to quantile bounds
        q_min, q_max = quantiles[:, w].min(), quantiles[:, w].max()
        assert np.all(ppf_vals >= q_min - 1e-5)
        assert np.all(ppf_vals <= q_max + 1e-5)
    

    
def test_logpdf_monotonicity_around_mode( ):
    """Test that logpdf has reasonable shape (single peak for normal)"""
    
    
    quantiles, quantiles_points, popsize = normal_quantiles()
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    
    # Get the median (mode for symmetric distribution)
    median_idx = np.argmin(np.abs(quantiles_points - 50))
    median_val = quantiles[median_idx, 0]
    
    # Test values around the median
    half_width = (quantiles[:, 0].max() - quantiles[:, 0].min()) / 4
    test_values = np.linspace(median_val - half_width, median_val + half_width, 30)
    worker_indices = np.zeros(len(test_values), dtype=int)
    
    logpdf_vals = dist.logpdf(test_values, worker_indices)
    
    # Find the peak
    peak_idx = np.argmax(logpdf_vals)
    
    # Should have a single peak, not oscillating
    assert peak_idx > 0 and peak_idx < len(logpdf_vals) - 1
    
def test_logpdf_consistency_across_workers():
    """Test that logpdf works consistently for different workers"""
    
    
    quantiles, quantiles_points, popsize = normal_quantiles()
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    
    test_values = np.array([0.0, 0.5, 1.0])
    
    for w in range(popsize):
        worker_indices = np.full(len(test_values), w, dtype=int)
        logpdf_vals = dist.logpdf(test_values, worker_indices)
        
        assert np.all(np.isfinite(logpdf_vals))
        assert len(logpdf_vals) == len(test_values)
    
def test_setup_as_described():
    """Test using the exact setup described in the requirements"""
    
    
    np.random.seed(42)
    popsize = 5
    
    # Generate synthetic data
    projected_live_points = np.random.normal(0, 1, size=(100, popsize))
    
       
    # Setup quantiles
    quantiles_points = np.linspace(0, 100, 10)
    quantiles = np.percentile(projected_live_points, quantiles_points, axis=0)
    #print(quantiles.shape, quantiles_points.shape,popsize)
    # Create distribution
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    worker_running = np.arange(popsize, dtype=int)
    
    # Create test data
    tleft_unitcube = np.random.uniform(0, 1, size=popsize)
    
    # Test the pipeline
    tleft_cdf = dist.cdf(tleft_unitcube, worker_running)
    tleft_reverse = dist.ppf(tleft_cdf, worker_running)
    logpdf = dist.logpdf(tleft_unitcube, worker_running)
    
    # Check outputs
    assert tleft_cdf.shape == tleft_unitcube.shape
    assert tleft_reverse.shape == tleft_unitcube.shape
    assert logpdf.shape == tleft_unitcube.shape
    
    # CDF should be in [0, 1]
    assert np.all(tleft_cdf >= 0) and np.all(tleft_cdf <= 1)
    
    # PPF should invert CDF
    assert np.allclose(tleft_unitcube,tleft_reverse)
    
    # Logpdf should be finite
    assert np.all(np.isfinite(logpdf))
 
 
def test_full_pipeline_with_normal_samples():
    """Test complete pipeline with normal distribution samples"""
    
    from scipy.stats import norm,rv_histogram 
    np.random.seed(50)
    popsize = 1
    n_samples = 1000
    
    # Generate samples from different normal distributions
    loc=np.random.uniform(-5, 5, popsize)
    scale=np.random.uniform(0.5, 2, popsize)
    samples = np.random.normal(loc=loc,
                   scale=scale,
                   size=(n_samples, popsize))
     
    quantiles_points = np.linspace(0, 100, 50)
    quantiles = np.percentile(samples, quantiles_points, axis=0)
    
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    Approx_dist = [rv_histogram(np.histogram(samples[:, w], bins=50, density=True),density=True) for w in range(popsize)]
    # Generate test points in probability space
    probs = np.random.uniform(0.01, 0.99, 500)
    
    # Test each worker
    for w in range(popsize):
        worker_indices = np.full(len(probs), w, dtype=int)
        
        # Forward and inverse transforms
        values = dist.ppf(probs, worker_indices)
        recovered_probs = dist.cdf(values, worker_indices)
        
        # Check round-trip
        np.testing.assert_allclose(recovered_probs, probs, rtol=1e-5)
        
        # Get log-pdf
        True_logpdf = norm.logpdf(values, loc=loc[w], scale=scale[w])
        rv_hist_logpdf = Approx_dist[w].logpdf(values)
        logpdf_vals = dist.logpdf(values, worker_indices)
        mean_diff_true = np.where(np.isfinite((True_logpdf-logpdf_vals)/True_logpdf), (True_logpdf-logpdf_vals)/True_logpdf, 0)
        mean_diff_rvhist = np.where(np.isfinite((rv_hist_logpdf-logpdf_vals)/rv_hist_logpdf), (rv_hist_logpdf-logpdf_vals)/rv_hist_logpdf, 0)
        mean_diff_true_rvhist = np.where(np.isfinite((True_logpdf-rv_hist_logpdf)/True_logpdf), (True_logpdf-rv_hist_logpdf)/True_logpdf, 0)
        print(f"Worker {w}: Mean relative difference (True vs Quantile): {np.mean(mean_diff_true)} Mean relative difference (rv_hist vs Quantile): {np.mean(mean_diff_rvhist)} Mean relative difference (True vs rv_hist): {np.mean(mean_diff_true_rvhist)}")
        assert np.all(np.isfinite(logpdf_vals))
    
def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    
    # Minimal quantiles
    quantiles = np.array([[0.0, 10.0], [5.0, 15.0], [10.0, 20.0]]).T
    quantiles_points = np.array([0.0, 50.0, 100.0])
    popsize = 2
    test_points = np.array([5.0,15.0])
    dist = QuantileDistribution(quantiles, quantiles_points, popsize)
    
    worker_indices = np.array([0, 1], dtype=int)
    
    # Test at mid points
    cdf_vals = dist.cdf(test_points, worker_indices)
    assert np.all(np.isfinite(cdf_vals))
    
    
 


    
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
    #test_slice_limit()
    #test_update_slice_sampler()
    #Test_SimpleSliceSampler(4)
    test_edge_cases()
    test_full_pipeline_with_normal_samples()
    test_setup_as_described()
    test_logpdf_consistency_across_workers()    
    test_logpdf_monotonicity_around_mode( )
    test_ppf_bounds( )
    test_cdf_bounds( )
    test_cdf_ppf_round_trip_data_space( )
    test_linear_vs_scipy()
    test_cdf_ppf_inverse_property()
    test_boundary_conditions()
    test_dense_interpolation()
    
    
    
