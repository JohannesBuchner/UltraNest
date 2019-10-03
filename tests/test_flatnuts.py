import numpy as np
from ultranest.mlfriends import AffineLayer, ScalingLayer, MLFriends
from ultranest.flatnuts import ClockedStepSampler, ClockedBisectSampler, ClockedNUTSSampler
from ultranest.flatnuts import SingleJumper, DirectJumper
from ultranest.samplingpath import SamplingPath, ContourSamplingPath
from numpy.testing import assert_allclose

def gap_free_path(sampler, ilo, ihi, transform, loglike, Lmin):
    """
    Check if sampling path at all intermediate points between ilo and ihi
    are above Lmin.
    """
    for i in range(ilo, ihi):
        xi, vi, Li, onpath = sampler.contourpath.samplingpath.interpolate(i)
        assert onpath
        if Li is None:
            pi = transform(xi)
            Li = loglike(pi)
            if not Li > Lmin:
                return False
    return True

def check_starting_point(sampler, startx, startL, transform, loglike, Lmin):
    """ Verify that if going 0 steps, should return start point. """
    assert sampler.goals == [], sampler.goals
    sampler.set_nsteps(0)
    Llast = None
    sample, is_independent = sampler.next(Llast)
    assert is_independent, (sample, is_independent)
    unew, Lnew = sample
    assert_allclose(startx, unew)
    assert_allclose(startL, Lnew)
    
    assert sampler.goals == [], sampler.goals
    sample, is_independent = sampler.next(Llast)
    assert sample is None and not is_independent, (sample, is_independent)

def test_detailed_balance():
    def loglike(x):
        x, y = x.transpose()
        return -0.5 * (x**2 + ((y - 0.5)/0.2)**2)
    def transform(u):
        return u

    Lmin = -0.5
    for i in range(1, 100):
        print()
        print("---- seed=%d ----" % i)
        print()
        np.random.seed(i)
        points = np.random.uniform(size=(10000, 2))
        L = loglike(points)
        mask = L > Lmin
        points = points[mask,:][:400,:]
        active_u = points
        active_values = L[mask][:400]

        transformLayer = AffineLayer(wrapped_dims=[])
        transformLayer.optimize(points, points)
        region = MLFriends(points, transformLayer)
        region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
        region.create_ellipsoid()
        nclusters = transformLayer.nclusters
        assert nclusters == 1
        assert np.allclose(region.unormed, region.transformLayer.transform(points)), "transform should be reproducible"
        assert region.inside(points).all(), "live points should lie near live points"

        v = np.random.normal(size=2)
        v /= (v**2).sum()**0.5
        v *= 0.04
        
        print("StepSampler ----")
        print("FORWARD SAMPLING FROM", 0, active_u[0], v, active_values[0])
        samplingpath = SamplingPath(active_u[0], v, active_values[0])
        problem = dict(loglike=loglike, transform=transform, Lmin=Lmin)
        sampler = ClockedStepSampler(ContourSamplingPath(samplingpath, region))
        check_starting_point(sampler, active_u[0], active_values[0], **problem)
        sampler.expand_onestep(fwd=True, **problem)
        sampler.expand_onestep(fwd=True, **problem)
        sampler.expand_onestep(fwd=True, **problem)
        sampler.expand_onestep(fwd=True, **problem)
        sampler.expand_onestep(fwd=False, **problem)
        sampler.expand_to_step(4, **problem)
        sampler.expand_to_step(-4, **problem)
        check_starting_point(sampler, active_u[0], active_values[0], **problem)
        
        starti, startx, startv, startL = max(sampler.points)
        
        print()
        print("BACKWARD SAMPLING FROM", starti, startx, startv, startL)
        samplingpath2 = SamplingPath(startx, -startv, startL)
        sampler2 = ClockedStepSampler(ContourSamplingPath(samplingpath2, region))
        check_starting_point(sampler2, startx, startL, **problem)
        sampler2.expand_to_step(starti, **problem)
        check_starting_point(sampler2, startx, startL, **problem)
        
        starti2, startx2, startv2, startL2 = max(sampler2.points)
        assert_allclose(active_u[0], startx2)
        assert_allclose(v, -startv2)
        
        starti, startx, startv, startL = min(sampler.points)
        print()
        print("BACKWARD SAMPLING FROM", starti, startx, startv, startL)
        samplingpath3 = SamplingPath(startx, startv, startL)
        sampler3 = ClockedStepSampler(ContourSamplingPath(samplingpath3, region))
        check_starting_point(sampler3, startx, startL, **problem)
        sampler3.expand_to_step(-starti, **problem)
        check_starting_point(sampler3, startx, startL, **problem)
        
        starti3, startx3, startv3, startL3 = max(sampler3.points)
        assert_allclose(active_u[0], startx3)
        assert_allclose(v, startv3)
        print()
        
        print("BisectSampler ----")
        log = dict(log=True)
        print("FORWARD SAMPLING FROM", 0, active_u[0], v, active_values[0])
        samplingpath = SamplingPath(active_u[0], v, active_values[0])
        sampler = ClockedBisectSampler(ContourSamplingPath(samplingpath, region), **log)
        check_starting_point(sampler, active_u[0], active_values[0], **problem)
        sampler.expand_to_step(10, **problem)
        check_starting_point(sampler, active_u[0], active_values[0], **problem)
        
        starti, startx, startv, startL = max(sampler.points)
        print()
        print("BACKWARD SAMPLING FROM", starti, startx, startv, startL)
        samplingpath2 = SamplingPath(startx, -startv, startL)
        sampler2 = ClockedBisectSampler(ContourSamplingPath(samplingpath2, region), **log)
        check_starting_point(sampler2, startx, startL, **problem)
        sampler2.expand_to_step(starti, **problem)
        check_starting_point(sampler2, startx, startL, **problem)
        
        starti2, startx2, startv2, startL2 = max(sampler2.points)
        if gap_free_path(sampler, 0, starti, **problem) and gap_free_path(sampler2, 0, starti2, **problem):
            assert_allclose(active_u[0], startx2)
            assert_allclose(v, -startv2)
        
        starti, startx, startv, startL = min(sampler.points)
        print()
        print("BACKWARD SAMPLING FROM", starti, startx, startv, startL)
        samplingpath3 = SamplingPath(startx, -startv, startL)
        sampler3 = ClockedBisectSampler(ContourSamplingPath(samplingpath3, region), **log)
        check_starting_point(sampler3, startx, startL, **problem)
        sampler3.expand_to_step(starti, **problem)
        check_starting_point(sampler3, startx, startL, **problem)
        
        starti3, startx3, startv3, startL3 = min(sampler3.points)
        if gap_free_path(sampler, 0, starti, **problem) and gap_free_path(sampler3, 0, starti3, **problem):
            assert_allclose(active_u[0], startx3)
            assert_allclose(v, -startv3)
        print()


        print("NUTSSampler ----")
        print("FORWARD SAMPLING FROM", 0, active_u[0], v, active_values[0])
        samplingpath = SamplingPath(active_u[0], v, active_values[0])
        np.random.seed(i)
        sampler = ClockedNUTSSampler(ContourSamplingPath(samplingpath, region))
        sampler.get_independent_sample(**problem)


def makejump(stepper, sampler, transform, loglike, Lmin):
    stepper.prepare_jump()
    Llast = None
    while not sampler.is_done():
        u, is_independent = sampler.next(Llast=Llast)
        if not is_independent:
            # should evaluate
            p = transform(u)
            L = loglike(p)
            if L > Lmin:
                Llast = L
            else:
                Llast = None
    return stepper.make_jump()

def make_region(ndim):
    us = np.random.uniform(size=(1000, ndim))
    if ndim > 1:
        transformLayer = AffineLayer()
    else:
        transformLayer = ScalingLayer()
    transformLayer.optimize(us, us)
    region = MLFriends(us, transformLayer)
    region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
    region.create_ellipsoid(minvol=1.0)
    return region

def test_singlejumper():
    Lmin = -1.0
    us = 0.5 + np.zeros((100, 2))
    #Ls = np.zeros(100)
    region = make_region(2)
    def transform(x): return x
    def loglike(x): return 0.0
    def gradient(x, plot=False):
        j = np.argmax(np.abs(x - 0.5))
        v = np.zeros(len(x))
        v[j] = -1 if x[j] > 0.5 else 1
        return v
    
    def nocall(x):
        assert False

    ui = us[np.random.randint(len(us)),:]
    v = np.array([0.01, 0.01])
    path = ContourSamplingPath(SamplingPath(ui, v, 0.0), region)
    path.gradient = nocall
    sampler = ClockedStepSampler(path)
    stepper = SingleJumper(sampler, 4)

    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 0, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.51, 0.51])
    assert (stepper.naccepts, stepper.nrejects) == (1, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 1, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.52, 0.52])
    assert (stepper.naccepts, stepper.nrejects) == (2, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 2, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.53, 0.53])
    assert (stepper.naccepts, stepper.nrejects) == (3, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 3, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.54, 0.54])
    assert (stepper.naccepts, stepper.nrejects) == (4, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 4, stepper.isteps
    
    print()
    print("make reflect")
    print()
    def loglike(x): return 0.0 if x[0] < 0.505 else -100
    path = ContourSamplingPath(SamplingPath(ui, v, 0.0), region)
    path.gradient = gradient
    sampler = ClockedStepSampler(path)
    stepper = SingleJumper(sampler, 4)
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 0, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.50, 0.52])
    assert (stepper.naccepts, stepper.nrejects) == (1, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 1, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.49, 0.53])
    assert (stepper.naccepts, stepper.nrejects) == (2, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 2, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.48, 0.54])
    assert (stepper.naccepts, stepper.nrejects) == (3, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 3, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.47, 0.55])
    assert (stepper.naccepts, stepper.nrejects) == (4, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 4, stepper.isteps
    
    print()
    print("make stuck")
    print()
    # make stuck
    def loglike(x): return -100
    path = ContourSamplingPath(SamplingPath(ui, v, 0.0), region)
    path.gradient = gradient
    sampler = ClockedStepSampler(path)
    stepper = SingleJumper(sampler, 4)
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 0, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.50, 0.50])
    assert (stepper.naccepts, stepper.nrejects) == (0, 1), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 1, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.50, 0.50])
    assert (stepper.naccepts, stepper.nrejects) == (0, 2), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 2, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.50, 0.50])
    assert (stepper.naccepts, stepper.nrejects) == (0, 3), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 3, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.50, 0.50])
    assert (stepper.naccepts, stepper.nrejects) == (0, 4), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 4, stepper.isteps
    
def test_directjumper():
    Lmin = -1.0
    us = 0.5 + np.zeros((100, 2))
    #Ls = np.zeros(100)
    region = make_region(2)
    def transform(x): return x
    def loglike(x): return 0.0
    def gradient(x, plot=False):
        j = np.argmax(np.abs(x - 0.5))
        v = np.zeros(len(x))
        v[j] = -1 if x[j] > 0.5 else 1
        return v
    
    def nocall(x):
        assert False

    ui = us[np.random.randint(len(us)),:]
    v = np.array([0.01, 0.01])
    path = ContourSamplingPath(SamplingPath(ui, v, 0.0), region)
    path.gradient = nocall
    sampler = ClockedBisectSampler(path)
    stepper = DirectJumper(sampler, 4)
    
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 0, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.54, 0.54])
    assert (stepper.naccepts, stepper.nrejects) == (4, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 4, stepper.isteps
    
    print()
    print("make reflect")
    print()
    def loglike(x): return 0.0 if x[0] < 0.505 else -100
    path = ContourSamplingPath(SamplingPath(ui, v, 0.0), region)
    path.gradient = gradient
    sampler = ClockedBisectSampler(path)
    stepper = DirectJumper(sampler, 4)
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 0, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.47, 0.55])
    assert (stepper.naccepts, stepper.nrejects) == (4, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 4, stepper.isteps
    
    print()
    print("make stuck")
    print()
    # make stuck
    def loglike(x): return -100
    path = ContourSamplingPath(SamplingPath(ui, v, 0.0), region)
    path.gradient = gradient
    sampler = ClockedBisectSampler(path)
    stepper = DirectJumper(sampler, 4)
    assert (stepper.naccepts, stepper.nrejects) == (0, 0), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 0, stepper.isteps
    x, L = makejump(stepper, sampler, transform, loglike, Lmin)
    assert_allclose(x, [0.50, 0.50])
    assert (stepper.naccepts, stepper.nrejects) == (0, 4), (stepper.naccepts, stepper.nrejects)
    assert stepper.isteps == 4, stepper.isteps

if __name__ == '__main__':
    test_detailed_balance(plot=True)
