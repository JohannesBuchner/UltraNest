import numpy as np
from mininest.mlfriends import AffineLayer, MLFriends
from mininest.flatnuts import ClockedStepSampler, ClockedBisectSampler, ClockedNUTSSampler
from mininest.samplingpath import SamplingPath, ContourSamplingPath
from numpy.testing import assert_allclose

def gap_free_path(sampler, ilo, ihi, transform, loglike, Lmin):
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
        print("FORWARD SAMPLING FROM", 0, active_u[0], v, active_values[0])
        samplingpath = SamplingPath(active_u[0], v, active_values[0])
        sampler = ClockedBisectSampler(ContourSamplingPath(samplingpath, region))
        check_starting_point(sampler, active_u[0], active_values[0], **problem)
        sampler.expand_to_step(10, **problem)
        check_starting_point(sampler, active_u[0], active_values[0], **problem)
        
        starti, startx, startv, startL = max(sampler.points)
        print()
        print("BACKWARD SAMPLING FROM", starti, startx, startv, startL)
        samplingpath2 = SamplingPath(startx, -startv, startL)
        sampler2 = ClockedBisectSampler(ContourSamplingPath(samplingpath2, region))
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
        sampler3 = ClockedBisectSampler(ContourSamplingPath(samplingpath3, region))
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
    
    
if __name__ == '__main__':
    test_detailed_balance(plot=True)
