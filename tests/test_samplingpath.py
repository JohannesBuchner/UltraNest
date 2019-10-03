import numpy as np
import matplotlib.pyplot as plt
from ultranest.mlfriends import AffineLayer, MLFriends
from ultranest.samplingpath import SamplingPath, ContourSamplingPath
from ultranest.samplingpath import box_line_intersection, nearest_box_intersection_line, linear_steps_with_reflection, angle, get_sphere_tangents, norm
from numpy.testing import assert_allclose

def test_horizontal():
    (c1, _, ax1), (c2, _, ax2) = box_line_intersection(np.array([0.5, 0.5]), np.array([0, 1.]))
    print((c1, ax1), (c2, ax2))
    assert ax1 == 1
    np.testing.assert_allclose(c1, [0.5, 0])
    assert ax2 == 1
    np.testing.assert_allclose(c2, [0.5, 1])

    (c1, _, ax1), (c2, _, ax2) = box_line_intersection(np.array([0.3, 0.3]), np.array([1, 0.]))
    print((c1, ax1), (c2, ax2))
    assert ax1 == 0
    np.testing.assert_allclose(c1, [0, 0.3])
    assert ax2 == 0
    np.testing.assert_allclose(c2, [1, 0.3])

def test_corner():
    start, direction = np.array([0.6, 0.5]), np.array([0.4, 0.5])
    print("starting ray:", start, direction)
    (c1, _, ax1), (c2, _, ax2) = box_line_intersection(start, direction)
    print((c1, ax1), (c2, ax2))
    np.testing.assert_allclose(c2, [1, 1])
    start = c2
    direction[ax2] *= -1
    print("restarting ray:", start, direction)
    (c1, _, ax1), (c2, _, ax2) = box_line_intersection(start, direction)
    print((c1, ax1), (c2, ax2))
    np.testing.assert_allclose(c1, [1., 1.])
    np.testing.assert_allclose(c2, [0.2, 0.])
    start = c2
    direction[ax2] *= -1
    (c1, _, ax1), (c2, _, ax2) = box_line_intersection(start, direction)
    print((c1, ax1), (c2, ax2))
    np.testing.assert_allclose(c1, [0.2, 0.])
    np.testing.assert_allclose(c2, [0., 0.25])


def test_wrap():
    start, direction = np.array([0.1, 0.89]), np.array([0.2, 0.1])
    wrap = np.array([False, False])
    newpoint, _ = linear_steps_with_reflection(start, direction, 0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.1, 0.89])
    newpoint, _ = linear_steps_with_reflection(start, direction, 1.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.3, 0.99])
    newpoint, _ = linear_steps_with_reflection(start, direction, 2.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.5, 0.91])
    newpoint, _ = linear_steps_with_reflection(start, direction, 3.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.7, 0.81])
    newpoint, _ = linear_steps_with_reflection(start, direction, 4.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.9, 0.71])
    newpoint, _ = linear_steps_with_reflection(start, direction, 5.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.9, 0.61])

    start, direction = np.array([0.1, 0.89]), np.array([0.2, 0.1])
    wrap = np.array([True, False])
    newpoint, _ = linear_steps_with_reflection(start, direction, 0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.1, 0.89])
    newpoint, _ = linear_steps_with_reflection(start, direction, 1.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.3, 0.99])
    newpoint, _ = linear_steps_with_reflection(start, direction, 2.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.5, 0.91])
    newpoint, _ = linear_steps_with_reflection(start, direction, 3.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.7, 0.81])
    newpoint, _ = linear_steps_with_reflection(start, direction, 4.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.9, 0.71])
    newpoint, _ = linear_steps_with_reflection(start, direction, 5.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.1, 0.61])

    start, direction = np.array([0.1, 0.89]), np.array([0.2, 0.1])
    wrap = np.array([False, True])
    newpoint, _ = linear_steps_with_reflection(start, direction, 0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.1, 0.89])
    newpoint, _ = linear_steps_with_reflection(start, direction, 1.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.3, 0.99])
    newpoint, _ = linear_steps_with_reflection(start, direction, 2.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.5, 0.09])
    newpoint, _ = linear_steps_with_reflection(start, direction, 3.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.7, 0.19])
    newpoint, _ = linear_steps_with_reflection(start, direction, 4.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.9, 0.29])
    newpoint, _ = linear_steps_with_reflection(start, direction, 5.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.9, 0.39])

    start, direction = np.array([0.1, 0.89]), np.array([0.2, 0.1])
    wrap = np.array([True, True])
    newpoint, _ = linear_steps_with_reflection(start, direction, 0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.1, 0.89])
    newpoint, _ = linear_steps_with_reflection(start, direction, 1.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.3, 0.99])
    newpoint, _ = linear_steps_with_reflection(start, direction, 2.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.5, 0.09])
    newpoint, _ = linear_steps_with_reflection(start, direction, 3.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.7, 0.19])
    newpoint, _ = linear_steps_with_reflection(start, direction, 4.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.9, 0.29])
    newpoint, _ = linear_steps_with_reflection(start, direction, 5.0, wrapped_dims=wrap)
    assert_allclose(newpoint, [0.1, 0.39])

def test_random():
    for i in range(100):
        ndim = 1 + i
        start = np.random.uniform(size=ndim)
        direction = np.random.normal(size=ndim)
        direction /= (direction**2).sum()**0.5
        
        reset = np.random.binomial(1, 0.1, size=ndim) == 1
        direction[reset] = -start[reset]
        # check that the returned result is symmetric to the direction
        (c1, _, ax1), (c2, _, ax2) = box_line_intersection(start, direction)
        (b1, _, ax1), (b2, _, ax2) = box_line_intersection(start, -direction)
        np.testing.assert_allclose(c1, b2)
        np.testing.assert_allclose(b1, c2)
        
        # check that the i+j step is consistent with making i steps and then j steps
        wrapped_dims = np.random.binomial(0.5, 1, size=ndim).astype(bool)
        a, b = linear_steps_with_reflection(start, direction, 1 * 0.04, wrapped_dims=wrapped_dims)
        c, d = linear_steps_with_reflection(a, b, 1 * 0.04, wrapped_dims=wrapped_dims)
        e, f = linear_steps_with_reflection(start, direction, 2 * 0.04, wrapped_dims=wrapped_dims)
        
        # check that the length is not infinite
        wrapped_dims = np.ones(ndim, dtype=bool)
        a, b = linear_steps_with_reflection(start, direction, 1000000, wrapped_dims=wrapped_dims)
        
        

def test_forward(plot=False):
    np.random.seed(1)
    for j in range(40):
        if j % 2 == 0:
            wrapped_dims = np.array([False, False])
        else:
            wrapped_dims = None
        start = np.random.uniform(size=2)
        direction = np.random.normal(size=2)
        direction /= (direction**2).sum()**0.5
        points = []
        for i in range(100):
            newpoint, _ = linear_steps_with_reflection(start, direction, i * 0.04, wrapped_dims=wrapped_dims)
            points.append(newpoint)
        points = np.array(points)
        
        a, b = linear_steps_with_reflection(start, direction, 1 * 0.04, wrapped_dims=wrapped_dims)
        c, d = linear_steps_with_reflection(a, b, 1 * 0.04, wrapped_dims=wrapped_dims)
        e, f = linear_steps_with_reflection(start, direction, 2 * 0.04, wrapped_dims=wrapped_dims)
        assert_allclose(c, e)
        assert_allclose(d, f)
        
        np.testing.assert_allclose(points[0], start)
        if plot:
            plt.plot(start[0], start[1], 'o ')
            plt.plot(points[:,0], points[:,1], 'x-')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig('flatnuts_test_forward_%02d.png' % j, bbox_inches='tight')
            plt.close()
        assert np.isfinite(points).all(), (j, points)
        assert (points > 0).all(), (j, points)
        assert (points < 1).all(), (j, points)
        delta = ((points[1:,:] - points[:-1,:])**2).sum(axis=1)**0.5
        #print(delta.max(), delta.min(), direction)
        assert (delta <= 0.04001).all(), (j, delta, np.where(delta > 0.1), points)
    
def test_samplingpath():
    x0 = np.array([0.5, 0.5])
    v0 = np.array([0.1, 0.0])
    L0 = 0.
    path = SamplingPath(x0, v0, L0)
    assert path.interpolate(0) == (x0, v0, L0, True)
    try:
        path.interpolate(1)
        assert False
    except KeyError:
        pass
    try:
        path.interpolate(-1)
        assert False
    except KeyError:
        pass
    
    path.add(-1, x0 - v0, v0, 1.0)
    x1, v1, L1, on_path = path.interpolate(-1)
    assert_allclose(x1, x0 - v0)
    assert_allclose(v1, v0)
    assert_allclose(L1, 1.0)
    assert on_path
    
    path.add(4, x0 + 4*v0, v0, 4.0)
    x1, v1, L1, on_path = path.interpolate(1)
    assert_allclose(x1, x0 + v0)
    assert_allclose(v1, v0)
    assert L1 is None, L1
    assert on_path

def test_samplingpath_cubereflect():
    x0 = np.array([0.1, 0.1])
    v0 = np.array([0.1, 0.01])
    L0 = 0.
    path = SamplingPath(x0, v0, L0)
    path.add(-1, x0 - v0, v0, 1.0)

def test_samplingpath_oddcase():
    x0 = np.array([0.19833663, 0.49931288, 0.62744967, 0.47308545, 0.48858042, 0.49025685,
 0.48481497, 0.49068977, 0.49562456, 0.51102634] )
    v0 = np.array([-0.00053468, -0.00106889,  0.0012165,   0.00737494,  0.00152363, -0.00164736,
  0.00371493,  0.02057758, -0.00260349,  0.01266826])
    L0 = 0.
    path = SamplingPath(x0, v0, L0)
    for i in range(-10, 11):
        if i != 0:
            path.extrapolate(i)
    


def get_reflection_angles(normal, v):
    angles = (normal * (v / norm(v))).sum(axis=1)
    #mask_forward1 = angles < 0
    
    # additionally, the reverse should work:
    vnew = -(v.reshape((1, -1)) - 2 * angles.reshape((-1, 1)) * normal)
    anglesnew = (normal * (vnew / norm(vnew, axis=1).reshape((-1, 1)))).sum(axis=1)
    assert anglesnew.shape == (len(normal),), (anglesnew.shape, normal.shape)
    mask_forward = np.logical_and(angles < 0, anglesnew < 0)
    return mask_forward, angles, anglesnew

def test_reversible_gradient(plot=False):
    def loglike(x):
        x, y = x.transpose()
        return -0.5 * (x**2 + ((y - 0.5)/0.2)**2)
    def transform(u):
        return u
    Lmin = -0.5

    for i in [84] + list(range(1, 100)):
        print("setting seed = %d" % i)
        np.random.seed(i)
        points = np.random.uniform(size=(10000, 2))
        L = loglike(points)
        mask = L > Lmin
        points = points[mask,:][:100,:]
        active_u = points
        active_values = L[mask][:100]

        transformLayer = AffineLayer(wrapped_dims=[])
        transformLayer.optimize(points, points)
        region = MLFriends(points, transformLayer)
        region.maxradiussq, region.enlarge = region.compute_enlargement(nbootstraps=30)
        region.create_ellipsoid()
        nclusters = transformLayer.nclusters
        assert nclusters == 1
        assert np.allclose(region.unormed, region.transformLayer.transform(points)), "transform should be reproducible"
        assert region.inside(points).all(), "live points should lie near live points"

        if i == 84:
            v = np.array([0.03477044, -0.01977415])
            reflpoint = np.array([0.09304075, 0.29114574])
        elif i == 4:
            v = np.array([0.03949306, -0.00634806])
            reflpoint = np.array([0.9934771, 0.55358031])
            
        else:
            v = np.random.normal(size=2)
            v /= (v**2).sum()**0.5
            v *= 0.04
            j = np.random.randint(len(active_u))
            reflpoint = np.random.normal(active_u[j,:], 0.04)
            if not (reflpoint < 1).all() and not (reflpoint > 0).all():
                continue
        
        
        bpts = region.transformLayer.transform(reflpoint).reshape((1, -1))
        tt = get_sphere_tangents(region.unormed, bpts)
        t = region.transformLayer.untransform(tt * 1e-3 + region.unormed) - region.u
        # compute new vector
        normal = t / norm(t, axis=1).reshape((-1, 1))
        print("reflecting at  ", reflpoint, "with direction", v)
        mask_forward1, angles, anglesnew = get_reflection_angles(normal, v)
        if mask_forward1.any():
            j = np.argmin(((region.unormed[mask_forward1,:] - bpts)**2).sum(axis=1))
            k = np.arange(len(normal))[mask_forward1][j]
            angles_used = angles[k]
            normal_used = normal[k,:]
            print("chose normal", normal_used, k)
            #chosen_point = region.u[k,:]
            vnew = -(v - 2 * angles_used * normal_used)
            assert vnew.shape == v.shape
            
            mask_forward2, angles2, anglesnew2 = get_reflection_angles(normal, vnew)
            #j2 = np.argmin(((region.unormed[mask_forward2,:] - bpts)**2).sum(axis=1))
            #chosen_point2 = region.u[mask_forward2,:][0,:]
            #assert j2 == j, (j2, j)
            assert mask_forward2[k]
            #assert_allclose(chosen_point, chosen_point2)
        
            #for m, a, b, m2, a2, b2 in zip(mask_forward1, angles, anglesnew, mask_forward2, angles2, anglesnew2):
            #    if m != m2:
            #        print('  ', m, a, b, m2, a2, b2)
        
            #print("using normal", normal)
            #print("changed v from", v, "to", vnew)
        
            #angles2 = -(normal * (vnew / norm(vnew))).sum(axis=1)
            #mask_forward2 = angles < 0
            if plot:
                plt.figure(figsize=(5,5))
                plt.title('%d' % mask_forward1.sum())
                plt.plot((reflpoint + v)[0], (reflpoint + v)[1], '^', color='orange')
                plt.plot((reflpoint + vnew)[:,0], (reflpoint + vnew)[:,1], '^ ', color='lime')
                plt.plot(reflpoint[0], reflpoint[1], '^ ', color='r')
                plt.plot(region.u[:,0], region.u[:,1], 'x ', ms=2, color='k')
                plt.plot(region.u[mask_forward1,0], region.u[mask_forward1,1], 'o ', ms=6, mfc='None', mec='b')
                plt.plot(region.u[mask_forward2,0], region.u[mask_forward2,1], 's ', ms=8, mfc='None', mec='g')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.savefig('test_flatnuts_reversible_gradient_%d.png' % i, bbox_inches='tight')
                plt.close()
            assert mask_forward1[k] == mask_forward2[k], (mask_forward1[k], mask_forward2[k])
                
            print("reflecting at  ", reflpoint, "with direction", v)
            # make that step, then try to go back
            j = np.arange(len(normal))[mask_forward1][0]
            normal = normal[j,:]
            angles = (normal * (v / norm(v))).sum()
            v2 = v - 2 * angle(normal, v) * normal
            
            print("reflecting with", normal, "new direction", v2)
            
            #newpoint = reflpoint + v2
            #angles2 = (normal * (v2 / norm(v2))).sum()
            v3 = v2 - 2 * angle(normal, v2) * normal
            
            print("re-reflecting gives direction", v3)
            assert_allclose(v3, v)
            
            print()
            print("FORWARD:", v, reflpoint)
            samplingpath = SamplingPath(reflpoint - v, v, active_values[0])
            contourpath = ContourSamplingPath(samplingpath, region)
            normal = contourpath.gradient(reflpoint)
            if normal is not None:
                assert normal.shape == v.shape, (normal.shape, v.shape)
                
                print("BACKWARD:", v, reflpoint)
                v2 = -(v - 2 * angle(normal, v) * normal)
                normal2 = contourpath.gradient(reflpoint)
                assert_allclose(normal, normal2)
                normal2 = normal
                v3 = -(v2 - 2 * angle(normal2, v2) * normal2)
                assert_allclose(v3, v)
        
if __name__ == '__main__':
    test_forward()
    test_horizontal()
    test_corner()
    test_random()
    test_samplingpath()
    test_samplingpath_cubereflect()
    test_reversible_gradient(plot=True)
    
    import sys
    if len(sys.argv) > 1:
        # estimate how many reflections we have before we u-turn
        
        ndim = int(sys.argv[1])
        seq = []
        tseq = []
        for j in range(100):
            start = np.random.uniform(size=ndim)
            initial_direction = np.random.normal(size=ndim)
            initial_direction /= (initial_direction**2).sum()**0.5
            direction = initial_direction.copy()
            _, t_initial, _ = nearest_box_intersection_line(start, direction, fwd=True)
            t_total = 0
            for i in range(10000):
                start, t, i = nearest_box_intersection_line(start, direction, fwd=True)
                direction[i] *= -1
                t_total += t
                if (direction * initial_direction).sum() <= 0:
                    break
            seq.append(i)
            tseq.append(t_total / t_initial)
        
        # print number of reflections before u-turn and distance compared to a slice sampling distance
        # the numbers are ~ndim/2 and ~ndim
        # which means that the track is a very long coherent walk!
        print(np.mean(seq), np.mean(tseq))
    



