import numpy as np
import os
import matplotlib.pyplot as plt
from mininest.flatnuts import SamplingPath, box_line_intersection, nearest_box_intersection_line, linear_steps_with_reflection
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


def test_random():
    for i in range(100):
        start = np.random.uniform(size=2)
        direction = np.random.normal(size=2)
        direction /= (direction**2).sum()**0.5
        
        reset = np.random.binomial(1, 0.1, size=2) == 1
        direction[reset] = -start[reset]
        (c1, _, ax1), (c2, _, ax2) = box_line_intersection(start, direction)
        (b1, _, ax1), (b2, _, ax2) = box_line_intersection(start, -direction)
        np.testing.assert_allclose(c1, b2)
        np.testing.assert_allclose(b1, c2)
    #pF, tF, iF = nearest_box_intersection_line(ray_origin, ray_direction, fwd=True)
    

def test_forward(plot=False):
    np.random.seed(1)
    for j in range(40):
        start = np.random.uniform(size=2)
        direction = np.random.normal(size=2)
        direction /= (direction**2).sum()**0.5
        points = []
        for i in range(100):
            newpoint, _ = linear_steps_with_reflection(start, direction, i * 0.04)
            points.append(newpoint)
        points = np.array(points)
        np.testing.assert_allclose(points[0], start)
        if plot:
            import matplotlib.pyplot as plt
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
    
    
if __name__ == '__main__':
    test_forward()
    test_horizontal()
    test_corner()
    test_random()
    test_samplingpath()
    test_samplingpath_cubereflect()
    
    
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
    



