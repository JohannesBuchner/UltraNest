from hypothesis import given
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import matplotlib.pyplot as plt
import time
import ultranest.mlfriends
from ultranest.mlfriends import pareto_front_filter, find_slope

MEPS = 1 + 1e-10
AEPS = 1e-10

def find_bounding_line_exhaustive(points):
    assert np.isfinite(points).all(), points
    indices = np.argsort(points[:,0])
    x = points[indices,0]
    y = points[indices,1]
    del points
    xmin = x[0]
    xmax = x[-1]

    yminopt = y[0]
    ymaxopt = y[0]
    Aopt = np.inf
    for j in range(len(x)):
        xj, yj = x[j], y[j]
        for i in range(j):
            assert i < j, (i, j)
            # for xi, yi in zip(x[:k], y[:k]):
            xi, yi = x[i], y[i]
            # i is left of j
            assert xi <= xj
            if yi > yj:
                # slope would be negative
                continue
            slope = (yj - yi) / (xj - xi)
            assert slope >= 0, slope
            # make interpolation 
            enclosed = y <= ((x - xi) * slope + yi) * MEPS
            # enclosed[j] = True
            if not enclosed.all():
                #print('1:', A, 'not enclosing all:', xi, yi, xj, yj, ':', x[~enclosed][0], y[~enclosed][0])
                continue
            
            ymin = (xmin - xi) * slope + yi
            ymax = (xmax - xi) * slope + yi
            A = (ymax + ymin) / 2
            #print('1:', A)
            # check if all points are contained
            if A >= 0 and A < Aopt:
                yminopt, ymaxopt, Aopt = ymin, ymax, A
            #elif A < Aopt:
            #   yminopt, ymaxopt = ymin, ymin
    #print('1: Aopt', Aopt)
    assert np.isfinite(yminopt)
    assert np.isfinite(ymaxopt)
    return yminopt, ymaxopt, xmin, xmax

def pareto_front(points):
    # pareto filtering:
    at_front = np.ones(len(points), dtype=bool)
    for xi, yi in points:
        dominated = np.logical_and(points[:,0] > xi, points[:,1] <= yi)
        at_front = np.logical_and(at_front, ~dominated)
    return at_front

def pareto_front_sorted(points):
    # pareto filtering:
    at_front = np.ones(len(points), dtype=bool)
    for i, (xi, yi) in enumerate(points):
        dominated = np.logical_and(points[i+1:,0] > xi, points[i+1:,1] <= yi)
        at_front[i+1:] = np.logical_and(at_front[i+1:], ~dominated)
        #dominated = np.logical_and(points[:,0] > xi, points[:,1] <= yi)
        #at_front = np.logical_and(at_front, ~dominated)
    return at_front

def pareto_front_sorted2(points):
    # pareto filtering:
    at_front = np.ones(len(points), dtype=bool)
    y = points[:, 1]
    #for i, yi in enumerate(y[:-1]):
    for i in range(len(y)-1):
        if not at_front[i]: continue
        #at_front[i+1:] = np.logical_and(at_front[i+1:], y[i+1:] > y[i])
        at_front[i+1:][y[i+1:] <= y[i]] = False
        #dominated = np.logical_and(points[:,0] > xi, points[:,1] <= yi)
        #at_front = np.logical_and(at_front, ~dominated)
    return at_front

def find_bounding_line_exhaustive_pareto(points, sorted=False):
    xmin = points[:,0].min()
    xmax = points[:,0].max()
    assert np.isfinite(points).all()
    points = points[pareto_front(points),:]
    if len(points) == 1:
        return points[0,1], points[0,1], xmin, xmax
    assert np.isfinite(points).all()

    if sorted:
        x = points[:,0]
        y = points[:,1]
    else:
        indices = np.argsort(points[:,0])
        x = points[indices,0]
        y = points[indices,1]
    del points
    yminopt = np.nan
    ymaxopt = np.nan
    Aopt = np.inf
    for j in range(len(x)):
        for i in range(j):
            # i is left of j
            assert x[i] <= x[j]
            # make interpolation 
            slope = (y[j] - y[i]) / (x[j] - x[i])
            ypred = (x - x[i]) * slope + y[i]
            enclosed = y <= ypred * MEPS
            #print(i, j, enclosed, y, (x - x[i]), (y[j] - y[i]) / (x[j] - x[i]), y[i])
            assert enclosed[i], (y[i], ypred[i], ypred[i] * MEPS, x, y)
            enclosed[j] = True
            #assert enclosed[j], (y[j], (x[j] - x[i]) * (y[j] - y[i]) / (x[j] - x[i]) + y[i])
            if not enclosed.all():
                continue
            
            ymin = (xmin - x[i]) * slope + y[i]
            ymax = (xmax - x[i]) * slope + y[i]
            A = (ymax + ymin) / 2
            #print('2:', A, x[i], y[i], x[j], y[j])
            # check if all points are contained
            if A >= 0 and A < Aopt:
                yminopt, ymaxopt, Aopt = ymin, ymax, A

    #print('2: Aopt:', Aopt)
    assert np.isfinite(yminopt), (yminopt, x, y)
    assert np.isfinite(ymaxopt), (ymaxopt, x, y)
    return yminopt, ymaxopt, xmin, xmax


def find_bounding_line_exhaustive_pareto_vectorized1(points, sorted=False):
    if sorted:
        sorted_points = points
    else:
        indices = np.argsort(points[:,0])
        sorted_points = points[indices,:]
    del points
    xmin, xmax = sorted_points[0,0], sorted_points[-1,0]
    
    points = sorted_points[pareto_front_sorted(sorted_points),:]
    del sorted_points
    if len(points) == 1:
        return points[0,1], points[0,1], xmin, xmax

    x = points[:,0]
    y = points[:,1]
    
    yminopt = np.nan
    ymaxopt = np.nan
    Aopt = np.inf
    for k, (xi, yi) in enumerate(points[:-1,:]):
        # i is left of j
        xj, yj = x[k+1:].reshape((-1, 1)), y[k+1:].reshape((-1, 1))
        slopes = (yj - yi) / (xj - xi)
        
        # make interpolation
        mask_enclosed = (y.reshape((1, -1)) <= MEPS * ((x.reshape((1, -1)) - xi) * slopes + yi)).all(axis=1)
        assert mask_enclosed.shape == (len(slopes),), (mask_enclosed.shape, slopes.shape)
        mask_enclosed = np.logical_and(mask_enclosed, slopes[:,0] >= 0)
        assert mask_enclosed.shape == (len(slopes),), (mask_enclosed.shape, slopes.shape)
        if not mask_enclosed.any():
            continue
        
        # xi, yi, slopes = xi[mask_enclosed,0], yi[mask_enclosed,0], slopes[mask_enclosed,0]
        slopes = slopes[mask_enclosed,0]
        
        ymin = (xmin - xi) * slopes + yi
        ymax = (xmax - xi) * slopes + yi
        A = (ymax + ymin) / 2
        # check if all points are contained
        if not (A < Aopt).any():
            continue
        i = np.argmin(A)
        assert A[i] >= 0 and A[i] < Aopt, (A[i], Aopt)
        yminopt, ymaxopt, Aopt = ymin[i], ymax[i], A[i]

    return yminopt, ymaxopt, xmin, xmax

def find_bounding_line_exhaustive_pareto_vectorized2(points, sorted=False):
    if sorted:
        sorted_points = points
    else:
        indices = np.argsort(points[:,0])
        sorted_points = points[indices,:]
    del points
    xmin, xmax = sorted_points[0,0], sorted_points[-1,0]
    
    points = sorted_points[pareto_front_sorted(sorted_points),:]
    del sorted_points
    if len(points) == 1:
        return points[0,1], points[0,1], xmin, xmax

    x = points[:,0]
    y = points[:,1]
    
    yminopt = np.nan
    ymaxopt = np.nan
    Aopt = np.inf
    for j in range(len(x)):
        for i in range(j):
            # i is left of j
            assert x[i] <= x[j]
            # make interpolation
            slope = (y[j] - y[i]) / (x[j] - x[i])
            enclosed = y <= ((x - x[i]) * slope + y[i]) * MEPS
            if not enclosed.all():
                continue
            
            ymin = (xmin - x[i]) * slope + y[i]
            ymax = (xmax - x[i]) * slope + y[i]
            A = (ymax + ymin) / 2
            # check if all points are contained
            if A >= 0 and A < Aopt:
                yminopt, ymaxopt, Aopt = ymin, ymax, A

    return yminopt, ymaxopt, xmin, xmax


find_bounding_line_exhaustive_pareto_vectorized = find_bounding_line_exhaustive_pareto_vectorized2

def orig_test_pareto(N, k, r):
    points = np.transpose([
        np.random.uniform(0, 1, size=N),
        k * np.random.uniform(0, 1, size=N) + 0.01,
    ])
    points.flags.writeable = False
    ppoints = points[pareto_front(points),:]
    sorted_ppoints = ppoints[np.argsort(ppoints[:,0]),:]
    sorted_points = points[np.argsort(points[:,0]),:]
    pspoints = sorted_points[pareto_front_sorted(sorted_points),:]
    pspoints2 = sorted_points[pareto_front_sorted2(sorted_points),:]
    pspoints3 = sorted_points[pareto_front_filter(sorted_points[:,0], sorted_points[:,1]),:]
    np.testing.assert_allclose(sorted_ppoints, pspoints)
    np.testing.assert_allclose(sorted_ppoints, pspoints2)
    np.testing.assert_allclose(sorted_ppoints, pspoints3)

#@h.settings(max_examples=5000)
@given(st.integers(min_value=2, max_value=1000), st.floats(min_value=0.0, max_value=1e5), st.random_module())
def test_pareto(N, k, r):
    return orig_test_pareto(N, k, r)

def speedtest_pareto():
    slopes = np.linspace(0, 1, 20)
    for N in [10, 40, 100, 400, 1000]:
        inputs = []
        for k in slopes:
            points = np.transpose([
                np.random.uniform(0, 1, size=N),
                k * np.random.uniform(0, 1, size=N) + 0.01,
            ])
            sorted_points = points[np.argsort(points[:,0]),:]
            inputs.append(sorted_points)

        t0 = time.time()
        for points in inputs:
            for i in range(10000 // N):
                pareto_front(points)
        t1 = time.time() - t0
        t0 = time.time()
        for points in inputs:
            for i in range(10000 // N):
                pareto_front_sorted(sorted_points)
        t2 = time.time() - t0
        t0 = time.time()
        for points in inputs:
            for i in range(100000 // N):
                pareto_front_sorted2(sorted_points)
        t3 = time.time() - t0
        t0 = time.time()
        for points in inputs:
            for i in range(100000 // N):
                pareto_front_filter(sorted_points[:,0], sorted_points[:,1])
        t4 = time.time() - t0
        print("%d %.3fs %.3fs %.3fs %.3fs (pareto filter)" % (N, t1 * 10, t2 * 10, t3, t4))

def speedtest_bounding_line():
    slopes = np.linspace(0, 1, 20)
    for N in [10, 40, 100, 400, 1000]:
        indicator = np.zeros(N, dtype=bool)
        inputs = []
        for k in slopes:
            points = np.transpose([
                np.random.uniform(0, 1, size=N),
                k * np.random.uniform(0, 1, size=N) + 0.01,
            ])
            inputs.append(points)
        results = []
        t0 = time.time()
        for points in inputs:
            results.append(find_bounding_line_exhaustive(points))
        texhaustive = time.time() - t0
        results2 = []
        t0 = time.time()
        for points in inputs:
            for i in range(100-1):
                find_bounding_line_exhaustive_pareto(points)
            results2.append(find_bounding_line_exhaustive_pareto(points))
        tpareto = time.time() - t0
        results3 = []
        t0 = time.time()
        for points in inputs:
            for i in range(100-1):
                find_bounding_line_exhaustive_pareto_vectorized1(points)
            results3.append(find_bounding_line_exhaustive_pareto_vectorized1(points))
        tparetov = time.time() - t0
        results4 = []
        t0 = time.time()
        for points in inputs:
            for i in range(100-1):
                find_bounding_line_exhaustive_pareto_vectorized2(points)
            results4.append(find_bounding_line_exhaustive_pareto_vectorized2(points))
        tparetov2 = time.time() - t0
        results5 = []
        t0 = time.time()
        for points in inputs:
            for i in range(1000-1):
                indices = np.argsort(points[:,0])
                sorted_x = points[indices,0]
                sorted_y = points[indices,1]
                find_slope(sorted_x, sorted_y, indicator)
            results5.append(find_slope(sorted_x, sorted_y, indicator)[:-1])
        tparetoc = time.time() - t0
        print("%d %.3fs %.3fs %.3fs %.3fs %.3fs (find_slope)" % (N, texhaustive, tpareto, tparetov, tparetov2, tparetoc / 10))
        for input, k, r1, r2 in zip(inputs, slopes, results, results2):
            np.testing.assert_allclose(r1, r2, err_msg="input slope: %s" % k)
        for input, k, r1, r3 in zip(inputs, slopes, results, results3):
            np.testing.assert_allclose(r1, r3, err_msg="input slope: %s" % k)
        for input, k, r1, r4 in zip(inputs, slopes, results, results4):
            np.testing.assert_allclose(r1, r4, err_msg="input slope: %s" % k)
        for input, k, r1, r5 in zip(inputs, slopes, results, results5):
            np.testing.assert_allclose(r1, r5, err_msg="input slope: %s" % k)


def speedtest_bounding_line_multid():
    print("speedtest_bounding_line_multid")
    d = 50
    use_sorted = True
    for N in [40, 100, 400, 1000]:
        indicator = np.ones(N, dtype=bool)
        slopes = np.linspace(0, 1, 2000 // N)
        inputs = []
        chunks = []
        for k in slopes:
            points = np.transpose([np.random.uniform(0, 1, size=N)] +
                [k * np.random.uniform(0, 1, size=N) + 0.01 for i in range(d)])
            sorted_points = points[np.argsort(points[:,0]),:]
            del points
            sorted_points.flags.writeable = False
            chunk = []
            for r in range(20):
                selected = np.zeros(N, dtype=bool)
                selected[np.random.randint(0, N, size=N)] = True
                #if selected.sum() > 1:
                chunk.append(sorted_points[selected,:])
            inputs += chunk
            chunks.append(chunk)
        results = []
        t0 = time.time()
        for points in inputs:
            for i in range(d):
                results.append(find_bounding_line_exhaustive_pareto(points[:,[0,1+i]], sorted=use_sorted))
        tpareto = time.time() - t0
        results2 = []
        t0 = time.time()
        for points in inputs:
            for i in range(d):
                results2.append(find_bounding_line_exhaustive_pareto_vectorized1(points[:,[0,1+i]], sorted=use_sorted))
        tparetov = time.time() - t0
        results3 = []
        t0 = time.time()
        for points in inputs:
            for i in range(d):
                results3.append(find_bounding_line_exhaustive_pareto_vectorized2(points[:,[0,1+i]], sorted=use_sorted))
        tparetov2 = time.time() - t0
        results4 = []
        t0 = time.time()
        for _ in range(10):
            for points in inputs:
                for i in range(d):
                    results4.append(find_slope(points[:,0], points[:,i+1], indicator[:len(points)])[:-1])
        tparetoc = time.time() - t0
        print("%d %.3fs %.3fs %.3fs %.3fs (find_slope multiple%s)" % (N, tpareto, tparetov, tparetov2, tparetoc / 10, ', assuming sorted' if use_sorted else ''))
        for input, k, r1, r2 in zip(inputs, slopes, results, results2):
            np.testing.assert_allclose(r1, r2, err_msg="input slope: %s" % k)
        for input, k, r1, r3 in zip(inputs, slopes, results, results3):
            np.testing.assert_allclose(r1, r3, err_msg="input slope: %s" % k)
        for input, k, r1, r4 in zip(inputs, slopes, results, results4):
            np.testing.assert_allclose(r1, r4, err_msg="input slope: %s" % k)

def orig_test_bounding_linep(N, k, r, plot):
    points = np.transpose([
        np.random.uniform(0, 1, size=N),
        k * np.random.uniform(0, 1, size=N) + 0.01,
    ])
    points.flags.writeable = False
    #print("points:", points)
    indicator = np.ones(N, dtype=bool)
    ppoints = points[pareto_front(points),:]
    sorted_ppoints = ppoints[np.argsort(ppoints[:,0]),:]
    sorted_points = points[np.argsort(points[:,0]),:]
    pspoints = sorted_points[pareto_front_sorted(sorted_points),:]
    if plot:
        plt.plot(points[:,0], points[:,1], 'o ')
        plt.plot(ppoints[:,0], ppoints[:,1], 'x ', ms=10)
        plt.plot(pspoints[:,0], pspoints[:,1], '+ ', ms=10)
        plt.savefig('test_boundingline.pdf', bbox_inches='tight')
        plt.close()
    
    r1 = find_bounding_line_exhaustive(points)
    r2 = find_bounding_line_exhaustive_pareto(points)
    r3 = find_bounding_line_exhaustive_pareto_vectorized1(points)
    r4 = find_bounding_line_exhaustive_pareto_vectorized2(points)
    r5 = find_slope(sorted_points[:,0], sorted_points[:,1], indicator)
    ymin, ymax, xmin, xmax = r1
    ymin2, ymax2, xmin2, xmax2 = r2
    ymin3, ymax3, xmin3, xmax3 = r3
    ymin4, ymax4, xmin4, xmax4 = r4
    if plot:
        plt.plot([xmin, xmax], [ymin, ymax], '--')
        plt.plot([xmin2, xmax2], [ymin2, ymax2], alpha=0.5, lw=4)
        #plt.plot([xmin3, xmax3], [ymin3, ymax3], alpha=0.5, lw=6)
        plt.plot(points[:,0], points[:,1], 'o ')
        plt.plot(ppoints[:,0], ppoints[:,1], 'x ', ms=10)
        plt.plot(pspoints[:,0], pspoints[:,1], '+ ', ms=10)
        plt.savefig('test_boundingline.pdf', bbox_inches='tight')
        plt.close()
        time.sleep(0.2)

    np.testing.assert_allclose(sorted_ppoints, pspoints)

    ypred = (sorted_points[:,0] - xmin) / (xmax - xmin) * (ymax - ymin) + ymin
    assert (ypred * MEPS >= sorted_points[:,1]).all(), (ypred, sorted_points[:,1], ypred >= sorted_points[:,1])

    ypred = (sorted_points[:,0] - xmin2) / (xmax2 - xmin2) * (ymax2 - ymin2) + ymin2
    assert (ypred * MEPS >= sorted_points[:,1]).all(), (ypred, sorted_points[:,1], ypred >= sorted_points[:,1])

    np.testing.assert_allclose(list(r1), list(r2))

    ypred = (points[:,0] - xmin3) / (xmax3 - xmin3) * (ymax3 - ymin3) + ymin3
    assert (ypred * MEPS >= points[:,1]).all(), (ypred, points[:,1])

    np.testing.assert_allclose(list(r1), list(r3))
    np.testing.assert_allclose(list(r1), list(r4))
    np.testing.assert_allclose(list(r1), list(r5)[:-1])
    assert r5[-1] == len(pspoints)

@h.settings(deadline=10000.0, max_examples=3, verbosity=h.Verbosity.verbose)
@given(st.integers(min_value=2, max_value=1000), st.floats(min_value=0.0, max_value=1e100), st.random_module())
def test_bounding_linep(N, k, r):
    return orig_test_bounding_linep(N, k, r, plot=False)

@given(st.integers(min_value=2, max_value=100), st.floats(min_value=0.0, max_value=1), st.random_module())
def test_bounding_linep(N, k, r):
    return orig_test_bounding_linep(N, k, r, plot=False)

def test_bounding_line(plot=False):
    print("test_bounding_line")
    #np.random.seed(1)
    np.random.seed(223)
    orig_test_bounding_linep(6, 29790.523779723226, 0, plot)
    np.random.seed(0)
    orig_test_bounding_linep(256, 0.0, 0, plot)
    np.random.seed(0)
    orig_test_bounding_linep(2, 1, 0, plot)
    np.random.seed(0)
    orig_test_bounding_linep(417, 0.2293582383128268, 0, plot)
    print("test_bounding_line done")

if __name__ == '__main__':
    test_bounding_line(plot=True)
    print("testing pareto filters...")
    test_pareto()
    print("testing bounding line functions...")
    test_bounding_linep()
    print("speed tests...")
    import sys
    if 'speedtest-pareto' in sys.argv[1:]:
        speedtest_pareto()
    if 'speedtest-slopes' in sys.argv[1:]:
        speedtest_bounding_line()
    if 'speedtest-slopes-multid' in sys.argv[1:]:
        speedtest_bounding_line_multid()
