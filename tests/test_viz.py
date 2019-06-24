import numpy as np

from mininest.viz import round_parameterlimits

def wrap_single_test(vlo, vhi, plo_expected, phi_expected):
    err_msg = 'for input values (%s, %s)' % (vlo, vhi)
    plo, phi, fmts = round_parameterlimits(np.asarray([vlo]), np.asarray([vhi]))
    np.testing.assert_allclose(plo, plo_expected, err_msg=err_msg)
    np.testing.assert_allclose(phi, phi_expected, err_msg=err_msg)

def test_rounding_pos():
    wrap_single_test(0.00003, 0.001, 0, 0.001)
    wrap_single_test(0.1, 0.9, 0, 1)
    wrap_single_test(1.5, 150, 0, 1000)
    wrap_single_test(20000, 100000, 0, 100000)

def test_rounding_u():
    # test 0-1 range
    wrap_single_test(0, 1., 0, 1)
    wrap_single_test(0.0001, 0.99, 0, 1)
    wrap_single_test(0.001, 0.99, 0, 1)
    wrap_single_test(0.01, 0.9999, 0, 1)

def test_rounding_negpos():
    wrap_single_test(-0.1, 0.9, -1, 1)
    wrap_single_test(-1.5, 150, -1000, 1000)
    wrap_single_test(-20000, 100000, -100000, 100000)

def test_rounding_withguess():
    plo, phi, fmts = round_parameterlimits(
        [-3.14, 0.01, 3000], 
        [0.9, 0.3, 100000], 
        [(-3.14, 3.14), (0, 1.0), (-2000, 10000)])
    assert np.allclose(plo, [-3.14, 0, 0]), plo
    assert np.allclose(phi, [3.14, 1, 100000]), phi

    plo, phi, fmts = round_parameterlimits(
        [1.4, 24.0, 0.4],
        [2.6, 25.5, 7.99],
        [(1, 3), (20, 26), (0.1, 8.0)])
    assert np.allclose(plo, [1, 20, 0.1]), plo
    assert np.allclose(phi, [3, 26, 8]), phi

if __name__ == '__main__':
    test_rounding_u()
    test_rounding_pos()
    test_rounding_negpos()
    test_rounding_withguess()
