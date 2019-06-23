import logging
import sys
import os
import numpy as np
import errno


def create_logger(module_name, log_dir=None, level=logging.INFO):
    logger = logging.getLogger(module_name)
    if log_dir is not None and logger.handlers == []:
        # create file handler which logs even debug messages
        handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
        formatter = logging.Formatter('[{}] [%(levelname)s] %(message)s'.format(module_name))
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    elif logger.handlers == [] and False:
        # if it is new, register to write to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('[{}] [%(levelname)s] %(message)s'.format(module_name))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def make_run_dir(log_dir, run_num=None, append_run_num=True):
    """Generates a new numbered directory for this run to store output"""
    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if run_num is None or run_num == '':
        run_num = (sum(os.path.isdir(os.path.join(log_dir,i))
                      for i in os.listdir(log_dir)) + 1)
    if append_run_num:
        run_dir = os.path.join(log_dir, 'run%s' % run_num)
    else:
        run_dir = log_dir
    if not os.path.isdir(run_dir):
        print('Creating directory for new run %s' % run_dir)
        os.makedirs(run_dir)
    if not os.path.isdir(os.path.join(run_dir, 'info')):
        os.makedirs(os.path.join(run_dir, 'info'))
        os.makedirs(os.path.join(run_dir, 'results'))
        os.makedirs(os.path.join(run_dir, 'chains'))
        os.makedirs(os.path.join(run_dir, 'extra'))
        os.makedirs(os.path.join(run_dir, 'plots'))

    return {'run_dir': run_dir,
            'info': os.path.join(run_dir, 'info'),
            'results': os.path.join(run_dir, 'results'),
            'chains': os.path.join(run_dir, 'chains'),
            'extra': os.path.join(run_dir, 'extra'),
            'plots': os.path.join(run_dir, 'plots')
            }

SQRTEPS = (float(np.finfo(np.float64).eps))**0.5

def resample_equal(samples, weights, rstate=None):
    """Resample the samples so that the final samples all have equal weight.

    Each input sample appears in the output array either
    `floor(weights[i] * N)` or `ceil(weights[i] * N)` times, with
    `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray`
        Unequally weight samples returned by the nested sampling algorithm.
        Shape is (N, ...), with N the number of samples.

    weights : `~numpy.ndarray`
        Weight of each sample. Shape is (N,).
    
    Returns
    -------
    equal_weight_samples : `~numpy.ndarray`
        Samples with equal weights, same shape as input samples.

    Examples
    --------

    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> nestle.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in
    `this PDF <http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf>`_.
    Another way to sample according to weights would be::

        N = len(weights)
        new_samples = samples[np.random.choice(N, size=N, p=weights)]

    However, the method used in this function is less "noisy".

    """

    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("weights do not sum to 1")

    if rstate is None:
        rstate = np.random

    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (rstate.random() + np.arange(N)) / N

    idx = np.zeros(N, dtype=np.int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


