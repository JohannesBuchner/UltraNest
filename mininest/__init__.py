"""
Performs nested sampling to calculate the Bayesian evidence and posterior samples
Some parts are from the Nestle library by Kyle Barbary (https://github.com/kbarbary/nestle)
Some parts are from the nnest library by Adam Moss (https://github.com/adammoss/nnest)
"""

from __future__ import print_function
from __future__ import division

from .integrator import NestedSampler, ReactiveNestedSampler



