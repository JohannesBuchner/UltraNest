# noqa: D400 D205
"""UltraNets performs nested sampling to calculate the Bayesian evidence and posterior samples."""

from .integrator import NestedSampler, ReactiveNestedSampler, read_file
from .utils import vectorize

__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '4.5.0'
