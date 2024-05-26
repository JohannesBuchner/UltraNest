# noqa: D400 D205
"""
Performs nested sampling to calculate the Bayesian evidence and posterior samples

Some ellipsoid code is adopted from the Nestle library by Kyle Barbary (https://github.com/kbarbary/nestle)
Some of the architecture and parallelisation is adopted from the nnest library by Adam Moss (https://github.com/adammoss/nnest)
Some visualisations are adopted from the dynesty library by Josh Speagle (https://github.com/joshspeagle/dynesty/)
"""

from .integrator import NestedSampler, ReactiveNestedSampler, read_file
from .utils import vectorize


__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '4.3.1'
