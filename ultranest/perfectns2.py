"""
Deprecated alias for :mod:`ultranest.perfectns`
===============================================

The contour-parameterized problems (:class:`ContourProblem`,
:class:`GaussContour`, :class:`HyperPyramidContour`, :class:`ShellContour`,
:class:`BananaContour`, :class:`SplitEllipsoidContour`) now live in
:mod:`ultranest.perfectns` together with the other perfect-sampling
problems. This module re-exports them for backwards compatibility.
"""
from .perfectns import (
    BananaContour,
    ContourProblem,
    GaussContour,
    HyperPyramidContour,
    ShellContour,
    SplitEllipsoidContour,
    logspace_sigma,
)

__all__ = [
    'ContourProblem',
    'GaussContour',
    'HyperPyramidContour',
    'ShellContour',
    'BananaContour',
    'SplitEllipsoidContour',
    'logspace_sigma',
]
