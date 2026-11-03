"""
Snowballing nested sampling
---------------------------

The :py:class:`SnowballingNestedSampler` advances threads (lineages) of
live points in batch advances, buries dead points through the burial
gate once every live point sits above their thresholds, and integrates
them with 1/K strips. Each snowball rotation widens K by doubling the
threads with fresh prior draws.

Completed dead points are written once to a growing HDF5 store; the
per-rotation derived records and the plots are written alongside.
"""
from .integrator import SnowballingNestedSampler
from .state import SnowballState, DeadPointStore, MemoryPointStore

__all__ = [
    'SnowballingNestedSampler',
    'SnowballState', 'DeadPointStore', 'MemoryPointStore',
]
