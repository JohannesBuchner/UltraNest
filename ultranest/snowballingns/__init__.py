"""
Snowballing nested sampling
---------------------------

Burial-gate incremental integrator with HDF5-backed completed data.

Threads (lineages) of live points are advanced in batch advances; the
nested sampling integrator is a separate observer advancing in threshold
order at its own speed, burying dead points once every live point sits
above them. Completed dead points are written once to a growing HDF5
store and re-integrated in the next snowball rotation as its front rises
past their thresholds.
"""
from .state import SnowballState, DeadPointStore, MemoryPointStore

__all__ = ['SnowballState', 'DeadPointStore', 'MemoryPointStore']
