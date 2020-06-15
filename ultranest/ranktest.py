#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions and classes for treating nested sampling exploration as a tree.

The root represents the prior volume, branches and sub-branches split the volume.
The leaves of the tree are the integration tail.

Nested sampling proceeds as a breadth first graph search,
with active nodes sorted by likelihood value.
The number of live points are the number of parallel edges (active nodes to do).

Most functions receive the argument "roots", which are the
children of the tree root (main branches).

The exploration is bootstrap-capable without requiring additional
computational effort: The roots are indexed, and the bootstrap explorer
can ignore the rootids it does not know about.


"""

from __future__ import print_function, division
import numpy as np

class RankAccumulator():
    def __init__(self, nsize):
        self.histogram = np.zeros(nsize, dtype=np.uint32)
    def reset(self):
        self.histogram[:] = 0
    def expand(self, nsize):
        if nsize > self.histogram.size:
            old_hist = self.histogram
            self.histogram = np.zeros(nsize, dtype=np.uint32)
            self.histogram[:old_hist.size] = old_hist
    def __iadd__(self, rank):
        """ add rank to histogram """
        self.histogram[rank] += 1
        return self
    def __sub__(self, other):
        """ subtract two accumulators: Mann-Whitney-Wilcoxon U test: """
        U = 0.0
        n1 = self.histogram.sum()
        n2 = other.histogram.sum()
        tsum = 0
        for i, n in enumerate(self.histogram):
            # count how many time this histogram wins over the other histogram:
            nties = other.histogram[i].sum() * n
            nwins = other.histogram[:i].sum() * n
            natrank = (other.histogram[i] + n)
            tsum += natrank**3 * (1 + natrank)
            U += nties * 0.5 + nwins
        
        m_U = (n1 * n2) / 2
        #sigma_U = (n1 * n2 * (n1 + n2 + 1) / 12)**0.5
        sigma_U_corr = (n1 * n2 * (n1 + n2 + 1 - tsum / ((n1 + n2) * (n1 + n2 - 1))) / 12)**0.5
        #print(sigma_U / sigma_U_corr)
        
        return (U - m_U) / sigma_U_corr

    def __len__(self):
        return self.histogram.sum()

class DifferenceRankAccumulator():
    """
    Store ranks (1 to N), with nsize allowed to vary, for comparison
    with a random rank.
    """
    def __init__(self, N):
        self.histogram = np.zeros(N, dtype=np.uint32)
        self.ref_histogram = np.zeros(N, dtype=np.uint32)
        self.U = 0.0

    def reset(self):
        """Set all counts to zero. """
        self.histogram[:] = 0
        self.ref_histogram[:] = 0
        self.U = 0.0

    def expand(self, N):
        if N > self.histogram.size:
            old_hist = self.histogram
            self.histogram = np.zeros(N, dtype=np.uint32)
            self.histogram[:old_hist.size] = old_hist
            old_ref_hist = self.histogram
            self.ref_histogram = np.zeros(N, dtype=np.uint32)
            self.ref_histogram[:old_ref_hist.size] = old_ref_hist

    def add(self, rank, N):
        """ add rank out of N to histogram. """
        assert rank <= N, (rank, N)
        if N >= self.histogram.size:
            self.expand(N+1)
        ref_rank = np.random.randint(0, N)
        # count how often this one value from rank wins over 
        # ref_histogram including new ref_rank
        self.ref_histogram[ref_rank] += 1
        nties1 = self.ref_histogram[rank].sum()
        nwins1 = self.ref_histogram[:rank].sum()
        # also count how often the existing values win over ref_rank
        nties2 = self.histogram[ref_rank].sum()
        nwins2 = self.histogram[ref_rank+1:].sum()
        # increment U statistic
        self.U += (nties1 + nties2) * 0.5 + (nwins1 + nwins2)
        self.histogram[rank] += 1
        return self
    
    @property
    def zscore(self):
        """ Mann-Whitney-Wilcoxon U test z-score, tie-corrected. """
        n1 = self.histogram.sum()
        n2 = self.ref_histogram.sum()
        natrank = (self.ref_histogram + self.histogram)
        tsum = (natrank**3 * (1 + natrank)).sum()
        m_U = (n1 * n2) / 2
        sigma_U_corr = (n1 * n2 * (n1 + n2 + 1 - tsum / ((n1 + n2) * (n1 + n2 - 1))) / 12)**0.5
        return (self.U - m_U) / sigma_U_corr

    def __len__(self):
        return self.histogram.sum()
