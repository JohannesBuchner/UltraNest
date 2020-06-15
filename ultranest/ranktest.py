#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This implements the same idea as https://arxiv.org/abs/2006.03371
except their KS test is invalid because the variable (insertion rank)
is not continuous. Instead, this implements a Mann-Whitney-Wilcoxon
U test, which also is in practice more sensitive than the KS test.
A highly efficient implementation is achieved by keeping only
a histogram of the insertion ranks and comparing those
to a histogram of uniformly randomly generated ranks.

To quantify the convergence of a run, one route is to apply this test
at the end of the run. Another approach is to reset the counters every
time the test exceeds a z-score of 3 sigma, and report the run lengths,
which quantify how many iterations nested sampling was able to proceed
without detection of a insertion rank problem.

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
