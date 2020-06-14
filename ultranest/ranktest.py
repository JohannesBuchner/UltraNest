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


