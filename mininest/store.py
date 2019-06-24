"""

Storage for nested sampling points.

The information stored is a table with 
- the likelihood threshold drawn from
- the likelihood, prior volume coordinates and physical coordinates of the point

"""


import numpy as np
import warnings
import os

class Point(object):
	def __init__(self, u, p, L):
		self.u = u
		self.p = p
		self.L = L

class NullPointStore(object):
	"""
	No storage
	"""
	def __init__(self, ncols):
		self.ncols = int(ncols)
		self.nrows = 0
		self.stack_empty = True
	
	def reset(self):
		pass

	def close(self):
		pass

	def flush(self):
		pass
	
	def add(self, row):
		self.nrows += 1
		return self.nrows - 1
	
	def pop(self, Lmin):
		return None, None

class FilePointStore(object):
	def reset(self):
		"""
		Reset stack to loaded data. Useful when Lmin is not reset to a 
		lower value
		"""
		#self.stack = sorted(self.stack + self.data, key=lambda e: (e[1][0], e[0]))
		self.stack_empty = len(self.stack) == 0
		#print("PointStore: have %d items" % len(self.stack))
	
	def close(self):
		self.fileobj.close()
	def flush(self):
		self.fileobj.flush()
	
	def pop(self, Lmin):
		"""
		Request from the storage a point sampled from <= Lmin with L > Lmin
		
		Returns the point if exists, None otherwise
		"""
		
		if self.stack_empty: 
			return None, None
		
		# look forward to see if there is an exact match
		# if we do not use the exact matches
		#   this causes a shift in the loglikelihoods
		for i, (idx, next_row) in enumerate(self.stack):
			row_Lmin = next_row[0]
			L = next_row[1]
			if row_Lmin <= Lmin and L > Lmin:
				idx, row = self.stack.pop(i)
				self.stack_empty = self.stack == []
				return idx, row
		
		self.stack_empty = len(self.stack) == 0
		return None, None
		
class TextPointStore(FilePointStore):
	"""
	Stores previously drawn points above some likelihood contour, 
	so that they can be reused in another run.
	
	Format is a TST text file.
	With fmt and delimiter the output can be altered.
	"""
	def __init__(self, filepath, ncols):
		"""
		Load and append to storage at filepath, which should contain
		ncols columns (Lmin, L, and others)
		"""
		
		self.ncols = int(ncols)
		self.nrows = 0
		self.stack_empty = True
		self._load(filepath)
		self.fileobj = open(filepath, 'ab')
		self.fmt = '%.18e'
		self.delimiter = '\t'
	
	def _load(self, filepath):
		"""
		Load from data file
		"""
		stack = []
		if os.path.exists(filepath):
			try:
				for line in open(filepath):
					try:
						parts = [float(p) for p in line.split()]
						if len(parts) != self.ncols:
							warnings.warn("skipping lines in '%s' with different number of columns" % (filepath))
							continue
						stack.append(parts)
					except ValueError:
						warnings.warn("skipping unparsable line in '%s'" % (filepath))
			except IOError:
				pass
		
		self.stack = list(enumerate(stack))
		self.reset()
	
	def add(self, row):
		"""
		Add data point row = [Lmin, L, *otherinfo] to storage
		"""
		if len(row) != self.ncols:
			raise ValueError("expected %d values, got %d in %s" % (self.ncols, len(row), row))
		np.savetxt(self.fileobj, [row], fmt=self.fmt, delimiter=self.delimiter)
		self.nrows += 1
		return self.nrows - 1
	

class HDF5PointStore(FilePointStore):
	"""
	Stores previously drawn points above some likelihood contour, 
	so that they can be reused in another run.
	
	Format is a HDF5 file, which grows.
	"""
	def __init__(self, filepath, ncols):
		"""
		Load and append to storage at filepath, which should contain
		ncols columns (Lmin, L, and others)
		"""
		import h5py
		self.ncols = int(ncols)
		self.stack_empty = True
		self.fileobj = h5py.File(filepath)
		self._load()
	
	def _load(self):
		"""
		Load from data file
		"""
		if 'points' not in self.fileobj:
			self.fileobj.create_dataset('points', shape=(0, self.ncols), 
				maxshape=(None, self.ncols), dtype=np.float)
		
		self.nrows, ncols = self.fileobj['points'].shape
		assert ncols == self.ncols
		points = self.fileobj['points'][:]
		self.stack = list(enumerate(points))
		self.reset()
	
	def add(self, row):
		"""
		Add data point row = [Lmin, L, *otherinfo] to storage
		"""
		if len(row) != self.ncols:
			raise ValueError("expected %d values, got %d in %s" % (self.ncols, len(row), row))
		
		# make space:
		self.fileobj['points'].resize(self.nrows + 1, axis=0)
		# insert:
		self.fileobj['points'][self.nrows,:] = row
		self.nrows += 1
		return self.nrows - 1
	

