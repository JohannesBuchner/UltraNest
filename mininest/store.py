import numpy as np
import operator
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
		self.stack = sorted(self.stack + self.data, key=lambda e: e[1][0])
		self.stack_empty = len(self.stack) == 0
		self.data = []
		assert self.stack_empty == (len(self.stack) == 0), (self.stack_empty, len(self.stack))
		print("PointStore: have %d items" % len(self.stack))
	
	def close(self):
		self.fileobj.close()
	def flush(self):
		self.fileobj.flush()
	
	def pop(self, Lmin):
		"""
		Request from the storage a point sampled from <= Lmin with L > Lmin
		
		Returns the point if exists, None otherwise
		"""
		while not self.stack_empty:
			row_Lmin = self.stack[0][1][0]
			if row_Lmin > Lmin:
				# the stored Lmin is above the request, so we do not 
				# have anything useful to offer at the moment
				#print("PointStore: next point is yet to come %.3f -> %.3f, need %.3f" % (self.stack[0][0], self.stack[0][1], Lmin))
				return None, None
			elif row_Lmin <= Lmin:
				# The stored arc is sampled from below
				idx, row = self.stack.pop(0)
				self.stack_empty = len(self.stack) == 0
				row_L = row[1]
				if row_L > Lmin:
					# and goes above the required limit
					#print("PointStore: popping & using   , %d left" % len(self.stack), row_L)
					return idx, row
				else:
					# but does not go above the required limit
					# we removed the point.
					#print("PointStore: popping & skipping, %d left" % len(self.stack), "(%.3f -> %.3f, need %.3f)" % (row[0], row[1], Lmin))
					self.data.append((idx, row))
					continue
		#print("PointStore: all points used up, none left")
		return None, None



class TextPointStore(FilePointStore):
	"""
	stores previously drawn points above some likelihood contour, 
	so that they can be reused in another run.
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
		self.fileobj = open(filepath, 'a')
		self.fmt = '%.18e'
		self.delimiter = '\t'
	
	def _load(self, filepath):
		"""
		Load from data file
		"""
		stack = []
		self.data = []
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
			self.data = list(enumerate(stack))
		self.stack = []
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
	stores previously drawn points above some likelihood contour, 
	so that they can be reused in another run.
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
		self.data = []
		if 'points' not in self.fileobj:
			print("PointStore: creating points dataset")
			self.fileobj.create_dataset('points', shape=(0, self.ncols), 
				maxshape=(None, self.ncols), dtype=np.float)
		
		self.nrows, ncols = self.fileobj['points'].shape
		assert ncols == self.ncols
		points = self.fileobj['points'][:]
		idx = np.argsort(points[:,0])
		self.data = list(enumerate(points[idx,:]))
		self.stack = []
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
	

