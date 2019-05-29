import numpy as np
import operator
import warnings
import os


class NullPointStore(object):
	"""
	No storage
	"""
	def __init__(self, ncols):
		self.ncols = int(ncols)
		self.stack_empty = True
	
	def reset(self):
		pass

	def close(self):
		pass

	def flush(self):
		pass
	
	def add(self, row):
		pass
	
	def pop(self, Lmin):
		return None
		

class PointStore(object):
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
			print("loading ...")
			try:
				for line in open(filepath):
					try:
						parts = [float(p) for p in line.split()]
						if len(parts) != self.ncols:
							warnings.warn("skipping lines in '%s' with different number of columns" % (filepath))
							continue
						stack.append(parts)
					except ValueError as e:
						warnings.warn("skipping unparsable line in '%s'" % (filepath))
			except IOError as e:
				pass
			self.data = sorted(stack, key=operator.itemgetter(0))
			print("loading done ...")
		self.reset()
	
	def reset(self):
		"""
		Reset stack to loaded data. Useful when Lmin is not reset to a 
		lower value
		"""
		self.stack = list(self.data)
		self.stack_empty = len(self.stack) == 0
		assert self.stack_empty == (len(self.stack) == 0), (self.stack_empty, len(self.stack))
		#print("PointStore: have %d items" % len(self.stack))
	
	def close(self):
		self.fileobj.close()
	def flush(self):
		self.fileobj.flush()
	
	def add(self, row):
		"""
		Add data point row = [Lmin, L, *otherinfo] to storage
		"""
		if len(row) != self.ncols:
			raise ValueError("expected %d values, got %d in %s" % (self.ncols, len(row), row))
		np.savetxt(self.fileobj, [row], fmt=self.fmt, delimiter=self.delimiter)
	
	def pop(self, Lmin):
		"""
		Request from the storage a point sampled from <= Lmin with L > Lmin
		
		Returns the point if exists, None otherwise
		"""
		while not self.stack_empty:
			row_Lmin = self.stack[0][0]
			if row_Lmin > Lmin:
				# the stored Lmin is above the request, so we do not 
				# have anything useful to offer at the moment
				#print("PointStore: next point is yet to come %.1f -> %.1f, need %.1f" % (self.stack[0][0], self.stack[0][0], Lmin))
				return None
			elif row_Lmin <= Lmin:
				# The stored arc is sampled from below
				row = self.stack.pop(0)
				self.stack_empty = len(self.stack) == 0
				row_L = row[1]
				if row_L > Lmin:
					# and goes above the required limit
					#print("PointStore: popping & using   , %d left" % len(self.stack))
					return row
				else:
					# but does not go above the required limit
					# we removed the point.
					#print("PointStore: popping & skipping, %d left" % len(self.stack))
					continue
		#print("PointStore: all points used up, none left")
		return None
		
	

