from __future__ import print_function, division
import numpy as np
import tempfile
import os
from ultranest.store import TextPointStore, HDF5PointStore, NullPointStore
import pytest

def test_text_store():
	PointStore = TextPointStore
	try:
		fobj, filepath = tempfile.mkstemp()
		os.close(fobj)
		
		ptst = PointStore(filepath, 4)
		assert ptst.stack_empty
		assert ptst.pop(-np.inf)[1] is None, "new store should not return anything"
		assert ptst.pop(100)[1] is None, "new store should not return anything"
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.pop(-np.inf)[1] is None, "empty store should not return anything"
		assert ptst.pop(100)[1] is None, "empty store should not return anything"
		ptst.close()

		ptst = PointStore(filepath, 4)
		with pytest.raises(ValueError):
			ptst.add([-np.inf, 123, 4], 1)
			assert False, "should not allow adding wrong length"
		
		ptst = PointStore(filepath, 4)
		assert ptst.stack_empty
		ptst.add([-np.inf, 123, 413, 213], 2)
		assert ptst.stack_empty
		ptst.close()
		
		ptst = PointStore(filepath, 4)
		assert not ptst.stack_empty
		entry = ptst.pop(-np.inf)[1]
		assert entry is not None, ("retrieving entry should succeed", entry)
		assert entry[1] == 123,  ("retrieving entry should succeed", entry)
		assert ptst.pop(100)[1] is None, "other queries should return None"
		assert ptst.stack_empty
		ptst.add([101, 155, 413, 213], 3)
		assert ptst.stack_empty
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.pop(-np.inf)[1] is not None, "retrieving entry should succeed"
		assert ptst.pop(-np.inf)[1] is None, "retrieving unknown entry should fail"
		assert ptst.pop(100)[1] is None, "retrieving unknown entry should fail"
		ptst.add([99, 156, 413, 213], 4)
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.pop(-np.inf)[1] is not None, "retrieving entry should succeed"
		assert ptst.pop(-np.inf)[1] is None, "retrieving unknown entry should fail"
		print(ptst.stack)
		entry = ptst.pop(100)[1]
		assert entry is not None, ("retrieving entry should succeed", entry)
		assert entry[1] == 156, ("retrieving entry should return correct value", entry)
		ptst.close()
		
		with pytest.warns(UserWarning):
			ptst = PointStore(filepath, 3)
			assert ptst.stack_empty
			ptst.close()
			
		with pytest.warns(UserWarning):
			ptst = PointStore(filepath, 5)
			assert ptst.stack_empty
			ptst.close()
			

		
	finally:
		os.remove(filepath)

def test_hdf5_store():
	PointStore = HDF5PointStore
	try:
		fobj, filepath = tempfile.mkstemp()
		os.close(fobj)
		
		ptst = PointStore(filepath, 4)
		assert ptst.stack_empty
		assert ptst.pop(-np.inf)[1] is None, "new store should not return anything"
		assert ptst.pop(100)[1] is None, "new store should not return anything"
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.pop(-np.inf)[1] is None, "empty store should not return anything"
		assert ptst.pop(100)[1] is None, "empty store should not return anything"
		ptst.close()

		ptst = PointStore(filepath, 4)
		with pytest.raises(ValueError):
			ptst.add([-np.inf, 123, 4], 1)
			assert False, "should not allow adding wrong length"
		
		ptst = PointStore(filepath, 4)
		assert ptst.stack_empty
		ptst.add([-np.inf, 123, 413, 213], 2)
		assert ptst.stack_empty
		ptst.close()
		
		ptst = PointStore(filepath, 4)
		assert not ptst.stack_empty
		entry = ptst.pop(-np.inf)[1]
		assert entry is not None, ("retrieving entry should succeed", entry)
		assert entry[1] == 123,  ("retrieving entry should succeed", entry)
		assert ptst.pop(100)[1] is None, "other queries should return None"
		assert ptst.stack_empty
		ptst.add([101, 155, 413, 213], 3)
		assert ptst.ncalls == 3, (ptst.ncalls)
		assert ptst.stack_empty
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.ncalls == 3, (ptst.ncalls)
		assert ptst.pop(-np.inf)[1] is not None, "retrieving entry should succeed"
		assert ptst.pop(-np.inf)[1] is None, "retrieving unknown entry should fail"
		assert ptst.pop(100)[1] is None, "retrieving unknown entry should fail"
		ptst.add([99, 156, 413, 213], 4)
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.ncalls == 4, (ptst.ncalls)
		assert ptst.pop(-np.inf)[1] is not None, "retrieving entry should succeed"
		assert ptst.pop(-np.inf)[1] is None, "retrieving unknown entry should fail"
		print(ptst.stack)
		entry = ptst.pop(100)[1]
		assert entry is not None, ("retrieving entry should succeed", entry)
		assert entry[1] == 156, ("retrieving entry should return correct value", entry)
		ptst.close()
		
		with pytest.raises(IOError):
			ptst = PointStore(filepath, 3)
		
		with pytest.raises(IOError):
			ptst = PointStore(filepath, 5)

		ptst = PointStore(filepath, 4, mode='w')
		assert ptst.ncalls == 0, (ptst.ncalls)
		assert ptst.pop(-np.inf)[1] is None, "overwritten store should be empty"
		assert ptst.pop(100)[1] is None, "overwritten store should not return anything"
		ptst.close()

		
	finally:
		os.remove(filepath)


def test_nullstore():
	ptst = NullPointStore(4)
	assert ptst.stack_empty
	assert ptst.pop(-np.inf)[1] is None, "new store should not return anything"
	assert ptst.pop(100)[1] is None, "new store should not return anything"
	ptst.close()

	ptst = NullPointStore(4)
	assert ptst.pop(-np.inf)[1] is None, "empty store should not return anything"
	assert ptst.pop(100)[1] is None, "empty store should not return anything"
	ptst.close()

	ptst = NullPointStore(4)
	assert ptst.stack_empty
	# no errors even if we give rubbish input
	ptst.add([-np.inf, 123, 413, 213], 1)
	ptst.add([10, 123, 413, 213], 2)
	ptst.add([10, 123, 413, 213, 123], 3)
	ptst.add([99, 123, 413], 4)
	assert ptst.stack_empty
	ptst.close()
	
	ptst = NullPointStore(4)
	assert ptst.stack_empty
	entry = ptst.pop(-np.inf)[1]
	assert entry is None
	ptst.close()


def test_storemany():
	for PointStore in TextPointStore, HDF5PointStore:
		for N in 1, 2, 10, 100:
			print()
			print("======== %s N=%d ========" % (PointStore, N))
			print()
			try:
				fobj, filepath = tempfile.mkstemp()
				os.close(fobj)

				print("writing...")
				ptst = PointStore(filepath, 3)
				for i in range(N):
					ptst.add([-np.inf, i-0.1, i-0.1], i)
				for i in range(N):
					ptst.add([i, i+1, i+1], i+N)
					print(i, i+1, "storing:", [i, i+0.1, i+.1])
				for i in range(N):
					ptst.add([-np.inf, i-0.1, i-0.1], i+2*N)
				for i in range(N-1,-1,-1):
					ptst.add([N-i, N-i+.5, N-i+.5], (N-i)+3*N)
					print(N-i, N-i+1, "storing:", [N-i, N-i+.5, N-i+.5])
				ptst.close()
				
				print("reading...")

				ptst = PointStore(filepath, 3)
				assert ptst.ncalls == 4*N, (ptst.ncalls, N, 4*N)
				print('stack[0]:', ptst.stack)
				assert len(ptst.stack) == 4 * N
				for i in range(N):
					idx, row = ptst.pop(-np.inf)
					assert row is not None
				assert len(ptst.stack) == 3 * N
				print('stack[1]:', ptst.stack)
				for i in range(N):
					idx, row = ptst.pop(i)
					print(i, i+.1, "reading:", row)
					assert row is not None
					assert row[0] == i
					assert row[1] >= i+.1
					#assert row == i+1
				ptst.reset()
				print('stack[2]:', ptst.stack)
				assert len(ptst.stack) == 2 * N
				for i in range(N):
					idx, row = ptst.pop(-np.inf)
					assert row is not None
				print('stack[3]:', ptst.stack)
				assert len(ptst.stack) == N
				for i in range(N-1,-1,-1):
					ptst.reset()
					idx, row = ptst.pop(N-i)
					print(N-i, N-i+.1, "reading:", row)
					assert row is not None
					assert row[0] == N-i
					assert row[1] >= N-i+.1
					#assert row == i+1
				assert len(ptst.stack) == 0
				assert ptst.stack_empty
				ptst.close()
			finally:
				os.remove(filepath)
