import numpy as np
import tempfile
import os
from mininest.store import PointStore, NullPointStore

def test_store():
	try:
		fobj, filepath = tempfile.mkstemp()
		os.close(fobj)
		
		ptst = PointStore(filepath, 4)
		assert ptst.stack_empty
		assert ptst.pop(-np.inf) is None, "new store should not return anything"
		assert ptst.pop(100) is None, "new store should not return anything"
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.pop(-np.inf) is None, "empty store should not return anything"
		assert ptst.pop(100) is None, "empty store should not return anything"
		ptst.close()

		ptst = PointStore(filepath, 4)
		try:
			ptst.add([-np.inf, 123, 4])
			assert False, "should not allow adding wrong length"
		except ValueError:
			pass
		
		ptst = PointStore(filepath, 4)
		assert ptst.stack_empty
		ptst.add([-np.inf, 123, 413, 213])
		assert ptst.stack_empty
		ptst.close()
		
		ptst = PointStore(filepath, 4)
		assert not ptst.stack_empty
		entry = ptst.pop(-np.inf)
		assert entry is not None, ("retrieving entry should succeed", entry)
		assert entry[1] == 123,  ("retrieving entry should succeed", entry)
		assert ptst.pop(100) is None, "other queries should return None"
		assert ptst.stack_empty
		ptst.add([101, 155, 413, 213])
		assert ptst.stack_empty
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.pop(-np.inf) is not None, "retrieving entry should succeed"
		assert ptst.pop(-np.inf) is None, "retrieving unknown entry should fail"
		assert ptst.pop(100) is None, "retrieving unknown entry should fail"
		ptst.add([99, 156, 413, 213])
		ptst.close()

		ptst = PointStore(filepath, 4)
		assert ptst.pop(-np.inf) is not None, "retrieving entry should succeed"
		assert ptst.pop(-np.inf) is None, "retrieving unknown entry should fail"
		print(ptst.stack)
		entry = ptst.pop(100)
		assert entry is not None, ("retrieving entry should succeed", entry)
		assert entry[1] == 156, ("retrieving entry should return correct value", entry)
		ptst.close()
		
		ptst = PointStore(filepath, 3)
		assert ptst.stack_empty
		ptst.close()
		
		ptst = PointStore(filepath, 5)
		assert ptst.stack_empty
		ptst.close()
			

		
	finally:
		os.remove(filepath)


def test_nullstore():
	ptst = NullPointStore(4)
	assert ptst.stack_empty
	assert ptst.pop(-np.inf) is None, "new store should not return anything"
	assert ptst.pop(100) is None, "new store should not return anything"
	ptst.close()

	ptst = NullPointStore(4)
	assert ptst.pop(-np.inf) is None, "empty store should not return anything"
	assert ptst.pop(100) is None, "empty store should not return anything"
	ptst.close()

	ptst = NullPointStore(4)
	assert ptst.stack_empty
	# no errors even if we give rubbish input
	ptst.add([-np.inf, 123, 413, 213])
	ptst.add([10, 123, 413, 213])
	ptst.add([10, 123, 413, 213, 123])
	ptst.add([99, 123, 413])
	assert ptst.stack_empty
	ptst.close()
	
	ptst = NullPointStore(4)
	assert ptst.stack_empty
	entry = ptst.pop(-np.inf)
	assert entry is None
	ptst.close()


