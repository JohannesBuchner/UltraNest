import numpy as np
import tempfile
import os
from ultranest.utils import vectorize, is_affine_transform, normalised_kendall_tau_distance, make_run_dir
from ultranest.utils import distributed_work_chunk_size
from numpy.testing import assert_allclose
import pytest

def test_vectorize():
	
	def myfunc(x):
		return (x**2).sum()

	myvfunc = vectorize(myfunc)
	
	a = np.array([1.2, 2.3, 3.4])
	
	assert_allclose(np.array([myfunc(a)]), myvfunc([a]))
	b = np.array([[1.2, 2.3, 3.4], [1.2, 2.3, 3.4]])
	assert_allclose(np.array([myfunc(b[0]), myfunc(b[1])]), myvfunc(b))
	
	class FuncClass(object):
		def __call__(self, x):
			return (x**2).sum()
		def foo(self, x):
			return x
	
	mycaller = FuncClass()
	vectorize(mycaller)
	vectorize(mycaller.foo)


def test_is_affine_transform():
	na = 2**np.random.randint(1, 10)
	d = 2**np.random.randint(1, 3)
	a = np.random.uniform(-1, 1, size=(na, d))
	
	assert is_affine_transform(a, a)
	assert is_affine_transform(a, a * 2.0)
	assert is_affine_transform(a, a - 1)
	assert is_affine_transform(a, a * 10000 - 5000.)
	assert not is_affine_transform(a, a**2)

def test_tau():
	
	assert normalised_kendall_tau_distance(np.arange(400), np.arange(400)) == 0
	assert normalised_kendall_tau_distance(np.arange(2000), np.arange(2000)) == 0
	a = np.array([1, 2, 3, 4, 5])
	b = np.array([3, 4, 1, 2, 5])
	assert normalised_kendall_tau_distance(a, b) == 0.4
	i, j = np.meshgrid(np.arange(len(a)), np.arange(len(b)))
	assert normalised_kendall_tau_distance(a, b, i, j) == 0.4
	assert normalised_kendall_tau_distance(a, a, i, j) == 0
	
	try:
		normalised_kendall_tau_distance(np.arange(5), np.arange(10))
		raise Exception("expect error")
	except AssertionError:
		pass


def test_make_log_dirs():
	import shutil
	try:
		filepath = tempfile.mkdtemp()
		make_run_dir(filepath, max_run_num=3)
		assert os.path.exists(os.path.join(filepath, 'run1'))
		make_run_dir(filepath, max_run_num=3)
		assert os.path.exists(os.path.join(filepath, 'run2'))
		try:
			make_run_dir(filepath, max_run_num=3)
			assert False
		except ValueError:
			pass
	finally:
		shutil.rmtree(filepath)

@pytest.mark.parametrize("mpi_size", [1, 4, 10, 37, 53, 100, 1000, 513])
@pytest.mark.parametrize("num_live_points_missing", [0, 1, 4, 10, 17, 31, 100, 1000, 513])
def test_distributed_work_chunk_size(mpi_size, num_live_points_missing):
    processes = range(mpi_size)
    todo = [distributed_work_chunk_size(num_live_points_missing, rank, mpi_size) for rank in processes]
    assert sum(todo) == num_live_points_missing
    assert max(todo) - min(todo) in {0, 1}
