import numpy as np
import tempfile
import os
from ultranest.utils import vectorize, is_affine_transform, normalised_kendall_tau_distance
from numpy.testing import assert_allclose


def test_vectorize():
	
	def myfunc(x):
		return (x**2).sum()

	myvfunc = vectorize(myfunc)
	
	a = np.array([1.2, 2.3, 3.4])
	
	assert_allclose(np.array([myfunc(a)]), myvfunc([a]))
	b = np.array([[1.2, 2.3, 3.4], [1.2, 2.3, 3.4]])
	assert_allclose(np.array([myfunc(b[0]), myfunc(b[1])]), myvfunc(b))


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
