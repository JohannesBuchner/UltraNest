import numpy as np
import tempfile
import os
from ultranest.utils import vectorize, is_affine_transform
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
