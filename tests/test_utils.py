import numpy as np
import tempfile
import os
from ultranest.utils import vectorize
from numpy.testing import assert_allclose


def test_vectorize():
	
	def myfunc(x):
		return (x**2).sum()

	myvfunc = vectorize(myfunc)
	
	a = np.array([1.2, 2.3, 3.4])
	
	assert_allclose(np.array([myfunc(a)]), myvfunc([a]))
	b = np.array([[1.2, 2.3, 3.4], [1.2, 2.3, 3.4]])
	assert_allclose(np.array([myfunc(b[0]), myfunc(b[1])]), myvfunc(b))

