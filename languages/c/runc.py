import numpy as np
import ctypes
from ultranest import ReactiveNestedSampler

myclib = ctypes.CDLL("mylib.so")  

# define the arguments of the functions and return values
myclib.my_c_transform.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t]

myclib.my_c_likelihood.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t]
myclib.my_c_likelihood.restype = ctypes.c_double


def mytransformwrapper(cube):
	params = cube.copy()
	myclib.my_c_transform(params, params.size)
	return params

def mylikelihoodwrapper(params):
	return myclib.my_c_likelihood(params, params.size)

paramnames = ["a", "b", "c"]
sampler = ReactiveNestedSampler(paramnames, mylikelihoodwrapper, transform=mytransformwrapper)
sampler.run()
sampler.print_results()
sampler.plot()
