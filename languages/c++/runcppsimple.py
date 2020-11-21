import numpy as np
import ctypes
from ultranest import ReactiveNestedSampler

# this version uses one parameter vector per function call
# because function calls are expensive, the runcpp.py way is more efficient and recommended

mycpplib = ctypes.CDLL("mycpplib.so")

# define the arguments of the functions and return values
mycpplib.my_cpp_transform.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t]

mycpplib.my_cpp_likelihood.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t]
mycpplib.my_cpp_likelihood.restype = ctypes.c_double


def mytransformwrapper(cube):
	params = cube.copy()
	mycpplib.my_cpp_transform(params, params.size)
	return params

def mylikelihoodwrapper(params):
	return mycpplib.my_cpp_likelihood(params, params.size)

paramnames = ["a", "b", "c"]
sampler = ReactiveNestedSampler(paramnames, mylikelihoodwrapper, transform=mytransformwrapper)
sampler.run()
sampler.print_results()
sampler.plot()
