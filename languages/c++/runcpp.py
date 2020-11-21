import numpy as np
import ctypes
from ultranest import ReactiveNestedSampler

mycpplib = ctypes.CDLL("mycpplib.so")

# define the arguments of the functions and return values
mycpplib.my_cpp_transform_vectorized.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_size_t, 
    ctypes.c_size_t]

mycpplib.my_cpp_likelihood_vectorized.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_size_t, 
    ctypes.c_size_t,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]


def mytransformwrapper(cube):
    params = cube.copy()
    mycpplib.my_cpp_transform_vectorized(params, params.shape[1], params.shape[0])
    return params

def mylikelihoodwrapper(params):
    l = np.zeros(len(params))
    mycpplib.my_cpp_likelihood_vectorized(params, params.shape[1], params.shape[0], l)
    return l

paramnames = ["a", "b", "c"]
sampler = ReactiveNestedSampler(paramnames, mylikelihoodwrapper, transform=mytransformwrapper, vectorized=True)
sampler.run()
sampler.print_results()
sampler.plot()
