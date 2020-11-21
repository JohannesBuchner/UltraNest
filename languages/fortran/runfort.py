import numpy as np
import ctypes
from ultranest import ReactiveNestedSampler

myfortlib = ctypes.CDLL("myfortlib.so")

# define the arguments of the functions and return values
myfortlib.my_fort_transform.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_size_t)]
myfortlib.my_fort_transform.restype = None

myfortlib.my_fort_likelihood.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
myfortlib.my_fort_likelihood.restype = ctypes.c_double

def mytransformwrapper(cube):
	params = cube.copy()
	ndim = ctypes.c_size_t(cube.size)
	myfortlib.my_fort_transform(params, ctypes.pointer(ndim))
	return params

print(mytransformwrapper(np.array([0.5, 0.5])))

def mylikelihoodwrapper(params):
	ndim = ctypes.c_size_t(params.size)
	l = ctypes.c_double(0.0)
	myfortlib.my_fort_likelihood(params, ctypes.pointer(ndim), ctypes.pointer(l))
	return l.value

print(mylikelihoodwrapper(np.array([0.1, 0.2])))

paramnames = ["a", "b", "c"]
sampler = ReactiveNestedSampler(paramnames, mylikelihoodwrapper, transform=mytransformwrapper)
sampler.run()
sampler.print_results()
sampler.plot()
