#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

extra_include_dirs = []
try:
    extra_include_dirs = [numpy.get_include()]
except:
    pass

setup(
    name = "mininest",
    version = "0.1",
    author = "Johannes Buchner",
    author_email = "johannes.buchner.acad@gmx.com",
    description = "Reactive Nested Sampling",
    url = "https://github.com/JohannesBuchner/mininest/",
    license = "GPL",
    packages = ['mininest'],
    install_requires = ['numpy'],
    requires = ["matplotlib", "numpy", "scipy", "corner"],
    ext_modules = [Extension('mininest.mlfriends', ["mininest/mlfriends.pyx"], 
        include_dirs=['.'] + extra_include_dirs)],
    provides = ['mininest'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    cmdclass={'build_ext': build_ext},
)
