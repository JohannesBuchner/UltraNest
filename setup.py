#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from setuptools import find_packages, setup
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = "mininest",
    version = "0.1",
    author = "Johannes Buchner",
    author_email = "johannes.buchner.acad@gmx.com",
    description = "Reactive Nested Sampling",
    url = "https://github.com/JohannesBuchner/mininest/",
    license = "GPL",
    packages = ['mininest'],
    requires = ["matplotlib", "numpy", "scipy"],
    ext_modules = [Extension('mininest.mlfriends', ["mininest/mlfriends.pyx"], include_dirs=['.'])],
    provides = ['mininest'],
    cmdclass={'build_ext': build_ext},
    script_args=['build_ext'],
    options={'build_ext':{'inplace':True, 'force':False}}
)
