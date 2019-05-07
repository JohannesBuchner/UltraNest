#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "mininest",
    version = "0.1",
    author = "Johannes Buchner",
    author_email = "johannes.buchner.acad@gmx.com",
    description = "Reactive Nested Sampling",
    url = "https://github.com/JohannesBuchner/mininest/",
    license = "GPL",
    packages = find_packages(),
    requires = ["matplotlib", "numpy", "scipy"],
    ext_modules = cythonize("mininest/mlfriends.pyx"),
    provides = ['mininest'],
)
