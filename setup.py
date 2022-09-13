#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

extra_include_dirs = ['.']
try:
    import numpy
    extra_include_dirs += [numpy.get_include()]
except:
    pass

ext_args = dict(
    include_dirs=extra_include_dirs,
    extra_compile_args=['-O3'],
    extra_link_args=['-O3'],
)


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'cython', 'matplotlib', 'corner']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Johannes Buchner",
    author_email='johannes.buchner.acad@gmx.com',
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Fit and compare complex models reliably and rapidly. Advanced Nested Sampling.",
    install_requires=requirements,
    ext_modules = cythonize([
        Extension('ultranest.mlfriends', ["ultranest/mlfriends.pyx"], 
            **ext_args),
        Extension('ultranest.stepfuncs', ["ultranest/stepfuncs.pyx"], 
            **ext_args),
    ]),
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ultranest',
    name='ultranest',
    packages=['ultranest'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JohannesBuchner/ultranest',
    version='3.5.5',
    zip_safe=False,
    cmdclass={'build_ext': build_ext},
)
