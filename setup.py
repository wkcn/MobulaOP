#!/usr/bin/env python

from setuptools import find_packages, setup, Extension
exec(open('./mobula/version.py').read())

setup(
    name="MobulaOP",
    version=__version__,
    url='https://github.com/wkcn/mobulaop',
    description='A Simple & Flexible Cross Framework Operators Toolkit',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'numpy',
        'portalocker',
    ],
)
