#!/usr/bin/env python

from setuptools import setup

setup(name='bayesquad',
      version='0.1',
      description='Bayesian Quadrature Library',
      author='Ed Wagstaff',
      author_email='ed@robots.ox.ac.uk',
      url='https://github.com/OxfordML/bayesquad',
      packages=['bayesquad'],
      install_requires=['scipy', 'numpy', 'matplotlib', 'GPy', 'multimethod'],
      python_requires='>=3.5'
      )
