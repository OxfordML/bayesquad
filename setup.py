#!/usr/bin/env python

from distutils.core import setup

setup(name='Bayesian Quadrature',
      version='0.1',
      description='Bayesian Quadrature Library',
      author='Ed Wagstaff',
      author_email='ed@robots.ox.ac.uk',
      url='https://github.com/OxfordML/bayesquad',
      packages=['bayesquad'],
      requires=['scipy', 'numpy', 'matplotlib', 'GPy', 'multimethod']
      )
