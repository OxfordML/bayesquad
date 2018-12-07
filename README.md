# Bayesian Quadrature

This library provides code for evaluating the integral of non-negative functions using Bayesian Quadrature, both serially and in a batched mode.

For some background on Bayesian quadrature, see:

- [Introductory slides by David Duvenaud](https://www.cs.toronto.edu/~duvenaud/talks/intro_bq.pdf), or [these slides by Roman Garnett](http://probabilistic-numerics.org/assets/pdf/nips2015_probint/roman_talk.pdf)

And for gory detail:

- [Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature, Gunter et al. 2014](https://papers.nips.cc/paper/5483-sampling-for-inference-in-probabilistic-models-with-fast-bayesian-quadrature.pdf) for details on the warped Gaussian Process model implemented here ("WSABI")
- [Batch Selection for Parallelisation of Bayesian Quadrature](https://arxiv.org/abs/1812.01553) for details on our batch selection process

## Installation

Check out this repository and run `pip install .` in the root directory of the repository (i.e. in the directory containing setup.py). You should then be able to run the example scripts.

## Documentation

Documentation is still a work in progress, but some docs are available at https://OxfordML.github.io/bayesquad

