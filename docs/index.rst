.. xfaster documentation master file, created by
   sphinx-quickstart on Wed Feb 10 11:38:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

XFaster Documentation
=====================
XFaster is a power spectrum and parameter likelihood estimator for cosmic microwave background (CMB) data sets.
It's a hybrid of two types of estimators: pseudo-:math:`C_\ell` Monte Carlo estimators, like MASTER/PolSpice, and iterative quadratic estimators.
The full description of the pipeline, including the math and assumptions used, are in the `pipeline paper <https://arxiv.org/abs/2104.01172>`_.
These docs focus on the `code implementation <https://github.com/SPIDERCMB/xfaster>`_.

The basics of running the code are given in :ref:`Quick Start<Quick Start>`.
More details of the math can be found in :ref:`Algorithm<Algorithm>`, and a detailed review of the code is given in :ref:`Tutorial<Tutorial>`.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  quickstart
  algorithm
  notebooks/XFaster_Tutorial
  gcorr
  api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
