API Reference
=============

batch_tools
-----------
.. automodule:: xfaster.batch_tools
  :members:
  :undoc-members:

  Tools for submitting to HPC systems: PBS or SLURM

parse_tools
-----------
.. automodule:: xfaster.parse_tools
  :members:
  :undoc-members:

  Tools for parsing XFasters outputs and intermediate data structures

xfaster_class
-------------
.. automodule:: xfaster.xfaster_class
  :members:
  :undoc-members:

  The main class where the algorithmic steps are defined

xfaster_exec
------------
.. automodule:: xfaster.xfaster_exec
  :members:
  :undoc-members:

  The module containing the master function that performs all the algorithm steps in order, and optionally
  submits the job to a grid system

spec_tools
----------
.. automodule:: xfaster.spec_tools
  :members:
  :undoc-members:

  More generic functions used by methods in :py:mod:`~xfaster.xfaster_class`

gcorr_tools
-----------
.. automodule:: xfaster.gcorr_tools
  :members:
  :undoc-members:

  Tools for iteratively computing corrections to the ``g_ell`` factor
