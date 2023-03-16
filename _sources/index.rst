====================
GPyReg Documentation
====================

What is it?
===========

GPyReg is a lightweight package for Gaussian process regression in Python. It was developed for use with :labrepos:`PyVBMC <pyvbmc>`, a Python package for efficient black-box Bayesian inference, but is usable as a standalone package.

Installation
============

GPyReg is available via ``pip`` and ``conda-forge``::

     python -m pip install gpyreg

or::

     conda install --channel=conda-forge gpyreg

GPyReg requires Python version 3.9 or newer.

Documentation
=============

The primary entry point for users is the :ref:`GP Class<\`\`GP\`\`>`, used to construct and fit Gaussian process models to data. More detailed information can be found in the links below:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gaussian_process
   covariance_functions
   mean_functions
   noise_functions
   slice_sample

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License and source
==================

GPyReg is released under the terms of the :mainbranch:`BSD 3-Clause License <LICENSE>`.
The source code is on :labrepos:`GitHub <gpyreg>`.

Acknowledgments
===============

GPyReg was developed by `members <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people>`_ (past and current) of the `Machine and Human Intelligence Lab <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/>`_ at the University of Helsinki. Development is being supported by the `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`_.
