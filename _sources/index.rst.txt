====================
GPyReg Documentation
====================

What is it?
===========

GPyReg is a lightweight package for Gaussian process regression in Python. It was developed for use with :labrepos:`PyVBMC <pyvbmc>`, a Python package for efficient black-box Bayesian inference, but is usable as a standalone package.

Installation
============

GPyReg is not yet available on ``pip`` / ``conda-forge``, but can be installed in a few steps:

1. Clone the GitHub repo locally::

     git clone https://github.com/acerbilab/gpyreg

2. (Optional) Create a new environment in ``conda`` and activate it. We recommend using Python 3.9 or newer, but older versions *might* work::

     conda create --name gpyreg-env python=3.9
     conda activate gpyreg-env

3. Install the package::

     cd ./gpyreg
     pip install -e .

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
