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

BLAS threading
==============

The cost of GPyReg is dominated by dense linear algebra, mainly the
Cholesky factorizations and triangular solves performed by NumPy and
SciPy. The installed BLAS/LAPACK library parallelizes these operations
across threads. The most efficient number of threads depends on the
workload and the environment. A single thread often works best for small
training sets, of a few hundred points or fewer, and avoids
oversubscription when the calling application already runs work in
parallel; several threads tend to pay off for larger training sets.
Benchmark a representative workload rather than assuming a setting.

GPyReg does not set a thread policy, leaving the choice to the user.
Depending on which library NumPy is built against, the thread count is
read from ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS`` or
``OMP_NUM_THREADS``. These variables take effect only when set in the
environment before NumPy is imported, for example with
``OPENBLAS_NUM_THREADS=1 python my_script.py`` in a POSIX shell.

The `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ package
changes the limits at runtime instead, for the duration of a ``with``
block::

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        hyp, optimization, sampling = gp.fit(X, y)

The OpenBLAS `usage notes
<https://github.com/OpenMathLib/OpenBLAS/blob/develop/USAGE.md>`_ and
`runtime variables
<https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/>`_ describe
thread control in more detail.

Documentation
=============

The primary entry point for users is the :ref:`GP Class<\`\`GP\`\`>`, used to construct and fit Gaussian process models to data. More detailed information can be found in the links below:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   release_notes
   gaussian_process
   covariance_functions
   mean_functions
   noise_functions
   slice_sample
   rng

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

License and source
==================

GPyReg is released under the terms of the :mainbranch:`BSD 3-Clause License <LICENSE>`.
The source code is on :labrepos:`GitHub <gpyreg>`.

Acknowledgments
===============

GPyReg is developed by `members <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people>`_ (past and current) of the `Machine and Human Intelligence Lab <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/>`_ at the University of Helsinki and `ELLIS Institute Finland <https://www.ellisinstitute.fi/>`_. Development of GPyReg from version 1.1 onwards has been assisted by coding agents, including Anthropic's `Claude Fable 5.1 <https://www.anthropic.com/claude-fable-and-mythos-5-1>`_ and OpenAI's `GPT-6 Astra <https://developers.openai.com/api/docs/models/gpt-6-astra>`_.
Work on the GPyReg package is supported by the Research Council of Finland (grants 356498 and 358980 to Luigi Acerbi) and its Flagship programme: `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`_.
