Covariance functions
====================
-------------------------------
``gpyreg.covariance_functions``
-------------------------------

Each covariance function is implemented as a subclass of :ref:`\`\`AbstractKernel\`\``, which defines their basic interface. An instance of one of these classes is passed to ``gpyreg.GP`` at initialization and defines the type of covariance function for that Gaussian process.

``AbstractKernel``
------------------
.. autoclass:: gpyreg.covariance_functions.AbstractKernel
    :members:
    :undoc-members:

``RationalQuadraticARD``
------------------------
.. autoclass:: gpyreg.covariance_functions.RationalQuadraticARD
    :members:
    :undoc-members:
    :show-inheritance:

``Matern``
----------
.. autoclass:: gpyreg.covariance_functions.Matern
    :members:
    :undoc-members:
    :show-inheritance:

``SquaredExponential``
----------------------
.. autoclass:: gpyreg.covariance_functions.SquaredExponential
    :members:
    :undoc-members:
    :show-inheritance:
