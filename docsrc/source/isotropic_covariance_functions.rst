Isotropic covariance functions
==============================
------------------------------------------
``gpyreg.isotropic_covariance_functions``
------------------------------------------

The isotropic kernels use a single length scale for every input dimension, so they have two hyperparameters whatever the dimensionality. Each is implemented as a subclass of :ref:`\`\`AbstractIsotropicKernel\`\``, which fixes those two hyperparameters and the bounds recommended for them, and of the kernel of the same family in :ref:`\`\`gpyreg.covariance_functions\`\``, from which it inherits everything but ``compute``. An instance is passed to ``gpyreg.GP`` at initialization in the same way as an anisotropic one.

``AbstractIsotropicKernel``
---------------------------
.. autoclass:: gpyreg.isotropic_covariance_functions.AbstractIsotropicKernel
    :members:
    :undoc-members:
    :show-inheritance:

``MaternIsotropic``
-------------------
.. autoclass:: gpyreg.isotropic_covariance_functions.MaternIsotropic
    :members:
    :undoc-members:
    :show-inheritance:

``SquaredExponentialIsotropic``
-------------------------------
.. autoclass:: gpyreg.isotropic_covariance_functions.SquaredExponentialIsotropic
    :members:
    :undoc-members:
    :show-inheritance:
