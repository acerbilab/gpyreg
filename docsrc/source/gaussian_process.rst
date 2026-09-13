==================
Gaussian processes
==================
---------------------------
``gpyreg.gaussian_process``
---------------------------

This module contains the ``GP`` class, which is the main entry point for working with Gaussian processes. A class instance is initialized with specified dimension and covariance/mean/noise functions. Then, its various methods can be used to fit data, make predictions, plot the GP, etc.

``GP``
============
.. autoclass:: gpyreg.GP
    :members:
    :undoc-members:

``GP.predict`` can optionally return the latent kernel matrices between the
training and prediction inputs.  Passing
``return_cross_covariance=True`` appends a tuple in hyperparameter-sample
order to the ordinary return values.  These matrices have shape
``(N_training, N_prediction)`` and are not averaged when prediction samples
are averaged.  See the method documentation for the complete return
contract, including prior-only Gaussian processes.

``Posterior``
===================
.. autoclass:: gpyreg.gaussian_process.Posterior
    :members:
    :undoc-members:
