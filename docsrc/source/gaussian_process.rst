==================
Gaussian processes
==================
---------------------------
``gpyreg.gaussian_process``
---------------------------

This module contains the ``GP`` class, which is the main entry point for working with Gaussian processes. A class instance is initialized with specified dimension and covariance/mean/noise functions. Then, its various methods can be used to fit data, make predictions, plot the GP, etc.

Reproducible fitting
====================

Since version 1.1.0, fitting accepts an ``rng`` argument. Given a GP instance
``gp`` and training arrays ``X`` and ``y``, an integer seed initializes the
stream used for both the space-filling design and hyperparameter sampling::

    hyp, optimization, sampling = gp.fit(X, y, rng=1234)

Pass a ``numpy.random.Generator`` to share a stream across calls. See
:doc:`rng` for stream ownership and the default behavior.

Reusing prediction kernels
==========================

Since version 1.2.0, :meth:`gpyreg.GP.predict` can return the latent kernel
matrices between training and prediction inputs. Given a fitted ``gp``
and an ``(M, D)`` array ``X_star``::

    mu, variance = gp.predict(X_star)
    mu, variance, kernels = gp.predict(
        X_star, return_cross_covariance=True
    )

``mu`` and ``variance`` have shape ``(M, 1)`` when predictions are averaged
across hyperparameter samples, or ``(M, S)`` with ``separate_samples=True``.
The extra ``kernels`` tuple always contains one entry for each of the ``S``
hyperparameter samples, independently of ``separate_samples``. Each entry
is ``K(X, X_star)`` with shape ``(N, M)``, where ``N`` is the training-set
size. A GP without training targets returns ``None`` for each sample.

These are unconditioned latent kernels. Observation noise is excluded even
with ``add_noise=True``. To obtain the posterior predictive covariance
between prediction points, use :meth:`gpyreg.GP.predict_full`.

With ``return_lpd=True``, the kernel tuple follows the log predictive density::

    mu, variance, lpd, kernels = gp.predict(
        X_star, y_star=y_star, return_lpd=True,
        return_cross_covariance=True
    )

Here ``y_star`` contains the observed values at the prediction points.
Treat returned kernels as read-only. Retaining all matrices requires
approximately ``8 * N * M * S`` bytes for float64 data, in addition to other
prediction allocations. GPyReg imposes no retention cap; request these
matrices only when needed and release them after use. The option defaults
to ``False``.

``GP``
============
.. autoclass:: gpyreg.GP
    :members:
    :undoc-members:

``Posterior``
===================
.. autoclass:: gpyreg.gaussian_process.Posterior
    :members:
    :undoc-members:
