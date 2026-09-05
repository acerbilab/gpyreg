========================
Random number generation
========================
--------------
``gpyreg.rng``
--------------

The functions of gpyreg that draw random numbers (``GP.fit``,
``GP.random_function``, ``SliceSampler``, ``f_min_fill``) take an ``rng``
argument resolved by the helpers below. ``rng=None`` (the default) keeps
NumPy's global legacy stream, as before generators were supported.

``resolve_rng``
---------------
.. autofunction:: gpyreg.rng.resolve_rng

``random_integer``
------------------
.. autofunction:: gpyreg.rng.random_integer
