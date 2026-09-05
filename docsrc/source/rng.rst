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
The legacy stream uses a stateless proxy, so copying or pickling a sampler
does not capture NumPy's global state; ``np.random.seed`` continues to
control its draws. Samplers pickled before the ``rng`` argument was added
also resume using the global stream. A sampler with an explicit generator
preserves that generator's state when copied or pickled.

Resolving a generator or an already resolved legacy proxy returns the same
object. ``GP.fit`` resolves its argument once and shares the stream between
the initial design and the sampler, including when given an integer seed.

``resolve_rng``
---------------
.. autofunction:: gpyreg.rng.resolve_rng

``random_integer``
------------------
.. autofunction:: gpyreg.rng.random_integer
