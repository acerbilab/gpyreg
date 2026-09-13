========================
Random number generation
========================
--------------
``gpyreg.rng``
--------------

Since version 1.1.0, the GPyReg functions that draw random numbers (``GP.fit``,
``GP.random_function``, ``SliceSampler``, ``f_min_fill``) take an ``rng``
argument resolved by the helpers below. ``rng=None`` (the default) keeps
NumPy's global legacy stream.
The legacy stream uses a stateless proxy, so copying or pickling a sampler
does not capture NumPy's global state; ``np.random.seed`` continues to
control its draws. Samplers pickled before the ``rng`` argument was added
also resume using the global stream. A sampler with an explicit generator
preserves that generator's state when copied or pickled.

Resolving a generator or an already resolved legacy proxy returns the same
object. ``GP.fit`` resolves its argument once and shares the stream between
the initial design and the sampler, including when given an integer seed.

Choosing a stream
=================

Pass an integer seed to initialize a new stream for a fit::

    gp.fit(X, y, rng=1234)

Pass a generator when several operations should consume successive draws
from the same stream::

    import numpy as np

    rng = np.random.default_rng(1234)
    gp.fit(X, y, rng=rng)
    draw = gp.random_function(X_star, rng=rng)

These examples assume ``gp`` is a GP instance, ``X`` and ``y`` are training
data, and ``X_star`` contains prediction inputs. Calls advance the supplied
generator. Passing the same integer seed to each call instead creates a new
stream each time. With ``rng=None``, use ``np.random.seed`` to initialize the
global stream; a generator passed explicitly is independent of that stream.

``resolve_rng``
---------------
.. autofunction:: gpyreg.rng.resolve_rng

``random_integer``
------------------
.. autofunction:: gpyreg.rng.random_integer
