"""Random-number generation helper shared by the gpyreg functions that draw
random numbers (the slice sampler, the space-filling design of ``GP.fit``
and ``GP.random_function``)."""

import numpy as np


def resolve_rng(rng=None):
    """Return the object to draw random numbers from.

    Parameters
    ----------
    rng : None, numpy.random.Generator, int, array_like[int], SeedSequence \
or BitGenerator, optional
        ``None`` (the default everywhere in gpyreg) returns the
        ``numpy.random`` module itself, that is NumPy's global legacy stream,
        which is what gpyreg drew from before generators were supported,
        call for call. A ``numpy.random.Generator`` is returned as is, so a
        generator can be shared with the caller. Anything else is passed to
        ``numpy.random.default_rng`` and seeds a new generator.

    Returns
    -------
    rng : numpy.random.Generator or the ``numpy.random`` module
        Both expose ``random()``, ``uniform(size=...)``,
        ``standard_normal(size)`` and ``shuffle(x)`` with the same
        meaning; see :func:`random_integer` for the one method whose name
        differs.
    """
    if rng is None:
        return np.random
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def random_integer(rng, high):
    """One integer drawn uniformly from ``0, ..., high - 1`` (``randint`` on
    the legacy module, ``integers`` on a ``Generator``)."""
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(0, high))
    return int(rng.randint(0, high))
