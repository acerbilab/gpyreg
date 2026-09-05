"""Random-number generation helper shared by the gpyreg functions that draw
random numbers (the slice sampler, the space-filling design of ``GP.fit``
and ``GP.random_function``)."""

import numpy as np


class _LegacyRNG:
    """Picklable, stateless access to NumPy's current global random stream."""

    def random(self, *args, **kwargs):
        return np.random.random(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return np.random.uniform(*args, **kwargs)

    def standard_normal(self, *args, **kwargs):
        return np.random.standard_normal(*args, **kwargs)

    def shuffle(self, *args, **kwargs):
        return np.random.shuffle(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return np.random.randint(*args, **kwargs)


_LEGACY_RNG = _LegacyRNG()


def resolve_rng(rng=None):
    """Return the object to draw random numbers from.

    Parameters
    ----------
    rng : None, numpy.random.Generator, int, array_like[int], SeedSequence \
or BitGenerator, optional
        ``None`` (the default everywhere in gpyreg) returns a picklable,
        stateless proxy for NumPy's global legacy stream, forwarding draws
        call for call. Copying or pickling the proxy does not capture the
        global random state. A ``numpy.random.Generator`` or an already
        resolved proxy is returned as is, so streams can be shared with
        the caller. Anything else is passed to ``numpy.random.default_rng``
        and seeds a new generator.

    Returns
    -------
    rng : numpy.random.Generator or legacy stream proxy
        Both expose ``random()``, ``uniform(size=...)``,
        ``standard_normal(size)`` and ``shuffle(x)`` with the same
        meaning; see :func:`random_integer` for the one method whose name
        differs.
    """
    if rng is None:
        return _LEGACY_RNG
    if isinstance(rng, (np.random.Generator, _LegacyRNG)):
        return rng
    return np.random.default_rng(rng)


def random_integer(rng, high):
    """One integer drawn uniformly from ``0, ..., high - 1`` (``randint`` on
    the legacy proxy, ``integers`` on a ``Generator``)."""
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(0, high))
    return int(rng.randint(0, high))
