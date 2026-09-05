import copy
import pickle

import numpy as np
import pytest

from gpyreg.rng import random_integer, resolve_rng


@pytest.mark.parametrize("rng", [None, np.random.default_rng(123)])
def test_resolve_rng_is_idempotent(rng):
    resolved = resolve_rng(rng)
    assert resolve_rng(resolved) is resolved


@pytest.mark.parametrize("serialization", ["none", "pickle", "deepcopy"])
def test_legacy_rng_forwards_draws_and_global_state(serialization):
    state = np.random.get_state()
    try:
        rng = resolve_rng()
        if serialization == "pickle":
            rng = pickle.loads(pickle.dumps(rng))
        elif serialization == "deepcopy":
            rng = copy.deepcopy(rng)
        assert resolve_rng(rng) is rng

        def draws(stream):
            shuffled = np.arange(24).reshape(6, 4)
            stream.shuffle(shuffled.T)
            return (
                shuffled,
                stream.random(),
                stream.random((2, 3)),
                stream.uniform(-2, 3, size=(3, 2)),
                stream.standard_normal((2, 4)),
                stream.randint(2, 9, size=4),
                random_integer(stream, 7),
            )

        np.random.seed(345)
        expected = draws(np.random)
        expected_state = np.random.get_state()
        np.random.seed(345)
        actual = draws(rng)
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
        assert all(
            np.array_equal(a, b)
            for a, b in zip(np.random.get_state(), expected_state)
        )
    finally:
        np.random.set_state(state)
