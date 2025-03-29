import numpy as np
import jax.numpy as jnp
import pytest

from chunkax import chunk


def identity(x):
    return x


def add(a, b):
    return a + b


def test_identity():
    x = jnp.arange(16).reshape(4, 4)
    out = chunk(identity, sizes=2, in_axes=(-2, -1), out_axes=(-2, -1))(x)
    np.testing.assert_array_equal(out, x)


def test_add():
    a = jnp.arange(11)
    b = jnp.arange(11)
    out = chunk(add, (5,), in_axes=(0,), out_axes=(0,))(a, b)
    np.testing.assert_array_equal(out, jnp.add(a, b))


def test_axes_int():
    x = jnp.arange(16)
    out = chunk(identity, sizes=4, in_axes=-1, out_axes=-1)(x)
    np.testing.assert_array_equal(out, x)


def test_axes_list():
    x = jnp.arange(16)
    out = chunk(identity, sizes=4, in_axes=[-1], out_axes=[-1])(x)
    np.testing.assert_array_equal(out, x)


def test_wrong_in_axes():
    x = jnp.arange(16)
    with pytest.raises(ValueError, match="in_axes must be a tuple of dimension"):
        _ = chunk(identity, sizes=4, in_axes=((0,), (0,)), out_axes=0)(x)


def test_multi_args_none():
    a = jnp.arange(11)
    b = jnp.ones(5)
    out = chunk(add, (5,), in_axes=((0,), None), out_axes=(0,))(a, b)
    np.testing.assert_array_equal(out, jnp.add(a, 1))


def test_wrong_sizes():
    def shrink(x):
        return x.mean()[None]

    x = jnp.arange(16)
    with pytest.raises(ValueError, match="Input chunk size has to be equal"):
        _ = chunk(shrink, sizes=4, in_axes=0, out_axes=0)(x)
