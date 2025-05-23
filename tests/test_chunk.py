import time

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from chunkax import chunk

jax.config.update('jax_platform_name', 'cpu')


def identity(x):
    return x


def add(a, b):
    return a + b


def check_size(x):
    return jnp.ones_like(x) * x.shape[0]


def test_identity():
    x = jnp.arange(16).reshape(4, 4)
    out = chunk(identity, sizes=2, in_axes=(-2, -1), out_axes=(-2, -1))(x)
    np.testing.assert_array_equal(out, x)


def test_identity_infer_out():
    x = jnp.arange(16).reshape(4, 4)
    out = chunk(identity, sizes=2, in_axes=(-2, -1))(x)
    np.testing.assert_array_equal(out, x)


def test_identity_with_static():
    x = jnp.arange(16).reshape(4, 4)
    out = chunk(
            lambda x, s: identity(x),
            sizes=2,
            in_axes=((-2, -1), None),
            out_axes=(-2, -1)
        )(x, 'static!')
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


def test_jit():
    x = jnp.arange(16).reshape(4, 4)
    out = jax.jit(chunk(identity, sizes=2, in_axes=(-2, -1), out_axes=(-2, -1)))(x)
    np.testing.assert_array_equal(out, x)


def test_jit_with_static():
    x = jnp.arange(16).reshape(4, 4)
    out = jax.jit(
        chunk(
            lambda x, s: identity(x),
            sizes=2,
            in_axes=((-2, -1), None),
            out_axes=(-2, -1)
        ),
        static_argnums=(1,),
    )(x, 'static!')
    np.testing.assert_array_equal(out, x)


def test_strategy_equal():
    x = jnp.arange(10)
    out = chunk(check_size, sizes=4, in_axes=-1, out_axes=-1, strategy='equal')(x)
    assert (out == 4).all()


def test_strategy_fit():
    x = jnp.arange(10)
    out = chunk(check_size, sizes=4, in_axes=-1, out_axes=-1, strategy='fit')(x)
    assert (out[:8] == 4).all() and (out[8:] == 2).all()


def test_identity_batched():
    x = jnp.arange(16).reshape(4, 4)
    out = chunk(identity, sizes=2, in_axes=(-2, -1), out_axes=(-2, -1), batch_size=3)(x)
    np.testing.assert_array_equal(out, x)


def test_identity_batched_jit():
    x = jnp.arange(49).reshape(7, 7)
    out = jax.jit(chunk(identity, sizes=3, in_axes=(-2, -1), out_axes=(-2, -1), strategy='fit',
                        batch_size=3))(x)
    np.testing.assert_array_equal(out, x)


def test_strategy_fit_batched():
    x = jnp.arange(10)
    out = chunk(check_size, sizes=4, in_axes=-1, out_axes=-1, strategy='fit', batch_size=3)(x)
    assert (out[:8] == 4).all() and (out[8:] == 2).all()
