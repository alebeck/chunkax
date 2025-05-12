from functools import wraps, partial
from itertools import groupby
from typing import Callable, Sequence, NamedTuple
import math

import numpy as np
import jax
import jax.numpy as jnp


def chunk(f: Callable,
          sizes: int | tuple,
          in_axes: int | tuple | Sequence[tuple] = (-1,),
          out_axes: None | int | tuple = None,
          strategy: str = 'equal',
          batch_size: int = 1,
          ) -> Callable:
    """
    Returns a new function that applies `fun` to sliced (chunked) segments of the
    input arrays along specified axes, then reassembles the partial outputs into a
    single array.

    Parameters
    ----------
    f : Callable
        The function to be applied on each chunk. It should accept the same positional
        and keyword arguments as the returned function, but on smaller slices.
    sizes : int or tuple
        The size(s) of each chunk along the axes in `in_axes`. If a single int is
        provided, it is repeated for each axis in `in_axes[0]`.
    in_axes : int or tuple or Sequence[tuple], optional
        The dimension indices along which inputs are chunked. If a single tuple is
        provided, it applies to every input array; otherwise, it should match the
        number of input arrays. `None` entries indicate no chunking for that dimension.
        Defaults to `(-1,)`.
    out_axes : None or int or tuple, optional
        The dimension indices along which the output chunks are placed. If `None`,
        reuses the value for `in_axes` if it's the same for all arguments.
        Defaults to `None`.
    strategy : str, optional
        The chunking strategy to use. Can be 'equal' or 'fit'. Defaults to 'equal'.
    batch_size : int
        Specify how many chunks should be processed in parallel at a time, parallelized via `vmap`.

    Returns
    -------
    Callable
        A wrapped version of `fun` that splits inputs into chunks, applies `fun` on each
        chunk, and combines the resulting patch outputs into a single array.

    Notes
    -----
    - Currently the transformation only works on functions whose output has the same size
      as the input along specified `in_axes` and `out_axes`, e.g. this is fine:
        (B, H, W, 3) -> (B, 10, H, W) with in_axes=(-3, -2) and out_axes=(-2, -1)
    """

    if isinstance(in_axes, int):
        in_axes = (in_axes,)
    else:
        # in case it is a list or other sequence
        in_axes = tuple(in_axes)

    in_axes_short = not any(isinstance(e, tuple) for e in in_axes)

    if out_axes is None:
        axes = (in_axes,) if in_axes_short else in_axes
        axes = {a for a in axes if a is not None}
        if len(axes) == 1:
            out_axes = axes.pop()
        else:
            raise ValueError("out_axes can only be None if non-None entries of in_axes "
                             f"are all equal, but got {axes}.")
    elif isinstance(out_axes, int):
        out_axes = (out_axes,)
    else:
        out_axes = tuple(out_axes)

    @wraps(f)
    def wrapper(*args, **kwargs):
        if in_axes_short:
            # single tuple (e.g., (0, 1)), repeat to number of arguments
            in_axes_inner = (in_axes,) * len(args)
        elif len(in_axes) != len(args):
            raise ValueError("in_axes must be a tuple of dimension indices or a tuple "
                             "of such tuples corresponding to the positional arguments "
                             f"passed to the function, but got {len(in_axes)=}, {len(args)=}")
        else:
            in_axes_inner = in_axes

        if len({len(t) for t in in_axes_inner if t is not None}) != 1:
            raise ValueError("All in_axes entries must have the same length.")

        in_shapes = _in_shapes_from_args(args, in_axes_inner)

        if isinstance(sizes, int):
            chunk_sizes = (sizes,) * len(in_shapes)
        else:
            chunk_sizes = tuple(sizes)
            if len(chunk_sizes) != len(in_shapes):
                raise ValueError("Sizes must match the number of chunked dimensions.")

        # check that chunk sizes are not larger than input shapes
        chunk_sizes = tuple(min(*s) for s in zip(chunk_sizes, in_shapes))

        num_patches = [math.ceil(sh / ps) for sh, ps in zip(in_shapes, chunk_sizes)]

        is_vmapped = batch_size > 1
        f_inner = f
        if is_vmapped:
            f_inner = jax.vmap(f_inner, [None if ax is None else 0 for ax in in_axes_inner])

        chunk_coords = _chunk_coordinates(num_patches)
        chunk_lows, act_chunk_sizes = _chunk_bounds(chunk_coords, chunk_sizes, in_shapes, strategy)
        batched_lows_and_sizes = _batch_chunks(chunk_lows, act_chunk_sizes, n=batch_size)

        context = _Context(
            f_inner, args, kwargs, in_shapes, chunk_sizes, in_axes_inner, out_axes, is_vmapped)

        # if tracing, we jit inner function once so it's not re-traced in each iteration
        is_tracing = any(isinstance(a, jax.core.Tracer) for a in args)
        process_fn = _process_traced if is_tracing else _process_eager

        return process_fn(batched_lows_and_sizes, context)

    return wrapper


class _Context(NamedTuple):
    apply_fn: Callable
    fn_args: Sequence
    fn_kwargs: dict
    in_shapes: tuple
    chunk_sizes: tuple
    in_axes: tuple
    out_axes: tuple
    is_vmapped: bool


def _in_shapes_from_args(args, in_axes):
    in_shapes, i_arg = None, 0
    for i_arg_, axes in enumerate(in_axes):
        if axes is None:
            continue
        shapes_ = tuple(np.array(args[i_arg_].shape)[list(axes)].tolist())
        if in_shapes is not None and in_shapes != shapes_:
            raise ValueError("Corresponding chunked dimensions of multiple input arrays "
                             f"must have equal shape, but got {in_shapes} and {shapes_} "
                             f"for arguments {i_arg} and {i_arg_}. This may be lifted "
                             "in the future.")
        in_shapes, i_arg = shapes_, i_arg_

    if in_shapes is None:
        raise ValueError("Not all in_axes elements can be None.")

    return in_shapes


def _chunk_coordinates(shape):
    return np.indices(shape).reshape(len(shape), -1).T


def _chunk_bounds(chunk_coords, chunk_sizes, shapes, strategy: str):
    chunk_sizes = np.array(chunk_sizes)
    if strategy == "equal":
        low = np.minimum(chunk_coords * chunk_sizes, shapes - chunk_sizes)
        high = np.minimum(low + chunk_sizes, shapes)
    elif strategy == "fit":
        low = chunk_coords * chunk_sizes
        high = np.minimum(low + chunk_sizes, shapes)
    else:
        raise ValueError(f"Chunking strategy {strategy} not understood.")
    return low, high - low


@partial(jax.jit, static_argnums=(2, 3), donate_argnums=0)
def _fetch_chunk(a, start, size, axes):
    start_full = [0] * a.ndim
    size_full = list(a.shape)
    for st, si, ax in zip(start, size, axes):
        start_full[ax] = st
        size_full[ax] = si
    return jax.lax.dynamic_slice(a, start_full, size_full)


_fetch_chunk_batched = jax.vmap(_fetch_chunk, in_axes=(None, 0, None, None))


def _slice_args(args, lows, sizes, in_axes):
    """
    Slices out chunks of an input argument list according to chunk bounds and axes description.
    """
    sliced = []
    fetch_fn = _fetch_chunk_batched if lows.ndim > 1 else _fetch_chunk
    for arg, axes in zip(args, in_axes):
        if axes is not None:
            arg_sliced = fetch_fn(arg, lows, sizes, axes)
            sliced.append(arg_sliced)
        else:
            sliced.append(arg)
    return sliced


def _init_out_array(shape, dtype, c: _Context):
    ndim = len(shape)
    if not all(-ndim <= i < ndim for i in c.out_axes):
        raise ValueError(f"Cannot index output of {shape=} with out_axes={c.out_axes}")
    out_shape = list(shape)
    for d, chunk_size, in_shape in zip(c.out_axes, c.chunk_sizes, c.in_shapes):
        if out_shape[d] != chunk_size:
            raise ValueError("Input chunk size has to be equal to output"
                             f" chunk size along chunked axes, but got {chunk_size} !="
                             f" {out_shape[d]} in axis {d}. This may be lifted in the"
                             f" future.")
        out_shape[d] = in_shape
    return jnp.zeros(out_shape, dtype=dtype)


def _batch_chunks_old(lows, sizes, *, n):
    """
    Special iterator that batches together up to `n` subsequent chunk
    descriptions (lows and sizes), but only if they have the same size.
    """
    batch, curr_size = [], None
    for l, s in zip(lows, sizes):
        s = tuple(s)
        if batch and (s != curr_size or len(batch) == n):
            yield np.stack(batch), curr_size
            batch = []
        batch.append(l)
        curr_size = s
    if batch:
        yield np.stack(batch), curr_size


def _batch_chunks(lows, sizes, *, n):
    """
    Special iterator that batches together up to `n` subsequent chunk
    descriptions (lows and sizes), but only if they have the same size.
    """
    sizes = [tuple(s) for s in sizes]
    lows_and_sizes = sorted(zip(lows, sizes), key=lambda x: x[1], reverse=True)  # big chunks first

    batch, curr_size = [], None
    for l, s in lows_and_sizes:
        if batch and (s != curr_size or len(batch) == n):
            yield np.stack(batch), curr_size
            batch = []
        batch.append(l)
        curr_size = s
    if batch:
        yield np.stack(batch), curr_size


def _process_eager(batched_lows_and_sizes, c: _Context):
    out = None
    for lows, size in batched_lows_and_sizes:
        out_batch = _forward_batch(lows, size, c)
        if out is None:
            out = _init_out_array(out_batch[0].shape, out_batch[0].dtype, c)

        # insert all batch chunks into output array sequentially, this spares us
        # of the jax.lax.scan overhead in eager mode.
        out = _update_batch(out, lows, out_batch)
    return out


def _process_traced(batched_lows_and_sizes, c: _Context):
    # group by batch size & chunk size so we can dispatch one lax.scan operation
    # per group of same-sized batches
    grouped_batched_lows_and_sizes = groupby(
        batched_lows_and_sizes, lambda x: (len(x[0]), x[1]))

    def process_group(out, batched_lows, size: tuple):
        def update_batch(out, scan_args):
            (lows,) = scan_args
            out_batch = _forward_batch(lows, size, c)
            return _update_batch(out, lows, out_batch), None
        return jax.lax.scan(update_batch, out, (np.array(batched_lows),))[0]

    # initialize output array based on traced output shape of dummy input
    forward_fn = partial(_forward_batch, size=c.chunk_sizes, c=c)
    dummy_lows = np.zeros((1, len(c.chunk_sizes)), dtype=np.int32)
    dummy_out = jax.eval_shape(forward_fn, dummy_lows)
    out = _init_out_array(dummy_out.shape[1:], dummy_out.dtype, c)

    for (_, size), batched_lows in grouped_batched_lows_and_sizes:
        # size has to be static, so we dispatch N scan operations over batches,
        # where N is the number of distinct chunk sizes.
        batched_lows = np.array([b[0] for b in batched_lows])
        out = process_group(out, batched_lows, size)

    return out


def _forward_batch(lows, size, c: _Context):
    # 1) assemble *args_ vector containing a batch of patches for each arg/dim,
    # extracted with vmap(dynamic_slice)
    if not c.is_vmapped:
        lows = lows[0]
    sliced_args = _slice_args(c.fn_args, lows, size, c.in_axes)

    # 2) apply to inner function vmapped over dynamic args
    out_batch = c.apply_fn(*sliced_args, **c.fn_kwargs)
    if not c.is_vmapped:
        out_batch = out_batch[None]

    return out_batch


@partial(jax.jit, donate_argnums=(0,))
def _update_batch(out, lows, batches):
    def insert_one(out, args):
        patch, low = args
        return jax.lax.dynamic_update_slice(out, patch, low), None

    return jax.lax.scan(insert_one, out, (batches, lows))[0]
