from functools import partial, wraps
from itertools import islice
from typing import Callable, Sequence
import math

import numpy as np
import jax
import jax.numpy as jnp


def _make_chunk_bounds(patch_idc, chunk_sizes, shapes, strategy: str):
    if strategy == "equal":
        low = np.minimum(patch_idc * chunk_sizes, shapes - chunk_sizes)
        high = np.minimum(low + chunk_sizes, shapes)
    elif strategy == "fit":
        low = patch_idc * chunk_sizes
        high = np.minimum(low + chunk_sizes, shapes)
    else:
        raise ValueError(f"Chunking strategy {strategy} not understood.")
    return low, high - low


@partial(jax.vmap, in_axes=(None, 0, None, None))
def _fetch_chunk(a, start, size, axes):
    start_full = [0] * a.ndim
    size_full = list(a.shape)
    for st, si, ax in zip(start, size, axes):
        start_full[ax] = st
        size_full[ax] = si
    return jax.lax.dynamic_slice(a, start_full, size_full)


def _make_out_arr_from_chunk(chunk, out_axes, chunk_sizes, in_shapes):
    if not all(-chunk.ndim <= i < chunk.ndim for i in out_axes):
        raise ValueError(f"Cannot index output of shape {chunk.shape} with {out_axes=}")
    out_shape = list(chunk.shape)
    for d, chunk_size, in_shape in zip(out_axes, chunk_sizes, in_shapes):
        if out_shape[d] != chunk_size:
            raise ValueError("Input chunk size has to be equal to output"
                             f" chunk size along chunked axes, but got {chunk_size} !="
                             f" {out_shape[d]} in axis {d}. This may be lifted in the"
                             f" future.")
        out_shape[d] = in_shape
    return jnp.zeros(out_shape, dtype=chunk.dtype)


def _batch_chunks(lows, sizes, *, n):
    """
    Special iterator that batches together up to `n` subsequent chunk
    descriptions (lows and sizes), but only if they have the same size.
    """
    batch, curr_size = [], None
    for l, s in zip(lows, sizes):
        if batch and ((s != curr_size).any() or len(batch) == n):
            yield np.stack(batch), curr_size
            batch = []
        batch.append(l)
        curr_size = s
    if batch:
        yield np.stack(batch), curr_size


def chunk(f: Callable,
          sizes: int | tuple,
          in_axes: int | tuple | Sequence[tuple] = (-1,),
          out_axes: None | int | tuple = None,
          strategy: str = 'equal',
          batch_size: int = 1,
          no_jit_under_trace: bool = False,
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
    no_jit_under_trace : bool, optional
        If `True`, will not jit the inner function when under a trace. Defaults to `False`.

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
        in_axes = tuple(in_axes)

    if out_axes is None:
        axes = {a for a in in_axes if a is not None}
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
        if not any(isinstance(e, tuple) for e in in_axes):
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

        in_shapes, i_arg = None, 0
        for i_arg_, axes in enumerate(in_axes_inner):
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
        in_shapes = np.array(in_shapes)

        if isinstance(sizes, int):
            chunk_sizes = (sizes,) * len(in_shapes)
        else:
            chunk_sizes = tuple(sizes)
            if len(chunk_sizes) != len(in_shapes):
                raise ValueError("Sizes must match the number of chunked dimensions.")

        chunk_sizes = np.array(tuple(min(*s) for s in zip(chunk_sizes, in_shapes)))

        num_patches = [math.ceil(sh / ps) for sh, ps in zip(in_shapes, chunk_sizes)]
        out = None

        f_inner = f
        # if tracing, we jit inner function once so it's not re-traced in each iteration
        is_tracer = [isinstance(a, jax.core.Tracer) for a in args]
        if not no_jit_under_trace and any(is_tracer):
            f_inner = jax.jit(f, static_argnums=[i for i, is_t in enumerate(is_tracer) if not is_t])

        if batch_size > 1:
            f_inner = jax.vmap(f_inner, [None if ax is None else 0 for ax in in_axes_inner])

        chunk_idc = np.stack(np.ndindex(*num_patches))
        chunk_lows, act_chunk_sizes = _make_chunk_bounds(chunk_idc, chunk_sizes, in_shapes, strategy)

        for lows, chunk_size in _batch_chunks(chunk_lows, act_chunk_sizes, n=batch_size):
            # 1) assemble *args_ vector containing a batch of patches for each arg/dim,
            # extracted with vmap(dynamic_slice)
            args_ = []
            for arg, axes in zip(args, in_axes_inner):
                if axes is not None:
                    arg_sliced = _fetch_chunk(arg, lows, chunk_size, axes)
                    if batch_size == 1:
                        arg_sliced = arg_sliced[0]
                    args_.append(arg_sliced)
                else:
                    args_.append(arg)

            # 2) apply to inner function vmapped over dynamic args
            out_batch = f_inner(*args_, **kwargs)
            if batch_size == 1:
                out_batch = out_batch[None]

            if out is None:
                out = _make_out_arr_from_chunk(out_batch[0], out_axes, chunk_sizes, in_shapes)

            # 3) insert all batch chunks into output array sequentially
            def insert_one(out, args):
                patch, low = args
                return jax.lax.dynamic_update_slice(out, patch, low), None
            out, _ = jax.lax.scan(insert_one, out, (out_batch, lows))

        return out

    return wrapper
