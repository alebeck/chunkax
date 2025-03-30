from functools import wraps
from typing import Callable, Sequence
import math

import numpy as np
import jax.numpy as jnp


def chunk(fun: Callable,
          sizes: int | tuple,
          in_axes: int | tuple | Sequence[tuple] = (-1,),
          out_axes: None | int | tuple = None,
          ) -> Callable:
    """
    Returns a new function that applies `fun` to sliced (chunked) segments of the
    input arrays along specified axes, then reassembles the partial outputs into a
    single array.

    Parameters
    ----------
    fun : Callable
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
        reuses the value for `in_axes` for the first argument. Defaults to `None`.

    Returns
    -------
    Callable
        A wrapped version of `fun` that splits inputs into chunks, applies `fun` on each
        chunk, and combines the resulting patch outputs into a single array.

    Notes
    -----
    - By default, the last chunk in a dimension will also be sized as in `sizes`, which
      creates overlaps if the array’s shape is not a multiple of `sizes`. Adjust logic
      if smaller final chunks or different stitching behavior is desired.
    - Currently the transformation only works on functions whose output has the same size
      as the input along specified `in_axes` and `out_axes`, e.g. this is fine:
        (B, H, W, 3) -> (B, 10, H, W) with in_axes=(-3, -2) and out_axes=(-2, -1)
    """

    if isinstance(in_axes, int):
        in_axes = (in_axes,)
    else:
        in_axes = tuple(in_axes)

    if out_axes is None:
        out_axes = in_axes[0]
    elif isinstance(out_axes, int):
        out_axes = (out_axes,)
    else:
        out_axes = tuple(out_axes)

    @wraps(fun)
    def wrapper(*args, **kwargs):
        if not isinstance(in_axes[0], tuple):
            # single tuple, extend to number of arguments
            in_axes_inner = (in_axes,) * len(args)
        elif len(in_axes) != len(args):
            raise ValueError("in_axes must be a tuple of dimension indices or a tuple "
                             "of such tuples corresponding to the positional arguments "
                             f"passed to the function, but got {len(in_axes)=}, {len(args)=}")
        else:
            in_axes_inner = in_axes

        if len({len(t) for t in in_axes_inner if t is not None}) != 1:
            raise ValueError("All in_axes entries must have the same length.")

        shapes, i_arg = None, 0
        for i_arg_, axes in enumerate(in_axes_inner):
            if axes is None:
                continue
            shapes_ = tuple(np.array(args[i_arg_].shape)[list(axes)].tolist())
            if shapes is not None and shapes != shapes_:
                raise ValueError("Corresponding chunked dimensions of multiple input arrays "
                                 f"must have equal shape, but got {shapes} and {shapes_} "
                                 f"for arguments {i_arg} and {i_arg_}. This may be lifted "
                                 "in the future.")
            shapes, i_arg = shapes_, i_arg_

        if shapes is None:
            raise ValueError("Not all in_axes elements can be None.")

        if isinstance(sizes, int):
            sizes_inner = (sizes,) * len(shapes)
        else:
            sizes_inner = tuple(sizes)
            if len(sizes_inner) != len(shapes):
                raise ValueError("Sizes must match the number of chunked dimensions.")

        sizes_inner = tuple(min(*s) for s in zip(sizes_inner, shapes))

        num_patches = [math.ceil(sh / ps) for sh, ps in zip(shapes, sizes_inner)]
        out = None

        for patch_idc in np.ndindex(*num_patches):
            mins = [min(pi * sizes_inner[i], shapes[i] - sizes_inner[i]) for i, pi in enumerate(patch_idc)]
            maxs = [min(min_ + sizes_inner[i], shapes[i]) for i, min_ in enumerate(mins)]

            args_ = []
            for arg, axes in zip(args, in_axes_inner):
                indexes = [slice(None) for _ in range(arg.ndim)]
                if axes is not None:
                    for d, min_, max_ in zip(axes, mins, maxs):
                        indexes[d] = slice(min_, max_)
                args_.append(arg[tuple(indexes)])

            out_ = fun(*args_, **kwargs)

            if out is None:
                # initialize output array based on output patch shape
                if not all(-out_.ndim <= i < out_.ndim for i in out_axes):
                    raise ValueError(f"Cannot index output of shape {out_.shape} "
                                     f"with {out_axes=}")
                out_shape = list(out_.shape)
                for d, size, shape in zip(out_axes, sizes_inner, shapes):
                    if out_shape[d] != size:
                        raise ValueError("Input chunk size has to be equal to output"
                                         f" chunk size, but got {size} != {out_shape[d]}"
                                         f" in axis {d}. This may be lifted in the future.")
                    out_shape[d] = shape
                out = jnp.zeros(out_shape, dtype=out_.dtype)

            indexes = [slice(None) for _ in range(out_.ndim)]
            for d, min_, max_ in zip(out_axes, mins, maxs):
                indexes[d] = slice(min_, max_)
            out = out.at[tuple(indexes)].set(out_)

        return out

    return wrapper
