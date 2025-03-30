# Chunkax

A JAX function transformation for applying functions to arbitrary-dimensional chunks of arrays. Fully compatible with `jit`, `vmap`, etc.

```python
from chunkax import chunk

def increment(x):
    return x + 1

apply_chunked = jit(chunk(
    increment,  # function to apply in chunks
    (32, 32),  # size of chunks
    in_axes=(-2, -1)  # dimensions to chunk over
))

out = apply_chunked(jnp.ones((B, H, W)))
assert out.shape == (B, H, W)
```

## Install

```bash
pip install git+https://github.com/alebeck/chunkax
```

## Test

```
pip install pytest
pytest
```

## More examples

If output chunk dimensions differ from input chunk dimensions:
```python
apply_chunked = chunk(
    fun, 
    chunk_sizes,
    in_axes=(-2, -1),  # input dimensions to chunk over
    out_axes=(-3, -2)  # outputs are reassembled along (-3, -2)
)
```

If transformed function has multiple arguments:
```python
apply_chunked = chunk(
    fun, 
    chunk_sizes,
    
    # chunk over first dimension of first argument,
    # don't chunk second argument (similar to static_argnums),
    # and chunk first dimension of third argument.
    in_axes=((0,), None, (0,)),
    
    # fun has only one output, so reassemble along first axis
    # (currently chunkax only supports single-output functions)
    out_axes=(0,)
)
```

The same over an image tensor of shape (B, H, W, 3), where we would like to apply some function of 128x128 pixel patches:
```python
apply_chunked = chunk(
    fun,
    (128, 128),
    # we assume that `fun` takes a second input that we don't
    # want to chunk.
    in_axes=((-2, -1), None),
    out_axes=(-2, -1)
)
```

## Chunking strategies

### `'equal'`
Default is `'equal'`, this makes all chunks equal size, and therefore the last chunk potentially overlapping the previous one if the axis' size is not divisible by the chunk size of that axis:
```
Array indices:   0 1 2 3 4 5 6 7 8 9
                 ───────────────────
Chunk 0:        [0 1 2 3]
Chunk 1:                [4 5 6 7]
Chunk 2:                    [6 7 8 9]   <- overlaps with chunk 1
```

### `'fit'`
Then there's `'fit'`, which will not produce overlaps but will lead to chunks of different sizes, in case an axis' size is not divisible by the chunk size of that axis:
```
Array indices:   0 1 2 3 4 5 6 7 8 9
                 ───────────────────
Chunk 0:        [0 1 2 3]
Chunk 1:                [4 5 6 7]
Chunk 2:                         [8 9]   <- smaller chunk
```