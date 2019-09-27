<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.gather_with_default" />
<meta itemprop="path" content="Stable" />
</div>

# text.gather_with_default

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/pointer_ops.py">View
source</a>

Gather slices with `indices=-1` mapped to `default`.

``` python
text.gather_with_default(
    params,
    indices,
    default,
    name=None,
    axis=0
)
```

<!-- Placeholder for "Used in" -->

This operation is similar to `tf.gather()`, except that any value of `-1`
in `indices` will be mapped to `default`.  Example:

```python
>>> gather_with_default(['a', 'b', 'c', 'd'], [2, 0, -1, 2, -1], '_').eval()
array(['c', 'a', '_', 'c', '_'], dtype=object)
```

#### Args:

*   <b>`params`</b>: The `Tensor` from which to gather values. Must be at least
    rank `axis + 1`.
*   <b>`indices`</b>: The index `Tensor`. Must have dtype `int32` or `int64`,
    and values must be in the range `[-1, params.shape[axis])`.
*   <b>`default`</b>: The value to use when `indices` is `-1`. `default.shape`
    must be equal to `params.shape[axis + 1:]`.
*   <b>`name`</b>: A name for the operation (optional).
*   <b>`axis`</b>: The axis in `params` to gather `indices` from. Must be a
    scalar `int32` or `int64`. Supports negative indices.

#### Returns:

A `Tensor` with the same type as `param`, and with shape `params.shape[:axis] +
indices.shape + params.shape[axis + 1:]`.
