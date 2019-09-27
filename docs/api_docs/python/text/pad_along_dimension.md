<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.pad_along_dimension" />
<meta itemprop="path" content="Stable" />
</div>

# text.pad_along_dimension

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/pad_along_dimension_op.py">View
source</a>

Add padding to the beginning and end of data in a specific dimension.

``` python
text.pad_along_dimension(
    data,
    axis=-1,
    left_pad=None,
    right_pad=None,
    name=None
)
```

<!-- Placeholder for "Used in" -->

Returns a tensor constructed from `data`, where each row in dimension `axis`
is replaced by the concatenation of the left padding followed by the row
followed by the right padding.  I.e., if `L=left_pad.shape[0]` and
`R=right_pad.shape[0]`, then:

```python
result[i1...iaxis, 0:L] = left_pad
result[i1...iaxis, L:-R] = data[i0...iaxis]
result[i1...iaxis, -R:] = right_pad
```

#### Args:

*   <b>`data`</b>: `<dtype>[O1...ON, A, I1...IM]` A potentially ragged `K`
    dimensional tensor with outer dimensions of size `O1...ON`; axis dimension
    of size `A`; and inner dimensions of size `I1...IM`. I.e. `K = N + 1 + M`,
    where `N>=0` and `M>=0`.
*   <b>`axis`</b>: An integer constant specifying the axis along which padding
    is added. Negative axis values from `-K` to `-1` are supported.
*   <b>`left_pad`</b>: `<dtype>[L, I1...IM]` An `M+1` dimensional tensor that
    should be prepended to each row along dimension `axis`; or `None` if no
    padding should be added to the left side.
*   <b>`right_pad`</b>: `<dtype>[R, I1...IM]` An `M+1` dimensional tensor that
    should be appended to each row along dimension `axis`; or `None` if no
    padding should be added to the right side.
*   <b>`name`</b>: The name of this op (optional).

#### Returns:

`<dtype>[O1...ON, L + A + R, I1...IM]` A potentially ragged `K` dimensional
tensor with outer dimensions of size `O1...ON`; padded axis dimension size
`L+A+R`; and inner dimensions of size `I1...IM`. If `data` is a `RaggedTensor`,
then the returned tensor is a `RaggedTensor` with the same `ragged_rank`.
