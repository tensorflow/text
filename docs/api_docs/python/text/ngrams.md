<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.ngrams" />
<meta itemprop="path" content="Stable" />
</div>

# text.ngrams

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/ngrams_op.py">View
source</a>

Create a tensor of n-grams based on the input data `data`.

``` python
text.ngrams(
    data,
    width,
    axis=-1,
    reduction_type=None,
    string_separator=' ',
    name=None
)
```

<!-- Placeholder for "Used in" -->

Creates a tensor of n-grams based on `data`. The n-grams are of width `width`
and are created along axis `axis`; the n-grams are created by combining
windows of `width` adjacent elements from `data` using `reduction_type`. This
op is intended to cover basic use cases; more complex combinations can be
created using the sliding_window op.

#### Args:

*   <b>`data`</b>: The data to reduce.
*   <b>`width`</b>: The width of the ngram window. If there is not sufficient
    data to fill out the ngram window, the resulting ngram will be empty.
*   <b>`axis`</b>: The axis to create ngrams along. Note that for string join
    reductions, only axis '-1' is supported; for other reductions, any positive
    or negative axis can be used. Should be a constant.
*   <b>`reduction_type`</b>: A member of the Reduction enum. Should be a
    constant. Currently supports:

    *   <a href="../text/Reduction.md#SUM"><code>Reduction.SUM</code></a>: Add
        values in the window.
    *   <a href="../text/Reduction.md#MEAN"><code>Reduction.MEAN</code></a>:
        Average values in the window.
    *   <a href="../text/Reduction.md#STRING_JOIN"><code>Reduction.STRING_JOIN</code></a>:
        Join strings in the window. Note that axis must be -1 here.

*   <b>`string_separator`</b>: The separator string used for
    <a href="../text/Reduction.md#STRING_JOIN"><code>Reduction.STRING_JOIN</code></a>.
    Ignored otherwise. Must be a string constant, not a Tensor.

*   <b>`name`</b>: The op name.

#### Returns:

A tensor of ngrams.

#### Raises:

*   <b>`InvalidArgumentError`</b>: if `reduction_type` is either None or not a
    Reduction, or if `reduction_type` is STRING_JOIN and `axis` is not -1.
