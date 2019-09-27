<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.wordshape" />
<meta itemprop="path" content="Stable" />
</div>

# text.wordshape

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordshape_ops.py">View
source</a>

Determine wordshape features for each input string.

``` python
text.wordshape(
    input_tensor,
    pattern,
    name=None
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`input_tensor`</b>: string `Tensor` with any shape.
*   <b>`pattern`</b>: A `tftext.WordShape` or a list of WordShapes.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

`<bool>[input_tensor.shape + pattern.shape]`: A tensor where
  `result[i1...iN, j]` is true if `input_tensor[i1...iN]` has the wordshape
  specified by `pattern[j]`.

#### Raises:

* <b>`ValueError`</b>: If `pattern` contains an unknown identifier.