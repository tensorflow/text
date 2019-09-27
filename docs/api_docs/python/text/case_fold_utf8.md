<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.case_fold_utf8" />
<meta itemprop="path" content="Stable" />
</div>

# text.case_fold_utf8

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/normalize_ops.py">View
source</a>

Applies case folding to every UTF-8 string in the input.

``` python
text.case_fold_utf8(
    input,
    name=None
)
```

<!-- Placeholder for "Used in" -->

The input is a `Tensor` or `RaggedTensor` of any shape, and the resulting output
has the same shape as the input. Note that NFKC normalization is implicitly
applied to the strings.

#### For example:

```python
>>> case_fold_utf8(['The   Quick-Brown',
...                 'CAT jumped over',
...                 'the lazy dog  !!  ']
tf.Tensor(['the   quick-brown' 'cat jumped over' 'the lazy dog  !!  '],
          shape=(3,), dtype=string)
```

#### Args:

*   <b>`input`</b>: A `Tensor` or `RaggedTensor` of UTF-8 encoded strings.
*   <b>`name`</b>: The name for this op (optional).

#### Returns:

A `Tensor` or `RaggedTensor` of type string, with case-folded contents.
