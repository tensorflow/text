<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.normalize_utf8" />
<meta itemprop="path" content="Stable" />
</div>

# text.normalize_utf8

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/normalize_ops.py">View
source</a>

Normalizes each UTF-8 string in the input tensor using the specified rule.

``` python
text.normalize_utf8(
    input,
    normalization_form='NFKC',
    name=None
)
```

<!-- Placeholder for "Used in" -->

See http://unicode.org/reports/tr15/

#### Args:

*   <b>`input`</b>: A `Tensor` or `RaggedTensor` of type string. (Must be
    UTF-8.)
*   <b>`normalization_form`</b>: One of the following string values ('NFC',
    'NFKC', 'NFD', 'NFKD'). Default is 'NFKC'.
*   <b>`name`</b>: The name for this op (optional).

#### Returns:

A `Tensor` or `RaggedTensor` of type string, with normalized contents.
