<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.Detokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="detokenize"/>
</div>

# text.Detokenizer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

## Class `Detokenizer`

Base class for detokenizer implementations.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

```python
detokenize(input)
```

Tokenizes the input tensor.

#### Args:

*   <b>`input`</b>: An N-dimensional UTF-8 string (or optionally integer)
    `Tensor` or `RaggedTensor`.

#### Returns:

An (N-1)-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.
