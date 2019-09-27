<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.TokenizerWithOffsets" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.TokenizerWithOffsets

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

## Class `TokenizerWithOffsets`

Base class for tokenizer implementations that return offsets.

Inherits From: [`Tokenizer`](../text/Tokenizer.md)

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

```python
tokenize(input)
```

Tokenizes the input tensor.

#### Args:

*   <b>`input`</b>: An N-dimensional UTF-8 string (or optionally integer)
    `Tensor` or `RaggedTensor`.

#### Returns:

An N+1-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

```python
tokenize_with_offsets(input)
```

Tokenizes the input tensor and returns the result with offsets.

#### Args:

*   <b>`input`</b>: An N-dimensional UTF-8 string (or optionally integer)
    `Tensor` or `RaggedTensor`.

#### Returns:

A tuple `(tokens, start_offsets, limit_offsets)` where:

*   `tokens` is an N+1-dimensional UTF-8 string or integer `Tensor` or
    `RaggedTensor`.
*   `start_offsets` is an N+1-dimensional integer `Tensor` or `RaggedTensor`
    containing the starting indices of each token (byte indices for input
    strings).
*   `limit_offsets` is an N+1-dimensional integer `Tensor` or `RaggedTensor`
    containing the exclusive ending indices of each token (byte indices for
    input strings).
