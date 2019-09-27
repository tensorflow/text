<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.WhitespaceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.WhitespaceTokenizer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/whitespace_tokenizer.py">View
source</a>

## Class `WhitespaceTokenizer`

Tokenizes a tensor of UTF-8 strings on whitespaces.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/whitespace_tokenizer.py">View
source</a>

```python
tokenize(input)
```

Tokenizes a tensor of UTF-8 strings on whitespaces.

The strings are split on ICU defined whitespace characters. These whitespace
characters are dropped.

#### Args:

*   <b>`input`</b>: A `RaggedTensor` or `Tensor` of UTF-8 strings with any
    shape.

#### Returns:

A `RaggedTensor` of tokenized text. The returned shape is the shape of the input
tensor with an added ragged dimension for tokens of each string.

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/whitespace_tokenizer.py">View
source</a>

```python
tokenize_with_offsets(input)
```

Tokenizes a tensor of UTF-8 strings on whitespaces.

The strings are split on ICU defined whitespace characters. These whitespace
characters are dropped.

#### Args:

*   <b>`input`</b>: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

#### Returns:

A tuple `(tokens, start_offsets, limit_offsets)` where:

*   `tokens`: A `RaggedTensor` of tokenized text.
*   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
*   `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset.
