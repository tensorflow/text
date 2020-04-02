<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.UnicodeCharTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.UnicodeCharTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 strings on Unicode character boundaries.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Detokenizer`](../text/Detokenizer.md)

```python
text.UnicodeCharTokenizer()
```

<!-- Placeholder for "Used in" -->

Resulting tokens are integers (unicode codepoints)

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

```python
detokenize(
    input, name=None
)
```

Detokenizes input codepoints (integers) to UTF-8 strings.

#### Args:

*   <b>`input`</b>: A `RaggedTensor` or `Tensor` of codepoints (ints) with a
    rank of at least 1.
*   <b>`name`</b>: The name argument that is passed to the op function.

#### Returns:

A N-1 dimensional string tensor of the detokenized text.

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

```python
tokenize(
    input
)
```

Tokenizes a tensor of UTF-8 strings on Unicode character boundaries.

Input strings are split on character boundaries using
unicode_decode_with_offsets.

#### Args:

*   <b>`input`</b>: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

#### Returns:

A `RaggedTensor` of tokenized text. The returned shape is the shape of the input
tensor with an added ragged dimension for tokens (characters) of each string.

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

```python
tokenize_with_offsets(
    input
)
```

Tokenizes a tensor of UTF-8 strings to Unicode characters.

Returned token tensors are of integer type.

#### Args:

*   <b>`input`</b>: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

#### Returns:

A tuple `(tokens, start_offsets, limit_offsets)` where:

*   `tokens`: A `RaggedTensor` of codepoints (integer type).
*   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
*   `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset.
