<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.UnicodeScriptTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.UnicodeScriptTokenizer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py">View
source</a>

## Class `UnicodeScriptTokenizer`

Tokenizes a tensor of UTF-8 strings on Unicode script boundaries.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py">View
source</a>

```python
__init__(keep_whitespace=False)
```

Initializes a new instance.

#### Args:

*   <b>`keep_whitespace`</b>: A boolean that specifices whether to emit
    whitespace tokens (default `False`).

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py">View
source</a>

```python
tokenize(input)
```

Tokenizes a tensor of UTF-8 strings on Unicode script boundaries.

The strings are split when a change in the Unicode script is detected between
sequential tokens. The script codes used correspond to International Components
for Unicode (ICU) UScriptCode values. See:
http://icu-project.org/apiref/icu4c/uscript_8h.html

ICU defined whitespace characters are dropped, unless the `keep_whitespace`
option was specified at construction time.

#### Args:

*   <b>`input`</b>: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

#### Returns:

A `RaggedTensor` of tokenized text. The returned shape is the shape of the input
tensor with an added ragged dimension for tokens of each string.

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py">View
source</a>

```python
tokenize_with_offsets(input)
```

Tokenizes a tensor of UTF-8 strings on Unicode script boundaries.

The strings are split when a change in the Unicode script is detected between
sequential tokens. The script codes used correspond to International Components
for Unicode (ICU) UScriptCode values. See:
http://icu-project.org/apiref/icu4c/uscript_8h.html

ICU defined whitespace characters are dropped, unless the keep_whitespace option
was specified at construction time.

#### Args:

*   <b>`input`</b>: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

#### Returns:

A tuple `(tokens, start_offsets, limit_offsets)` where:

*   `tokens`: A `RaggedTensor` of tokenized text.
*   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
*   `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset.
