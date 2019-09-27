<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.SentencepieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="id_to_string"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
<meta itemprop="property" content="vocab_size"/>
</div>

# text.SentencepieceTokenizer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

## Class `SentencepieceTokenizer`

Tokenizes a tensor of UTF-8 strings.

Inherits From: [`Tokenizer`](../text/Tokenizer.md),
[`Detokenizer`](../text/Detokenizer.md)

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

```python
__init__(
    model=None,
    out_type=dtypes.int32,
    nbest_size=0,
    alpha=1.0,
    reverse=False,
    add_bos=False,
    add_eos=False,
    name=None
)
```

Creates & initializes a Sentencepiece processor.

#### Args:

*   <b>`model`</b>: The sentencepiece model serialized proto.
*   <b>`out_type`</b>: output type. tf.int32 or tf.string (Default = tf.int32)
    Setting tf.int32 directly encodes the string into an id sequence.
*   <b>`nbest_size`</b>: A scalar for sampling. nbest_size = {0,1}: No sampling
    is performed. (default) nbest_size > 1: samples from the nbest_size results.
    nbest_size < 0: assuming that nbest_size is infinite and samples from the
    all hypothesis (lattice) using forward-filtering-and-backward-sampling
    algorithm.
*   <b>`alpha`</b>: A scalar for a smoothing parameter. Inverse temperature for
    probability rescaling.
*   <b>`reverse`</b>: Reverses the tokenized sequence (Default = false)
*   <b>`add_bos`</b>: Add <s> to the result (Default = false)
*   <b>`add_eos`</b>: Add </s> to the result (Default = false) <s>/</s> is added
    after reversing (if enabled).
*   <b>`name`</b>: The name argument that is passed to the op function.

#### Returns:

*   <b>`pieces`</b>: A SentencepieceTokenizer.

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

```python
detokenize(
    input,
    name=None
)
```

Detokenizes tokens into preprocessed text.

#### Args:

*   <b>`input`</b>: A `RaggedTensor` or `Tensor` of UTF-8 string tokens with a
    rank of at least 1.
*   <b>`name`</b>: The name argument that is passed to the op function.

#### Returns:

A N-1 dimensional string Tensor or RaggedTensor of the detokenized text.

<h3 id="id_to_string"><code>id_to_string</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

```python
id_to_string(
    input,
    name=None
)
```

Converts vocabulary id into a token.

#### Args:

*   <b>`input`</b>: An arbitrary tensor of int32 representing the token IDs.
*   <b>`name`</b>: The name argument that is passed to the op function.

#### Returns:

A tensor of string with the same shape as input.

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

```python
tokenize(
    input,
    name=None
)
```

Tokenizes a tensor of UTF-8 strings.

#### Args:

*   <b>`input`</b>: A `RaggedTensor` or `Tensor` of UTF-8 strings with any
    shape.
*   <b>`name`</b>: The name argument that is passed to the op function.

#### Returns:

A `RaggedTensor` of tokenized text. The returned shape is the shape of the input
tensor with an added ragged dimension for tokens of each string.

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

```python
tokenize_with_offsets(
    input,
    name=None
)
```

Tokenizes a tensor of UTF-8 strings.

#### Args:

*   <b>`input`</b>: A `RaggedTensor` or `Tensor` of UTF-8 strings with any
    shape.
*   <b>`name`</b>: The name argument that is passed to the op function.

#### Returns:

A `RaggedTensor` of tokenized text. The returned shape is the shape of the input
tensor with an added ragged dimension for tokens of each string.

<h3 id="vocab_size"><code>vocab_size</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

```python
vocab_size(name=None)
```

Returns the vocabulary size.

#### Args:

*   <b>`name`</b>: The name argument that is passed to the op function.

#### Returns:

A scalar representing the vocabulary size.
