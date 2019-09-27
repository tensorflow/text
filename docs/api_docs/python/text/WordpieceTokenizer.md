<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.WordpieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.WordpieceTokenizer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

## Class `WordpieceTokenizer`

Tokenizes a tensor of UTF-8 string tokens into subword pieces.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

```python
__init__(
    vocab_lookup_table,
    suffix_indicator='##',
    max_bytes_per_word=100,
    max_chars_per_token=None,
    token_out_type=dtypes.int64,
    unknown_token='[UNK]',
    split_unknown_characters=False
)
```

Initializes the WordpieceTokenizer.

#### Args:

*   <b>`vocab_lookup_table`</b>: A lookup table implementing the LookupInterface
    containing the vocabulary of subwords.
*   <b>`suffix_indicator`</b>: (optional) The characters prepended to a
    wordpiece to indicate that it is a suffix to another subword. Default is
    '##'.
*   <b>`max_bytes_per_word`</b>: (optional) Max size of input token. Default
    is 100.
*   <b>`max_chars_per_token`</b>: (optional) Max size of subwords, excluding
    suffix indicator. If known, providing this improves the efficiency of
    decoding long words.
*   <b>`token_out_type`</b>: (optional) The type of the token to return. This
    can be `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
*   <b>`unknown_token`</b>: (optional) The value to use when an unknown token is
    found. Default is "[UNK]". If this is set to a string, and `token_out_type`
    is `tf.int64`, the `vocab_lookup_table` is used to convert the
    `unknown_token` to an integer. If this is set to `None`, out-of-vocabulary
    tokens are left as is.
*   <b>`split_unknown_characters`</b>: (optional) Whether to split out single
    unknown characters as subtokens. If False (default), words containing
    unknown characters will be treated as single unknown tokens.

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

```python
tokenize(input)
```

Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

### Example:
```python
>>> tokens = [["they're", "the", "greatest"]],
>>> tokenizer = WordpieceTokenizer(vocab, token_out_type=tf.string)
>>> tokenizer.tokenize(tokens)
[[['they', "##'", '##re'], ['the'], ['great', '##est']]]
```

#### Args:

*   <b>`input`</b>: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8
    strings.

#### Returns:

A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string contents (or
ID in the vocab_lookup_table representing that string) of the `jth` token in
`input[i1...iN]`

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

```python
tokenize_with_offsets(input)
```

Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

### Example:

```python
>>> tokens = [["they're", "the", "greatest"]],
>>> tokenizer = WordpieceTokenizer(vocab, token_out_type=tf.string)
>>> result = tokenizer.tokenize_with_offsets(tokens)
>>> result[0].to_list()  # subwords
[[['they', "##'", '##re'], ['the'], ['great', '##est']]]
>>> result[1].to_list()  # offset starts
[[[0, 4, 5], [0], [0, 5]]]
>>> result[2].to_list()  # offset limits
[[[4, 5, 7], [3], [5, 8]]]
```

#### Args:

*   <b>`input`</b>: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8
    strings.

#### Returns:

A tuple `(tokens, start_offsets, limit_offsets)` where:

*   `tokens[i1...iN, j]` is a `RaggedTensor` of the string contents (or ID in
    the vocab_lookup_table representing that string) of the `jth` token in
    `input[i1...iN]`.
*   `start_offsets[i1...iN, j]` is a `RaggedTensor` of the byte offsets for the
    start of the `jth` token in `input[i1...iN]`.
*   `limit_offsets[i1...iN, j]` is a `RaggedTensor` of the byte offsets for the
    end of the `jth` token in `input[i`...iN]`.
