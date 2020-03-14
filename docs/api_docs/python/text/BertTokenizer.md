<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.BertTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.BertTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

Tokenizer used for BERT.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

```python
text.BertTokenizer(
    vocab_lookup_table, suffix_indicator='##', max_bytes_per_word=100,
    max_chars_per_token=None, token_out_type=dtypes.int64, unknown_token='[UNK]',
    split_unknown_characters=False, lower_case=False, keep_whitespace=False,
    normalization_form=None, preserve_unused_token=False
)
```

<!-- Placeholder for "Used in" -->

This tokenizer applies an end-to-end, text string to wordpiece tokenization. It
first applies basic tokenization, and then follwed by wordpiece tokenization.

See BasicTokenizer and WordpieceTokenizer for their respective details.

#### Attributes:

*   <b>`vocab_lookup_table`</b>: A lookup table implementing the LookupInterface
    containing the vocabulary of subwords or a string which is the file path to
    the vocab.txt file.
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
*   <b>`lower_case`</b>: bool - If true, a preprocessing step is added to
    lowercase the text, apply NFD normalization, and strip accents characters.
*   <b>`keep_whitespace`</b>: bool - If true, preserves whitespace characters
    instead of stripping them away.
*   <b>`normalization_form`</b>: If true and lower_case=False, the input text
    will be normalized to `normalization_form`. See normalize_utf8() op for a
    list of valid values.
*   <b>`preserve_unused_token`</b>: If true, text in the regex format
    `\\[unused\\d+\\]` will be treated as a token and thus remain preserved as
    is to be looked up in the vocabulary.

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

```python
tokenize(
    text_input
)
```

Performs untokenized text to wordpiece tokenization for BERT.

#### Args:

*   <b>`text_input`</b>: input: A `Tensor` or `RaggedTensor` of untokenized
    UTF-8 strings.

#### Returns:

A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string contents (or
ID in the vocab_lookup_table representing that string) of the `jth` token in
`input[i1...iN]`

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

```python
tokenize_with_offsets(
    text_input
)
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
