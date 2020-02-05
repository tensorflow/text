<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.SplitMergeTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.SplitMergeTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_tokenizer.py">View
source</a>

## Class `SplitMergeTokenizer`

Tokenizes a tensor of UTF-8 string into words according to labels.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

<!-- Placeholder for "Used in" -->


## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_tokenizer.py">View
source</a>

```python
tokenize(
    input,
    labels,
    force_split_at_break_character=True
)
```

Tokenizes a tensor of UTF-8 strings according to labels.

### Example:
```python
>>> strings = ["HelloMonday", "DearFriday"],
>>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
>>> tokenizer = SplitMergeTokenizer()
>>> tokenizer.tokenize(strings, labels)
[['Hello', 'Monday'], ['Dear', 'Friday']]
```

#### Args:

*   <b>`input`</b>: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8
    strings.
*   <b>`labels`</b>: An (N+1)-dimensional `Tensor` or `RaggedTensor` of int32,
    with labels[i1...iN, j] being the split(0)/merge(1) label of the j-th
    character for input[i1...iN]. Here split means create a new word with this
    character and merge means adding this character to the previous word.
*   <b>`force_split_at_break_character`</b>: bool indicates whether to force
    start a new word after seeing a ICU defined whitespace character. When
    seeing one or more ICU defined whitespace character: -if
    force_split_at_break_character is set true, then create a new word at the
    first non-space character, regardless of the label of that character, for
    instance input="New York", labels=[0, 1, 1, 0, 1, 1, 1, 1] output
    tokens=["New", "York"] input="New York", labels=[0, 1, 1, 1, 1, 1, 1, 1]
    output tokens=["New", "York"] input="New York", labels=[0, 1, 1, 1, 0, 1, 1,
    1] output tokens=["New", "York"]

    -otherwise, whether to create a new word or not for the first non-space
    character depends on the label of that character, for instance input="New
    York", labels=[0, 1, 1, 0, 1, 1, 1, 1] output tokens=["NewYork"] input="New
    York", labels=[0, 1, 1, 1, 1, 1, 1, 1] output tokens=["NewYork"] input="New
    York", labels=[0, 1, 1, 1, 0, 1, 1, 1] output tokens=["New", "York"]

#### Returns:

A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string contents (or
ID in the vocab_lookup_table representing that string) of the `jth` token in
`input[i1...iN]`

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_tokenizer.py">View
source</a>

```python
tokenize_with_offsets(
    input,
    labels,
    force_split_at_break_character=True
)
```

Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

### Example:

```python
>>> strings = ["HelloMonday", "DearFriday"],
>>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
>>> tokenizer = SplitMergeTokenizer()
>>> result = tokenizer.tokenize_with_offsets(strings, labels)
>>> result[0].to_list()
[['Hello', 'Monday'], ['Dear', 'Friday']]
>>> result[1].to_list()
>>> [[0, 5], [0, 4]]
>>> result[2].to_list()
>>> [[5, 11], [4, 10]]
```

#### Args:

*   <b>`input`</b>: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8
    strings.
*   <b>`labels`</b>: An (N+1)-dimensional `Tensor` or `RaggedTensor` of int32,
    with labels[i1...iN, j] being the split(0)/merge(1) label of the j-th
    character for input[i1...iN]. Here split means create a new word with this
    character and merge means adding this character to the previous word.
*   <b>`force_split_at_break_character`</b>: bool indicates whether to force
    start a new word after seeing a ICU defined whitespace character. When
    seeing one or more ICU defined whitespace character: -if
    force_split_at_break_character is set true, then create a new word at the
    first non-space character, regardless of the label of that character, for
    instance input="New York", labels=[0, 1, 1, 0, 1, 1, 1, 1] output
    tokens=["New", "York"] input="New York", labels=[0, 1, 1, 1, 1, 1, 1, 1]
    output tokens=["New", "York"] input="New York", labels=[0, 1, 1, 1, 0, 1, 1,
    1] output tokens=["New", "York"]

    -otherwise, whether to create a new word or not for the first non-space
    character depends on the label of that character, for instance input="New
    York", labels=[0, 1, 1, 0, 1, 1, 1, 1] output tokens=["NewYork"] input="New
    York", labels=[0, 1, 1, 1, 1, 1, 1, 1] output tokens=["NewYork"] input="New
    York", labels=[0, 1, 1, 1, 0, 1, 1, 1] output tokens=["New", "York"]

#### Returns:

A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string contents (or
ID in the vocab_lookup_table representing that string) of the `jth` token in
`input[i1...iN]`

#### Returns:

A tuple `(tokens, start_offsets, limit_offsets)` where:

*   `tokens[i1...iN, j]` is a `RaggedTensor` of the string contents (or ID in
    the vocab_lookup_table representing that string) of the `jth` token in
    `input[i1...iN]`.
*   `start_offsets[i1...iN, j]` is a `RaggedTensor` of the byte offsets for the
    start of the `jth` token in `input[i1...iN]`.
*   `limit_offsets[i1...iN, j]` is a `RaggedTensor` of the byte offsets for the
    end of the `jth` token in `input[i`...iN]`.
