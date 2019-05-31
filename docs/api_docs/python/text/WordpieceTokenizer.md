<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.WordpieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# text.WordpieceTokenizer

## Class `WordpieceTokenizer`

Creates a wordpiece tokenizer.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

Defined in
[`python/ops/wordpiece_tokenizer.py`](https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py).

<!-- Placeholder for "Used in" -->

It tokenizes utf-8 encoded tokens into subword pieces based off of a vocab.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    vocab_lookup_table,
    suffix_indicator='##',
    max_bytes_per_word=100,
    token_out_type=dtypes.int64,
    unknown_token='[UNK]'
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
*   <b>`token_out_type`</b>: (optional) The type of the token to return. This
    can be `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
*   <b>`unknown_token`</b>: (optional) The value to use when an unknown token is
    found. Default is "[UNK]". If this is set to a string, and `token_out_type`
    is `tf.int64`, the `vocab_lookup_table` is used to convert the
    `unknown_token` to an integer. If this is set to `None`, out-of-vocabulary
    tokens are left as is.

## Properties

<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes parent
module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.

<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
a = tf.Module()
b = tf.Module()
c = tf.Module()
a.b = b
b.c = c
assert list(a.submodules) == [b, c]
assert list(b.submodules) == [c]
assert list(c.submodules) == []
```

#### Returns:

A sequence of all submodules.

<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute name)
followed by variables from all submodules recursively (breadth first).

<h3 id="variables"><code>variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute name)
followed by variables from all submodules recursively (breadth first).

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

```python
tokenize(input)
```

"Splits tokens further into wordpiece tokens.

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

A `RaggedTensor`s `tokens` where `tokens[i1...iN, j]` is the string contents, or
ID in the vocab_lookup_table representing that string, of the `j`th token in
`input[i1...iN]`

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

```python
tokenize_with_offsets(input)
```

Tokenizes utf-8 encoded tokens into subword pieces based off of a vocab.

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

A tuple of `RaggedTensor`s `tokens`, `start_offsets`, and `limit_offsets`

*   <b>`where`</b>: * `tokens[i1...iN, j]` is the string contents, or ID in the
    vocab_lookup_table representing that string, of the `j`th token in
    `input[i1...iN]`
    *   `start_offsets[i1...iN, j]` is the byte offset for the start of the
        `j`th token in `input[i1...iN]`
    *   `limit_offsets[i1...iN, j]` is the byte offset for the end of the

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

```python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
class MyModule(tf.Module):
  @tf.Module.with_name_scope
  def __call__(self, x):
    if not hasattr(self, 'w'):
      self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
    return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose names
included the module name:

```
mod = MyModule()
mod(tf.ones([8, 32]))
# ==> <tf.Tensor: ...>
mod.w
# ==> <tf.Variable ...'my_module/w:0'>
```

#### Args:

*   <b>`method`</b>: The method to wrap.

#### Returns:

The original method wrapped such that it enters the module's name scope.
