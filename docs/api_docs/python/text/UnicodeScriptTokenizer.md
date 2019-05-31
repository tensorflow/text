<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.UnicodeScriptTokenizer" />
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

# text.UnicodeScriptTokenizer

## Class `UnicodeScriptTokenizer`

Tokenizes a tensor of UTF-8 strings on Unicode script boundaries.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

Defined in
[`python/ops/unicode_script_tokenizer.py`](https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py).

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(name=None)
```

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

Tokenizes a tensor of UTF-8 strings on Unicode script boundaries.

The strings are split when a change in the Unicode script is detected between
sequential tokens. The script codes used correspond to International Components
for Unicode (ICU) UScriptCode values. See:
http://icu-project.org/apiref/icu4c/uscript_8h.html

ICU defined whitespace characters are dropped.

#### Args:

*   <b>`input`</b>: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

#### Returns:

A RaggedTensor of tokenized text. The returned shape is the shape of the input
tensor with an added ragged dimension for tokens of each string.

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

```python
tokenize_with_offsets(input)
```

Tokenizes a tensor of UTF-8 strings on Unicode script boundaries.

The strings are split when a change in the Unicode script is detected between
sequential tokens. The script codes used correspond to International Components
for Unicode (ICU) UScriptCode values. See:
http://icu-project.org/apiref/icu4c/uscript_8h.html

ICU defined whitespace characters are dropped.

#### Args:

*   <b>`input`</b>: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

#### Returns:

A tuple of `RaggedTensor`s `tokens`, `start_offsets`, and `limit_offsets`

*   <b>`where`</b>: * `tokens`: A `RaggedTensor` of tokenized text.
    *   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
    *   `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset.

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
