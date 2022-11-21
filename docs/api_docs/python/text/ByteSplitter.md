description: Splits a string tensor into bytes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.ByteSplitter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
</div>

# text.ByteSplitter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/byte_splitter.py">View
source</a>

Splits a string tensor into bytes.

Inherits From: [`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.ByteSplitter()
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/byte_splitter.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    input
)
</code></pre>

Splits a string tensor into bytes.

The strings are split bytes. Thus, some unicode characters may be split into
multiple bytes.

#### Example:

```
>>> ByteSplitter().split("hello")
<tf.Tensor: shape=(5,), dtype=uint8, numpy=array([104, 101, 108, 108, 111],
dtype=uint8)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` of strings with any shape.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of bytes. The returned shape is the shape of the
input tensor with an added ragged dimension for the bytes that make up
each string.
</td>
</tr>

</table>

<h3 id="split_with_offsets"><code>split_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/byte_splitter.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split_with_offsets(
    input
)
</code></pre>

Splits a string tensor into bytes.

The strings are split bytes. Thus, some unicode characters may be split into
multiple bytes.

#### Example:

```
>>> splitter = ByteSplitter()
>>> bytes, starts, ends = splitter.split_with_offsets("hello")
>>> print(bytes.numpy(), starts.numpy(), ends.numpy())
[104 101 108 108 111] [0 1 2 3 4] [1 2 3 4 5]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` of strings with any shape.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of bytest. The returned shape is the shape of the
input tensor with an added ragged dimension for the bytes that make up
each string.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple `(bytes, offsets)` where:

*   `bytes`: A `RaggedTensor` of bytes.
*   `start_offsets`: A `RaggedTensor` of the bytes' starting byte offset.
*   `end_offsets`: A `RaggedTensor` of the bytes' ending byte offset. </td>
    </tr>

</table>
