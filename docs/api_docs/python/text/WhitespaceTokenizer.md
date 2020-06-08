<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.WhitespaceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.WhitespaceTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/whitespace_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 strings on whitespaces.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.WhitespaceTokenizer()
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/whitespace_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 strings on whitespaces.

The strings are split on ICU defined whitespace characters. These whitespace
characters are dropped.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` of UTF-8 strings with any shape.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of tokenized text. The returned shape is the shape of the
input tensor with an added ragged dimension for tokens of each string.
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/whitespace_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 strings on whitespaces.

The strings are split on ICU defined whitespace characters. These whitespace
characters are dropped.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple `(tokens, start_offsets, limit_offsets)` where:

*   `tokens`: A `RaggedTensor` of tokenized text.
*   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
*   `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset. </td>
    </tr>

</table>
