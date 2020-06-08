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

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.UnicodeCharTokenizer()
</code></pre>

<!-- Placeholder for "Used in" -->

Resulting tokens are integers (unicode codepoints)

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    input, name=None
)
</code></pre>

Detokenizes input codepoints (integers) to UTF-8 strings.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` of codepoints (ints) with a rank of at
least 1.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name argument that is passed to the op function.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A N-1 dimensional string tensor of the detokenized text.
</td>
</tr>

</table>

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 strings on Unicode character boundaries.

Input strings are split on character boundaries using
unicode_decode_with_offsets.

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
A `RaggedTensor` of tokenized text. The returned shape is the shape of the
input tensor with an added ragged dimension for tokens (characters) of
each string.
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 strings to Unicode characters.

Returned token tensors are of integer type.

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

*   `tokens`: A `RaggedTensor` of codepoints (integer type).
*   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
*   `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset. </td>
    </tr>

</table>
