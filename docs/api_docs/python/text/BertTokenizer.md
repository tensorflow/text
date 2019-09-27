<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.BertTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="tokenize"/>
</div>

# text.BertTokenizer

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

## Class `BertTokenizer`

Basic tokenizer for BERT preprocessing.

Inherits From: [`Tokenizer`](../text/Tokenizer.md)

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

```python
__init__(
    lower_case=False,
    keep_whitespace=False
)
```

## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

```python
tokenize(text_input)
```

Performs basic word tokenization for BERT.

#### Args:

*   <b>`text_input`</b>: A Tensor of untokenized strings with shape `[N]`.

#### Returns:

A RaggedTensor of tokens with shape `[N, (num_tokens)]`.
