<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.Tokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="tokenize"/>
</div>

# text.Tokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View source</a>



Base class for tokenizer implementations.

<!-- Placeholder for "Used in" -->


## Methods

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View source</a>

```python
tokenize(
    input
)
```

Tokenizes the input tensor.


#### Args:


* <b>`input`</b>: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
  `RaggedTensor`.


#### Returns:

An N+1-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.




