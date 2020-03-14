<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.keras.layers.ToDense" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# text.keras.layers.ToDense

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/keras/layers/todense.py">View
source</a>

Layer that makes padding and masking a Composite Tensors effortless.

```python
text.keras.layers.ToDense(
    pad_value=0, mask=False, **kwargs
)
```

<!-- Placeholder for "Used in" -->

The layer takes a RaggedTensor or a SparseTensor and converts it to a uniform
tensor by right-padding it or filling in missing values.

#### Example:

```python
x = tf.keras.layers.Input(shape=(None, None), ragged=True)
y = tf_text.keras.layers.ToDense(mask=True)(x)
model = tf.keras.Model(x, y)

rt = tf.RaggedTensor.from_nested_row_splits(
  flat_values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  nested_row_splits=([0, 1, 1, 5], [0, 3, 3, 5, 9, 10]))
model.predict(rt)

[[[10, 11, 12,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],
 [[ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],
 [[ 0,  0,  0,  0], [13, 14,  0,  0], [15, 16, 17, 18], [19,  0,  0,  0]]]
```

#### Arguments:

*   <b>`pad_value`</b>: A value used to pad and fill in the missing values.
    Should be a meaningless value for the input data. Default is '0'.
*   <b>`mask`</b>: A Boolean value representing whether to mask the padded
    values. If true, no any downstream Masking layer or Embedding layer with
    mask_zero=True should be added. Default is 'False'.
*   <b>`**kwargs`</b>: kwargs of parent class. Input shape: Any Ragged or Sparse
    Tensor is accepted, but it requires the type of input to be specified via
    the Input or InputLayer from the Keras API. Output shape: The output is a
    uniform tensor having the same shape, in case of a ragged input or the same
    dense shape, in case of a sparse input.
