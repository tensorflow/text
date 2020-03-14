<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.sliding_window" />
<meta itemprop="path" content="Stable" />
</div>

# text.sliding_window

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sliding_window_op.py">View
source</a>

Builds a sliding window for `data` with a specified width.

```python
text.sliding_window(
    data, width, axis=-1, name=None
)
```

<!-- Placeholder for "Used in" -->

Returns a tensor constructed from `data`, where each element in
dimension `axis` is a slice of `data` starting at the corresponding
position, with the given width and step size.  I.e.:

* `result.shape.ndims = data.shape.ndims + 1`
* `result[i1..iaxis, a] = data[i1..iaxis, a:a+width]`
  (where `0 <= a < data[i1...iaxis].shape[0] - (width - 1)`).

Note that each result row (along dimension `axis`) has `width - 1` fewer items
than the corresponding `data` row.  If a `data` row has fewer than `width`
items, then the corresponding `result` row will be empty.  If you wish for
the `result` rows to be the same size as the `data` rows, you can use
`pad_along_dimension` to add `width - 1` padding elements before calling
this op.

#### Args:

*   <b>`data`</b>: `<dtype> [O1...ON, A, I1...IM]` A potentially ragged
    K-dimensional tensor with outer dimensions of size `O1...ON`; axis dimension
    of size `A`; and inner dimensions of size `I1...IM`. I.e. `K = N + 1 + M`,
    where `N>=0` and `M>=0`.

*   <b>`width`</b>: An integer constant specifying the width of the window. Must
    be greater than zero.

*   <b>`axis`</b>: An integer constant specifying the axis along which sliding
    window is computed. Negative axis values from `-K` to `-1` are supported.

*   <b>`name`</b>: The name for this op (optional).

#### Returns:

A `K+1` dimensional tensor with the same dtype as `data`, where:

*   `result[i1..iaxis, a]` = `data[i1..iaxis, a:a+width]`
*   `result.shape[:axis]` = `data.shape[:axis]`
*   `result.shape[axis]` = `data.shape[axis] - (width - 1)`
*   `result.shape[axis + 1]` = `width`
*   `result.shape[axis + 2:]` = `data.shape[axis + 1:]`

#### Examples:

  Sliding window (width=3) across a sequence of tokens:

```python
  >>> # input: <string>[sequence_length]
  >>> input = tf.constant(["one", "two", "three", "four", "five", "six"])
  >>> # output: <string>[sequence_length-2, 3]
  >>> output = sliding_window(data=input, width=3, axis=0)
  >>> print output.eval()
  [["one", "two", "three"],
   ["two", "three", "four"],
   ["three", "four", "five"],
   ["four", "five", "six"]]
  >>> print("Shape: %s -> %s" % (input.shape, output.shape))
  Shape: (6,) -> (4, 3)
```

  Sliding window (width=2) across the inner dimension of a ragged matrix
  containing a batch of token sequences:

```python
  >>> # input: <string>[num_sentences, (num_words)]
  >>> input = tf.ragged.constant(
  ...     [['Up', 'high', 'in', 'the', 'air'],
  ...      ['Down', 'under', 'water'],
  ...      ['Away', 'to', 'outer', 'space']]
  >>> # output: <string>[num_sentences, (num_word-1), 2]
  >>> output = sliding_window(input, width=2, axis=-1)
  >>> print output.eval()
  [[['Up', 'high'], ['high', 'in'], ['in', 'the'], ['the', 'air']],
   [['Down', 'under'], ['under', 'water']],
   [['Away', 'to'], ['to', 'outer'], ['outer', 'space']]]
  >>> print("Shape: %s -> %s" % (input.shape, output.shape))
  Shape: (3, ?) -> (3, ?, 2)
```

  Sliding window across the second dimension of a 3-D tensor containing
  batches of sequences of embedding vectors:

```python
  >>> # input: <int32>[num_sequences, sequence_length, embedding_size]
  >>> input = tf.constant([
  ...     [[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]],
  ...     [[1, 1, 2], [2, 2, 2], [3, 3, 2], [4, 4, 2], [5, 5, 2]]])
  >>> # output: <int32>[num_sequences, sequence_length-1, 2, embedding_size]
  >>> output = sliding_window(data=input, width=2, axis=1)
  >>> print output.eval()
  [[[[1, 1, 1], [2, 2, 1]],
    [[2, 2, 1], [3, 3, 1]],
    [[3, 3, 1], [4, 4, 1]],
    [[4, 4, 1], [5, 5, 1]]],
   [[[1, 1, 2], [2, 2, 2]],
    [[2, 2, 2], [3, 3, 2]],
    [[3, 3, 2], [4, 4, 2]],
    [[4, 4, 2], [5, 5, 2]]]]
  >>> print("Shape: %s -> %s" % (input.shape, output.shape))
  Shape: (2, 5, 3) -> (2, 4, 2, 3)
```
