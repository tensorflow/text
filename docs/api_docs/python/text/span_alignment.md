<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.span_alignment" />
<meta itemprop="path" content="Stable" />
</div>

# text.span_alignment

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/pointer_ops.py">View
source</a>

Return an alignment from a set of source spans to a set of target spans.

```python
text.span_alignment(
    source_start, source_limit, target_start, target_limit, contains=False,
    contained_by=False, partial_overlap=False, multivalent_result=False, name=None
)
```

<!-- Placeholder for "Used in" -->

The source and target spans are specified using B+1 dimensional tensors,
with `B>=0` batch dimensions followed by a final dimension that lists the
span offsets for each span in the batch:

* The `i`th source span in batch `b1...bB` starts at
  `source_start[b1...bB, i]` (inclusive), and extends to just before
  `source_limit[b1...bB, i]` (exclusive).
* The `j`th target span in batch `b1...bB` starts at
  `target_start[b1...bB, j]` (inclusive), and extends to just before
  `target_limit[b1...bB, j]` (exclusive).

`result[b1...bB, i]` contains the index (or indices) of the target span that
overlaps with the `i`th source span in batch `b1...bB`.  The
`multivalent_result` parameter indicates whether the result should contain
a single span that aligns with the source span, or all spans that align with
the source span.

* If `multivalent_result` is false (the default), then `result[b1...bB, i]=j`
  indicates that the `j`th target span overlaps with the `i`th source span
  in batch `b1...bB`.  If no target spans overlap with the `i`th target span,
  then `result[b1...bB, i]=-1`.

* If `multivalent_result` is true, then `result[b1...bB, i, n]=j` indicates
  that the `j`th target span is the `n`th span that overlaps with the `i`th
  source span in in batch `b1...bB`.

For a definition of span overlap, see the docstring for `span_overlaps()`.

#### Args:

*   <b>`source_start`</b>: A B+1 dimensional potentially ragged tensor with
    shape `[D1...DB, source_size]`: the start offset of each source span.
*   <b>`source_limit`</b>: A B+1 dimensional potentially ragged tensor with
    shape `[D1...DB, source_size]`: the limit offset of each source span.
*   <b>`target_start`</b>: A B+1 dimensional potentially ragged tensor with
    shape `[D1...DB, target_size]`: the start offset of each target span.
*   <b>`target_limit`</b>: A B+1 dimensional potentially ragged tensor with
    shape `[D1...DB, target_size]`: the limit offset of each target span.
*   <b>`contains`</b>: If true, then a source span is considered to overlap a
    target span when the source span contains the target span.
*   <b>`contained_by`</b>: If true, then a source span is considered to overlap
    a target span when the source span is contained by the target span.
*   <b>`partial_overlap`</b>: If true, then a source span is considered to
    overlap a target span when the source span partially overlaps the target
    span.
*   <b>`multivalent_result`</b>: Whether the result should contain a single
    target span index (if `multivalent_result=False`) or a list of target span
    indices (if `multivalent_result=True`) for each source span.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

An int64 tensor with values in the range: `-1 <= result < target_size`. If
`multivalent_result=False`, then the returned tensor has shape `[source_size]`,
where `source_size` is the length of the `source_start` and `source_limit` input
tensors. If `multivalent_result=True`, then the returned tensor has shape
`[source_size, (num_aligned_target_spans)].

#### Examples:

  Given the following source and target spans (with no batch dimensions):

```python
  >>> #         0    5    10   15   20   25   30   35   40   45   50   55   60
  >>> #         |====|====|====|====|====|====|====|====|====|====|====|====|
  >>> # Source: [-0-]     [-1-] [2] [3]    [4][-5-][-6-][-7-][-8-][-9-]
  >>> # Target: [-0-][-1-]     [-2-][-3-][-4-] [5] [6]    [7]  [-8-][-9-][10]
  >>> #         |====|====|====|====|====|====|====|====|====|====|====|====|
  >>> source_start=[0, 10, 16, 20, 27, 30, 35, 40, 45, 50]
  >>> source_limit=[5, 15, 19, 23, 30, 35, 40, 45, 50, 55]
  >>> target_start=[0,  5, 15, 20, 25, 31, 35, 42, 47, 52, 57]
  >>> target_limit=[5, 10, 20, 25, 30, 34, 38, 45, 52, 57, 61]

```

> > > span_alignment_lists(source_starts, source_limits, target_starts,
> > > target_limits) [0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
> > > span_alignment_lists(source_starts, source_limits, ... target_starts,
> > > target_limits, ... multivalent_result=True) [[0], [], [], [], [], [], [],
> > > [], [], []] ```

```
  >>> span_alignment_lists(source_starts, source_limits,
  ...                      target_starts, target_limits,
  ...                      contains=True)
  [ 0, -1, -1, -1, -1, 5, 6, 7, -1, -1]
```

```
  >>> span_alignment_lists(source_starts, source_limits,
  ...                      target_starts, target_limits,
  ...                      partial_overlap=True,
  ...                      multivalent_result=True)
  [[0], [], [2], [3], [4], [5], [6], [7], [8], [8, 9]]
```
