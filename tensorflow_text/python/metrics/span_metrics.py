# coding=utf-8
# Copyright 2020 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Span-based metrics used for evaluating the models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import metrics
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import sets_impl
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_map_ops


def _ragged_set_op(set_op,
                   pred_begin,
                   pred_end,
                   gold_begin,
                   gold_end,
                   pred_label=None,
                   gold_label=None):
  """Computes a sets op on RaggedTensor of prediction and labelled spans.

  This op computes `set_op` (an op from tf.sets) on prediction and gold labelled
  spans and returns the number of results from `set_op`.
  Args:
    set_op: A callable tf.sets operation
    pred_begin: A `RaggedTensor` of shape [batch, (num_spans)] containing the
      beginning indices of a prediction span.
    pred_end: A `RaggedTensor` of shape [batch, (num_spans)] containing the
      ending indices of a prediction span.
    gold_begin: A `RaggedTensor` of shape [batch, (num_spans)] containing the
      beginning indices of a gold labelled span.
    gold_end: A `RaggedTensor` of shape [batch, (num_spans)] containing the
      ending indices of a gold labelled span.
    pred_label: (optional) A `RaggedTensor` of shape [batch, (num_spans)]
      containing the prediction label types. If not provided, assumes all spans
      are of the same type.
    gold_label: (optional) A `RaggedTensor` of shape [batch, (num_spans)]
      containing the gold label types. If not provided, assumes all spans are of
      the same type.

  Returns:
    A 1-D Tensor containing the number of elements in the results of
    `set_op`.
  """
  op = functools.partial(_per_batch_set_op, set_op)
  if pred_label is None:
    pred_label = pred_begin.with_flat_values(
        array_ops.zeros_like(pred_begin.flat_values))
  if gold_label is None:
    gold_label = gold_begin.with_flat_values(
        array_ops.zeros_like(gold_begin.flat_values))
  results = ragged_map_ops.map_fn(
      op, (pred_begin, pred_end, pred_label, gold_begin, gold_end, gold_label),
      dtype=(dtypes.int32),
      infer_shape=False)
  return results


def _per_batch_set_op(set_op, x):
  """Computes a set operation on a single batch of prediction & labelled span.

  Args:
    set_op: A callable function which is a tf.sets operation.
    x: A tuple of (pred_begin, pred_end, pred_label, gold_begin, gold_end,
      gold_label) which are the prediction and gold labelled spans for a single
      batch.  Each element of the tuple is a 1-D Tensor. pred_* Tensors should
      have the same size, and gold_* as well, but pred_* and gold_* Tensors may
      have different sizes.

  Returns:
    Performs the sets operation and returns the number of results.
  """
  x = tuple(math_ops.cast(i, dtypes.int64) for i in x)
  pred_begin, pred_end, pred_label, gold_begin, gold_end, gold_label = x
  # Combine spans together so they can be compared as one atomic unit. For
  # example:
  # If pred_begin = [0, 3, 5], pred_end = [2, 4, 6]
  #    gold_begin = [0, 5], gold_end = [2, 7]
  # Then we combine the spans into:
  #    pred = [[0, 2], [3, 4], [5, 6]]
  #    gold = [[0, 2], [5, 7]]
  #
  # In the sets operation, we want [0, 2] to be treated as one atomic comparison
  # unit (both begin=0 and end=2 offsets must match). Conversely, partial
  # matches (like [5, 6] and [5, 7]) are not a match.
  #
  # This is done by constructing a SparseTensor (containing span begin, end,
  # label points) for predictions and labels.
  pred_begin = array_ops.expand_dims(pred_begin, 1)
  pred_end = array_ops.expand_dims(pred_end, 1)
  gold_begin = array_ops.expand_dims(gold_begin, 1)
  gold_end = array_ops.expand_dims(gold_end, 1)
  # Because the last dimension is ignored in comparisons for tf.sets operations,
  # we add an unused last dimension.
  unused_last_pred_dim = array_ops.zeros_like(pred_begin)
  unused_last_gold_dim = array_ops.zeros_like(gold_begin)
  pred_indices = array_ops.concat([pred_begin, pred_end, unused_last_pred_dim],
                                  1)
  gold_indices = array_ops.concat([gold_begin, gold_end, unused_last_gold_dim],
                                  1)

  # set_ops require the bounding shape to match. Find the bounding shape
  # with the max number
  max_shape = math_ops.reduce_max(
      array_ops.concat([pred_indices, gold_indices], 0), 0)
  max_shape = max_shape + array_ops.ones_like(max_shape)

  pred = sparse_tensor.SparseTensor(pred_indices, pred_label, max_shape)
  pred = sparse_ops.sparse_reorder(pred)
  gold = sparse_tensor.SparseTensor(gold_indices, gold_label, max_shape)
  gold = sparse_ops.sparse_reorder(gold)
  results = set_op(pred, gold).indices
  num_results = control_flow_ops.cond(
      array_ops.size(results) > 0,
      true_fn=lambda: array_ops.shape(results)[0],
      false_fn=lambda: constant_op.constant(0))
  return num_results


def calculate_true_positive(pred_begin, pred_end, gold_begin, gold_end):
  """Calculates true positive given prediction and gold labelled spans."""
  with ops.name_scope("TruePositive"):
    op_results = _ragged_set_op(sets_impl.set_intersection, pred_begin,
                                pred_end, gold_begin, gold_end)
    return math_ops.reduce_sum(op_results)


def _update_confusion_matrix(pred_begin, pred_end, gold_begin, gold_end):
  """Updates internal variables of the confusion matrix."""
  with ops.name_scope("UpdateConfusionMatrix"):
    total_true_pos = metrics_impl.metric_variable([],
                                                  dtypes.int32,
                                                  name="total_true_pos")
    total_false_pos = metrics_impl.metric_variable([],
                                                   dtypes.int32,
                                                   name="total_false_pos")
    total_false_neg = metrics_impl.metric_variable([],
                                                   dtypes.int32,
                                                   name="total_false_neg")

    num_gold = ragged_array_ops.size(gold_begin)
    num_pred = ragged_array_ops.size(pred_begin)
    tp = calculate_true_positive(pred_begin, pred_end, gold_begin, gold_end)
    fp = num_pred - tp
    fn = num_gold - tp
    tp_op = state_ops.assign_add(total_true_pos, tp)
    fp_op = state_ops.assign_add(total_false_pos, fp)
    fn_op = state_ops.assign_add(total_false_neg, fn)
    return (total_true_pos, total_false_pos,
            total_false_neg), control_flow_ops.group(tp_op, fp_op, fn_op)


def span_recall(pred_begin, pred_end, gold_begin, gold_end):
  """Calculates the recall metric given prediction and labelled spans.

  Computes an Estimator-style metric for recall given begin and end spans of
  predictions and golden labels.

  Args:
    pred_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the starting positions of the predicted spans.
    pred_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the ending positions of the predicted spans.
    gold_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the starting positions of the golden labelled spans.
    gold_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the ending positions of the golden labelled spans.

  Returns:
    A tuple of (recall_value, recall_update_op) where `recall_value`
    returns the recall metric value and `recall_update_op` updates the
    internal variables.
  """
  with ops.name_scope("Recall"):
    counts, update_op = _update_confusion_matrix(pred_begin, pred_end,
                                                 gold_begin, gold_end)
    tp, _, fn = counts
    value = math_ops.div_no_nan(
        math_ops.cast(tp, dtypes.float32),
        math_ops.cast(tp + fn, dtypes.float32))
    return value, update_op


def span_precision(pred_begin, pred_end, gold_begin, gold_end):
  """Calculates the precision metric given prediction and labelled spans.

  Computes an Estimator-style metric for precision given begin and end spans of
  predictions and golden labels.

  Args:
    pred_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the starting positions of the predicted spans.
    pred_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the ending positions of the predicted spans.
    gold_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the starting positions of the golden labelled spans.
    gold_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the ending positions of the golden labelled spans.

  Returns:
    A tuple of (precision_value, precision_update_op) where `f1_value` returns
    the precision metric value and `precision_update_op` updates the internal
    variables.
  """
  with ops.name_scope("Precision"):
    counts, update_op = _update_confusion_matrix(pred_begin, pred_end,
                                                 gold_begin, gold_end)
    tp, fp, _ = counts
    value = math_ops.div_no_nan(
        math_ops.cast(tp, dtypes.float32),
        math_ops.cast(tp + fp, dtypes.float32))
    return value, update_op


def span_f1(pred_begin, pred_end, gold_begin, gold_end):
  """Calculates the F1 metric given prediction and labelled spans.

  Computes an Estimator-style metric for F1 given begin and end spans of
  predictions and golden labels.

  Args:
    pred_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the starting positions of the predicted spans.
    pred_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the ending positions of the predicted spans.
    gold_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the starting positions of the golden labelled spans.
    gold_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This contains
      the ending positions of the golden labelled spans.

  Returns:
    A tuple of (f1_value, f1_update_op) where `f1_value` returns the F1 metric
    value and `f1_update_op` updates the internal variables.
  """
  with ops.name_scope("F1"):
    precision_value, prec_update_op = span_precision(pred_begin, pred_end,
                                                     gold_begin, gold_end)
    recall_value, recall_update_op = span_recall(pred_begin, pred_end,
                                                 gold_begin, gold_end)
    value = 2 * math_ops.div_no_nan(precision_value * recall_value,
                                    precision_value + recall_value)
    update_op = control_flow_ops.group(prec_update_op, recall_update_op)
    return value, update_op


class _SpanMetricsBase(metrics.Metric):
  """Base class for Keras-style span metrics."""

  def __init__(self, name):
    super(_SpanMetricsBase, self).__init__(name=name)
    with ops.init_scope():
      self.true_positive = self.add_weight(
          "true_positive",
          initializer=init_ops.zeros_initializer,
          dtype=dtypes.float32)
      self.false_positive = self.add_weight(
          "false_positive",
          initializer=init_ops.zeros_initializer,
          dtype=dtypes.float32)
      self.false_negative = self.add_weight(
          "false_negative",
          initializer=init_ops.zeros_initializer,
          dtype=dtypes.float32)

  def update_state(self, prediction_begin, prediction_end, label_begin,
                   label_end):
    """Updates metric given prediction and labelled spans.

    Args:
      prediction_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the starting positions of the predicted spans.
      prediction_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the ending positions of the predicted spans.
      label_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the starting positions of the golden labelled spans.
      label_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the ending positions of the golden labelled spans.
    """
    tp = math_ops.cast(
        calculate_true_positive(prediction_begin, prediction_end, label_begin,
                                label_end), dtypes.float32)
    num_pred = math_ops.cast(
        ragged_array_ops.size(prediction_begin), dtypes.float32)
    num_gold = math_ops.cast(ragged_array_ops.size(label_begin), dtypes.float32)
    fp = num_pred - tp
    fn = num_gold - tp
    self.true_positive.assign_add(tp)
    self.false_positive.assign_add(fp)
    self.false_negative.assign_add(fn)


class SpanPrecision(_SpanMetricsBase):
  """A Keras-style metric that computes precision from spans."""

  def __init__(self, name="span_precision"):
    super(SpanPrecision, self).__init__(name=name)

  def result(self):
    """Returns a `Tensor` of the computed metric value."""
    tp = self.true_positive
    fp = self.false_positive
    return math_ops.div_no_nan(tp, tp + fp)


class SpanRecall(_SpanMetricsBase):
  """A Keras-style metric that computes recall from prediction and gold spans."""

  def __init__(self, name="span_recall"):
    super(SpanRecall, self).__init__(name=name)

  def result(self):
    """Returns a `Tensor` of the computed metric value."""
    tp = self.true_positive.read_value()
    fn = self.false_negative.read_value()
    return math_ops.div_no_nan(tp, tp + fn)


class SpanF1(metrics.Metric):
  """A Keras-style metric that computes F1 measure across spans."""

  def __init__(self, name="span_f1"):
    super(SpanF1, self).__init__(name=name)
    with ops.init_scope():
      self.precision = SpanPrecision()
      self.recall = SpanRecall()

  def update_state(self, prediction_begin, prediction_end, label_begin,
                   label_end):
    """Updates metric given prediction and labelled spans.

    Args:
      prediction_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the starting positions of the predicted spans.
      prediction_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the ending positions of the predicted spans.
      label_begin: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the starting positions of the golden labelled spans.
      label_end: A `RaggedTensor` w/ `ragged_rank`=1 of type int64. This
        contains the ending positions of the golden labelled spans.
    """
    self.precision.update_state(prediction_begin, prediction_end, label_begin,
                                label_end)
    self.recall.update_state(prediction_begin, prediction_end, label_begin,
                             label_end)

  def result(self):
    """Returns a `Tensor` of the computed metric value."""
    prec_value = self.precision.result()
    recall_value = self.recall.result()
    return 2 * math_ops.div_no_nan(prec_value * recall_value,
                                   prec_value + recall_value)
