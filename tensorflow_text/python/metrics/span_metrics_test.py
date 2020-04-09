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
"""Tests for span metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.metrics import span_metrics


@test_util.run_all_in_graph_and_eager_modes
class SpanMetricsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          predict_begin=[[1, 3], [5]],
          predict_end=[[2, 4], [6]],
          gold_begin=[[1], [5, 7]],
          gold_end=[[3], [6, 8]],
          expected_tp=1.0,
          expected_fp=2.0,
          expected_fn=2.0,
          description="FN: (1, 3)"),
      dict(
          predict_begin=[[1, 3], [5, 7]],
          predict_end=[[2, 4], [6, 8]],
          gold_begin=[[1], [5, 7]],
          gold_end=[[3], [6, 8]],
          expected_tp=2.0,
          expected_fp=2.0,
          expected_fn=1.0,
          description="TP: [5, 6], [7, 8]"),
      dict(
          predict_begin=[[1, 3], [5, 7]],
          predict_end=[[2, 4], [6, 8]],
          gold_begin=[[1, 3], [5, 7]],
          gold_end=[[3, 4], [6, 10]],
          expected_tp=2.0,
          expected_fp=2.0,
          expected_fn=2.0,
          description="TP: [5, 6], and [3, 4], FN: (1, 3), and (7, 10)"),
      dict(
          predict_begin=[[1, 3], [5, 7]],
          predict_end=[[2, 5], [7, 10]],
          gold_begin=[[1, 3], [5, 7]],
          gold_end=[[3, 4], [6, 9]],
          expected_tp=0.0,
          expected_fp=4.0,
          expected_fn=4.0,
          description="Partial matches in different batches, no TPs, all FPs " +
          "and FNs."),
      dict(
          predict_begin=[[1, 3, 4], [5, 7]],
          predict_end=[[2, 5, 5], [7, 10]],
          gold_begin=[[1, 3], [5, 7]],
          gold_end=[[2, 4], [7, 10]],
          expected_tp=3.0,
          expected_fp=2.0,
          expected_fn=1.0,
          description="Different inner ragged sizes; TPs: (1, 2), (5, 7), " +
          "(7, 10); FPs: (3, 5), (4, 5); FNs: (3, 4)"),
  ])
  def testSpanMetrics(self, predict_begin, predict_end, gold_begin, gold_end,
                      expected_tp, expected_fp, expected_fn, description):
    predict_begin = ragged_factory_ops.constant(predict_begin)
    predict_end = ragged_factory_ops.constant(predict_end)
    gold_begin = ragged_factory_ops.constant(gold_begin)
    gold_end = ragged_factory_ops.constant(gold_end)

    def _safe_divide(x, y):
      try:
        return x / y
      except ZeroDivisionError:
        return 0

    expected_precision = _safe_divide(expected_tp, (expected_tp + expected_fp))
    expected_recall = _safe_divide(expected_tp, (expected_tp + expected_fn))
    expected_f1 = _safe_divide((2 * expected_precision * expected_recall),
                               (expected_precision + expected_recall))

    actual_tp = span_metrics.calculate_true_positive(predict_begin, predict_end,
                                                     gold_begin, gold_end)
    actual_fp = ragged_array_ops.size(predict_begin) - actual_tp
    actual_fn = ragged_array_ops.size(gold_begin) - actual_tp

    self.assertAllEqual(actual_tp, expected_tp)
    self.assertAllEqual(actual_fp, expected_fp)
    self.assertAllEqual(actual_fn, expected_fn)

    actual_recall, recall_update_op = span_metrics.span_recall(
        predict_begin, predict_end, gold_begin, gold_end)
    actual_precision, precision_update_op = span_metrics.span_precision(
        predict_begin, predict_end, gold_begin, gold_end)
    actual_f1, f1_update_op = span_metrics.span_f1(predict_begin, predict_end,
                                                   gold_begin, gold_end)

    self.evaluate(variables.local_variables_initializer())
    self.evaluate((precision_update_op, recall_update_op, f1_update_op))
    self.assertAllClose(actual_f1, expected_f1)
    self.assertAllClose(actual_recall, expected_recall)
    self.assertAllClose(actual_precision, expected_precision)

    # Test Keras metric ops.
    f1_metric = span_metrics.SpanF1()
    precision_metric = span_metrics.SpanPrecision()
    recall_metric = span_metrics.SpanRecall()

    # Init all the variables.
    all_variables = []
    all_variables.extend(f1_metric.variables)
    all_variables.extend(recall_metric.variables)
    all_variables.extend(precision_metric.variables)
    self.evaluate([v.initializer for v in all_variables])

    # Update states.
    self.evaluate(
        f1_metric.update_state(predict_begin, predict_end, gold_begin,
                               gold_end))
    self.evaluate(
        recall_metric.update_state(predict_begin, predict_end, gold_begin,
                                   gold_end))
    self.evaluate(
        precision_metric.update_state(predict_begin, predict_end, gold_begin,
                                      gold_end))
    self.assertAllClose(f1_metric.result(), expected_f1)
    self.assertAllClose(recall_metric.result(), expected_recall)
    self.assertAllClose(precision_metric.result(), expected_precision)


if __name__ == "__main__":
  test.main()
