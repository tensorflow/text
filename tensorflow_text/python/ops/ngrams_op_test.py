# coding=utf-8
# Copyright 2019 TF.Text Authors.
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

"""Tests for ngram ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf  # tf
import tensorflow_text as text

from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class NgramsOpTest(ragged_test_util.RaggedTensorTestCase):

  def testSumReduction(self):
    test_data = tf.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=1, reduction_type=text.Reduction.SUM)
    expected_values = [[3.0, 5.0], [30.0, 50.0]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testRaggedSumReduction(self):
    test_data = tf.ragged.constant([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=1, reduction_type=text.Reduction.SUM)
    expected_values = [[3.0, 5.0, 7.0], [30.0, 50.0]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testRaggedSumReductionAxisZero(self):
    test_data = tf.ragged.constant([[1.0, 2.0, 3.0, 4.0],
                                    [10.0, 20.0, 30.0, 40.0]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=0, reduction_type=text.Reduction.SUM)
    expected_values = [[11.0, 22.0, 33.0, 44.0]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testMeanReduction(self):
    test_data = tf.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=1, reduction_type=text.Reduction.MEAN)
    expected_values = [[1.5, 2.5], [15.0, 25.0]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testRaggedMeanReduction(self):
    test_data = tf.ragged.constant([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=-1, reduction_type=text.Reduction.MEAN)
    expected_values = [[1.5, 2.5, 3.5], [15.0, 25.0]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testStringJoinReduction(self):
    test_data = tf.constant([["a", "b", "c"], ["dd", "ee", "ff"]])
    ngrams_op = text.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=text.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [["a|b", "b|c"], ["dd|ee", "ee|ff"]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testStringJoinReductionAxisZero(self):
    test_data = tf.constant(["a", "b", "c"])
    ngrams_op = text.ngrams(
        test_data,
        width=2,
        axis=-1,  # The -1 axis is the zero axis here.
        reduction_type=text.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = ["a|b", "b|c"]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testRaggedStringJoinReduction(self):
    test_data = tf.ragged.constant([["a", "b", "c"], ["dd", "ee"]])
    ngrams_op = text.ngrams(
        test_data,
        width=2,
        axis=-1,
        reduction_type=text.Reduction.STRING_JOIN,
        string_separator="|")
    expected_values = [["a|b", "b|c"], ["dd|ee"]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testReductionWithNegativeAxis(self):
    test_data = tf.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=-1, reduction_type=text.Reduction.SUM)
    expected_values = [[3.0, 5.0], [30.0, 50.0]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testReductionOnInnerAxis(self):
    test_data = tf.constant([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                             [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=-2, reduction_type=text.Reduction.SUM)
    expected_values = [[[11.0, 22.0, 33.0]], [[44.0, 55.0, 66.0]]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testRaggedReductionOnInnerAxis(self):
    test_data = tf.ragged.constant([[[1.0, 2.0, 3.0, 4.0],
                                     [10.0, 20.0, 30.0, 40.0]],
                                    [[100.0, 200.0], [300.0, 400.0]]])
    ngrams_op = text.ngrams(
        test_data, width=2, axis=-2, reduction_type=text.Reduction.SUM)
    expected_values = [[[11.0, 22.0, 33.0, 44.0]], [[400.0, 600.0]]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testReductionOnAxisWithInsufficientValuesReturnsEmptySet(self):
    test_data = tf.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    ngrams_op = text.ngrams(
        test_data, width=4, axis=-1, reduction_type=text.Reduction.SUM)
    expected_values = [[], []]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testRaggedReductionOnAxisWithInsufficientValuesReturnsEmptySet(self):
    test_data = tf.ragged.constant([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0]])
    ngrams_op = text.ngrams(
        test_data, width=4, axis=1, reduction_type=text.Reduction.SUM)
    expected_values = [[], [100.0]]

    self.assertRaggedEqual(expected_values, ngrams_op)

  def testStringJoinReductionFailsWithImproperAxis(self):
    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        r".*requires that ngrams' 'axis' parameter be -1."):
      _ = text.ngrams(
          data=[], width=2, axis=0, reduction_type=text.Reduction.STRING_JOIN)

  def testUnspecifiedReductionTypeFails(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 r"reduction_type must be specified."):
      _ = text.ngrams(data=[], width=2, axis=0)

  def testBadReductionTypeFails(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 r"reduction_type must be a Reduction."):
      _ = text.ngrams(data=[], width=2, axis=0, reduction_type="SUM")


if __name__ == "__main__":
  test.main()
