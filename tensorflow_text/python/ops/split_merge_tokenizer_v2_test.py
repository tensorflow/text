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

# -*- coding: utf-8 -*-
"""Tests for split_merge_tokenizer op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test

# pylint: disable=line-too-long
from tensorflow_text.python.ops.split_merge_tokenizer_v2 import SplitMergeTokenizerV2


def _Utf8(char):
  return char.encode('utf-8')


def _RaggedSubstr(text_input, begin, end):
  text_input_flat = None
  if ragged_tensor.is_ragged(text_input):
    text_input_flat = text_input.flat_values
  else:
    text_input_flat = ops.convert_to_tensor(text_input)

  if ragged_tensor.is_ragged(begin):
    broadcasted_text = array_ops.gather_v2(text_input_flat,
                                           begin.nested_value_rowids()[-1])

    # convert boardcasted_text into a 1D tensor.
    broadcasted_text = array_ops.reshape(broadcasted_text, [-1])
    size = math_ops.sub(end.flat_values, begin.flat_values)
    new_tokens = string_ops.substr_v2(broadcasted_text, begin.flat_values, size)
    return begin.with_flat_values(new_tokens)
  else:
    assert begin.shape.ndims == 1
    assert text_input_flat.shape.ndims == 0
    size = math_ops.sub(end, begin)
    new_tokens = string_ops.substr_v2(text_input_flat, begin, size)
    return new_tokens


@test_util.run_all_in_graph_and_eager_modes
class SplitMergeTokenizerV2Test(test.TestCase):

  def setUp(self):
    super(SplitMergeTokenizerV2Test, self).setUp()
    self.tokenizer = SplitMergeTokenizerV2()

  def testVectorSingleValueSplitMerge(self):
    test_text = constant_op.constant([b'IloveFlume!'])
    test_logits = constant_op.constant([
        [
            # I
            [5.0, -3.2],  # split
            # love
            [2.2, -1.0],  # split
            [0.2, 12.0],  # merge
            [0.0, 11.0],  # merge
            [-3.0, 3.0],  # merge
            # Flume
            [10.0, 0.0],  # split
            [0.0, 11.0],  # merge
            [0.0, 11.0],  # merge
            [0.0, 12.0],  # merge
            [0.0, 12.0],  # merge
            # !
            [5.2, -7.0],  # split
        ]])
    expected_tokens = [[b'I', b'love', b'Flume', b'!']]
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_text, test_logits))
    self.assertAllEqual(tokens, expected_tokens)
    extracted_tokens = _RaggedSubstr(test_text, starts, limits)
    self.assertAllEqual(extracted_tokens, expected_tokens)

  def testVectorSingleValueTokenCrossSpace(self):
    test_text = constant_op.constant([b'I love Flume!'])
    test_logits = constant_op.constant([
        [
            # I
            [1.0, 0.0],
            # ' '
            [0.0, 1.0],
            # love
            [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            # ' '
            [1.0, 0.0],
            # Flume
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            # !
            [1.0, 0.0]
        ]])

    # By default force_split_at_break_character is set True, so we start new
    # tokens after break characters regardless of the SPLIT/MERGE label of the
    # break character.
    expected_tokens = [[b'I', b'love', b'Flume', b'!']]
    expected_offset_starts = [[0, 2, 7, 12]]
    expected_offset_limits = [[1, 6, 12, 13]]
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_text, test_logits))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(limits, expected_offset_limits)

    # When force_split_at_break_character is set false, we may combine two tokens
    # together to form a word according to the label of the first non-space
    # character.
    expected_tokens = [[b'I', b'loveFlume', b'!']]
    expected_offset_starts = [[0, 2, 12]]
    expected_offset_limits = [[1, 11, 13]]
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(
            test_text, test_logits, force_split_at_break_character=False))
    self.assertAllEqual(tokens, expected_tokens)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(limits, expected_offset_limits)

  def testVectorSingleValueTokenChinese(self):
    test_text = constant_op.constant([_Utf8(u'我在谷歌　写代码')])
    test_logits = constant_op.constant([
        [
            # 我
            [10.0, 0.0],  # split
            # 在
            [10.0, 0.0],  # split
            # 谷歌
            [10.0, 0.0],  # split
            [0.0, 10.0],  # merge
            # '　', note this is a full-width space which contains 3 bytes.
            [10.0, 0.0],  # split
            # 写代码
            [10.0, 0.0],  # split
            [0.0, 10.0],  # merge
            [0.0, 10.0],  # merge
        ]])

    # By default force_split_at_break_character is set True, so we start new
    # tokens after break characters regardless of the SPLIT/MERGE label of the
    # break character.
    expected_tokens = [[
        _Utf8(u'我'), _Utf8(u'在'), _Utf8(u'谷歌'), _Utf8(u'写代码')]]
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_text, test_logits))
    self.assertAllEqual(tokens, expected_tokens)

    # Extract tokens according to the returned starts, limits.
    tokens_by_offsets = _RaggedSubstr(test_text, starts, limits)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)

    # Although force_split_at_break_character is set false we actually predict a
    # SPLIT at '写', so we still start a new token: '写代码'.
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(
            test_text, test_logits, force_split_at_break_character=False))
    self.assertAllEqual(tokens, expected_tokens)

    # Extract tokens according to the returned starts, limits.
    tokens_by_offsets = _RaggedSubstr(test_text, starts, limits)
    self.assertAllEqual(expected_tokens, tokens_by_offsets)

  def testVectorMultipleValue(self):
    test_text = constant_op.constant([b'IloveFlume!',
                                      b'and tensorflow'])
    test_logits = constant_op.constant([
        [
            # I
            [7.2, -5.3],  # split
            # love
            [7.2, -5.3],  # split
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            # Flume
            [7.2, -5.3],  # split
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            # !
            [7.2, -5.3],  # split
            # paddings
            [9.1, -4.3],  # split
            [8.2, -5.1],  # split
            [7.5, -5.3]   # split
        ], [
            # and
            [7.2, -5.3],  # split
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            # ' '
            [2.3, 16.1],  # merge
            # tensorflow
            [7.2, -5.3],  # split
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
            [2.3, 16.1],  # merge
        ]])
    expected_tokens = [[b'I', b'love', b'Flume', b'!'],
                       [b'and', b'tensorflow']]
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_text, test_logits))
    self.assertAllEqual(tokens, expected_tokens)
    tokens_by_offsets = _RaggedSubstr(test_text, starts, limits)
    self.assertAllEqual(tokens_by_offsets, expected_tokens)


if __name__ == '__main__':
  test.main()
