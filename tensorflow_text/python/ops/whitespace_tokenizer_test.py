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

# -*- coding: utf-8 -*-
"""Tests for whitespace_tokenizer_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import test
from tensorflow_text.python.ops.whitespace_tokenizer import WhitespaceTokenizer


@test_util.run_all_in_graph_and_eager_modes
class WhitespaceTokenizerOpTest(ragged_test_util.RaggedTensorTestCase):

  def setUp(self):
    self.whitespace_tokenizer = WhitespaceTokenizer()

  def testScalar(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        self.whitespace_tokenizer.tokenize('I love Flume!')

  def testVectorSingleValue(self):
    test_value = constant_op.constant(['I love Flume!'])
    expected_tokens = [['I', 'love', 'Flume!']]
    expected_offset_starts = [[0, 2, 7]]
    expected_offset_limits = [[1, 6, 13]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testVector(self):
    test_value = constant_op.constant(['I love Flume!', 'Good day'])
    expected_tokens = [['I', 'love', 'Flume!'], ['Good', 'day']]
    expected_offset_starts = [[0, 2, 7], [0, 5]]
    expected_offset_limits = [[1, 6, 13], [4, 8]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testMatrix(self):
    test_value = constant_op.constant([['I love Flume!', 'Good day'],
                                       ['I don\'t want', 'no scrubs']])
    expected_tokens = [[['I', 'love', 'Flume!'], ['Good', 'day']],
                       [['I', 'don\'t', 'want'], ['no', 'scrubs']]]
    expected_offset_starts = [[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 3]]]
    expected_offset_limits = [[[1, 6, 13], [4, 8]], [[1, 7, 12], [2, 9]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testMatrixRagged(self):
    test_value = ragged_factory_ops.constant([['I love Flume!'],
                                              ['I don\'t want', 'no scrubs']])
    expected_tokens = [[['I', 'love', 'Flume!']],
                       [['I', 'don\'t', 'want'], ['no', 'scrubs']]]
    expected_offset_starts = [[[0, 2, 7]], [[0, 2, 8], [0, 3]]]
    expected_offset_limits = [[[1, 6, 13]], [[1, 7, 12], [2, 9]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def test3DimMatrix(self):
    test_value = constant_op.constant([[['I love Flume!', 'Good day'],
                                        ['I don\'t want', 'no scrubs']],
                                       [['I love Zhu!', 'Good night'],
                                        ['A scrub is', 'a guy']]])
    expected_tokens = [[[['I', 'love', 'Flume!'], ['Good', 'day']],
                        [['I', 'don\'t', 'want'], ['no', 'scrubs']]],
                       [[['I', 'love', 'Zhu!'], ['Good', 'night']],
                        [['A', 'scrub', 'is'], ['a', 'guy']]]]
    expected_offset_starts = [[[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 3]]],
                              [[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 2]]]]
    expected_offset_limits = [[[[1, 6, 13], [4, 8]], [[1, 7, 12], [2, 9]]],
                              [[[1, 6, 11], [4, 10]], [[1, 7, 10], [1, 5]]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def test3DimMatrixRagged(self):
    test_value = ragged_factory_ops.constant([[['I love Flume!'],
                                               ['I don\'t want', 'no scrubs']],
                                              [['I love Zhu!', 'Good night']]])
    expected_tokens = [[[['I', 'love', 'Flume!']],
                        [['I', 'don\'t', 'want'], ['no', 'scrubs']]],
                       [[['I', 'love', 'Zhu!'], ['Good', 'night']]]]
    expected_offset_starts = [[[[0, 2, 7]], [[0, 2, 8], [0, 3]]],
                              [[[0, 2, 7], [0, 5]]]]
    expected_offset_limits = [[[[1, 6, 13]], [[1, 7, 12], [2, 9]]],
                              [[[1, 6, 11], [4, 10]]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testInternationalization(self):
    test_value = constant_op.constant(
        [u"J'adore la灯".encode('utf8'), u'¡Escríbeme!'.encode('utf8')])
    expected_tokens = [['J\'adore', u'la灯'.encode('utf8')],
                       [u'¡Escríbeme!'.encode('utf8')]]
    expected_offset_starts = [[0, 8], [0]]
    expected_offset_limits = [[7, 13], [13]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testSpaceBoundaries(self):
    test_value = constant_op.constant([' Hook em! ', ' .Ok.   Go  '])
    expected_tokens = [['Hook', 'em!'], ['.Ok.', 'Go']]
    expected_offset_starts = [[1, 6], [1, 8]]
    expected_offset_limits = [[5, 9], [5, 10]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testOnlySpaces(self):
    test_value = constant_op.constant([' ', '     ', ' \t\r\n'])
    expected_tokens = [[], [], []]
    expected_offset_starts = [[], [], []]
    expected_offset_limits = [[], [], []]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testWhitespaceCharacters(self):
    test_value = constant_op.constant(['things:\tcarpet\rdesk\nlamp\r\nlove'])
    expected_tokens = [['things:', 'carpet', 'desk', 'lamp', 'love']]
    expected_offset_starts = [[0, 8, 15, 20, 26]]
    expected_offset_limits = [[7, 14, 19, 24, 30]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testEmptyStringSingle(self):
    test_value = constant_op.constant([''])
    expected_tokens = [[]]
    expected_offset_starts = [[]]
    expected_offset_limits = [[]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testEmptyString(self):
    test_value = constant_op.constant(['', 'I love Flume!', '', 'O hai', ''])
    expected_tokens = [[], ['I', 'love', 'Flume!'], [], ['O', 'hai'], []]
    expected_offset_starts = [[], [0, 2, 7], [], [0, 2], []]
    expected_offset_limits = [[], [1, 6, 13], [], [1, 5], []]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testEmptyDimensions(self):
    test_value = ragged_factory_ops.constant(
        [[['I love Flume!', 'Good day. . .'], []], [],
         [['I love Zhu!', 'Good night'], ['A scrub is', 'a guy']]])
    expected_tokens = [[[['I', 'love', 'Flume!'], ['Good', 'day.', '.', '.']],
                        []], [],
                       [[['I', 'love', 'Zhu!'], ['Good', 'night']],
                        [['A', 'scrub', 'is'], ['a', 'guy']]]]
    expected_offset_starts = [[[[0, 2, 7], [0, 5, 10, 12]], []], [],
                              [[[0, 2, 7], [0, 5]], [[0, 2, 8], [0, 2]]]]
    expected_offset_limits = [[[[1, 6, 13], [4, 9, 11, 13]], []], [],
                              [[[1, 6, 11], [4, 10]], [[1, 7, 10], [1, 5]]]]
    tokens = self.whitespace_tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.whitespace_tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)


if __name__ == '__main__':
  test.main()
