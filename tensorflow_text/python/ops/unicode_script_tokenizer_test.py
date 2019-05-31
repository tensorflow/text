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
"""Tests for unicode_script_tokenizer_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import test
from tensorflow_text.python.ops.unicode_script_tokenizer import UnicodeScriptTokenizer


@test_util.run_all_in_graph_and_eager_modes
class UnicodeScriptTokenizerOpTest(ragged_test_util.RaggedTensorTestCase):

  def setUp(self):
    self.tokenizer = UnicodeScriptTokenizer()

  def testRequireParams(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        self.tokenizer.tokenize()

  def testScalar(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        self.tokenizer.tokenize('I love Flume!')

  def testVectorSingleValue(self):
    test_value = constant_op.constant(['I love Flume!'])
    expected_tokens = [['I', 'love', 'Flume', '!']]
    expected_offset_starts = [[0, 2, 7, 12]]
    expected_offset_limits = [[1, 6, 12, 13]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testVector(self):
    test_value = constant_op.constant(['I love Flume!', 'Good day'])
    expected_tokens = [['I', 'love', 'Flume', '!'], ['Good', 'day']]
    expected_offset_starts = [[0, 2, 7, 12], [0, 5]]
    expected_offset_limits = [[1, 6, 12, 13], [4, 8]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testMatrix(self):
    test_value = constant_op.constant(
        [['I love Flume!', 'Good day'], ['I don\'t want', 'no scrubs']])
    expected_tokens = [[['I', 'love', 'Flume', '!'], ['Good', 'day']],
                       [['I', 'don', '\'', 't', 'want'], ['no', 'scrubs']]]
    expected_offset_starts = [[[0, 2, 7, 12], [0, 5]],
                              [[0, 2, 5, 6, 8], [0, 3]]]
    expected_offset_limits = [[[1, 6, 12, 13], [4, 8]],
                              [[1, 5, 6, 7, 12], [2, 9]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testMatrixRagged(self):
    test_value = ragged_factory_ops.constant(
        [['I love Flume!'], ['I don\'t want', 'no scrubs']])
    expected_tokens = [[['I', 'love', 'Flume', '!']],
                       [['I', 'don', '\'', 't', 'want'], ['no', 'scrubs']]]
    expected_offset_starts = [[[0, 2, 7, 12]],
                              [[0, 2, 5, 6, 8], [0, 3]]]
    expected_offset_limits = [[[1, 6, 12, 13]],
                              [[1, 5, 6, 7, 12], [2, 9]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def test3DimMatrix(self):
    test_value = constant_op.constant(
        [[['I love Flume!', 'Good day'], ['I don\'t want', 'no scrubs']],
         [['I love Zhu!', 'Good night'], ['A scrub is', 'a guy']]])
    expected_tokens = [[[['I', 'love', 'Flume', '!'], ['Good', 'day']],
                        [['I', 'don', '\'', 't', 'want'], ['no', 'scrubs']]],
                       [[['I', 'love', 'Zhu', '!'], ['Good', 'night']],
                        [['A', 'scrub', 'is'], ['a', 'guy']]]]
    expected_offset_starts = [[[[0, 2, 7, 12], [0, 5]],
                               [[0, 2, 5, 6, 8], [0, 3]]],
                              [[[0, 2, 7, 10], [0, 5]],
                               [[0, 2, 8], [0, 2]]]]
    expected_offset_limits = [[[[1, 6, 12, 13], [4, 8]],
                               [[1, 5, 6, 7, 12], [2, 9]]],
                              [[[1, 6, 10, 11], [4, 10]],
                               [[1, 7, 10], [1, 5]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def test3DimMatrixRagged(self):
    test_value = ragged_factory_ops.constant(
        [[['I love Flume!'], ['I don\'t want', 'no scrubs']],
         [['I love Zhu!', 'Good night']]])
    expected_tokens = [[[['I', 'love', 'Flume', '!']],
                        [['I', 'don', '\'', 't', 'want'], ['no', 'scrubs']]],
                       [[['I', 'love', 'Zhu', '!'], ['Good', 'night']]]]
    expected_offset_starts = [[[[0, 2, 7, 12]],
                               [[0, 2, 5, 6, 8], [0, 3]]],
                              [[[0, 2, 7, 10], [0, 5]]]]
    expected_offset_limits = [[[[1, 6, 12, 13]],
                               [[1, 5, 6, 7, 12], [2, 9]]],
                              [[[1, 6, 10, 11], [4, 10]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testInternationalization(self):
    test_value = constant_op.constant([u"J'adore la灯".encode('utf8'),
                                       u'¡Escríbeme!'.encode('utf8')])
    expected_tokens = [['J', "'", 'adore', 'la', u'灯'.encode('utf8')],
                       [u'¡'.encode('utf8'), u'Escríbeme'.encode('utf8'), '!']]
    expected_offset_starts = [[0, 1, 2, 8, 10], [0, 2, 12]]
    expected_offset_limits = [[1, 2, 7, 10, 13], [2, 12, 13]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testSpaceBoundaries(self):
    test_value = constant_op.constant([' Hook em! ', ' .Ok.   Go  '])
    expected_tokens = [['Hook', 'em', '!'], ['.', 'Ok', '.', 'Go']]
    expected_offset_starts = [[1, 6, 8], [1, 2, 4, 8]]
    expected_offset_limits = [[5, 8, 9], [2, 4, 5, 10]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testOnlySpaces(self):
    test_value = constant_op.constant([' ', '     '])
    expected_tokens = [[], []]
    expected_offset_starts = [[], []]
    expected_offset_limits = [[], []]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testWhitespaceCharacters(self):
    test_value = constant_op.constant(['things:\tcarpet\rdesk\nlamp'])
    expected_tokens = [['things', ':', 'carpet', 'desk', 'lamp']]
    expected_offset_starts = [[0, 6, 8, 15, 20]]
    expected_offset_limits = [[6, 7, 14, 19, 24]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testEmptyStringSingle(self):
    test_value = constant_op.constant([''])
    expected_tokens = [[]]
    expected_offset_starts = [[]]
    expected_offset_limits = [[]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testEmptyString(self):
    test_value = constant_op.constant(['', 'I love Flume!', '', 'O hai', ''])
    expected_tokens = [[], ['I', 'love', 'Flume', '!'], [], ['O', 'hai'], []]
    expected_offset_starts = [[], [0, 2, 7, 12], [], [0, 2], []]
    expected_offset_limits = [[], [1, 6, 12, 13], [], [1, 5], []]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)

  def testEmptyDimensions(self):
    test_value = ragged_factory_ops.constant(
        [[['I love Flume!', 'Good day. . .'],
          []],
         [],
         [['I love Zhu!', 'Good night'], ['A scrub is', 'a guy']]])
    expected_tokens = [[[['I', 'love', 'Flume', '!'], ['Good', 'day', '...']],
                        []],
                       [],
                       [[['I', 'love', 'Zhu', '!'], ['Good', 'night']],
                        [['A', 'scrub', 'is'], ['a', 'guy']]]]
    expected_offset_starts = [[[[0, 2, 7, 12], [0, 5, 8]],
                               []],
                              [],
                              [[[0, 2, 7, 10], [0, 5]],
                               [[0, 2, 8], [0, 2]]]]
    expected_offset_limits = [[[[1, 6, 12, 13], [4, 8, 13]],
                               []],
                              [],
                              [[[1, 6, 10, 11], [4, 10]],
                               [[1, 7, 10], [1, 5]]]]
    tokens = self.tokenizer.tokenize(test_value)
    self.assertRaggedEqual(tokens, expected_tokens)
    (tokens, starts, limits) = (
        self.tokenizer.tokenize_with_offsets(test_value))
    self.assertRaggedEqual(tokens, expected_tokens)
    self.assertRaggedEqual(starts, expected_offset_starts)
    self.assertRaggedEqual(limits, expected_offset_limits)


if __name__ == '__main__':
  test.main()
