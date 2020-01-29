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

# coding=utf-8
"""Tests for SentencePieceProcessor Tensorflow op."""

import sys
from absl.testing import parameterized
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow_text.python.ops import ragged_test_util
from tensorflow_text.python.ops.sentencepiece_tokenizer import SentencepieceTokenizer


def _utf8(tokens):
  if sys.version_info[0] == 2:
    return tokens
  if isinstance(tokens, list):
    return [_utf8(t) for t in tokens]
  else:
    return tokens.encode('utf-8')


@test_util.run_all_in_graph_and_eager_modes
class SentencepieceTokenizerOpTest(ragged_test_util.RaggedTensorTestCase,
                                   parameterized.TestCase):

  def getTokenizerAndSetOptions(self, reverse, add_bos, add_eos, out_type):
    return SentencepieceTokenizer(
        self.model,
        reverse=reverse,
        add_bos=add_bos,
        add_eos=add_eos,
        out_type=out_type)

  def setUp(self):
    super(SentencepieceTokenizerOpTest, self).setUp()
    sentencepiece_model_file = (
        'third_party/tensorflow_text/python/ops/test_data/test_oss_model.model')
    self.model = gfile.GFile(sentencepiece_model_file, 'r').read()

  def testGetVocabSize(self):
    sp = SentencepieceTokenizer(self.model)
    self.assertAllEqual(1000, sp.vocab_size())

  def testIdToStringScalar(self):
    sp = SentencepieceTokenizer(self.model)
    result = sp.id_to_string(125)
    self.assertAllEqual('ve', result)

  def testIdToStringVector(self):
    sp = SentencepieceTokenizer(self.model)
    pieces = _utf8([['▁I', '▁l', 'o', 've', '▁c', 'ar', 'pe', 't'],
                    ['▁I', '▁l', 'o', 've', '▁desk', '.'],
                    ['▁I', '▁l', 'o', 've', '▁l', 'amp', '.']])
    ids = [[9, 169, 21, 125, 78, 48, 132, 15], [9, 169, 21, 125, 727, 6],
           [9, 169, 21, 125, 169, 579, 6]]
    result = sp.id_to_string(ragged_factory_ops.constant(ids))
    self.assertAllEqual(pieces, result)

  def testIdToStringRagged(self):
    sp = SentencepieceTokenizer(self.model)
    pieces = _utf8(
        [[['▁I', '▁l', 'o', 've', '▁c', 'ar', 'pe', 't'],
          ['▁I', '▁l', 'o', 've', '▁desk', '.'],
          ['▁I', '▁l', 'o', 've', '▁l', 'amp', '.']],
         [['▁', 'N', 'ever', '▁tell', '▁me', '▁the', '▁', 'o', 'd', 'd', 's']]])
    ids = [[[9, 169, 21, 125, 78, 48, 132, 15], [9, 169, 21, 125, 727, 6],
            [9, 169, 21, 125, 169, 579, 6]],
           [[4, 199, 363, 310, 33, 7, 4, 21, 17, 17, 8]]]
    result = sp.id_to_string(ragged_factory_ops.constant(ids, dtypes.int32))
    self.assertRaggedEqual(pieces, result)

  @parameterized.parameters([
      (False, False, False, dtypes.int32),
      (False, False, True, dtypes.int32),
      (False, True, False, dtypes.int32),
      (False, True, True, dtypes.int32),
      (True, False, False, dtypes.int32),
      (True, False, True, dtypes.int32),
      (True, True, False, dtypes.int32),
      (True, True, True, dtypes.int32),
      (False, False, False, dtypes.string),
      (False, False, True, dtypes.string),
      (False, True, False, dtypes.string),
      (False, True, True, dtypes.string),
      (True, False, False, dtypes.string),
      (True, False, True, dtypes.string),
      (True, True, False, dtypes.string),
      (True, True, True, dtypes.string),
  ])
  def testTokenizeAndDetokenizeScalar(self, reverse, add_bos, add_eos,
                                      out_type):
    sp = self.getTokenizerAndSetOptions(reverse, add_bos, add_eos, out_type)
    sentence = b'I love lamp.'
    expected = []
    if out_type == dtypes.int32:
      expected = [9, 169, 21, 125, 169, 579, 6]
    else:
      expected = _utf8(['▁I', '▁l', 'o', 've', '▁l', 'amp', '.'])
    result = sp.tokenize(sentence)
    self.assertRaggedEqual(expected, result)
    detokenized = sp.detokenize(result)
    self.assertAllEqual(sentence, detokenized)

  @parameterized.parameters([
      (False, False, False, dtypes.int32),
      (False, False, True, dtypes.int32),
      (False, True, False, dtypes.int32),
      (False, True, True, dtypes.int32),
      (True, False, False, dtypes.int32),
      (True, False, True, dtypes.int32),
      (True, True, False, dtypes.int32),
      (True, True, True, dtypes.int32),
      (False, False, False, dtypes.string),
      (False, False, True, dtypes.string),
      (False, True, False, dtypes.string),
      (False, True, True, dtypes.string),
      (True, False, False, dtypes.string),
      (True, False, True, dtypes.string),
      (True, True, False, dtypes.string),
      (True, True, True, dtypes.string),
  ])
  def testTokenizeAndDetokenizeVec(self, reverse, add_bos, add_eos, out_type):
    sp = self.getTokenizerAndSetOptions(reverse, add_bos, add_eos, out_type)
    sentences = ['I love carpet', 'I love desk.', 'I love lamp.']
    expected = []
    if out_type == dtypes.int32:
      expected = [[9, 169, 21, 125, 78, 48, 132, 15], [9, 169, 21, 125, 727, 6],
                  [9, 169, 21, 125, 169, 579, 6]]
    else:
      expected = _utf8([['▁I', '▁l', 'o', 've', '▁c', 'ar', 'pe', 't'],
                        ['▁I', '▁l', 'o', 've', '▁desk', '.'],
                        ['▁I', '▁l', 'o', 've', '▁l', 'amp', '.']])
    result = sp.tokenize(sentences)
    self.assertRaggedEqual(expected, result)
    detokenized = sp.detokenize(result)
    self.assertAllEqual(sentences, detokenized)

  @parameterized.parameters([
      (False, False, False, dtypes.int32),
      (False, False, True, dtypes.int32),
      (False, True, False, dtypes.int32),
      (False, True, True, dtypes.int32),
      (True, False, False, dtypes.int32),
      (True, False, True, dtypes.int32),
      (True, True, False, dtypes.int32),
      (True, True, True, dtypes.int32),
      (False, False, False, dtypes.string),
      (False, False, True, dtypes.string),
      (False, True, False, dtypes.string),
      (False, True, True, dtypes.string),
      (True, False, False, dtypes.string),
      (True, False, True, dtypes.string),
      (True, True, False, dtypes.string),
      (True, True, True, dtypes.string),
  ])
  def testTokenizeAndDetokenizeUniformTensorMatrix(self, reverse, add_bos,
                                                   add_eos, out_type):
    sp = self.getTokenizerAndSetOptions(reverse, add_bos, add_eos, out_type)
    sentences = [['I love carpet', 'I love desk.'],
                 ['I love lamp.', 'Never tell me the odds']]
    expected = []
    if out_type == dtypes.int32:
      expected = [[[9, 169, 21, 125, 78, 48, 132, 15],
                   [9, 169, 21, 125, 727, 6]],
                  [[9, 169, 21, 125, 169, 579, 6],
                   [4, 199, 363, 310, 33, 7, 4, 21, 17, 17, 8]]]
    else:
      expected = _utf8(
          [[['▁I', '▁l', 'o', 've', '▁c', 'ar', 'pe', 't'],
            ['▁I', '▁l', 'o', 've', '▁desk', '.']],
           [['▁I', '▁l', 'o', 've', '▁l', 'amp', '.'],
            ['▁', 'N', 'ever', '▁tell', '▁me', '▁the', '▁', 'o', 'd', 'd',
             's']]])
    result = sp.tokenize(constant_op.constant(sentences))
    self.assertRaggedEqual(expected, result)
    detokenized = sp.detokenize(result)
    self.assertAllEqual(sentences, detokenized)

  @parameterized.parameters([
      (False, False, False, dtypes.int32),
      (False, False, True, dtypes.int32),
      (False, True, False, dtypes.int32),
      (False, True, True, dtypes.int32),
      (True, False, False, dtypes.int32),
      (True, False, True, dtypes.int32),
      (True, True, False, dtypes.int32),
      (True, True, True, dtypes.int32),
      (False, False, False, dtypes.string),
      (False, False, True, dtypes.string),
      (False, True, False, dtypes.string),
      (False, True, True, dtypes.string),
      (True, False, False, dtypes.string),
      (True, False, True, dtypes.string),
      (True, True, False, dtypes.string),
      (True, True, True, dtypes.string),
  ])
  def testTokenizeAndDetokenizeRaggedMatrix(self, reverse, add_bos, add_eos,
                                            out_type):
    sp = self.getTokenizerAndSetOptions(reverse, add_bos, add_eos, out_type)
    sentences = [['I love carpet', 'I love desk.', 'I love lamp.'],
                 ['Never tell me the odds']]
    expected = []
    if out_type == dtypes.int32:
      expected = [[[9, 169, 21, 125, 78, 48, 132, 15],
                   [9, 169, 21, 125, 727, 6], [9, 169, 21, 125, 169, 579, 6]],
                  [[4, 199, 363, 310, 33, 7, 4, 21, 17, 17, 8]]]
    else:
      expected = _utf8(
          [[['▁I', '▁l', 'o', 've', '▁c', 'ar', 'pe', 't'],
            ['▁I', '▁l', 'o', 've', '▁desk', '.'],
            ['▁I', '▁l', 'o', 've', '▁l', 'amp', '.']],
           [['▁', 'N', 'ever', '▁tell', '▁me', '▁the', '▁', 'o', 'd', 'd',
             's']]])
    result = sp.tokenize(ragged_factory_ops.constant(sentences))
    self.assertRaggedEqual(expected, result)
    detokenized = sp.detokenize(result)
    self.assertRaggedEqual(sentences, detokenized)

  @parameterized.parameters([
      (
          False,
          False,
          False,
          dtypes.int32,
          [0, 1, 3, 4, 6, 8, 11],
          [1, 3, 4, 6, 8, 11, 12]),
      (
          False,
          False,
          True,
          dtypes.int32,
          [0, 1, 3, 4, 6, 8, 11, 0],
          [1, 3, 4, 6, 8, 11, 12, 0]),
      (
          False,
          True,
          False,
          dtypes.int32,
          [0, 0, 1, 3, 4, 6, 8, 11],
          [0, 1, 3, 4, 6, 8, 11, 12]),
      (
          False,
          True,
          True,
          dtypes.int32,
          [0, 0, 1, 3, 4, 6, 8, 11, 0],
          [0, 1, 3, 4, 6, 8, 11, 12, 0]),
      (
          True,
          False,
          False,
          dtypes.int32,
          [11, 8, 6, 4, 3, 1, 0],
          [12, 11, 8, 6, 4, 3, 1]),
      (
          True,
          False,
          True,
          dtypes.int32,
          [0, 11, 8, 6, 4, 3, 1, 0],
          [0, 12, 11, 8, 6, 4, 3, 1]),
      (
          True,
          True,
          False,
          dtypes.int32,
          [11, 8, 6, 4, 3, 1, 0, 0],
          [12, 11, 8, 6, 4, 3, 1, 0]),
      (
          True,
          True,
          True,
          dtypes.int32,
          [0, 11, 8, 6, 4, 3, 1, 0, 0],
          [0, 12, 11, 8, 6, 4, 3, 1, 0]),
      (
          False,
          False,
          False,
          dtypes.string,
          [0, 1, 3, 4, 6, 8, 11],
          [1, 3, 4, 6, 8, 11, 12]),
      (
          False,
          False,
          True,
          dtypes.string,
          [0, 1, 3, 4, 6, 8, 11, 0],
          [1, 3, 4, 6, 8, 11, 12, 0]),
      (
          False,
          True,
          False,
          dtypes.string,
          [0, 0, 1, 3, 4, 6, 8, 11],
          [0, 1, 3, 4, 6, 8, 11, 12]),
      (
          False,
          True,
          True,
          dtypes.string,
          [0, 0, 1, 3, 4, 6, 8, 11, 0],
          [0, 1, 3, 4, 6, 8, 11, 12, 0]),
      (
          True,
          False,
          False,
          dtypes.string,
          [11, 8, 6, 4, 3, 1, 0],
          [12, 11, 8, 6, 4, 3, 1]),
      (
          True,
          False,
          True,
          dtypes.string,
          [0, 11, 8, 6, 4, 3, 1, 0],
          [0, 12, 11, 8, 6, 4, 3, 1]),
      (
          True,
          True,
          False,
          dtypes.string,
          [11, 8, 6, 4, 3, 1, 0, 0],
          [12, 11, 8, 6, 4, 3, 1, 0]),
      (
          True,
          True,
          True,
          dtypes.string,
          [0, 11, 8, 6, 4, 3, 1, 0, 0],
          [0, 12, 11, 8, 6, 4, 3, 1, 0]),
  ])  # pyformat: disable
  def testTokenizeAndDetokenizeWithOffsetsScalar(self, reverse, add_bos,
                                                 add_eos, out_type,
                                                 expected_starts,
                                                 expected_limits):
    sp = self.getTokenizerAndSetOptions(reverse, add_bos, add_eos, out_type)
    sentence = 'I love lamp.'
    expected_tok = []
    if out_type == dtypes.int32:
      expected_tok = [9, 169, 21, 125, 169, 579, 6]
    else:
      expected_tok = _utf8(['▁I', '▁l', 'o', 've', '▁l', 'amp', '.'])
    (tokens, starts,
     limits) = sp.tokenize_with_offsets(ragged_factory_ops.constant(sentence))
    self.assertRaggedEqual(expected_tok, tokens)
    self.assertRaggedEqual(expected_starts, starts)
    self.assertRaggedEqual(expected_limits, limits)

  def testTokenizeAndDetokenizeWithOffsetsSingleElementVector(self):
    sp = SentencepieceTokenizer(self.model, out_type=dtypes.string)
    sentences = ['I love lamp.']
    expected_tokens = [['▁I', '▁l', 'o', 've', '▁l', 'amp', '.']]
    expected_tokens = _utf8(expected_tokens)
    expected_starts = [[0, 1, 3, 4, 6, 8, 11]]
    expected_limits = [[1, 3, 4, 6, 8, 11, 12]]
    (tokens, starts,
     limits) = sp.tokenize_with_offsets(ragged_factory_ops.constant(sentences))
    self.assertRaggedEqual(expected_tokens, tokens)
    self.assertRaggedEqual(expected_starts, starts)
    self.assertRaggedEqual(expected_limits, limits)

  def testTokenizeAndDetokenizeWithOffsetsVector(self):
    sp = SentencepieceTokenizer(self.model, out_type=dtypes.string)
    sentences = ['I love carpet.', 'I love desk.', 'I love lamp.']
    expected_tokens = [['▁I', '▁l', 'o', 've', '▁c', 'ar', 'pe', 't', '.'],
                       ['▁I', '▁l', 'o', 've', '▁desk', '.'],
                       ['▁I', '▁l', 'o', 've', '▁l', 'amp', '.']]
    expected_tokens = _utf8(expected_tokens)
    expected_starts = [[0, 1, 3, 4, 6, 8, 10, 12, 13], [0, 1, 3, 4, 6, 11],
                       [0, 1, 3, 4, 6, 8, 11]]
    expected_limits = [[1, 3, 4, 6, 8, 10, 12, 13, 14], [1, 3, 4, 6, 11, 12],
                       [1, 3, 4, 6, 8, 11, 12]]
    (tokens, starts,
     limits) = sp.tokenize_with_offsets(ragged_factory_ops.constant(sentences))
    self.assertRaggedEqual(expected_tokens, tokens)
    self.assertRaggedEqual(expected_starts, starts)
    self.assertRaggedEqual(expected_limits, limits)

  def testTokenizeAndDetokenizeWithOffsetsMatrix(self):
    sp = SentencepieceTokenizer(self.model, out_type=dtypes.string)
    sentences = [['I love carpet.', 'I love desk.', 'I love lamp.'],
                 ['Never tell me the odds']]
    expected_tokens = [[['▁I', '▁l', 'o', 've', '▁c', 'ar', 'pe', 't', '.'],
                        ['▁I', '▁l', 'o', 've', '▁desk', '.'],
                        ['▁I', '▁l', 'o', 've', '▁l', 'amp', '.']],
                       [[
                           '▁', 'N', 'ever', '▁tell', '▁me', '▁the', '▁', 'o',
                           'd', 'd', 's'
                       ]]]
    expected_tokens = _utf8(expected_tokens)
    expected_starts = [[[0, 1, 3, 4, 6, 8, 10, 12, 13], [0, 1, 3, 4, 6, 11],
                        [0, 1, 3, 4, 6, 8, 11]],
                       [[0, 0, 1, 5, 10, 13, 17, 18, 19, 20, 21]]]
    expected_limits = [[[1, 3, 4, 6, 8, 10, 12, 13, 14], [1, 3, 4, 6, 11, 12],
                        [1, 3, 4, 6, 8, 11, 12]],
                       [[0, 1, 5, 10, 13, 17, 18, 19, 20, 21, 22]]]
    (tokens, starts,
     limits) = sp.tokenize_with_offsets(ragged_factory_ops.constant(sentences))
    self.assertRaggedEqual(expected_tokens, tokens)
    self.assertRaggedEqual(expected_starts, starts)
    self.assertRaggedEqual(expected_limits, limits)

  @parameterized.parameters([
      (-1, 0.1, dtypes.int32),
      (64, 0.1, dtypes.int32),
      (0, 0.0, dtypes.int32),
      (-1, 0.1, dtypes.string),
      (64, 0.1, dtypes.string),
      (0, 0.0, dtypes.string),
  ])
  def testSampleTokenizeAndDetokenize(self, nbest_size, alpha, out_type):
    sp = SentencepieceTokenizer(
        self.model, nbest_size=nbest_size, alpha=alpha, out_type=out_type)
    sentences = [['I love carpet', 'I love desk.', 'I love lamp.'],
                 ['Never tell me the odds']]
    result = sp.tokenize(ragged_factory_ops.constant(sentences))
    detokenized = sp.detokenize(result)
    self.assertRaggedEqual(sentences, detokenized)

  def testEmptyModel(self):
    with self.cached_session():
      with self.assertRaises(errors.InvalidArgumentError):
        sp = SentencepieceTokenizer()
        result = sp.tokenize('whatever')
        result.eval()

  def testInvalidModel(self):
    with self.cached_session():
      with self.assertRaises(errors.InternalError):
        sp = SentencepieceTokenizer('invalid model')
        result = sp.tokenize('whatever')
        result.eval()


if __name__ == '▁_main__':
  test.main()
