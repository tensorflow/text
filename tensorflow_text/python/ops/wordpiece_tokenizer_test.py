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

# encoding=utf-8
"""Tests for wordpiece_tokenized op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import test
from tensorflow_text.python.ops.wordpiece_tokenizer import WordpieceTokenizer


def _Utf8(char):
  return char.encode("utf-8")


def _CreateTable(vocab, num_oov=1):
  init = lookup_ops.KeyValueTensorInitializer(
      vocab,
      math_ops.range(
          array_ops.size(vocab, out_type=dtypes.int64), dtype=dtypes.int64),
      key_dtype=dtypes.string,
      value_dtype=dtypes.int64)
  return lookup_ops.StaticVocabularyTableV1(
      init, num_oov, lookup_key_dtype=dtypes.string)


_ENGLISH_VOCAB = [
    "don",
    "##'",
    "##t",
    "tread",
    "##ness",
    "hel",
    "##lo",
    "there",
    "my",
    "na",
    "##me",
    "is",
    "ter",
    "##ry",
    "what",
    "##cha",
    "##ma",
    "##call",
    "##it?",
    "you",
    "said",
]

_CHINESE_VOCAB = [
    _Utf8(u"貿"),
    _Utf8(u"易"),
    _Utf8(u"戰"),
    _Utf8(u"最"),
    _Utf8(u"大"),
    _Utf8(u"受"),
    _Utf8(u"益"),
    _Utf8(u"者"),
    _Utf8(u"越"),
    _Utf8(u"南"),
    _Utf8(u"總"),
    _Utf8(u"理"),
    _Utf8(u"阮"),
    _Utf8(u"春"),
    _Utf8(u"福"),
]

_MIXED_LANG_VOCAB = [
    "don",
    "##'",
    "##t",
    "tread",
    "##ness",
    "hel",
    "##lo",
    "there",
    "my",
    "na",
    "##me",
    "is",
    "ter",
    "##ry",
    "what",
    "##cha",
    "##ma",
    "##call",
    "##it?",
    "you",
    "said",
    _Utf8(u"貿"),
    _Utf8(u"易"),
    _Utf8(u"戰"),
    _Utf8(u"最"),
    _Utf8(u"大"),
    _Utf8(u"受"),
    _Utf8(u"益"),
    _Utf8(u"者"),
    _Utf8(u"越"),
    _Utf8(u"南"),
    _Utf8(u"總"),
    _Utf8(u"理"),
    _Utf8(u"阮"),
    _Utf8(u"春"),
    _Utf8(u"福"),
]

_RUSSIAN_VOCAB = [
    _Utf8(u"к"),
    _Utf8(u"##уп"),
    _Utf8(u"##иха"),
]

_DEATH_VOCAB = [
    _Utf8(u"क"),
    _Utf8(u"##र"),
    _Utf8(u"##े"),
    _Utf8(u"##ं"),
    "##*",
    _Utf8(u"##👇"),
]


def _GetTokensFromWordpieceOffsets(tokens, begin_indices, end_indices):
  tokens = tokens.to_list()
  begin_indices = begin_indices.to_list()
  end_indices = end_indices.to_list()
  result = []
  for docs_idx in xrange(len(tokens)):
    tokens_in_doc = []
    for tokens_idx in xrange(len(tokens[docs_idx])):
      token = bytes(tokens[docs_idx][tokens_idx])
      begin_offsets = begin_indices[docs_idx][tokens_idx]
      end_offsets = end_indices[docs_idx][tokens_idx]
      tokens_in_doc.append("".join(
          [token[begin:end] for begin, end in zip(begin_offsets, end_offsets)]))
    result.append(tokens_in_doc)
  return result


class WordpieceOpTest(ragged_test_util.RaggedTensorTestCase,
                      parameterized.TestCase):

  @parameterized.parameters([
      # Basic case
      dict(
          tokens=[[_Utf8(u"купиха")]],
          expected_subwords=[[[
              _Utf8(u"к"),
              _Utf8(u"##уп"),
              _Utf8(u"##иха"),
          ]]],
          vocab=_RUSSIAN_VOCAB,
      ),
      dict(
          tokens=[["don't", "treadness"]],
          expected_subwords=[[["don", "##'", "##t"], ["tread", "##ness"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[["hello", "there", "my", "name", "is", "terry"],
                  ["whatchamacallit?", "you", "said"]],
          expected_subwords=[[["hel", "##lo"], ["there"], ["my"],
                              ["na", "##me"], ["is"], ["ter", "##ry"]],
                             [["what", "##cha", "##ma", "##call", "##it?"],
                              ["you"], ["said"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      # Basic case w/ unknown token
      dict(
          tokens=[["don't", "tread", "cantfindme", "treadcantfindme"]],
          expected_subwords=[[["don", "##'", "##t"], ["tread"], ["[UNK]"],
                              ["[UNK]"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      # Basic case w/o unknown token
      dict(
          tokens=[["don't", "tread", "cantfindme", "treadcantfindme"]],
          expected_subwords=[[["don", "##'", "##t"], ["tread"], ["cantfindme"],
                              ["treadcantfindme"]]],
          unknown_token=None,
          vocab=_ENGLISH_VOCAB,
      ),
      # Basic case w/ int id lookup
      dict(
          tokens=[["don't", "tread", "cantfindme", "treadcantfindme"]],
          token_out_type=dtypes.int64,
          expected_subwords=[[[0, 1, 2], [3], [21], [21]]],
          vocab=_ENGLISH_VOCAB,
      ),
      # Chinese test case
      dict(
          tokens=[[
              _Utf8(u"貿"),
              _Utf8(u"易"),
              _Utf8(u"戰"),
              _Utf8(u"最"),
              _Utf8(u"大"),
              _Utf8(u"受"),
              _Utf8(u"益"),
              _Utf8(u"者")
          ],
                  [
                      _Utf8(u"越"),
                      _Utf8(u"南"),
                      _Utf8(u"總"),
                      _Utf8(u"理"),
                      _Utf8(u"阮"),
                      _Utf8(u"春"),
                      _Utf8(u"福")
                  ]],
          expected_subwords=[[[_Utf8(u"貿")], [_Utf8(u"易")], [_Utf8(u"戰")],
                              [_Utf8(u"最")], [_Utf8(u"大")], [_Utf8(u"受")],
                              [_Utf8(u"益")], [_Utf8(u"者")]],
                             [[_Utf8(u"越")], [_Utf8(u"南")], [_Utf8(u"總")],
                              [_Utf8(u"理")], [_Utf8(u"阮")], [_Utf8(u"春")],
                              [_Utf8(u"福")]]],
          vocab=_CHINESE_VOCAB,
      ),
      # Mixed lang test cases
      dict(
          tokens=[
              [
                  _Utf8(u"貿"),
                  _Utf8(u"易"),
                  _Utf8(u"戰"),
                  _Utf8(u"最"),
                  _Utf8(u"大"),
                  _Utf8(u"受"),
                  _Utf8(u"益"),
                  _Utf8(u"者")
              ],
              [
                  _Utf8(u"越"),
                  _Utf8(u"南"),
                  _Utf8(u"總"),
                  _Utf8(u"理"),
                  _Utf8(u"阮"),
                  _Utf8(u"春"),
                  _Utf8(u"福")
              ],
              ["don't", "treadness"],
          ],
          expected_subwords=[
              [[_Utf8(u"貿")], [_Utf8(u"易")], [_Utf8(u"戰")], [_Utf8(u"最")],
               [_Utf8(u"大")], [_Utf8(u"受")], [_Utf8(u"益")], [_Utf8(u"者")]],
              [[_Utf8(u"越")], [_Utf8(u"南")], [_Utf8(u"總")], [_Utf8(u"理")],
               [_Utf8(u"阮")], [_Utf8(u"春")], [_Utf8(u"福")]],
              [["don", "##'", "##t"], ["tread", "##ness"]],
          ],
          vocab=_MIXED_LANG_VOCAB,
      ),
      # Test token whose size is > max_bytes_per_word
      dict(
          tokens=[["don't", "treadness"]],
          expected_subwords=[[["don", "##'", "##t"], ["[UNK]"]]],
          vocab=_ENGLISH_VOCAB,
          max_bytes_per_word=5,
          # Explicitly specify the offsets here because the current way of
          # testing offsets would require '[UNK]' to be part of tokens.
          expected_start=[[[0, 3, 4], [0]]],
          expected_limit=[[[3, 4, 5], [5]]],
      ),
      # Test the token of death usecase.
      dict(
          tokens=[[_Utf8(u"करें*👇👇")]],
          token_out_type=dtypes.string,
          expected_subwords=[[[
              _Utf8(u"क"),
              _Utf8(u"##र"),
              _Utf8(u"##े"),
              _Utf8(u"##ं"), "##*",
              _Utf8(u"##👇"),
              _Utf8(u"##👇")
          ]]],
          vocab=_DEATH_VOCAB,
          max_bytes_per_word=40,
      ),
  ])
  def testWordPieceOpAndVerifyOffsets(self,
                                      tokens,
                                      expected_subwords,
                                      vocab,
                                      expected_start=None,
                                      expected_limit=None,
                                      use_unknown_token=True,
                                      unknown_token="[UNK]",
                                      token_out_type=dtypes.string,
                                      max_bytes_per_word=100):
    tokens = ragged_factory_ops.constant(tokens)
    vocab_table = _CreateTable(vocab)
    self.evaluate(vocab_table.initializer)
    tokenizer = WordpieceTokenizer(
        vocab_table,
        unknown_token=unknown_token,
        token_out_type=token_out_type,
        max_bytes_per_word=max_bytes_per_word,
    )
    subwords, begin, end = tokenizer.tokenize_with_offsets(tokens)
    self.assertRaggedEqual(subwords, expected_subwords)

    # Verify the indices by performing the following:
    # - Extract the subwords and join them together to form the original tokens.
    # - Then compare the extracted tokens and original tokens.
    tokens, begin, end = (self.evaluate((tokens, begin, end)))

    # If expected start/limit offsets were provided, check them explicitly.
    # Otherwise test the offsets by extracting subwords using token offsets
    # from the original 'tokens' input.
    if expected_start is None or expected_limit is None:
      extracted_tokens = _GetTokensFromWordpieceOffsets(tokens, begin, end)
      self.assertRaggedEqual(extracted_tokens, tokens)
    else:
      self.assertRaggedEqual(begin, expected_start)
      self.assertRaggedEqual(end, expected_limit)

  @parameterized.parameters([
      dict(
          tokens=[[["don't"], ["treadness"],
                   ["whatchamacallit?", "you", "hello"]], [["treadness"]]],
          expected_subwords=[[[["don", "##'", "##t"]], [["tread", "##ness"]],
                              [["what", "##cha", "##ma", "##call", "##it?"],
                               ["you"], ["hel", "##lo"]]],
                             [[["tread", "##ness"]]]],
          vocab=_ENGLISH_VOCAB,
      ),
  ])
  def testWordPieceOpWithMultipleRaggedRank(self,
                                            tokens,
                                            expected_subwords,
                                            vocab,
                                            expected_start=None,
                                            expected_limit=None,
                                            use_unknown_token=True,
                                            token_out_type=dtypes.string):
    for row_splits_dtype in (dtypes.int32, dtypes.int64):
      ragged_tokens = ragged_factory_ops.constant(
          tokens, row_splits_dtype=row_splits_dtype)
      vocab_table = _CreateTable(vocab)
      self.evaluate(vocab_table.initializer)
      tokenizer = WordpieceTokenizer(vocab_table, token_out_type=token_out_type)
      subwords = tokenizer.tokenize(ragged_tokens)
      self.assertRaggedEqual(subwords, expected_subwords)

  def testWordPieceOpWithIdReturned(self):
    """Let the table determine how to do a lookup on unknown tokens."""
    tokens = ragged_factory_ops.constant(
        [["don't", "tread", "cantfindme", "treadcantfindme"]])
    vocab_table = _CreateTable(
        _ENGLISH_VOCAB,
        100  # OOV values
    )
    self.evaluate(vocab_table.initializer)
    tokenizer = WordpieceTokenizer(
        vocab_table, unknown_token=None, token_out_type=dtypes.int64)
    subwords, _, _ = tokenizer.tokenize_with_offsets(tokens)

    self.assertRaggedEqual(subwords, [[[0, 1, 2], [3], [96], [46]]])

  @parameterized.parameters([
      dict(
          tokens=[["don't", "treadness", "whatchamacallit?"]],
          expected_subwords=[[["don", "##'", "##t"], ["tread", "##ness"],
                              ["what", "##cha", "##ma", "##call", "##it?"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[[["don't"], ["treadness"], ["whatchamacallit?"]]],
          expected_subwords=[[[["don", "##'", "##t"]], [["tread", "##ness"]],
                              [["what", "##cha", "##ma", "##call", "##it?"]]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[[["don't", _Utf8(u"貿")], ["treadness",
                                            _Utf8(u"大")],
                   ["whatchamacallit?", _Utf8(u"福")]]],
          expected_subwords=[[[["don", "##'", "##t"], [_Utf8(u"貿")]],
                              [["tread", "##ness"], [_Utf8(u"大")]],
                              [["what", "##cha", "##ma", "##call", "##it?"],
                               [_Utf8(u"福")]]]],
          vocab=_MIXED_LANG_VOCAB,
      ),
      # Vector input
      dict(
          tokens=[_Utf8(u"купиха")],
          expected_subwords=[[
              _Utf8(u"к"),
              _Utf8(u"##уп"),
              _Utf8(u"##иха"),
          ]],
          vocab=_RUSSIAN_VOCAB,
      ),
      # Scalar input
      dict(
          tokens=_Utf8(u"купиха"),
          expected_subwords=[
              _Utf8(u"к"),
              _Utf8(u"##уп"),
              _Utf8(u"##иха"),
          ],
          vocab=_RUSSIAN_VOCAB,
      ),
      # 3D input with 1 ragged dimension.
      dict(
          tokens=[["don't", "treadness", "whatchamacallit?"]],
          expected_subwords=[[["don", "##'", "##t"], ["tread", "##ness"],
                              ["what", "##cha", "##ma", "##call", "##it?"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=ragged_factory_ops.constant_value(
              [[["don't"], ["treadness"], ["whatchamacallit?"]]],
              ragged_rank=1),
          expected_subwords=[[[["don", "##'", "##t"]], [["tread", "##ness"]],
                              [["what", "##cha", "##ma", "##call", "##it?"]]]],
          vocab=_ENGLISH_VOCAB,
      ),
  ])
  def testTensors(self,
                  tokens,
                  expected_subwords,
                  vocab,
                  expected_start=None,
                  expected_limit=None,
                  use_unknown_token=True,
                  token_out_type=dtypes.string):
    vocab_table = _CreateTable(vocab)
    self.evaluate(vocab_table.initializer)
    tokenizer = WordpieceTokenizer(vocab_table, token_out_type=token_out_type)
    subwords = tokenizer.tokenize(tokens)
    self.assertRaggedEqual(subwords, expected_subwords)


if __name__ == "__main__":
  test.main()
