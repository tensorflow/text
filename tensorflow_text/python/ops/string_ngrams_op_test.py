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

"""Tests for google3.third_party.tensorflow_text.python.ops.string_ngrams."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import string_ngrams_op


class StringNgramsTest(test_util.TensorFlowTestCase):

  def test_unpadded_ngrams(self):
    data = [["aa", "bb", "cc", "dd"], ["ee", "ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = string_ngrams_op.string_ngrams(
        data_tensor, ngram_width=3, separator="|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [["aa|bb|cc", "bb|cc|dd"], []]
    self.assertAllEqual(expected_ngrams, result)

  def test_fully_padded_ngrams(self):
    data = [["aa"], ["bb", "cc", "dd"], ["ee", "ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = string_ngrams_op.string_ngrams(
        data_tensor,
        ngram_width=3,
        separator="|",
        ngram_pad_values=("LP", "RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [["LP|LP|aa", "LP|aa|RP", "aa|RP|RP"],
                       [
                           "LP|LP|bb", "LP|bb|cc", "bb|cc|dd", "cc|dd|RP",
                           "dd|RP|RP"
                       ], ["LP|LP|ee", "LP|ee|ff", "ee|ff|RP", "ff|RP|RP"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_singly_padded_ngrams(self):
    data = [["a"], ["b", "c", "d"], ["e", "f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = string_ngrams_op.string_ngrams(
        data_tensor,
        ngram_width=5,
        separator="|",
        ngram_pad_values=("LP", "RP"),
        extend_padding=False)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[], ["LP|b|c|d|RP"], []]
    self.assertAllEqual(expected_ngrams, result)


if __name__ == "__main__":
  test.main()
