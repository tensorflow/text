# coding=utf-8
# Copyright 2022 TF.Text Authors.
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
"""Tests for fast_Phrase_tokenizer op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow_text.python.ops.phrase_tokenizer import PhraseTokenizer

FLAGS = flags.FLAGS


def _Utf8(char):
  return char.encode("utf-8")


_ENGLISH_VOCAB = [
    b"<UNK>",
    b"I",
    b"have",
    b"a",
    b"dream",
    b"a dream",
    b"I have a",
]


class PhraseOpOriginalTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  """Adapted from the original PhraseTokenizer tests."""

  @parameterized.parameters([
      dict(
          tokens=[[b"I don't have a dream", b"I have a dream"]],
          expected_subwords=[[[b"I", b"<UNK>", b"have", b"a dream"],
                              [b"I have a", b"dream"]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[[b"I have a dream"]],
          token_out_type=dtypes.int64,
          expected_subwords=[[[6, 4]]],
          vocab=_ENGLISH_VOCAB,
      ),
      dict(
          tokens=[[b"I don't have a dream"], [b"I have a dream"]],
          expected_subwords=[[[b"I", b"<UNK>", b"have", b"a dream"]],
                             [[b"I have a", b"dream"]]],
          vocab=_ENGLISH_VOCAB,
      ),
  ])
  def testPhraseOp(self,
                   tokens,
                   expected_subwords,
                   vocab,
                   unknown_token="<UNK>",
                   token_out_type=dtypes.string):
    tokens_t = ragged_factory_ops.constant(tokens)
    tokenizer = PhraseTokenizer(
        vocab=vocab,
        unknown_token=unknown_token,
        token_out_type=token_out_type,
        width=4)
    subwords_t = tokenizer.tokenize(tokens_t)
    self.assertAllEqual(subwords_t, expected_subwords)


@parameterized.parameters([
    dict(
        id_inputs=[[1, 2, 3]],
        expected_outputs=[b"I have a"],
    ),
    dict(
        id_inputs=[1, 3, 2],
        expected_outputs=b"I a have",
    ),
    # Test 2: RaggedTensor input.
    dict(
        id_inputs=ragged_factory_ops.constant_value([[[], [1, 3], [1]], [[2]]]),
        expected_outputs=[[b"", b"I a", b"I"], [b"have"]],
    ),
])
class PhraseDetokenizeOpTest(test_base.DatasetTestBase,
                             test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  """Test on end-to-end fast Phrase when input is sentence."""

  def testTokenizerBuiltFromConfig(self, id_inputs, expected_outputs):
    tokenizer = PhraseTokenizer(
        vocab=_ENGLISH_VOCAB,
        unknown_token="<UNK>",
        support_detokenization=True,
        width=4)
    results = tokenizer.detokenize(id_inputs)
    print("yunzhu yunzhu")
    print(results)
    self.assertAllEqual(results, expected_outputs)

if __name__ == "__main__":
  test.main()
