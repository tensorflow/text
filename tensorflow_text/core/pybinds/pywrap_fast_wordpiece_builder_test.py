# coding=utf-8
# Copyright 2021 TF.Text Authors.
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

r"""Tests for pywrap_fast_wordpiece_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags

from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow_text.core.pybinds import pywrap_fast_wordpiece_builder

FLAGS = flags.FLAGS

EXPECTED_MODEL_BUFFER_PATH = "google3/third_party/tensorflow_text/core/kernels/testdata/fast_wordpiece_tokenizer_config.fb"


class PywrapFastWordpieceBuilderTest(test_util.TensorFlowTestCase):

  def test_build(self):
    vocab = [
        "a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f", "##ghz",
        "<unk>"
    ]
    max_bytes_per_token = 100
    suffix_indicator = "##"
    unk_token = "<unk>"
    expected_model_buffer = gfile.GFile(
        os.path.join(FLAGS.test_srcdir, EXPECTED_MODEL_BUFFER_PATH),
        "rb").read()
    self.assertEqual(
        pywrap_fast_wordpiece_builder.build_fast_wordpiece_model(
            vocab, max_bytes_per_token, suffix_indicator, unk_token, False,
            False), expected_model_buffer)

  def test_build_throw_exception_unk_token_not_in_vocab(self):
    vocab = [
        "a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f", "##ghz"
    ]
    max_bytes_per_token = 100
    suffix_indicator = "##"
    unk_token = "<unk>"
    with self.assertRaisesRegex(RuntimeError,
                                "Cannot find unk_token in the vocab!"):
      pywrap_fast_wordpiece_builder.build_fast_wordpiece_model(
          vocab, max_bytes_per_token, suffix_indicator, unk_token, False, False)


if __name__ == "__main__":
  test.main()
