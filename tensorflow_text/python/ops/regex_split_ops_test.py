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

# encoding=utf-8
# Lint as: python3
"""Tests for regex_split and regex_split_with_offsets ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow_text.python.ops import regex_split_ops


def _utf8(char):
  return char.encode("utf-8")


# TODO(thuang513): It appears there isn't a Ragged version of substr; consider
#               checking this into core TF.
def _ragged_substr(text_input, begin, end):
  text_input_flat = None
  if ragged_tensor.is_ragged(text_input):
    text_input_flat = text_input.flat_values
  else:
    text_input_flat = text_input
  broadcasted_text = array_ops.gather_v2(text_input_flat,
                                         begin.nested_value_rowids()[-1])
  size = math_ops.sub(end.flat_values, begin.flat_values)
  new_tokens = string_ops.substr_v2(broadcasted_text, begin.flat_values, size)
  return begin.with_flat_values(new_tokens)


@test_util.run_all_in_graph_and_eager_modes
class RegexSplitOpsTest(parameterized.TestCase, test.TestCase):

  @parameterized.parameters([
      # Test doc string examples
      dict(
          text_input=[r"hello there"],
          delim_regex_pattern=r"\s",
          keep_delim_regex_pattern=r"\s",
          expected=[[b"hello", b" ", b"there"]],
      ),
      # Test simple whitespace
      dict(
          text_input=[r"hello there"],
          delim_regex_pattern=r"\s",
          expected=[[b"hello", b"there"]],
      ),
      # Two delimiters in a row
      dict(
          text_input=[r"hello  there"],
          delim_regex_pattern=r"\s",
          expected=[[b"hello", b"there"]],
      ),
      # Test Hiragana
      dict(
          text_input=[_utf8(u"では４日")],
          delim_regex_pattern=r"\p{Hiragana}",
          keep_delim_regex_pattern=r"\p{Hiragana}",
          expected=[[_utf8(u"で"), _utf8(u"は"),
                     _utf8(u"４日")]],
      ),
      # Test symbols and punctuation
      dict(
          text_input=[r"hello! (:$) there"],
          delim_regex_pattern=r"[\p{S}|\p{P}]+|\s",
          keep_delim_regex_pattern=r"[\p{S}|\p{P}]+",
          expected=[[b"hello", b"!", b"(:$)", b"there"]],
      ),
      # Test numbers
      dict(
          text_input=[r"hello12345there"],
          delim_regex_pattern=r"\p{N}+",
          keep_delim_regex_pattern=r"\p{N}+",
          expected=[[b"hello", b"12345", b"there"]],
      ),
      # Test numbers and symbols
      dict(
          text_input=[r"show me some $100 bills yo!"],
          delim_regex_pattern=r"\s|\p{S}",
          keep_delim_regex_pattern=r"\p{S}",
          expected=[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"]],
      ),
      # Test input Tensor with shape = [2], rank = 1
      dict(
          text_input=[
              r"show me some $100 bills yo!",
              r"hello there",
          ],
          delim_regex_pattern=r"\s|\p{S}",
          keep_delim_regex_pattern=r"\p{S}",
          expected=[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"],
                    [b"hello", b"there"]],
      ),
      # Test input RaggedTensor with ragged ranks; shape = [2, (1, 2)]
      dict(
          text_input=[
              [b"show me some $100 bills yo!",
               _utf8(u"では４日")],
              [b"hello there"],
          ],
          delim_regex_pattern=r"\s|\p{S}|\p{Hiragana}",
          keep_delim_regex_pattern=r"\p{S}|\p{Hiragana}",
          expected=[[[b"show", b"me", b"some", b"$", b"100", b"bills", b"yo!"],
                     [_utf8(u"で"), _utf8(u"は"),
                      _utf8(u"４日")]], [[b"hello", b"there"]]],
      ),
  ])
  def testRegexSplitOp(self,
                       text_input,
                       delim_regex_pattern,
                       expected,
                       keep_delim_regex_pattern=r""):
    text_input = ragged_factory_ops.constant(text_input)
    actual_tokens, start, end = regex_split_ops.regex_split_with_offsets(
        input=text_input,
        delim_regex_pattern=delim_regex_pattern,
        keep_delim_regex_pattern=keep_delim_regex_pattern,
    )
    self.assertAllEqual(actual_tokens, expected)

    # Use the offsets to extract substrings and verify that the substrings match
    # up with the expected tokens
    extracted_tokens = _ragged_substr(text_input, start, end)
    self.assertAllEqual(extracted_tokens, expected)


if __name__ == "__main__":
  test.main()
