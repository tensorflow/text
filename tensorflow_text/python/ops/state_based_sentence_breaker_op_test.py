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

"""Tests for sentence_breaking_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import state_based_sentence_breaker_op


class SentenceFragmenterTestCasesV2(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          test_description="Test acronyms",
          doc=["Welcome to the U.S. don't be surprised."],
          expected_fragment_text=[[
              b"Welcome to the U.S.", b"don't be surprised."
          ]],
      ),
      dict(
          test_description="Test batch containing acronyms",
          doc=["Welcome to the U.S. don't be surprised.", "I.B.M. yo"],
          expected_fragment_text=[[
              b"Welcome to the U.S.", b"don't be surprised."
          ], [b"I.B.M.", b"yo"]],
      ),
      dict(
          test_description="Test when rank > 1.",
          doc=[["Welcome to the U.S. don't be surprised."], ["I.B.M. yo"]],
          expected_fragment_text=[[
              b"Welcome to the U.S.", b"don't be surprised."
          ], [b"I.B.M.", b"yo"]],
      ),
      dict(
          test_description="Test semicolons",
          doc=["Welcome to the US; don't be surprised."],
          expected_fragment_text=[[b"Welcome to the US; don't be surprised."]],
      ),
      dict(
          test_description="Basic test",
          doc=["Hello. Foo bar!"],
          expected_fragment_text=[[b"Hello.", b"Foo bar!"]],
      ),
      dict(
          test_description="Basic ellipsis test",
          doc=["Hello...foo bar"],
          expected_fragment_text=[[b"Hello...", b"foo bar"]],
      ),
      dict(
          test_description="Parentheses and ellipsis test",
          doc=["Hello (who are you...) foo bar"],
          expected_fragment_text=[[b"Hello (who are you...)", b"foo bar"]],
      ),
      dict(
          test_description="Punctuation after parentheses test",
          doc=["Hello (who are you)? Foo bar!"],
          expected_fragment_text=[[b"Hello (who are you)?", b"Foo bar!"]],
      ),
      dict(
          test_description="MidFragment Parentheses test",
          doc=["Hello (who are you) world? Foo bar"],
          expected_fragment_text=[[b"Hello (who are you) world?", b"Foo bar"]],
      ),
      dict(
          test_description="Many final punctuation test",
          doc=["Hello!!!!! Who are you??"],
          expected_fragment_text=[[b"Hello!!!!!", b"Who are you??"]],
      ),
      dict(
          test_description="Test emoticons within text",
          doc=["Hello world :) Oh, hi :-O"],
          expected_fragment_text=[[b"Hello world :)", b"Oh, hi :-O"]],
      ),
      dict(
          test_description="Test emoticons with punctuation following",
          doc=["Hello world :)! Hi."],
          expected_fragment_text=[[b"Hello world :)!", b"Hi."]],
      ),
      dict(
          test_description="Test emoticon list",
          doc=[b":) :-\\ (=^..^=) |-O"],
          expected_fragment_text=[[b":)", b":-\\", b"(=^..^=)", b"|-O"]],
      ),
      dict(
          test_description="Test emoticon batch",
          doc=[":)", ":-\\", "(=^..^=)", "|-O"],
          expected_fragment_text=[[b":)"], [b":-\\"], [b"(=^..^=)"], [b"|-O"]],
      ),
  ])
  def testStateBasedSentenceBreaker(self, test_description, doc,
                                    expected_fragment_text):

    doc = constant_op.constant(doc)
    sentence_breaker = (
        state_based_sentence_breaker_op.StateBasedSentenceBreaker())
    fragment_text, fragment_starts, fragment_ends = self.evaluate(
        sentence_breaker.break_sentences_with_offsets(doc))

    # Reshape tensors for fragment extraction.
    doc_reshape = array_ops.reshape(doc, [-1, 1])
    doc_reshape = self.evaluate(doc_reshape)
    fragment_starts = fragment_starts.to_list()
    fragment_ends = fragment_ends.to_list()
    offset_fragment_text = []

    # Extract the fragments from doc based on the offsets obtained
    for line, starts, ends in zip(doc_reshape, fragment_starts, fragment_ends):
      temp_offset_fragment_text = []
      for frag in line:
        for start_index, end_index in zip(starts, ends):
          if end_index < len(frag):
            temp_offset_fragment_text.append(frag[start_index:end_index])
          else:
            temp_offset_fragment_text.append(frag[start_index:])
      offset_fragment_text.append(temp_offset_fragment_text)

    # Check that the expected_fragment_text matches both the fragments returned
    # by the op and the fragments extracted from the doc using the offsets
    # returned by the op.
    self.assertAllEqual(expected_fragment_text, offset_fragment_text)
    self.assertAllEqual(expected_fragment_text, fragment_text)


if __name__ == "__main__":
  test.main()
