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
"""Tests for sentence predictors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized

from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import segment_extractor_ops


@test_util.run_all_in_graph_and_eager_modes
class NextSentencePredictorExtractorTest(parameterized.TestCase, test.TestCase):

  @parameterized.parameters([
      # Basic example - all next sentences
      dict(
          test_description="all next sentences",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."]],
          expected_segment_a=[b"Hello there.", b"La la la.", b"Such is life."],
          expected_segment_b=[b"La la la.", b"Such is life.", b"Hello there."],
          expected_labels=[True, True, True],
          random_next_sentence_threshold=0.0,
      ),
      # Basic example - all random sentences
      dict(
          test_description="all random sentences",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."]],
          expected_segment_a=[b"Hello there.", b"La la la.", b"Such is life."],
          expected_segment_b=[b"Such is life.", b"La la la.", b"Hello there."],
          expected_labels=[False, False, False],
          random_next_sentence_threshold=1.0,
      ),
      # all random - sentences is a RaggedTensor w/ shape [2, (3, 2)]
      dict(
          test_description="all random - sentence has batch = 2",
          sentences=[[b"Hello there.", b"La la la.", b"Such is life."],
                     [b"Who let the dogs out?", b"Who?."]],
          expected_segment_a=[
              b"Hello there.", b"La la la.", b"Such is life.",
              b"Who let the dogs out?", b"Who?."
          ],
          expected_segment_b=[
              b"Who?.", b"Who let the dogs out?", b"Such is life.",
              b"La la la.", b"Hello there."
          ],
          expected_labels=[
              False,
              False,
              False,
              False,
              False,
          ],
          random_next_sentence_threshold=1.0,
      ),
  ])
  def testNextSentencePredictionExtractor(self,
                                          sentences,
                                          expected_segment_a,
                                          expected_segment_b,
                                          expected_labels,
                                          random_next_sentence_threshold=0.5,
                                          test_description=""):
    sentences = ragged_factory_ops.constant(sentences)
    # Set seed and rig the shuffle function to a deterministic reverse function
    # instead. This is so that we have consistent and deterministic results.
    random_seed.set_seed(1234)
    nsp = segment_extractor_ops.NextSentencePredictionExtractor(
        shuffle_fn=functools.partial(array_ops.reverse, axis=[-1]),
        random_next_sentence_threshold=random_next_sentence_threshold,
    )
    results = nsp.get_segments(sentences)
    actual_segment_a, actual_segment_b, actual_labels = results
    self.assertAllEqual(expected_segment_a, actual_segment_a)
    self.assertAllEqual(expected_segment_b, actual_segment_b)
    self.assertAllEqual(expected_labels, actual_labels)


if __name__ == "__main__":
  test.main()
