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

# Lint as: python3
"""Module for extracting segments from sentences in documents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_where_op


class NextSentencePredictionExtractor(object):
  """A class for extracting next sentence prediction task segments and labels.

  `NextSentencePredictionExtractor` takes in batches of sentences and extracts
  either the subsequent sentence or injects a random sentence in the document.
  It also returns a label the describes whether or not a random injection
  occured.
  """

  def __init__(self,
               shuffle_fn=None,
               random_fn=None,
               random_next_sentence_threshold=0.5):
    """Creates an instance of the `NextSentencePredictionExtractor`.

    Args:
      shuffle_fn: An op that shuffles the sentences in a random order. Default
        uses tf.random.shuffle. The output of `shuffle_fn` are the candidates
        used for the random next sentence.
      random_fn: An op that returns a random float from [0, 1]. If the results
        of this function passes `random_next_sentence_threshold` then a random
        sentence is swapped in as it's accompanying segment.  Default uses
        tf.random.uniform.
      random_next_sentence_threshold: A float threshold that determines whether
        or not a random sentence is injected instead of the next sentence. The
        higher the threshold, the higher the likelihood of inserting a random
        sentence.
    """
    self._shuffle_fn = shuffle_fn or random_ops.random_shuffle
    self._random_fn = random_fn or random_ops.random_uniform

    self._random_next_sentence_threshold = random_next_sentence_threshold

  def get_segments(self, sentences):
    """Extracts the next sentence label from sentences.

    Args:
      sentences: A `RaggedTensor` of strings w/ shape [batch, (num_sentences)].

    Returns:
      A tuple of (segment_a, segment_b, is_next_sentence) where:

      segment_a: A `Tensor` of strings w/ shape [total_num_sentences] that
        contains all the original sentences.
      segment_b:  A `Tensor` with shape [num_sentences] that contains
        either the subsequent sentence of `segment_a` or a randomly injected
        sentence.
      is_next_sentence: A `Tensor` of bool w/ shape [num_sentences]
        that contains whether or not `segment_b` is truly a subsequent sentence
        or not.
    """
    next_sentence = ragged_map_ops.map_fn(
        functools.partial(manip_ops.roll, axis=0, shift=-1),
        sentences,
        dtype=ragged_tensor.RaggedTensorType(dtypes.string, 1),
        infer_shape=False)
    random_sentence = sentences.with_flat_values(
        self._shuffle_fn(sentences.flat_values))
    is_next_sentence_labels = (
        self._random_fn(sentences.flat_values.shape) >
        self._random_next_sentence_threshold)
    is_next_sentence = sentences.with_flat_values(is_next_sentence_labels)

    # Randomly decide if we should use next sentence or throw in a random
    # sentence.
    segment_two = ragged_where_op.where(
        is_next_sentence, x=next_sentence, y=random_sentence)

    # Get rid of the docs dimensions
    sentences = sentences.merge_dims(-2, -1)
    segment_two = segment_two.merge_dims(-2, -1)
    is_next_sentence = is_next_sentence.merge_dims(-2, -1)
    is_next_sentence = math_ops.cast(is_next_sentence, dtypes.int64)
    return sentences, segment_two, is_next_sentence
