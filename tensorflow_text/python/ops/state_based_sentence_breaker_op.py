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

"""Break sentence ops."""

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_state_based_sentence_breaker_op = load_library.load_op_library(resource_loader.get_path_to_datafile('_state_based_sentence_breaker_op.so'))
from tensorflow_text.python.ops import sentence_breaking_ops


class StateBasedSentenceBreaker(sentence_breaking_ops.SentenceBreakerWithOffsets
                               ):
  """A `Splitter` that uses a state machine to determine sentence breaks.

  `StateBasedSentenceBreaker` splits text into sentences by using a state
  machine to determine when a sequence of characters indicates a potential
  sentence break.

  The state machine consists of an `initial state`, then transitions to a
  `collecting terminal punctuation state` once an acronym, an emoticon, or
  terminal punctuation (ellipsis, question mark, exclamation point, etc.), is
  encountered.

  It transitions to the `collecting close punctuation state` when a close
  punctuation (close bracket, end quote, etc.) is found.

  If non-punctuation is encountered in the collecting terminal punctuation or
  collecting close punctuation states, then the state machine exits, returning
  false, indicating it has moved past the end of a potential sentence fragment.
  """

  def break_sentences(self, doc):
    """Splits `doc` into sentence fragments and returns the fragments' text.

    Args:
      doc: A string `Tensor` of shape [batch] with a batch of documents.

    Returns:
      results: A string `RaggedTensor` of shape [batch, (num_sentences)]
      with each input broken up into its constituent sentence fragments.

    """
    results, _, _ = self.break_sentences_with_offsets(doc)
    return results

  def break_sentences_with_offsets(self, doc):
    """Splits `doc` into sentence fragments, returns text, start & end offsets.

    Example:
                      1                  1         2         3
            012345678901234    01234567890123456789012345678901234567
      doc: 'Hello...foo bar', 'Welcome to the U.S. don't be surprised'

      fragment_text: [['Hello...', 'foo bar'], ['Welcome to the U.S.' , 'don't
      be surprised']]
      start: [[0, 8],[0, 20]]
      end: [[8, 15],[19, 38]]

    Args:
      doc: A string `Tensor` of shape [batch] with a batch of documents.

    Returns:
      A tuple of (fragment_text, start, end) where:

      fragment_text: A string `RaggedTensor` of shape [batch, (num_sentences)]
      with each input broken up into its constituent sentence fragments.
      start: A int64 `RaggedTensor` of shape [batch, (num_sentences)]
        where each entry is the inclusive beginning byte offset of a sentence.
      end: A int64 `RaggedTensor` of shape [batch, (num_sentences)]
        where each entry is the exclusive ending byte offset of a sentence.
    """
    doc = ragged_tensor.convert_to_tensor_or_ragged_tensor(doc)
    if doc.shape.ndims > 1:
      doc = ragged_tensor.RaggedTensor.from_tensor(doc)
      doc = doc.flat_values

    # Run sentence fragmenter op v2
    fragment = gen_state_based_sentence_breaker_op.sentence_fragments_v2(doc)
    start, end, properties, terminal_punc_token, row_lengths = fragment

    # Pack and create `RaggedTensor`s
    start, end, properties, terminal_punc_token = tuple(
        ragged_tensor.RaggedTensor.from_row_lengths(value, row_lengths)
        for value in [start, end, properties, terminal_punc_token])

    # Helper for use within map_fn (function must only take in one argument)
    def _substring(x):
      s, pos, length = x
      return string_ops.substr(s, pos, length)

    # Extract fragment text using offsets
    fragment_text = ragged_map_ops.map_fn(
        _substring, (doc, start, math_ops.subtract(end, start)),
        infer_shape=False,
        dtype=dtypes.string)

    return fragment_text, start, end
