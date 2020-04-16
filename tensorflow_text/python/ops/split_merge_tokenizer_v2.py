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

"""Ops to tokenize words into subwords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_split_merge_tokenizer_v2 = load_library.load_op_library(resource_loader.get_path_to_datafile('_split_merge_tokenizer_v2.so'))

_tf_text_split_merge_tokenizer_v2_op_create_counter = monitoring.Counter(
    '/nlx/api/python/split_merge_tokenizer_v2_create_counter',
    'Counter for number of SplitMergeTokenizerV2s created in Python.')


class SplitMergeTokenizerV2(TokenizerWithOffsets):
  """Tokenizes a tensor of UTF-8 string into words according to labels."""

  def __init__(self):
    """Initializes a new instance.
    """
    super(SplitMergeTokenizerV2, self).__init__()
    _tf_text_split_merge_tokenizer_v2_op_create_counter.get_cell().increase_by(
        1)

  def tokenize(self, text, logits, force_split_at_break_character=True):
    """Tokenizes a tensor of UTF-8 strings according to labels.

    ### Example:
    ```python
    >>> text = ["HelloMonday", "DearFriday"],
    >>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
    >>> tokenizer = SplitMergeTokenizerV2()
    >>> tokenizer.tokenize(strings, labels)
    [['Hello', 'Monday'], ['Dear', 'Friday']]
    ```

    Args:
      text: A 1D `Tensor` of UTF-8 strings.

      logits: A 3D `Tensor` of int32, with logits[i, j, 0] / logits[i, j, 1]
        being the logits for the split / merge actions at the j-th character for
        text[i].  "Split" means create a new word starting with this character
        and "merge" means adding this character to the previous word.

      force_split_at_break_character: bool indicates whether to force start a
        new word after seeing a ICU defined whitespace character.  When seeing
        one or more ICU defined whitespace character:

    Returns:
      A `RaggedTensor` of strings where `tokens[i1...iN, j]` is the string
      content of the `j-th` token in `text[i1...iN]`
    """
    subword, _, _ = self.tokenize_with_offsets(text, logits)
    return subword

  def tokenize_with_offsets(self, text, logits,
                            force_split_at_break_character=True):
    """Tokenizes a tensor of UTF-8 strings into tokens with [start,end) offsets.

    ### Example:

    ```python
    >>> text = ["HelloMonday", "DearFriday"],
    >>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
    >>> tokenizer = SplitMergeTokenizerV2()
    >>> result = tokenizer.tokenize_with_offsets(strings, labels)
    >>> result[0].to_list()
    [['Hello', 'Monday'], ['Dear', 'Friday']]
    >>> result[1].to_list()
    >>> [[0, 5], [0, 4]]
    >>> result[2].to_list()
    >>> [[5, 11], [4, 10]]
    ```

    Args:
      text: A 1D `Tensor` of UTF-8 strings.

      logits: A 3D `Tensor` of int32, with logits[i, j, 0] being the logit for
        the split action at the j-th character for text[i] and logit[i, j, 1]
        being the logit for the merge action for the same character.  "Split"
        means create a new word starting with this character and "merge" means
        adding this character to the previous word.

      force_split_at_break_character: bool indicates whether to force start a
        new word after seeing a ICU-defined whitespace character.  When seeing
        one or more ICU-defined whitespace character:

    Returns:
      A tuple `(tokens, start_offsets, limit_offsets)` where:
        * `tokens` is a `RaggedTensor` of strings where `tokens[i, j]` is
          the string content of the `j-th` token in `text[i]`
        * `begin_offsets` is a `RaggedTensor` of int64s where
          `begin_offsets[i, j]` is the byte offset for the beginning of the
          `j-th` token in `text[i]`.
        * `end_offsets` is a `RaggedTensor` of int64s where
          `end_offsets[i, j]` is the byte offset immediately after the
          end of the `j-th` token in `text[i]`.
    """
    name = None
    with ops.name_scope(name, 'SplitMergeTokenizeWithOffsetsV2',
                        [text, logits, force_split_at_break_character]):
      # Check that the types are expected and the ragged rank is appropriate.
      text_rank = text.shape.ndims
      if text_rank != 1:
        raise ValueError('text must have rank 1.')

      # Tokenize the strings into tokens.
      tokens, begin_offsets, end_offsets, row_splits = (
          gen_split_merge_tokenizer_v2.split_merge_tokenize_with_offsets_v2(
              text=text,
              logits=logits,
              force_split_at_break_character=force_split_at_break_character))

      # Use row_splits to convert output info to RaggedTensors.
      def _to_ragged_tensor(dense_1d_tensor):
        return RaggedTensor.from_row_splits(dense_1d_tensor, row_splits,
                                            validate=False)

      tokens = _to_ragged_tensor(tokens)
      begin_offsets = _to_ragged_tensor(begin_offsets)
      end_offsets = _to_ragged_tensor(end_offsets)
      return tokens, begin_offsets, end_offsets
