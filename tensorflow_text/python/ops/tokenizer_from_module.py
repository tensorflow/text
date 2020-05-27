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

"""Segmenter that uses a hub module to tokenize the input strings."""

import tensorflow_hub as hub
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

_tf_text_segmenter_class_create_counter = monitoring.Counter(
    '/nlx/api/python/segmenter_create_counter',
    'Counter for number of Segmenter created in Python.')


class TokenizerFromModule(TokenizerWithOffsets):
  """Segmenter that uses a hub module to tokenize the input strings."""

  def __init__(self,
               segmenter_module_handle,
               force_split_at_break_character=True):
    """Initializes a new segmenter instance.

    Args:
      segmenter_module_handle: A ModuleSpec defining the Module to instantiate
        or a path where to load a ModuleSpec.  The underling module must be
        exported by running
        google3/learning/tfx/users/nlx/segmentation/models/train.py.
      force_split_at_break_character: Indicates whether to force start a new
        word after seeing an ICU defined whitespace character.  If set true,
        space characters are treated as word boundaries.  For instance, if the
        input is "New  York" the segmenter always start a new word at "Y".
        Otherwise, the segmenter could produce a word that concatenats multiple
        space sparated sub-strings, which is useful for languages such as
        Vietnamese.
    """
    super(TokenizerFromModule, self).__init__()
    empty_tags = set()
    hub_module = hub.load(segmenter_module_handle, tags=empty_tags)
    self._hub_module_signature = hub_module.signatures['default']
    self._force_split_at_break_character = framework_ops.convert_to_tensor(
        force_split_at_break_character, dtype=dtypes.bool)
    _tf_text_segmenter_class_create_counter.get_cell().increase_by(1)

  def _predict_tokens(self, input_strs):
    output_dict = self._hub_module_signature(
        text=input_strs,
        force_split_at_break_char=self._force_split_at_break_character)
    tokens = output_dict['tokens']
    num_tokens = output_dict['num_tokens']
    starts = output_dict['starts']
    ends = output_dict['ends']
    starts = ragged_tensor.RaggedTensor.from_row_lengths(
        starts, row_lengths=num_tokens)
    ends = ragged_tensor.RaggedTensor.from_row_lengths(
        ends, row_lengths=num_tokens)
    tokens = ragged_tensor.RaggedTensor.from_row_lengths(
        tokens, row_lengths=num_tokens)
    return tokens, starts, ends

  def tokenize_with_offsets(self, input_strs):
    """Tokenizes a tensor of UTF-8 strings into words with [start,end) offsets.

    Args:
      input_strs: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A tuple `(tokens, start_offsets, limit_offsets)` where:
        * `tokens` is a `RaggedTensor` of strings where `tokens[i1...iN, j]` is
          the string content of the `j-th` token in `input_strs[i1...iN]`
        * `start_offsets` is a `RaggedTensor` of int64s where
          `start_offsets[i1...iN, j]` is the byte offset for the start of the
          `j-th` token in `input_strs[i1...iN]`.
        * `limit_offsets` is a `RaggedTensor` of int64s where
          `limit_offsets[i1...iN, j]` is the byte offset immediately after the
          end of the `j-th` token in `input_strs[i...iN]`.
    """
    input_strs = ragged_tensor.convert_to_tensor_or_ragged_tensor(input_strs)
    rank = input_strs.shape.ndims
    if rank is None:
      raise ValueError('input must have a known rank.')

    # Currently, the hub_module accepts only rank 1 input tensors, and outputs
    # rank 2 tokens/starts/ends.  To handle input of different ranks (0, 2, 3,
    # etc), we first convert the input into a rank 1 tensor, then run the
    # module, and finally convert the output back to the expected shape.
    if rank == 0:
      # Build a rank 1 input batch with one string.
      input_batch = array_ops.stack([input_strs])
      # [1, (number codepoints)]
      tokens, starts, ends = self._predict_tokens(input_batch)
      return tokens.flat_values, starts.flat_values, ends.flat_values
    elif rank == 1:
      return self._predict_tokens(input_strs)
    else:
      if not ragged_tensor.is_ragged(input_strs):
        input_strs = ragged_tensor.RaggedTensor.from_tensor(
            input_strs, ragged_rank=rank - 1)

      # [number strings, (number codepoints)]
      tokens, starts, limits = self._predict_tokens(input_strs.flat_values)
      tokens = input_strs.with_flat_values(tokens)
      starts = input_strs.with_flat_values(starts)
      limits = input_strs.with_flat_values(limits)
    return tokens, starts, limits

  def tokenize(self, input_strs):
    """Tokenizes a tensor of UTF-8 strings into words.

    Args:
      input_strs: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A `RaggedTensor` of segmented text. The returned shape is the shape of the
      input tensor with an added ragged dimension for tokens of each string.
    """
    tokens, _, _ = self.tokenize_with_offsets(input_strs)
    return tokens
