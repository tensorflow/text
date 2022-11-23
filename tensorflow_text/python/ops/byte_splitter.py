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

"""Byte splitter for string tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.python.ops.tokenization import SplitterWithOffsets

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_byte_splitter = load_library.load_op_library(resource_loader.get_path_to_datafile('_byte_splitter.so'))

_tf_text_byte_splitter_op_create_counter = monitoring.Counter(
    "/nlx/api/python/byte_splitter_create_counter",
    "Counter for number of ByteSplitters created in Python.")


class ByteSplitter(SplitterWithOffsets):
  """Splits a string tensor into bytes."""

  def __init__(self):
    """Initializes the ByteSplitter.
    """
    super(ByteSplitter, self).__init__()
    _tf_text_byte_splitter_op_create_counter.get_cell().increase_by(1)

  def split(self, input):  # pylint: disable=redefined-builtin
    """Splits a string tensor into bytes.

    The strings are split bytes. Thus, some unicode characters may be split
    into multiple bytes.

    Example:

    >>> ByteSplitter().split("hello")
    <tf.Tensor: shape=(5,), dtype=uint8, numpy=array([104, 101, 108, 108, 111],
    dtype=uint8)>

    Args:
      input: A `RaggedTensor` or `Tensor` of strings with any shape.

    Returns:
      A `RaggedTensor` of bytes. The returned shape is the shape of the
      input tensor with an added ragged dimension for the bytes that make up
      each string.
    """
    (bytez, _, _) = self.split_with_offsets(input)
    return bytez

  def split_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Splits a string tensor into bytes.

    The strings are split bytes. Thus, some unicode characters may be split
    into multiple bytes.

    Example:

    >>> splitter = ByteSplitter()
    >>> bytes, starts, ends = splitter.split_with_offsets("hello")
    >>> print(bytes.numpy(), starts.numpy(), ends.numpy())
    [104 101 108 108 111] [0 1 2 3 4] [1 2 3 4 5]

    Args:
      input: A `RaggedTensor` or `Tensor` of strings with any shape.

    Returns:
      A `RaggedTensor` of bytest. The returned shape is the shape of the
      input tensor with an added ragged dimension for the bytes that make up
      each string.

    Returns:
      A tuple `(bytes, offsets)` where:

        * `bytes`: A `RaggedTensor` of bytes.
        * `start_offsets`: A `RaggedTensor` of the bytes' starting byte offset.
        * `end_offsets`: A `RaggedTensor` of the bytes' ending byte offset.
    """
    name = None
    with ops.name_scope(name, "ByteSplitter", [input]):
      input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      if input_tensor.shape.ndims is None:
        raise ValueError("Rank of input_tensor must be statically known.")
      if ragged_tensor.is_ragged(input_tensor):
        if input_tensor.flat_values.shape.ndims > 1:
          # If the flat_values of our ragged tensor is multi-dimensional, we can
          # process it separately and our output will have the same nested
          # splits as our input.
          (bytez, start_offsets, end_offsets) = self.split_with_offsets(
              input_tensor.flat_values)
          return (input_tensor.with_flat_values(bytez),
                  input_tensor.with_flat_values(start_offsets),
                  input_tensor.with_flat_values(end_offsets))
        else:
          # Recursively process the values of the ragged tensor.
          (bytez, start_offsets, end_offsets) = self.split_with_offsets(
              input_tensor.values)
          return (input_tensor.with_values(bytez),
                  input_tensor.with_values(start_offsets),
                  input_tensor.with_values(end_offsets))
      else:
        if input_tensor.shape.ndims > 1:
          # Convert the input tensor to ragged and process it.
          return self.split_with_offsets(
              ragged_conversion_ops.from_tensor(input_tensor))
        elif input_tensor.shape.ndims == 0:
          (bytez, start_offsets, end_offsets) = self.split_with_offsets(
              array_ops.stack([input_tensor]))
          return bytez.values, start_offsets.values, end_offsets.values
        else:
          # Our rank 1 tensor is the correct shape, so we can process it as
          # normal.
          return self._byte_split_with_offsets(input_tensor)

  def _byte_split_with_offsets(self, input_tensor):
    """Splits a tensor of strings into bytes.

    Args:
      input_tensor: Single-dimension Tensor of strings to split.

    Returns:
      Tuple of tokenized codepoints with offsets relative to the codepoints have
      a shape of [num_strings, (num_tokens or num_offsets)].
    """
    (values, row_splits, start_offsets, end_offsets) = (
        gen_byte_splitter.tf_text_byte_split_with_offsets(
            input_values=input_tensor))
    values = RaggedTensor.from_nested_row_splits(
        flat_values=values,
        nested_row_splits=[row_splits])
    start_offsets = RaggedTensor.from_nested_row_splits(
        flat_values=start_offsets,
        nested_row_splits=[row_splits])
    end_offsets = RaggedTensor.from_nested_row_splits(
        flat_values=end_offsets,
        nested_row_splits=[row_splits])
    return (values, start_offsets, end_offsets)
