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

"""Tensorflow lowercasing operation for UTF8 strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_normalize_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_normalize_ops.so'))


# pylint: disable=redefined-builtin
def case_fold_utf8(input, name=None):
  """Applies case folding to every UTF-8 string in the input.

  The input is a `Tensor` or `RaggedTensor` of any shape, and the resulting
  output has the same shape as the input. Note that NFKC normalization is
  implicitly applied to the strings.

  For example:

  ```python
  >>> case_fold_utf8(['The   Quick-Brown',
  ...                 'CAT jumped over',
  ...                 'the lazy dog  !!  ']
  tf.Tensor(['the   quick-brown' 'cat jumped over' 'the lazy dog  !!  '],
            shape=(3,), dtype=string)
  ```

  Args:
    input: A `Tensor` or `RaggedTensor` of UTF-8 encoded strings.
    name: The name for this op (optional).

  Returns:
    A `Tensor` or `RaggedTensor` of type string, with case-folded contents.
  """
  with ops.name_scope(name, "CaseFoldUTF8", [input]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string)
    if ragged_tensor.is_ragged(input_tensor):
      result = gen_normalize_ops.case_fold_utf8(input_tensor.flat_values)
      return input_tensor.with_flat_values(result)
    else:
      return gen_normalize_ops.case_fold_utf8(input_tensor)


# pylint: disable=redefined-builtin)
def normalize_utf8(input, normalization_form="NFKC", name=None):
  """Normalizes each UTF-8 string in the input tensor using the specified rule.

  See http://unicode.org/reports/tr15/

  Args:
    input: A `Tensor` or `RaggedTensor` of type string. (Must be UTF-8.)
    normalization_form: One of the following string values ('NFC', 'NFKC',
      'NFD', 'NFKD'). Default is 'NFKC'.
    name: The name for this op (optional).

  Returns:
    A `Tensor` or `RaggedTensor` of type string, with normalized contents.
  """
  with ops.name_scope(name, "NormalizeUTF8", [input]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string)
    if ragged_tensor.is_ragged(input_tensor):
      result = gen_normalize_ops.normalize_utf8(input_tensor.flat_values,
                                                normalization_form)
      return input_tensor.with_flat_values(result)
    else:
      return gen_normalize_ops.normalize_utf8(input_tensor, normalization_form)


# pylint: disable=redefined-builtin)
def normalize_utf8_with_offsets(input, normalization_form="NFKC", name=None):
  """Normalizes each UTF-8 string in the input tensor using the specified rule.

  Returns normalized string and a handle used by another operation to map
  offsets from the normalized string to source string.

  See http://unicode.org/reports/tr15/

  Args:
    input: A `Tensor` or `RaggedTensor` of type string. (Must be UTF-8.)
    normalization_form: One of the following string values ('NFC', 'NFKC',
      'NFD', 'NFKD'). Default is 'NFKC'.
    name: The name for this op (optional).

  Returns:
    A tuple of (results, handle) where:

    results: `Tensor` or `RaggedTensor` of type string, with normalized
      contents.
    handle: `ResourceHandle` used to retrieve the resource for mapping offsets
      to source strings
  """
  with ops.name_scope(name, "NormalizeUTF8WithOffsets", [input]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string)
    if ragged_tensor.is_ragged(input_tensor):
      result, handle = gen_normalize_ops.normalize_utf8_with_offsets(
          input_tensor.flat_values, normalization_form)
      return input_tensor.with_flat_values(result), handle
    else:
      return gen_normalize_ops.normalize_utf8_with_offsets(
          input_tensor, normalization_form)


def find_source_offsets(handle, input_starts, input_limits, name=None):
  """Returns the source offsets mapped with handle resource.

  Given the offsets (starts, limits) returned by a tokenizer and the resource
  handle returned by normalize_utf8_with_offsets function, returns the offsets
  in the source string.

  Args:
    handle: `ResourceHandle` used to retrieve the resource for mapping offsets
      to source strings
    input_starts: A `Tensor` or `RaggedTensor` of type int32, indicating the
      beginning offset indices of each token
    input_limits: A `Tensor` or `RaggedTensor` of type int32, indicating the
      ending offset indices of each token
    name: The name for this op (optional).

  Returns:
    results: `Tensor` or `RaggedTensor` of type int32, with result offsets
  """
  with ops.name_scope(name, "FindSourceOffsets", [input_starts, input_limits]):
    input_starts_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input_starts, dtype=dtypes.int32)
    input_limits_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input_limits, dtype=dtypes.int32)

    if input_starts_tensor.shape.as_list() != input_limits_tensor.shape.as_list(
    ):
      raise ValueError("Input starts and limits must have the same shape.")

    if ragged_tensor.is_ragged(input_starts_tensor):
      (output_values_starts, output_values_limits,
       output_splits) = gen_normalize_ops.find_source_offsets(
           resource_handle=handle,
           input_starts_values=input_starts_tensor.flat_values,
           input_starts_splits=input_starts_tensor.row_splits,
           input_limits_values=input_limits_tensor.flat_values,
           input_limits_splits=input_limits_tensor.row_splits)
      output_starts = ragged_tensor.RaggedTensor.from_row_splits(
          values=output_values_starts, row_splits=output_splits)
      output_limits = ragged_tensor.RaggedTensor.from_row_splits(
          values=output_values_limits, row_splits=output_splits)
      return output_starts, output_limits
    else:
      return find_source_offsets(
          handle, ragged_conversion_ops.from_tensor(input_starts_tensor),
          ragged_conversion_ops.from_tensor(input_limits_tensor))
