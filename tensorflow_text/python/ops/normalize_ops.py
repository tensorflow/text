# coding=utf-8
# Copyright 2019 TF.Text Authors.
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
