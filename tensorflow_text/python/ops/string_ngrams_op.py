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

"""Tensorflow string ngram operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_string_ngrams_op = load_library.load_op_library(resource_loader.get_path_to_datafile('_string_ngrams_op.so'))


def string_ngrams(data,
                  ngram_width,
                  separator="",
                  ngram_pad_values=None,
                  extend_padding=True,
                  name=None):
  """Create a tensor of n-grams based on the input data `data`.

  Creates a tensor of n-grams based on `data`. The n-grams are of width `width`
  and are created along the inner axis; the n-grams are created by concatenating
  windows of `width` adjacent elements from `data` using `string_separator`.

  The input data can be padded on both the start and end of the sequence, if
  desired, using the `pad' argument. If set, `pad` should contain a tuple of
  strings; the 0th element of the tuple will be used to pad the left side of the
  sequence and the 1st element of the tuple will be used to pad the right side
  of the sequence. In addition, users can control whether to pad with one value
  or with (width-1) values by setting `extend_padding` to False or True,
  respectively.

  If this op is configured to not have padding, or if it is configured to add
  padding with `extend_padding` set to False, it is possible that the sequence,
  or the sequence plus the non-extended padding, is smaller than the ngram
  width. In that case, no ngrams will be generated for that sequence.

  Args:
    data: A RaggedTensor containing data to reduce.
    ngram_width: The width of the ngram window. Must be an integer constant, not
      a Tensor.
    separator: The separator string used between ngram elements. Must be a
      string constant, not a Tensor.
    ngram_pad_values: A tuple of (left_pad_value, right_pad_value) or None. If
      None, no padding will be added.
    extend_padding: True to pad (width - 1) times on either side of the sequence
      or False to pad 1 time on either side of the sequence.
    name: The op name.

  Returns:
    A RaggedTensor of ngrams.

  Raises:
    InvalidArgumentError: if `pad` is set to something that is not a tuple of
      strings.
    RuntimeError: If `data` is not a RaggedTensor.
  """

  with ops.name_scope(name, "StringNGrams", [data]):
    left_pad = "" if ngram_pad_values is None else ngram_pad_values[0]
    right_pad = "" if ngram_pad_values is None else ngram_pad_values[1]
    if not isinstance(left_pad, str):
      raise errors.InvalidArgumentError(None, None, "pad[0] must be a string.")
    if not isinstance(right_pad, str):
      raise errors.InvalidArgumentError(None, None, "pad[0] must be a string.")
    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name="data")

    if not isinstance(data, ragged_tensor.RaggedTensor):
      raise RuntimeError("`data` input to StringNGrams must be a RaggedTensor.")

    # TODO(momernick): Extract internal RTs.

    output, output_splits = gen_string_ngrams_op.internal_string_n_grams(
        data=data.flat_values,
        data_splits=data.row_splits,
        separator=separator,
        ngram_width=ngram_width,
        left_pad=left_pad,
        right_pad=right_pad,
        use_pad=(ngram_pad_values is not None),
        extend_pad=extend_padding)

    return ragged_tensor.RaggedTensor.from_row_splits(
        values=output, row_splits=output_splits)
