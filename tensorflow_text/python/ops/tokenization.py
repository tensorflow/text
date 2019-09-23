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

"""Base classes (abstract class) for all tokenizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


# TODO(broken): Have this extend Module when it becomes public
class Tokenizer(object):
  """Base class for tokenizer implementations."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def tokenize(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes the input tensor.

    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
        `RaggedTensor`.

    Returns:
      An N+1-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.
    """
    raise NotImplementedError("Abstract method")


class TokenizerWithOffsets(Tokenizer):
  """Base class for tokenizer implementations that return offsets."""

  @abc.abstractmethod
  def tokenize_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes the input tensor and returns the result with offsets.

    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
        `RaggedTensor`.

    Returns:
      A tuple `(tokens, start_offsets, limit_offsets)` where:

        * `tokens` is an N+1-dimensional UTF-8 string or integer `Tensor` or
            `RaggedTensor`.
        * `start_offsets` is an N+1-dimensional integer `Tensor` or
            `RaggedTensor` containing the starting indices of each token (byte
            indices for input strings).
        * `limit_offsets` is an N+1-dimensional integer `Tensor` or
            `RaggedTensor` containing the exclusive ending indices of each token
            (byte indices for input strings).
    """
    raise NotImplementedError("Abstract method")


# TODO(broken): Have this extend Module when it becomes public
class Detokenizer(object):
  """Base class for detokenizer implementations."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def detokenize(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes the input tensor.

    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
        `RaggedTensor`.

    Returns:
      An (N-1)-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.
    """
    raise NotImplementedError("Abstract method")
