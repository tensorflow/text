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

"""Basic tokenization ops for BERT preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow_text.python.ops import wordshape_ops
from tensorflow_text.python.ops.normalize_ops import case_fold_utf8
from tensorflow_text.python.ops.normalize_ops import normalize_utf8
from tensorflow_text.python.ops.tokenization import Tokenizer
from tensorflow_text.python.ops.unicode_script_tokenizer import UnicodeScriptTokenizer

_CLS_TOKEN = "[CLS]"
_SEP_TOKEN = "[SEP]"
_PAD_TOKEN = "[PAD]"
_MASK_TOKEN = "[MASK]"


class BertTokenizer(Tokenizer):
  """Basic tokenizer for BERT preprocessing."""

  def __init__(self, lower_case=False, keep_whitespace=False):
    self._lower_case = lower_case
    self._keep_whitespace = keep_whitespace

  def _collapse_dims(self, rt, axis=0):
    """Collapses the specified axis of a RaggedTensor.

    Suppose we have a RaggedTensor like this:
    [[1, 2, 3],
     [4, 5],
     [6]]

    If we flatten the 0th dimension, it becomes:
    [1, 2, 3, 4, 5, 6]

    Args:
      rt: a RaggedTensor.
      axis: the dimension to flatten.

    Returns:
      A flattened RaggedTensor, which now has one less dimension.
    """
    to_expand = rt.nested_row_lengths()[axis]
    to_elim = rt.nested_row_lengths()[axis + 1]

    bar = ragged_tensor.RaggedTensor.from_row_lengths(
        to_elim, row_lengths=to_expand)
    new_row_lengths = math_ops.reduce_sum(bar, axis=1)
    return ragged_tensor.RaggedTensor.from_nested_row_lengths(
        rt.flat_values,
        rt.nested_row_lengths()[:axis] + (new_row_lengths,))

  def tokenize(self, text_input):
    """Performs basic word tokenization for BERT.

    Args:
      text_input: A Tensor of untokenized strings with shape `[N]`.
    Returns:
      A RaggedTensor of tokens with shape `[N, (num_tokens)]`.
    """
    # lowercase and strip accents (if option is set)
    if self._lower_case:
      text_input = case_fold_utf8(text_input)

    # normalize by NFD
    text_input = normalize_utf8(text_input, "NFD")

    # strip out control characters
    text_input = string_ops.regex_replace(text_input, r"\p{Cc}|\p{Cf}|\p{Mn}",
                                          "")

    # For chinese and emoji characters, tokenize by unicode codepoints
    unicode_tokenizer = UnicodeScriptTokenizer(
        keep_whitespace=self._keep_whitespace)
    script_tokenized = unicode_tokenizer.tokenize(text_input)
    token_script_ids = string_ops.unicode_script(
        string_ops.unicode_decode(script_tokenized.flat_values, "UTF-8"))

    is_cjkt = math_ops.logical_or(
        math_ops.logical_or(
            math_ops.equal(token_script_ids, 17),   # Han (Chinese) script
            math_ops.equal(token_script_ids, 38)),  # Thai script
        math_ops.logical_or(
            math_ops.equal(token_script_ids, 105),  # Japanese script
            math_ops.equal(token_script_ids, 119))  # Korean script
        )[:, :1].values
    is_emoji = wordshape_ops.wordshape(script_tokenized.flat_values,
                                       wordshape_ops.WordShape.HAS_EMOJI)
    is_punct = wordshape_ops.wordshape(
        script_tokenized.flat_values,
        wordshape_ops.WordShape.IS_PUNCT_OR_SYMBOL)
    split_cond = is_cjkt | is_emoji | is_punct
    unicode_char_split = string_ops.unicode_split(script_tokenized, "UTF-8")

    unicode_split_tokens = array_ops.where(
        split_cond,
        y=array_ops.expand_dims(script_tokenized.flat_values, 1),
        x=unicode_char_split.values)

    # Pack back into a [batch, (num_tokens), (num_unicode_chars)] RT
    chinese_mix_tokenized = ragged_tensor.RaggedTensor.from_row_lengths(
        values=unicode_split_tokens, row_lengths=script_tokenized.row_lengths())

    # Squeeze out to a [batch, (num_tokens)] RT
    return self._collapse_dims(chinese_mix_tokenized)
