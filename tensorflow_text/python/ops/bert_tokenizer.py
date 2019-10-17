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
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow_text.python.ops import wordshape_ops
from tensorflow_text.python.ops.normalize_ops import case_fold_utf8
from tensorflow_text.python.ops.normalize_ops import normalize_utf8
from tensorflow_text.python.ops.tokenization import Tokenizer
from tensorflow_text.python.ops.unicode_script_tokenizer import UnicodeScriptTokenizer
from tensorflow_text.python.ops.wordpiece_tokenizer import WordpieceTokenizer


class BasicTokenizer(Tokenizer):
  """Basic tokenizer for for tokenizing text.

  A basic tokenizer that tokenizes using some deterministic rules:
  - For most languages, this tokenizer will split on whitespace.
  - For Chinese, Japanese, and Korean characters, this tokenizer will split on
      Unicode characters.

  Attributes:
    lower_case: bool - If true, a preprocessing step is added to lowercase the
      text, apply NFD normalization, and strip accents characters.
    keep_whitespace: bool - If true, preserves whitespace characters instead of
      stripping them away.
    normalization_form: If true and lower_case=False, the input text will be
      normalized to `normalization_form`. See normalize_utf8() op for a list
      of valid values.
  """

  def __init__(self,
               lower_case=False,
               keep_whitespace=False,
               normalization_form=None):
    self._lower_case = lower_case
    self._keep_whitespace = keep_whitespace
    self._normalization_form = normalization_form

  def _is_cjk(self, script_ids):
    is_chinese = math_ops.logical_or(
        math_ops.equal(script_ids, 17),  # Han (Chinese) script
        math_ops.equal(script_ids, 74))  # Traditional Han (Chinese) script
    is_korean = math_ops.logical_or(
        math_ops.equal(script_ids, 119),  # Korean script
        math_ops.equal(script_ids, 18))
    return math_ops.logical_or(
        is_chinese,
        math_ops.logical_or(
            math_ops.equal(script_ids, 105),  # Japanese script
            is_korean))

  def _should_split(self, script_tokenized):
    token_script_ids = string_ops.unicode_script(
        ragged_string_ops.unicode_decode(script_tokenized.flat_values,
                                         "UTF-8"))[:, :1]

    token_script_ids_flat = token_script_ids.flat_values
    is_cjk = self._is_cjk(token_script_ids_flat)
    is_emoji = wordshape_ops.wordshape(script_tokenized.flat_values,
                                       wordshape_ops.WordShape.HAS_EMOJI)
    is_punct = wordshape_ops.wordshape(
        script_tokenized.flat_values,
        wordshape_ops.WordShape.IS_PUNCT_OR_SYMBOL)
    split_cond = is_cjk | is_emoji | is_punct
    return split_cond

  # TODO(thuang513): add support for tokenize_with_offsets()
  def tokenize(self, text_input):
    """Performs basic word tokenization for BERT.

    Args:
      text_input: A `Tensor` or `RaggedTensor` of untokenized UTF-8 strings.
    Returns:
      A `RaggedTensor` of tokenized strings from text_input.
    """
    # lowercase and strip accents (if option is set)
    if self._lower_case:
      text_input = case_fold_utf8(text_input)
      text_input = normalize_utf8(text_input, "NFD")
      text_input = string_ops.regex_replace(text_input, r"\p{Mn}", "")
    else:
      # utf8 normalization
      if self._normalization_form is not None:
        text_input = normalize_utf8(text_input, self._normalization_form)

    # strip out control characters
    text_input = string_ops.regex_replace(text_input, r"\p{Cc}|\p{Cf}", " ")

    # For chinese and emoji characters, tokenize by unicode codepoints
    unicode_tokenizer = UnicodeScriptTokenizer(
        keep_whitespace=self._keep_whitespace)
    script_tokenized = unicode_tokenizer.tokenize(text_input)

    split_cond = self._should_split(script_tokenized)

    unicode_char_split = ragged_string_ops.unicode_split(
        script_tokenized, "UTF-8")
    unicode_split_tokens = array_ops.where(
        array_ops.squeeze(split_cond),
        y=array_ops.expand_dims(script_tokenized.values, axis=1),
        x=unicode_char_split.values)
    final_tokens = script_tokenized.with_flat_values(unicode_split_tokens)
    return final_tokens.merge_dims(-2, -1)


class BertTokenizer(Tokenizer):
  """Tokenizer used for BERT.

    This tokenizer applies an end-to-end, text string to wordpiece tokenization.
    It first applies basic tokenization, and then follwed by wordpiece
    tokenization.

    See BasicTokenizer and WordpieceTokenizer for their respective details.

  Attributes:
    vocab_lookup_table: A lookup table implementing the LookupInterface
      containing the vocabulary of subwords.
    suffix_indicator: (optional) The characters prepended to a wordpiece to
      indicate that it is a suffix to another subword. Default is '##'.
    max_bytes_per_word: (optional) Max size of input token. Default is 100.
    max_chars_per_token: (optional) Max size of subwords, excluding suffix
      indicator. If known, providing this improves the efficiency of decoding
      long words.
    token_out_type: (optional) The type of the token to return. This can be
      `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
    unknown_token: (optional) The value to use when an unknown token is found.
      Default is "[UNK]". If this is set to a string, and `token_out_type` is
      `tf.int64`, the `vocab_lookup_table` is used to convert the
      `unknown_token` to an integer. If this is set to `None`,
      out-of-vocabulary tokens are left as is.
    split_unknown_characters: (optional) Whether to split out single unknown
      characters as subtokens. If False (default), words containing unknown
      characters will be treated as single unknown tokens.
    lower_case: bool - If true, a preprocessing step is added to lowercase the
      text, apply NFD normalization, and strip accents characters.
    keep_whitespace: bool - If true, preserves whitespace characters instead of
      stripping them away.
    normalization_form: If true and lower_case=False, the input text will be
      normalized to `normalization_form`. See normalize_utf8() op for a list
      of valid values.
  """

  def __init__(self,
               vocab_lookup_table,
               suffix_indicator="##",
               max_bytes_per_word=100,
               max_chars_per_token=None,
               token_out_type=dtypes.int64,
               unknown_token="[UNK]",
               split_unknown_characters=False,
               lower_case=False,
               keep_whitespace=False,
               normalization_form=None):
    self._basic_tokenizer = BasicTokenizer(lower_case, keep_whitespace,
                                           normalization_form)
    self._wordpiece_tokenizer = WordpieceTokenizer(
        vocab_lookup_table, suffix_indicator, max_bytes_per_word,
        max_chars_per_token, token_out_type, unknown_token,
        split_unknown_characters)

  def tokenize(self, text_input):
    """Performs untokenized text to wordpiece tokenization for BERT.

    Args:
      text_input: input: A `Tensor` or `RaggedTensor` of untokenized UTF-8
        strings.
    Returns:
      A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string
      contents (or ID in the vocab_lookup_table representing that string)
      of the `jth` token in `input[i1...iN]`
    """
    tokens = self._basic_tokenizer.tokenize(text_input)
    return self._wordpiece_tokenizer.tokenize(tokens)
