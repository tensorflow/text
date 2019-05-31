// Copyright 2019 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/tensorflow/core/framework/op.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("WordpieceTokenizeWithOffsets")
    .Input("input_values: string")
    .Input("vocab_lookup_table: resource")
    .Attr("suffix_indicator: string")
    .Attr("max_bytes_per_word: int")
    .Attr("use_unknown_token: bool")
    .Attr("unknown_token: string")
    .Output("output_values: string")
    .Output("output_row_lengths: int64")
    .Output("start_values: int64")
    .Output("limit_values: int64")
    .Doc(R"doc(
  Tokenizes tokens into sub-word pieces based off of a vocabulary.

  `wordpiece_tokenize_with_offsets` returns the relative offsets.

  ### Example:
  tokens = ['don', '\'t', 'treadness']
  wordpiece, start, end = wordpiece_tokenize_with_offset(tokens)
  wordpiece = [['don', '\'', 't'], ['tread', '##ness']]
  start = [[[0, 3, 4], [0, 5]]]
  end = [[[3, 4, 5], [5, 10]]]
  Args:
    tokens: <string>[num_batch, (num_tokens)] a `RaggedTensor` of UTF-8 token
      strings
    vocab_lookup_table: A lookup table implementing the LookupInterface
    word_split_char: Character used to define prefixes in the vocab.
    return_ids: A bool indicating whether the op returns int64 ids or tokenized
      subword strings.

  Returns:
    A tuple of `RaggedTensor`s `subword`, `subword_offset_starts`,
    `subword_offset_limit` where:

    `subword`: <string>[num_batch, (num_tokens), (num_subword_pieces)] is the
      wordpiece token string encoded in UTF-8.
    `subword_offset_starts`: <int64>[num_batch, (num_tokens),
      (num_subword_pieces)] is the word piece token's starting byte offset.
    `subword_offset_limit`: <int64>[num_batch, (num_tokens),
      (num_subword_pieces)] is the word piece token's ending byte offset.
)doc");

}  // namespace tensorflow
