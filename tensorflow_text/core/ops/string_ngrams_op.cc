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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("InternalStringNGrams")
    .Attr("separator: string")
    .Attr("ngram_width: int >= 1")
    .Attr("left_pad: string")
    .Attr("right_pad: string")
    .Attr("use_pad: bool")
    .Attr("extend_pad: bool")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Input("data: string")
    .Input("data_splits: Tsplits")
    .Output("ngrams: string")
    .Output("ngrams_splits: Tsplits")

    // TODO(b/122968457): Implement a shape function.
    .Doc(R"doc(
  Creates ngrams from ragged string data.
  
  This op accepts a ragged tensor with 1 ragged dimension containing only
  strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
  of that string, joined along the innermost axis.

  Attributes:
    separator: The string to append between elements of the token. Use "" for no
      separator.
    ngram_width: The size of the ngrams to create.
    left_pad: The string to use to pad the left side of the ngram sequence. Only
      used if 'use_pad' is True.
    right_pad: The string to use to pad the right side of the ngram sequence.
      Only used if 'use_pad' is True.
    use_pad: True if the op should pad the beginning and end of sequences. If
      false, sequences shorter than the ngram width will result in the empty
      string as output.
    extend_pad: Only valid if use_pad is true. This controls whether the op will
      repeatedly pad sequences. If True, there will be ngram_size-1 padding
      tokens on either side of the sequence. If False, there will be only one
      padding token on either side of the sequence. If this is false, sequences
      where the sequence plus the two padding tokens is shorter than the ngram
      width will result in the empty string as output.
    Tsplits: The data type of the splits tensor.
  
  Args:
    data: The values tensor of the ragged string tensor to make ngrams out of.
    data_splits:  The splits tensor of the ragged string tensor to make ngrams
      out of.
  ngrams: The values tensor of the output ngrams ragged tensor.
  ngrams_splits: The splits tensor of the output ngrams ragged tensor.
)doc");

}  // namespace tensorflow
