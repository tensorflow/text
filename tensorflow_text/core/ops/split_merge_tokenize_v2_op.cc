// Copyright 2020 TF.Text Authors.
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
namespace text {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status SplitMergeTokenizeWithOffsetsV2ShapeFn(InferenceContext* c);

REGISTER_OP("SplitMergeTokenizeWithOffsetsV2")
    .Input("text: string")
    .Input("logits: float")
    .Input("force_split_at_break_character: bool")
    .Output("token_values: string")
    .Output("begin_values: int64")
    .Output("end_values: int64")
    .Output("row_splits: int64")
    .SetShapeFn(SplitMergeTokenizeWithOffsetsV2ShapeFn)
    .Doc(R"doc(
  Segment input strings according to the split / merge actions indicated by
  logits.  There are 2 logits for each character from each input strings: first
  one (#0) is the logit for split, and the second one (#1) is the logit for
  merge; we perform the action with the biggest logit.

  ### Example:

  ```python
  >>> strs = ["Itis",
              "thanksgiving"]
  >>> labels = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
  >>> row_splits = [0, 4, 16]
  >>> words, row_splits, begin, end = create_token(strs, labels)
  >>> RaggedTensor.from_row_splits(words, row_splits)
  [['It', 'is'], ['thanks', 'giving']]
  >>> RaggedTensor.from_row_splits(begin, row_splits)
  begin = [[[0, 2], [0, 6]]]
  >>> RaggedTensor.from_row_splits(end, row_splits)
  end = [[[2, 4], [6, 11]]]
  ```

  Args:
    text: 1D Tensor of strings to tokenize.
    logits: 3D Tensor; logits[i,j,0] is the logit for the split action for j-th
      character of text[i].  logits[i,j1] is the logit for the merge action for
      that same character.  For each character, we pick the action with the
      largest logit.
    force_split_at_break_character: bool scalar, indicates whether to force
      start a new word after seeing an ICU defined whitespace character.

  Returns:
    * token_values: 1D tensor containing the tokens for all input strings.
      A 2D RaggedTensor can be constructed from this and row_splits.
    * begin_values: 1D tensor containing the inclusive begin byte offset for
      each token in all input strings.  Corresponds 1:1 with tokens.
      A 2D RaggedTensor can be constructed from this and row_splits.
    * end_values: 1D tensor containing the exclusive end byte offset for
      each token in all input strings.  Corresponds 1:1 with tokens.
      A 2D RaggedTensor can be constructed from this and row_splits.
    * row_splits: 1D tensor containing row split offsets indicating the
      begin and end offsets in the output values for each input string.
)doc");

Status SplitMergeTokenizeWithOffsetsV2ShapeFn(InferenceContext* c) {
  // Check shapes for the input tensors.
  ShapeHandle text = c->input(0);
  ShapeHandle logits = c->input(1);
  ShapeHandle force_split_at_break_character = c->input(2);
  TF_RETURN_IF_ERROR(c->WithRank(text, 1, &text));
  TF_RETURN_IF_ERROR(c->WithRank(logits, 3, &logits));
  TF_RETURN_IF_ERROR(c->WithRank(force_split_at_break_character, 0,
                                 &force_split_at_break_character));

  // Set shapes for the output tensors.
  c->set_output(0, c->UnknownShapeOfRank(1));  // token_values
  c->set_output(1, c->UnknownShapeOfRank(1));  // begin_values
  c->set_output(2, c->UnknownShapeOfRank(1));  // end_values
  DimensionHandle num_splits;
  DimensionHandle num_text = c->Dim(text, 0);
  TF_RETURN_IF_ERROR(c->Add(num_text, 1, &num_splits));
  c->set_output(3, c->Vector(num_splits));  // row_splits
  return Status::OK();
}

}  // namespace text
}  // namespace tensorflow
