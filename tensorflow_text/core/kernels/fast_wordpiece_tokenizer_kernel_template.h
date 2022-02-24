// Copyright 2022 TF.Text Authors.
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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_KERNEL_TEMPLATE_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {

// See `kDoc` data member for the documentation on this op kernel.
//
// This template class can be instantiated into a kernel for either TF or
// TFLite. See
// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/shim
// for more info on how this works.
template <tflite::shim::Runtime Rt>
class FastWordpieceTokenizeWithOffsetsOp
    : public tflite::shim::OpKernelShim<FastWordpieceTokenizeWithOffsetsOp,
                                        Rt> {
 private:
  enum Inputs { kInputValues = 0, kWpModel };
  enum Outputs {
    kOutputSubwords = 0,
    kOutputIds,
    kOutputRowSplits,
    kStartValues,
    kEndValues
  };

  using Shape = tflite::shim::Shape;
  using typename tflite::shim::OpKernelShim<FastWordpieceTokenizeWithOffsetsOp,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<FastWordpieceTokenizeWithOffsetsOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<FastWordpieceTokenizeWithOffsetsOp,
                                            Rt>::ShapeInferenceContext;
  static const char kGetPiecesAttr[];
  static const char kGetOffsetsAttr[];

  // The real work of the invoke operation.
  template <bool kGetPieces, bool kGetOffsets>
  absl::Status InvokeRealWork(InvokeContext* context);

  bool get_pieces_;
  bool get_offsets_;

 public:
  FastWordpieceTokenizeWithOffsetsOp() = default;
  static const char kOpName[];
  static const char kDoc[];

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs();

  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context);

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

////////////////////////// Implementation

template <tflite::shim::Runtime Rt>
const char FastWordpieceTokenizeWithOffsetsOp<Rt>::kGetPiecesAttr[] =
    "get_pieces";

template <tflite::shim::Runtime Rt>
const char FastWordpieceTokenizeWithOffsetsOp<Rt>::kGetOffsetsAttr[] =
    "get_offsets";

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastWordpieceTokenizeWithOffsetsOp<Rt>::Attrs() {
  return {
      absl::StrCat(kGetPiecesAttr, ": bool = false"),
      absl::StrCat(kGetOffsetsAttr, ": bool = false"),
  };
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastWordpieceTokenizeWithOffsetsOp<Rt>::Inputs() {
  return {"input_values: string", "wp_model: uint8"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastWordpieceTokenizeWithOffsetsOp<Rt>::Outputs() {
  return {"output_subwords: string", "output_ids: int64",
          "output_row_splits: int64", "start_values: int64",
          "end_values: int64"};
}

template <tflite::shim::Runtime Rt>
absl::Status FastWordpieceTokenizeWithOffsetsOp<Rt>::Init(
    InitContext* context) {
  SH_RETURN_IF_ERROR(context->GetAttr(kGetPiecesAttr, &get_pieces_));
  SH_RETURN_IF_ERROR(context->GetAttr(kGetOffsetsAttr, &get_offsets_));
  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status FastWordpieceTokenizeWithOffsetsOp<Rt>::Invoke(
    InvokeContext* context) {
  if (get_pieces_) {
    if (get_offsets_) {
      return InvokeRealWork</*kGetPieces=*/true, /*kGetOffsets=*/true>(context);
    } else {
      return InvokeRealWork</*kGetPieces=*/true, /*kGetOffsets=*/false>(
          context);
    }
  } else {
    if (get_offsets_) {
      return InvokeRealWork</*kGetPieces=*/false, /*kGetOffsets=*/true>(
          context);
    } else {
      return InvokeRealWork</*kGetPieces=*/false, /*kGetOffsets=*/false>(
          context);
    }
  }
}

template <tflite::shim::Runtime Rt>
template <bool kGetPieces, bool kGetOffsets>
absl::Status FastWordpieceTokenizeWithOffsetsOp<Rt>::InvokeRealWork(
    InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(const auto input_values, context->GetInput(kInputValues));
  const auto& values_vec = input_values->template As<tstring, 1>();

  SH_ASSIGN_OR_RETURN(const auto wp_model, context->GetInput(kWpModel));
  // OK to create on every call because FastWordpieceTokenizer is a
  // lightweight, memory-mapped wrapper on `wp_model` tensor, and thus
  // Create() is very cheap.
  auto fast_wordpiece_tokenizer =
      ::tensorflow::text::FastWordpieceTokenizer::Create(
          wp_model->template Data<uint8>().data());
  SH_RETURN_IF_ERROR(fast_wordpiece_tokenizer.status());

  std::vector<std::string> subwords;
  std::vector<int> subword_ids;
  std::vector<int> begin_offset;
  std::vector<int> end_offset;
  std::vector<int> row_splits;
  int subwords_size;
  row_splits.push_back(0);

  // Iterate through all the values and wordpiece tokenize them.
  for (int i = 0; i < values_vec.Dim(0); ++i) {
    // Tokenize into subwords and record the offset locations.
    int delta_num_wordpieces = 0;
    if constexpr (kGetPieces) {
      const int original_num_wordpieces = subwords.size();
      if constexpr (kGetOffsets) {
        fast_wordpiece_tokenizer->Tokenize(values_vec(i), &subwords,
                                           &begin_offset, &end_offset);
      } else {
        fast_wordpiece_tokenizer->Tokenize(values_vec(i), &subwords);
      }
      delta_num_wordpieces = subwords.size() - original_num_wordpieces;
    } else {
      const int original_num_wordpieces = subword_ids.size();
      if constexpr (kGetOffsets) {
        fast_wordpiece_tokenizer->Tokenize(values_vec(i), &subword_ids,
                                           &begin_offset, &end_offset);
      } else {
        fast_wordpiece_tokenizer->Tokenize(values_vec(i), &subword_ids);
      }
      delta_num_wordpieces = subword_ids.size() - original_num_wordpieces;
    }
    // Record the row splits.
    row_splits.push_back(delta_num_wordpieces + row_splits.back());
  }

  if constexpr (kGetPieces) {
    subwords_size = subwords.size();
  } else {
    subwords_size = subword_ids.size();
  }

  if constexpr (kGetPieces) {
    // Copy subwords to output.
    SH_ASSIGN_OR_RETURN(
        auto output_subwords,
        context->GetOutput(kOutputSubwords, Shape({subwords_size})));
    auto output_subwords_vec =
        output_subwords->template As<tensorflow::tstring, 1>();

    for (int i = 0; i < subwords.size(); ++i) {
      output_subwords_vec(i) = subwords[i];
    }
  } else {
    // Copy subword ids to output.
    SH_ASSIGN_OR_RETURN(
        auto output_ids,
        context->GetOutput(
            kOutputIds,
            Shape({static_cast<int>(
                subword_ids.size())}))); /* same shape as `output_subwords` */
    auto output_ids_vec = output_ids->template As<int64, 1>();
    for (int i = 0; i < subword_ids.size(); ++i) {
      output_ids_vec(i) = subword_ids[i];
    }
  }

  // Copy row splits to output.
  SH_ASSIGN_OR_RETURN(
      auto output_row_splits,
      context->GetOutput(kOutputRowSplits,
                         Shape({static_cast<int>(row_splits.size())})));
  auto output_row_splits_vec = output_row_splits->template As<int64, 1>();
  for (int i = 0; i < row_splits.size(); ++i) {
    output_row_splits_vec(i) = row_splits[i];
  }

  if constexpr (kGetOffsets) {
    // Copy offsets to output.
    SH_ASSIGN_OR_RETURN(
        auto start_values,
        context->GetOutput(kStartValues, Shape({subwords_size})));
    auto start_values_vec = start_values->template As<int64, 1>();

    SH_ASSIGN_OR_RETURN(auto end_values,
                        context->GetOutput(kEndValues, Shape({subwords_size})));
    auto end_values_vec = end_values->template As<int64, 1>();
    // The sizes of `begin_offset` and `end_offset` are always equal.
    for (int i = 0; i < begin_offset.size(); ++i) {
      start_values_vec(i) = begin_offset[i];
      end_values_vec(i) = end_offset[i];
    }
  }
  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status FastWordpieceTokenizeWithOffsetsOp<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  SH_ASSIGN_OR_RETURN(const Shape input_values_shape,
                      c->GetInputShape(kInputValues));
  SH_ASSIGN_OR_RETURN(const auto wp_model_shape, c->GetInputShape(kWpModel));
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  // TODO(b/204148042): Compatible & ToString are not exported by TF
  /*if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", input_values_shape.ToString()));
  }
  if (!wp_model_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", wp_model_shape.ToString()));
  }*/
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputSubwords, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputIds, rank_1_shape));
  // row splits size
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowSplits, Shape({num_splits})));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kStartValues, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kEndValues, rank_1_shape));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
const char FastWordpieceTokenizeWithOffsetsOp<Rt>::kOpName[] =
    "FastWordpieceTokenizeWithOffsets";

template <tflite::shim::Runtime Rt>
const char FastWordpieceTokenizeWithOffsetsOp<Rt>::kDoc[] = R"doc(
  Tokenizes tokens into sub-word pieces based off of a vocabulary using the fast
  linear WordPiece algorithm.

  `wordpiece_tokenize_with_offsets` returns the relative offsets.

  ### Example:

  ```python
  >>> tokens = ['don', '\'t', 'treadness']
  >>> wordpiece, ids, row_splits, start, end = (
  ...       fast_wordpiece_tokenize_with_offsets(tokens, model_buffer))
  >>> RaggedTensor.from_row_splits(wordpiece, row_splits)
  [['don', '\'', 't'], ['tread', '##ness']]
  >>> RaggedTensor.from_row_splits(ids, row_splits)
  [[0, 1, 2], [3, 4]]  # Dummy ids.
  >>> RaggedTensor.from_row_splits(start, row_splits)
  start = [[[0, 3, 4], [0, 5]]]
  >>> RaggedTensor.from_row_splits(end, row_splits)
  end = [[[3, 4, 5], [5, 10]]]
  ```
  
  Args:
    * get_pieces: bool - If true, generate `output_values` (i.e., the wordpiece
      strings) in the output, and `output_ids` are not touched. If false,
      generate `output_ids` (i.e., the wordpiece ids) and `output_pieces` are
      not touched.
    * get_offsets: bool - Generate `start_values`, and `end_values` in the output (so as to get the offsets of each wordpiece).
      If false, those tensors are not touched. Note that `output_row_splits` is
      always populated (to infer the shape of the ragged tensor `output_values`
      or `output_ids`).
  Inputs:
    * input_values: 1D Tensor of strings to tokenize with.
    * wp_model: Buffer tensor for the FastWordpieceTokenizerConfig flatbuffer.

  Returns:
    * output_values: 1D tensor containing the wordpieces for all input strings.
      A 2D RaggedTensor can be constructed from this and output_row_splits.
    * output_ids: 1D tensor containing the wordpiece ids for all input strings.
      A 2D RaggedTensor can be constructed from this and output_row_splits.
    * output_row_splits: 1D int tensor with the row splits that allow us to
      build RaggedTensors from output_values, output_ids, start_values, and
      end_values.
    * start_values: 1D tensor containing the inclusive start byte offset for
      each wordpiece in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and output_row_splits.
    * end_values: 1D tensor containing the exclusive end byte offset for
      each wordpiece in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and output_row_splits.
)doc";

// See `kDoc` data member for the documentation on this op kernel.
//
// This template class can be instantiated into a kernel for either TF or
// TFLite. See
// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/shim
// for more info on how this works.
template <tflite::shim::Runtime Rt>
class FastWordpieceDetokenizeOp
    : public tflite::shim::OpKernelShim<FastWordpieceDetokenizeOp, Rt> {
 private:
  enum Inputs { kInputValues = 0, kInputRowSplits, kWpModel };
  enum Outputs { kOutputWords = 0 };

  using Shape = tflite::shim::Shape;
  using typename tflite::shim::OpKernelShim<FastWordpieceDetokenizeOp,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<FastWordpieceDetokenizeOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<FastWordpieceDetokenizeOp,
                                            Rt>::ShapeInferenceContext;

 public:
  FastWordpieceDetokenizeOp() = default;
  static const char kOpName[];
  static const char kDoc[];

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() { return {}; }

  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context) { return absl::OkStatus(); }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

////////////////////////// Implementation

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastWordpieceDetokenizeOp<Rt>::Inputs() {
  return {"input_values: int32", "input_row_splits: int64", "wp_model: uint8"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastWordpieceDetokenizeOp<Rt>::Outputs() {
  return {"output_words: string"};
}

template <tflite::shim::Runtime Rt>
absl::Status FastWordpieceDetokenizeOp<Rt>::Invoke(InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(const auto input_values, context->GetInput(kInputValues));
  const auto& values_vec = input_values->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_row_splits,
                      context->GetInput(kInputRowSplits));
  const auto& row_splits_vec = input_row_splits->template As<int64, 1>();

  SH_ASSIGN_OR_RETURN(const auto wp_model, context->GetInput(kWpModel));
  // OK to create on every call because FastWordpieceTokenizer is a
  // lightweight, memory-mapped wrapper on `wp_model` tensor, and thus
  // Create() is very cheap.
  auto fast_wordpiece_tokenizer =
      ::tensorflow::text::FastWordpieceTokenizer::Create(
          wp_model->template Data<uint8>().data());
  SH_RETURN_IF_ERROR(fast_wordpiece_tokenizer.status());

  std::vector<std::string> sentences;

  // Iterate through row_splits to split input_values.
  for (int i = 0; i < row_splits_vec.Dim(0) - 1; ++i) {
    auto single_input =
        absl::Span<const int>(values_vec.Ptr() + row_splits_vec(i),
                              row_splits_vec(i + 1) - row_splits_vec(i));
    SH_ASSIGN_OR_RETURN(auto sentence,
                        fast_wordpiece_tokenizer->Detokenize(single_input));
    sentences.push_back(sentence);
  }

  const int words_size = sentences.size();
  SH_ASSIGN_OR_RETURN(auto output_words,
                      context->GetOutput(kOutputWords, Shape({words_size})));
  auto output_words_vec = output_words->template As<tensorflow::tstring, 1>();

  for (int i = 0; i < words_size; ++i) {
    output_words_vec(i) = sentences[i];
  }

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status FastWordpieceDetokenizeOp<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  SH_ASSIGN_OR_RETURN(const Shape input_values_shape,
                      c->GetInputShape(kInputValues));
  SH_ASSIGN_OR_RETURN(const Shape input_row_splits_shape,
                      c->GetInputShape(kInputRowSplits));
  SH_ASSIGN_OR_RETURN(const auto wp_model_shape, c->GetInputShape(kWpModel));
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  // TODO(b/204148042): Compatible & ToString are not exported by TF
  /*if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", input_values_shape.ToString()));
  }
  if (!input_row_splits_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_row_splits_shape.ToString()));
  }
  if (!wp_model_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", wp_model_shape.ToString()));
  }*/
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputWords, rank_1_shape));
  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
const char FastWordpieceDetokenizeOp<Rt>::kOpName[] =
    "TFText>FastWordpieceDetokenize";

template <tflite::shim::Runtime Rt>
const char FastWordpieceDetokenizeOp<Rt>::kDoc[] = R"doc(
  Detokenizes sub-word ids into sentences.

  ### Example:

  ```python
  >>> # Vocab of the model_buffer: ['a', 'ab', '##c', 'abc', '##d'].
  >>> wordpiece_ids = [0, 1, 2, 3, 4]
  >>> row_splits = [0, 3, 5]
  >>> tokens = fast_wordpiece_tokenizer_detokenize(tokens, row_splits, model_buffer)
  >>> tokens
  ['a abc', 'abcd']
  ```

  Args:
    input_values: 1D Tensor of sub-word ids.
    input_row_splits: 1D Tensor of row splits that denotes the boundary of each
      sentence in the `input_values`.
    wp_model: Buffer tensor for the FastWordpieceTokenizerConfig flatbuffer.

  Returns:
    * output_values: 1D tensor containing all the sentences.
)doc";

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_KERNEL_TEMPLATE_H_
