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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_KERNEL_TEMPLATE_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/fast_bert_normalizer.h"

namespace tensorflow {
namespace text {

// See `kDoc` data member for the documentation on this op kernel.
//
// This template class can be instantiated into a kernel for either TF or
// TFLite. See go/tfshim for more info on how this works.
template <tflite::shim::Runtime Rt>
class FastBertNormalizeOp
    : public tflite::shim::OpKernelShim<FastBertNormalizeOp, Rt> {
 private:
  enum Inputs { kInputValues = 0, kFastBertNormalizerModel };
  enum Outputs {
    kOutputValues = 0,
    kOutputOffsetMappings,
    kOutputRowSplitsOfOffsetMappings,
  };

  using Shape = tflite::shim::Shape;
  using
      typename tflite::shim::OpKernelShim<FastBertNormalizeOp, Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<FastBertNormalizeOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<FastBertNormalizeOp,
                                            Rt>::ShapeInferenceContext;

  static const char kGetOffsetMappingsAttr[];

  // The real work of the invoke operation.
  template <bool kGetOffsetMappings>
  absl::Status InvokeRealWork(InvokeContext* context);

  bool get_offset_mappings_;

 public:
  FastBertNormalizeOp() = default;
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
const char FastBertNormalizeOp<Rt>::kOpName[] =
    "FastBertNormalize";

template <tflite::shim::Runtime Rt>
const char FastBertNormalizeOp<Rt>::kDoc[] = R"doc(
  Tokenizes tokens into sub-word pieces based off of a vocabulary using the fast
  linear WordPiece algorithm.
)doc";

template <tflite::shim::Runtime Rt>
const char FastBertNormalizeOp<Rt>::kGetOffsetMappingsAttr[] =
    "get_offset_mappings";

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastBertNormalizeOp<Rt>::Attrs() {
  return {
      absl::StrCat(kGetOffsetMappingsAttr, ": bool = false"),
  };
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastBertNormalizeOp<Rt>::Inputs() {
  return {"input_values: string", "fast_bert_normalizer_model: uint8"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastBertNormalizeOp<Rt>::Outputs() {
  return {"output_values: string", "output_offset_mappings: int64",
          "output_row_splits: int64"};
}

template <tflite::shim::Runtime Rt>
absl::Status FastBertNormalizeOp<Rt>::Init(InitContext* context) {
  SH_RETURN_IF_ERROR(
      context->GetAttr(kGetOffsetMappingsAttr, &get_offset_mappings_));
  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status FastBertNormalizeOp<Rt>::Invoke(InvokeContext* context) {
  if (get_offset_mappings_) {
    return InvokeRealWork</*kGetOffsetMappings=*/true>(context);
  } else {
    return InvokeRealWork</*kGetOffsetMappings=*/false>(context);
  }
}

template <tflite::shim::Runtime Rt>
template <bool kGetOffsetMappings>
absl::Status FastBertNormalizeOp<Rt>::InvokeRealWork(InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(const auto input_values, context->GetInput(kInputValues));
  const auto& values_vec = input_values->template As<tstring, 1>();

  SH_ASSIGN_OR_RETURN(const auto fast_bert_normalizer_model,
                      context->GetInput(kFastBertNormalizerModel));
  // OK to create on every call because FastBertNormalizer is a lightweight,
  // memory-mapped wrapper on `fast_bert_normalizer_model` tensor, and thus
  // Create() is very cheap.
  auto text_normalizer = FastBertNormalizer::Create(
      fast_bert_normalizer_model->template Data<uint8>().data());
  SH_RETURN_IF_ERROR(text_normalizer.status());

  SH_ASSIGN_OR_RETURN(
      auto output_values,
      context->GetOutput(kOutputValues, Shape(input_values->Shape())));
  auto output_values_vec = output_values->template As<tensorflow::tstring, 1>();
  std::vector<int> offset_mappings;
  std::vector<int> row_splits;

  if constexpr (kGetOffsetMappings) {
    row_splits.push_back(0);
  }

  // Iterate through all the values and normalize them.
  for (int i = 0; i < values_vec.Dim(0); ++i) {
    // Normalize and record the offset locations.
    std::string normalized_string;
    bool is_normalized_string_identical;
    const int original_size = offset_mappings.size();

    text_normalizer->template NormalizeText</*kGetOffsets=*/kGetOffsetMappings>(
        values_vec(i), &is_normalized_string_identical, &normalized_string,
        &offset_mappings);
    if (is_normalized_string_identical) {
      // When the input string is not changed after normalization,
      // `normalized_string` is empty and `offset_mappings` is not changed by
      // the above function. So here we construct the corresponding result and
      // append to the final output.
      output_values_vec(i) = values_vec(i);  // The normalized text.
      if constexpr (kGetOffsetMappings) {
        // The offset mapping will be the identy mapping.
        for (int j = 0; j < values_vec(i).size(); ++j) {
          offset_mappings.push_back(j);
        }
        // The mapping from the end of the output to the end of the input.
        offset_mappings.push_back(values_vec(i).size());
      }
    } else {
      output_values_vec(i) = normalized_string;
    }

    if constexpr (kGetOffsetMappings) {
      // Record the row splits.
      const int delta_size = offset_mappings.size() - original_size;
      row_splits.push_back(delta_size + row_splits.back());
    }
  }

  if constexpr (kGetOffsetMappings) {
    SH_ASSIGN_OR_RETURN(
        auto output_offset_mappings,
        context->GetOutput(kOutputOffsetMappings,
                           Shape({static_cast<int>(offset_mappings.size())})));
    auto output_offset_mappings_vec =
        output_offset_mappings->template As<int64, 1>();

    SH_ASSIGN_OR_RETURN(
        auto output_row_splits,
        context->GetOutput(kOutputRowSplitsOfOffsetMappings,
                           Shape({static_cast<int>(row_splits.size())})));
    auto output_row_splits_vec = output_row_splits->template As<int64, 1>();

    for (int i = 0; i < row_splits.size(); ++i) {
      output_row_splits_vec(i) = row_splits[i];
    }

    for (int i = 0; i < offset_mappings.size(); ++i) {
      output_offset_mappings_vec(i) = offset_mappings[i];
    }
  }
  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status FastBertNormalizeOp<Rt>::ShapeInference(ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  SH_ASSIGN_OR_RETURN(const Shape input_values_shape,
                      c->GetInputShape(kInputValues));
  SH_ASSIGN_OR_RETURN(const auto fast_bert_normalizer_model_shape,
                      c->GetInputShape(kFastBertNormalizerModel));
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Input values shape must be rank 1: ", input_values_shape.ToString()));
  }
  if (!fast_bert_normalizer_model_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Fast BERT normalizer model shape must be rank 1: ",
                     fast_bert_normalizer_model_shape.ToString()));
  }
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputValues, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputOffsetMappings, rank_1_shape));
  // row splits size
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  SH_RETURN_IF_ERROR(
      c->SetOutputShape(kOutputRowSplitsOfOffsetMappings, Shape({num_splits})));

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_KERNEL_TEMPLATE_H_
