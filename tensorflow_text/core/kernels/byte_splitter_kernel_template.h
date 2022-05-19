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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_SPLITTER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_SPLITTER_KERNEL_TEMPLATE_H_

#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/byte_splitter.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt>
class ByteSplitterWithOffsetsOp
    : public tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp, Rt> {
 private:
  enum Inputs {
    kInputValues = 0
  };
  enum Outputs {
    kOutputBytes = 0,
    kOutputRowSplits,
    kOutputStartOffsets,
    kOutputEndOffsets
  };

  using typename tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<ByteSplitterWithOffsetsOp,
                                            Rt>::ShapeInferenceContext;

 public:
  ByteSplitterWithOffsetsOp() = default;
  static const char kOpName[];
  static const char kDoc[];

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() { return {}; }

  // Inputs declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Outputs declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context) { return absl::OkStatus(); }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);

 protected:
  template <typename BufferType, typename DType>
  inline absl::Status FillOutputTensor(
    const std::vector<BufferType>& buffer, const int index,
    InvokeContext* context);
};

template <tflite::shim::Runtime Rt>
std::vector<std::string> ByteSplitterWithOffsetsOp<Rt>::Inputs() {
  return {"input_values: string"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> ByteSplitterWithOffsetsOp<Rt>::Outputs() {
  return {"output_bytes: uint8", "output_row_splits: int64",
          "output_start_offsets: int32", "output_end_offsets: int32"};
}

template <tflite::shim::Runtime Rt>
absl::Status ByteSplitterWithOffsetsOp<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto input_values_shape_status = c->GetInputShape(kInputValues);
  if (!input_values_shape_status.ok()) {
    return input_values_shape_status.status();
  }
  const Shape& input_values_shape = *input_values_shape_status;

  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", input_values_shape.ToString()));
  }

  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputBytes, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputStartOffsets, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputEndOffsets, rank_1_shape));
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowSplits, Shape({num_splits})));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
    absl::Status ByteSplitterWithOffsetsOp<Rt>
        ::Invoke(InvokeContext* context) {
  // Inputs
  const auto values_statusor = context->GetInput(kInputValues);
  if (!values_statusor.ok()) {
    return values_statusor.status();
  }
  const auto values = (*values_statusor)->template As<tensorflow::tstring, 1>();

  ByteSplitter splitter;

  // Outputs
  std::vector<unsigned char> bytes;
  std::vector<int64_t> row_splits;
  std::vector<int32_t> start_offsets;
  std::vector<int32_t> end_offsets;

  // Iterate through all the string values and split them.
  row_splits.push_back(0);
  for (int i = 0; i < values.Dim(0); ++i) {
    // Split into bytes and record the offset locations.
    const int orig_num_bytes = bytes.size();
    splitter.Split(values(i), &bytes, &start_offsets, &end_offsets);
    const int delta_num_bytes = bytes.size() - orig_num_bytes;
    // Record the row splits.
    row_splits.push_back(delta_num_bytes + row_splits.back());
  }

  // Allocate output & fill output tensors.
  SH_RETURN_IF_ERROR(FillOutputTensor<unsigned char, tensorflow::uint8>(
      bytes, kOutputBytes, context));
  SH_RETURN_IF_ERROR(FillOutputTensor<int64_t, int64_t>(
      row_splits, kOutputRowSplits, context));
  SH_RETURN_IF_ERROR(FillOutputTensor<int32_t, int32_t>(
      start_offsets, kOutputStartOffsets, context));
  SH_RETURN_IF_ERROR(FillOutputTensor<int32_t, int32_t>(
      end_offsets, kOutputEndOffsets, context));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
template <typename BufferType, typename DType>
absl::Status ByteSplitterWithOffsetsOp<Rt>::FillOutputTensor(
    const std::vector<BufferType>& buffer, const int index,
    InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(const auto tensorview, context->GetOutput(
      index, tflite::shim::Shape({static_cast<int>(buffer.size())})));
  auto data = tensorview->template As<DType, 1>();
  // TODO(broken): investigate using memcpy like previous WST
  for (int i = 0; i < buffer.size(); ++i) data(i) = buffer.at(i);
  return absl::OkStatus();
}

// Static member definitions.
// These can be inlined once the toolchain is bumped up to C++17

template <tflite::shim::Runtime Rt>
const char ByteSplitterWithOffsetsOp<Rt>::kOpName[] =
    "TFText>ByteSplitWithOffsets";

template <tflite::shim::Runtime Rt>
const char ByteSplitterWithOffsetsOp<Rt>::kDoc[] = R"doc(
  Splits a string into bytes
  )doc";

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_SPLITTER_KERNEL_TEMPLATE_H_
