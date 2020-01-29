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

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow_text/core/kernels/regex_split.h"

namespace tensorflow {
namespace text {

using ::tensorflow::Status;

void GetRegexFromInput(tensorflow::OpKernelContext* ctx,
                       const string& input_name, std::unique_ptr<RE2>* result) {
  const Tensor* pattern_tensor;
  OP_REQUIRES_OK(ctx, ctx->input(input_name, &pattern_tensor));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(pattern_tensor->shape()),
              errors::InvalidArgument("Pattern must be scalar, but received ",
                                      pattern_tensor->shape().DebugString()));
  const string regex_pattern = pattern_tensor->flat<tstring>()(0);
  *result = absl::make_unique<RE2>(regex_pattern);
  OP_REQUIRES(ctx, (*result)->ok(),
              errors::InvalidArgument("Invalid pattern: ", regex_pattern,
                                      ", error: ", (*result)->error()));
}

class RegexSplitOp : public tensorflow::OpKernel {
 public:
  explicit RegexSplitOp(tensorflow::OpKernelConstruction* ctx)
      : tensorflow::OpKernel(ctx) {}

  void Compute(tensorflow::OpKernelContext* ctx) override {
    bool should_keep_delim;

    std::unique_ptr<RE2> delim_re;
    GetRegexFromInput(ctx, "delim_regex_pattern", &delim_re);

    std::unique_ptr<RE2> keep_delim_re;
    GetRegexFromInput(ctx, "keep_delim_regex_pattern", &keep_delim_re);
    should_keep_delim = keep_delim_re->pattern().empty() ? false : true;

    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<tstring>();

    std::vector<int64> begin_offsets;
    std::vector<int64> end_offsets;
    std::vector<absl::string_view> tokens;
    std::vector<int64> row_splits;
    row_splits.push_back(0);

    for (size_t i = 0; i < input_flat.size(); ++i) {
      RegexSplit(input_flat(i), *delim_re, should_keep_delim, *keep_delim_re,
                 &tokens, &begin_offsets, &end_offsets);
      row_splits.push_back(begin_offsets.size());
    }

    // Emit the flat Tensors needed to construct RaggedTensors for tokens,
    // start, end offsets.
    std::vector<int64> tokens_shape;
    tokens_shape.push_back(tokens.size());

    std::vector<int64> offsets_shape;
    offsets_shape.push_back(begin_offsets.size());

    std::vector<int64> row_splits_shape;
    row_splits_shape.push_back(row_splits.size());

    Tensor* output_tokens_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("tokens", TensorShape(tokens_shape),
                                        &output_tokens_tensor));
    auto output_tokens = output_tokens_tensor->flat<tstring>();

    Tensor* output_begin_offsets_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("begin_offsets", TensorShape(offsets_shape),
                                  &output_begin_offsets_tensor));
    auto output_begin_offsets = output_begin_offsets_tensor->flat<int64>();

    Tensor* output_end_offsets_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("end_offsets", TensorShape(offsets_shape),
                                  &output_end_offsets_tensor));
    auto output_end_offsets = output_end_offsets_tensor->flat<int64>();

    Tensor* output_row_splits_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("row_splits", TensorShape(row_splits_shape),
                                  &output_row_splits_tensor));
    auto output_row_splits = output_row_splits_tensor->flat<int64>();

    // Copy outputs to Tensors.
    for (size_t i = 0; i < tokens.size(); ++i) {
      const auto& token = tokens[i];
      output_tokens(i) = tstring(token.data(), token.length());
    }

    for (size_t i = 0; i < begin_offsets.size(); ++i) {
      output_begin_offsets(i) = begin_offsets[i];
    }

    for (size_t i = 0; i < end_offsets.size(); ++i) {
      output_end_offsets(i) = end_offsets[i];
    }

    for (size_t i = 0; i < row_splits.size(); ++i) {
      output_row_splits(i) = row_splits[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("RegexSplitWithOffsets").Device(tensorflow::DEVICE_CPU), RegexSplitOp);

}  // namespace text
}  // namespace tensorflow
