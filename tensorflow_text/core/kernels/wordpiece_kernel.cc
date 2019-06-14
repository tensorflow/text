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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {

namespace {
string GetWordSplitChar(OpKernelConstruction* ctx) {
  string suffix_indicator;
  ([=](string* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("suffix_indicator", c));
  })(&suffix_indicator);
  return suffix_indicator;
}

int32 GetMaxCharsPerWord(OpKernelConstruction* ctx) {
  int32 max_chars_per_word;
  ([=](int32* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_bytes_per_word", c));
  })(&max_chars_per_word);
  return max_chars_per_word;
}

bool GetShouldUseUnknownToken(OpKernelConstruction* ctx) {
  bool use_unknown_token;
  ([=](bool* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_unknown_token", c));
  })(&use_unknown_token);
  return use_unknown_token;
}

string GetUnknownToken(OpKernelConstruction* ctx) {
  string unknown_token;
  ([=](string* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unknown_token", c));
  })(&unknown_token);
  return unknown_token;
}

Status GetTableHandle(const string& input_name, OpKernelContext* ctx,
                      string* container, string* table_handle) {
  {
    mutex* mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Lookup table handle must be scalar, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<tstring>();
    *container = h(0);
    *table_handle = h(1);
  }
  return Status::OK();
}

// Gets the LookupTable stored in the ctx->resource_manager() with key
// passed by attribute with name input_name, returns null if the table
// doesn't exist.
Status GetLookupTable(const string& input_name, OpKernelContext* ctx,
                      lookup::LookupInterface** table) {
  string container;
  string table_handle;
  DataType handle_dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
  if (handle_dtype == DT_RESOURCE) {
    ResourceHandle handle;
    TF_RETURN_IF_ERROR(HandleFromInput(ctx, input_name, &handle));
    return LookupResource(ctx, handle, table);
  } else {
    TF_RETURN_IF_ERROR(
        GetTableHandle(input_name, ctx, &container, &table_handle));
    return ctx->resource_manager()->Lookup(container, table_handle, table);
  }
}

}  // namespace

class WordpieceTokenizeWithOffsetsOp : public OpKernel {
 public:
  explicit WordpieceTokenizeWithOffsetsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        suffix_indicator_(GetWordSplitChar(ctx)),
        max_bytes_per_word_(GetMaxCharsPerWord(ctx)),
        use_unknown_token_(GetShouldUseUnknownToken(ctx)),
        unknown_token_(GetUnknownToken(ctx)) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_values;
    OP_REQUIRES_OK(ctx, ctx->input("input_values", &input_values));
    const auto& values_vec = input_values->flat<tstring>();

    lookup::LookupInterface* lookup_table;
    // MKH: subject to change....
    OP_REQUIRES_OK(ctx,
                   GetLookupTable("vocab_lookup_table", ctx, &lookup_table));
    LookupTableVocab vocab_map(lookup_table, ctx);

    std::vector<string> subwords;
    std::vector<int> begin_offset;
    std::vector<int> end_offset;
    std::vector<int> row_lengths;

    // Iterate through all the values and wordpiece tokenize them.
    for (int i = 0; i < values_vec.size(); ++i) {
      // Tokenize into subwords and record the offset locations.
      int num_wordpieces = 0;
      OP_REQUIRES_OK(
          ctx, WordpieceTokenize(values_vec(i), max_bytes_per_word_,
                                 suffix_indicator_, use_unknown_token_,
                                 unknown_token_, &vocab_map, &subwords,
                                 &begin_offset, &end_offset, &num_wordpieces));

      // Record the row splits.
      row_lengths.push_back(num_wordpieces);
    }

    std::vector<int64> output_subwords_shape;
    output_subwords_shape.push_back(subwords.size());

    std::vector<int64> output_row_lengths_shape;
    output_row_lengths_shape.push_back(row_lengths.size());

    Tensor* output_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output_values",
                                             TensorShape(output_subwords_shape),
                                             &output_values));
    auto output_values_vec = output_values->vec<tstring>();

    Tensor* output_row_lengths;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("output_row_lengths",
                                        TensorShape(output_row_lengths_shape),
                                        &output_row_lengths));
    auto output_row_lengths_vec = output_row_lengths->vec<int64>();

    Tensor* start_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("start_values",
                                             TensorShape(output_subwords_shape),
                                             &start_values));
    auto start_values_vec = start_values->vec<int64>();

    Tensor* limit_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("limit_values",
                                             TensorShape(output_subwords_shape),
                                             &limit_values));
    auto limit_values_vec = limit_values->vec<int64>();

    for (int i = 0; i < subwords.size(); ++i) {
      output_values_vec(i) = subwords[i];
    }

    for (int i = 0; i < row_lengths.size(); ++i) {
      output_row_lengths_vec(i) = row_lengths[i];
    }

    for (int i = 0; i < begin_offset.size(); ++i) {
      start_values_vec(i) = begin_offset[i];
    }

    for (int i = 0; i < end_offset.size(); ++i) {
      limit_values_vec(i) = end_offset[i];
    }
  }

 private:
  const string suffix_indicator_;
  const int max_bytes_per_word_;
  const bool use_unknown_token_;
  const string unknown_token_;

  TF_DISALLOW_COPY_AND_ASSIGN(WordpieceTokenizeWithOffsetsOp);
};

REGISTER_KERNEL_BUILDER(Name("WordpieceTokenizeWithOffsets").Device(DEVICE_CPU),
                        WordpieceTokenizeWithOffsetsOp);

}  // namespace text
}  // namespace tensorflow
