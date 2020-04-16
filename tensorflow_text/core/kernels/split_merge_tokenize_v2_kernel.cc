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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "icu4c/source/common/unicode/unistr.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

namespace {

// Returns the length (number of bytes) of the UTF8 code point starting at src,
// by reading only the byte from address src.
//
// The result is a number from the set {1, 2, 3, 4}.
int OneCharLen(const char* src) {
  // On most platforms, char is unsigned by default, but iOS is an exception.
  // The cast below makes sure we always interpret *src as an unsigned char.
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"
      [(*(reinterpret_cast<const unsigned char*>(src)) & 0xFF) >> 4];
}

bool GetUTF8Chars(absl::string_view text,
                  std::vector<absl::string_view>* chars) {
  const char* start = text.data();
  const char* end = text.data() + text.size();
  while (start < end) {
    const int char_length = OneCharLen(start);
    if (char_length <= 0) {
      return false;
    }
    chars->emplace_back(start, char_length);
    start += char_length;
  }
  return true;
}

bool IsBreakChar(absl::string_view text) {
  icu::UnicodeString ustr(text.data(), text.length());
  return ustr.length() == 1 && u_isUWhiteSpace(ustr[0]);
}

Status TokenizeByLabel(const absl::string_view& text,
                       const Tensor& logits_tensor,
                       bool force_split_at_break_character,
                       std::vector<std::string>* tokens,
                       std::vector<int>* begin_offset,
                       std::vector<int>* end_offset, int* num_tokens) {
  std::vector<absl::string_view> chars;
  if (!GetUTF8Chars(text, &chars)) {
    return Status(error::Code::INVALID_ARGUMENT,
                  absl::StrCat("Input string is not utf8 valid: ", text));
  }

  if (chars.size() > logits_tensor.dim_size(0)) {
    return Status(error::Code::INVALID_ARGUMENT,
                  absl::StrCat("Number of logits ", logits_tensor.dim_size(0),
                               " is insufficient for text ", text));
  }

  bool last_character_is_break_character = false;
  int start = 0;
  bool has_new_token_generated_for_text = false;
  const auto& logits = logits_tensor.unaligned_flat<float>();
  for (int i = 0; i < chars.size(); ++i) {
    const bool is_break_character = IsBreakChar(chars[i]);
    if ((logits(2 * i) > logits(2 * i + 1)) ||
        !has_new_token_generated_for_text ||
        (last_character_is_break_character && force_split_at_break_character)) {
      // Start a new token with chars[i].
      if (!is_break_character) {
        tokens->emplace_back(chars[i].data(), chars[i].length());
        begin_offset->push_back(start);
        end_offset->push_back(start + chars[i].length());
        *num_tokens += 1;
        has_new_token_generated_for_text = true;
      }
    } else {
      // Append chars[i] to the last token.
      if (!is_break_character) {
        tokens->back().append(chars[i].data(), chars[i].length());
        end_offset->back() += chars[i].length();
      }
    }

    start += chars[i].length();
    last_character_is_break_character = is_break_character;
  }

  return Status::OK();
}

}  // namespace

class SplitMergeTokenizeWithOffsetsV2Op : public OpKernel {
 public:
  explicit SplitMergeTokenizeWithOffsetsV2Op(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    // Nothing.
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* text;
    OP_REQUIRES_OK(ctx, ctx->input("text", &text));
    const Tensor* logits;
    OP_REQUIRES_OK(ctx, ctx->input("logits", &logits));
    const Tensor* force_split_at_break_character;
    OP_REQUIRES_OK(ctx, ctx->input("force_split_at_break_character",
                                   &force_split_at_break_character));
    const bool force_split_at_break_character_bool =
        force_split_at_break_character->flat<bool>()(0);

    std::vector<string> tokens;
    std::vector<int> begin_offsets;
    std::vector<int> end_offsets;
    std::vector<int> row_splits(1, 0);

    // Iterate through all the values and tokenize them.
    const auto& text_vec = text->flat<tstring>();
    for (int i = 0; i < text_vec.size(); ++i) {
      // Tokenize into tokens and record the offset locations.
      int num_tokens = 0;
      OP_REQUIRES_OK(
          ctx, TokenizeByLabel(
                   text_vec(i),
                   logits->SubSlice(i),
                   force_split_at_break_character_bool, &tokens, &begin_offsets,
                   &end_offsets, &num_tokens));

      // Record the row splits.
      row_splits.push_back(num_tokens + row_splits.back());
    }

    std::vector<int64> tokens_shape;
    tokens_shape.push_back(tokens.size());
    Tensor* token_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("token_values",
                                             TensorShape(tokens_shape),
                                             &token_values));
    auto token_values_vec = token_values->vec<tstring>();

    std::vector<int64> row_splits_shape;
    row_splits_shape.push_back(row_splits.size());
    Tensor* row_splits_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("row_splits",
                                        TensorShape(row_splits_shape),
                                        &row_splits_tensor));
    auto row_splits_vec = row_splits_tensor->vec<int64>();

    Tensor* begin_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("begin_values",
                                             TensorShape(tokens_shape),
                                             &begin_values));
    auto begin_values_vec = begin_values->vec<int64>();

    Tensor* end_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("end_values",
                                             TensorShape(tokens_shape),
                                             &end_values));
    auto end_values_vec = end_values->vec<int64>();

    OP_REQUIRES(ctx, tokens.size() == begin_offsets.size(),
                errors::Internal(tokens.size(), " size vs ",
                                 begin_offsets.size(), " begin offsets"));
    OP_REQUIRES(ctx, tokens.size() == end_offsets.size(),
                errors::Internal(tokens.size(), " size vs ",
                                 end_offsets.size(), " end offsets"));
    for (int i = 0; i < tokens.size(); ++i) {
      token_values_vec(i) = tokens[i];
      begin_values_vec(i) = begin_offsets[i];
      end_values_vec(i) = end_offsets[i];
    }
    for (int i = 0; i < row_splits.size(); ++i) {
      row_splits_vec(i) = row_splits[i];
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SplitMergeTokenizeWithOffsetsV2Op);
};

REGISTER_KERNEL_BUILDER(
    Name("SplitMergeTokenizeWithOffsetsV2").Device(DEVICE_CPU),
    SplitMergeTokenizeWithOffsetsV2Op);

}  // namespace text
}  // namespace tensorflow
