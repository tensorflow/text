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

#include <locale>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "icu4c/source/common/unicode/errorcode.h"
#include "icu4c/source/common/unicode/normalizer2.h"
#include "icu4c/source/common/unicode/utypes.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace text {

class CaseFoldUTF8Op : public tensorflow::OpKernel {
 public:
  explicit CaseFoldUTF8Op(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_vec = input_tensor->flat<tstring>();

    // TODO(gregbillock): support forwarding
    tensorflow::Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));
    auto output_vec = output_tensor->flat<tstring>();

    icu::ErrorCode icu_error;
    const icu::Normalizer2* nfkc_cf = icu::Normalizer2::getNFKCCasefoldInstance(
        icu_error);
    OP_REQUIRES(context, icu_error.isSuccess(), errors::Internal(
        absl::StrCat(icu_error.errorName(),
                     ": Could not retrieve ICU NFKC_CaseFold normalizer")));

    for (int64 i = 0; i < input_vec.size(); ++i) {
      string output_text;
      icu::StringByteSink<string> byte_sink(&output_text);
      const auto& input = input_vec(i);
      nfkc_cf->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                             byte_sink, nullptr, icu_error);
      OP_REQUIRES(context, !U_FAILURE(icu_error), errors::Internal(
          "Could not normalize input string: " + input_vec(i)));
      output_vec(i) = output_text;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CaseFoldUTF8").Device(tensorflow::DEVICE_CPU),
                        CaseFoldUTF8Op);

namespace {

string GetNormalizationForm(OpKernelConstruction* context) {
  string normalization_form;
  ([=](string* c) -> void {
    OP_REQUIRES_OK(context, context->GetAttr("normalization_form", c));
  })(&normalization_form);
  return absl::AsciiStrToUpper(normalization_form);
}

}  // namespace

class NormalizeUTF8Op : public tensorflow::OpKernel {
 public:
  explicit NormalizeUTF8Op(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context),
        normalization_form_(GetNormalizationForm(context)) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_vec = input_tensor->flat<tstring>();

    tensorflow::Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));
    auto output_vec = output_tensor->flat<tstring>();

    icu::ErrorCode icu_error;
    const icu::Normalizer2* normalizer = nullptr;
    if (normalization_form_ == "NFKC") {
      normalizer = icu::Normalizer2::getNFKCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(), errors::Internal(
          absl::StrCat(icu_error.errorName(),
                       ": Could not retrieve ICU NFKC normalizer")));
    } else if (normalization_form_ == "NFC") {
      normalizer = icu::Normalizer2::getNFCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(), errors::Internal(
          absl::StrCat(icu_error.errorName(),
                       ": Could not retrieve ICU NFC normalizer")));
    } else if (normalization_form_ == "NFD") {
      normalizer = icu::Normalizer2::getNFDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(), errors::Internal(
          absl::StrCat(icu_error.errorName(),
                       ": Could not retrieve ICU NFD normalizer")));
    } else if (normalization_form_ == "NFKD") {
      normalizer = icu::Normalizer2::getNFKDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(), errors::Internal(
          absl::StrCat(icu_error.errorName(),
                       ": Could not retrieve ICU NFKd normalizer")));
    } else {
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument(absl::StrCat(
              "Unknown normalization form requrested: ", normalization_form_)));
    }

    for (int64 i = 0; i < input_vec.size(); ++i) {
      string output_text;
      icu::StringByteSink<string> byte_sink(&output_text);
      const auto& input = input_vec(i);
      normalizer->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                                byte_sink, nullptr, icu_error);
      OP_REQUIRES(
          context, !U_FAILURE(icu_error),
          errors::Internal(absl::StrCat(icu_error.errorName(),
                                        ": Could not normalize input string: ",
                                        absl::string_view(input_vec(i)))));
      output_vec(i) = output_text;
    }
  }

 private:
  string normalization_form_;
};

REGISTER_KERNEL_BUILDER(Name("NormalizeUTF8").Device(tensorflow::DEVICE_CPU),
                        NormalizeUTF8Op);

}  // namespace text
}  // namespace tensorflow
