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
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "icu4c/source/common/unicode/edits.h"
#include "icu4c/source/common/unicode/errorcode.h"
#include "icu4c/source/common/unicode/normalizer2.h"
#include "icu4c/source/common/unicode/utypes.h"
#include "tensorflow/core/framework//node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

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
    const icu::Normalizer2* nfkc_cf =
        icu::Normalizer2::getNFKCCasefoldInstance(icu_error);
    OP_REQUIRES(context, icu_error.isSuccess(),
                errors::Internal(absl::StrCat(
                    icu_error.errorName(),
                    ": Could not retrieve ICU NFKC_CaseFold normalizer")));

    for (int64 i = 0; i < input_vec.size(); ++i) {
      string output_text;
      icu::StringByteSink<string> byte_sink(&output_text);
      const auto& input = input_vec(i);
      nfkc_cf->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                             byte_sink, nullptr, icu_error);
      OP_REQUIRES(context, !U_FAILURE(icu_error),
                  errors::Internal("Could not normalize input string: " +
                                   input_vec(i)));
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
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(absl::StrCat(
                      icu_error.errorName(),
                      ": Could not retrieve ICU NFKC normalizer")));
    } else if (normalization_form_ == "NFC") {
      normalizer = icu::Normalizer2::getNFCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(
                      absl::StrCat(icu_error.errorName(),
                                   ": Could not retrieve ICU NFC normalizer")));
    } else if (normalization_form_ == "NFD") {
      normalizer = icu::Normalizer2::getNFDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(
                      absl::StrCat(icu_error.errorName(),
                                   ": Could not retrieve ICU NFD normalizer")));
    } else if (normalization_form_ == "NFKD") {
      normalizer = icu::Normalizer2::getNFKDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(absl::StrCat(
                      icu_error.errorName(),
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

namespace {
// Our resource object that will hold the Edits object.
struct NormalizeResource : public ResourceBase {
  std::vector<icu::Edits> edits_vec;
  int64 memory_used;

  string DebugString() const override { return "Edits Resource"; }

  int64 MemoryUsed() const override { return memory_used; }
};
}  // namespace

class NormalizeUTF8WithOffsetsOp : public tensorflow::OpKernel {
 public:
  explicit NormalizeUTF8WithOffsetsOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context),
        normalization_form_(GetNormalizationForm(context)) {
    OP_REQUIRES_OK(context, context->GetAttr("use_node_name_sharing",
                                             &use_node_name_sharing_));
  }

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
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(absl::StrCat(
                      icu_error.errorName(),
                      ": Could not retrieve ICU NFKC normalizer")));
    } else if (normalization_form_ == "NFC") {
      normalizer = icu::Normalizer2::getNFCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(
                      absl::StrCat(icu_error.errorName(),
                                   ": Could not retrieve ICU NFC normalizer")));
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument(absl::StrCat(
                      "Offset not supported for this normalization form: ",
                      normalization_form_)));
    }

    edits_vec.clear();
    for (int64 i = 0; i < input_vec.size(); ++i) {
      string output_text;
      icu::Edits edits;
      icu::StringByteSink<string> byte_sink(&output_text);
      const auto& input = input_vec(i);
      normalizer->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                                byte_sink, &edits, icu_error);
      OP_REQUIRES(
          context, !U_FAILURE(icu_error),
          errors::Internal(absl::StrCat(icu_error.errorName(),
                                        ": Could not normalize input string: ",
                                        absl::string_view(input_vec(i)))));
      output_vec(i) = output_text;
      edits_vec.push_back(edits);
    }

    absl::MutexLock lock(&mu_);

    Tensor* handle;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &handle));

    OP_REQUIRES_OK(context, cinfo_.Init(context->resource_manager(), def(),
                                        use_node_name_sharing_));

    auto creator = [ this](NormalizeResource** resource)
                       ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                         NormalizeResource* sp = new NormalizeResource();
                         *resource = sp;
                         sp->edits_vec = this->edits_vec;
                         return Status::OK();
                       };

    // Register the ResourceType alias.
    NormalizeResource* resource = nullptr;
    OP_REQUIRES_OK(
        context,
        cinfo_.resource_manager()->template LookupOrCreate<NormalizeResource>(
            cinfo_.container(), cinfo_.name(), &resource, creator));
    core::ScopedUnref unref_me(resource);

    // Put a handle to resource in the output tensor (the other aliases will
    // have the same handle).
    auto new_handle = MakeResourceHandle<NormalizeResource>(
        context, cinfo_.container(), cinfo_.name());
    handle->scalar<ResourceHandle>()() = new_handle;
  }

 private:
  string normalization_form_;
  absl::Mutex mu_;
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;
  std::vector<icu::Edits> edits_vec;
};

REGISTER_KERNEL_BUILDER(
    Name("NormalizeUTF8WithOffsets").Device(tensorflow::DEVICE_CPU),
    NormalizeUTF8WithOffsetsOp);

template <typename SPLITS_TYPE>
class FindSourceOffsetsOp : public tensorflow::OpKernel {
 public:
  explicit FindSourceOffsetsOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    NormalizeResource* nr;
    const Tensor& resource_tensor = context->input(0);
    const tensorflow::Tensor& input_starts_values = context->input(1);
    const tensorflow::Tensor& input_starts_splits = context->input(2);
    const tensorflow::Tensor& input_limits_values = context->input(3);

    const auto& input_starts_values_vec = input_starts_values.flat<int32>();
    const auto& input_starts_splits_vec =
        input_starts_splits.flat<SPLITS_TYPE>();
    const auto& input_limits_values_vec = input_limits_values.flat<int32>();

    // Retrieve edits object
    ResourceHandle resource_handle(resource_tensor.scalar<ResourceHandle>()());
    OP_REQUIRES_OK(
        context, context->resource_manager()->Lookup<NormalizeResource, true>(
                     resource_handle.container(), resource_handle.name(), &nr));
    core::ScopedUnref unref_me(nr);

    icu::ErrorCode icu_error;
    int64 cur_split_index_begin = 0;
    int64 cur_split_index_end = 0;
    std::vector<int32> output_values_starts;
    std::vector<int32> output_values_limits;
    for (int64 i = 0; i < input_starts_splits_vec.size() - 1; ++i) {
      cur_split_index_begin = input_starts_splits_vec(i);
      cur_split_index_end = input_starts_splits_vec(i + 1);
      auto iter = nr->edits_vec[i].getFineChangesIterator();
      for (int64 j = cur_split_index_begin; j < cur_split_index_end; ++j) {
        output_values_starts.push_back(iter.sourceIndexFromDestinationIndex(
            input_starts_values_vec(j), icu_error));
        output_values_limits.push_back(iter.sourceIndexFromDestinationIndex(
            input_limits_values_vec(j), icu_error));
      }
    }

    tensorflow::Tensor* output_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "output_splits", input_starts_splits.shape(),
                                &output_splits_tensor));
    auto output_splits_flat = output_splits_tensor->flat<SPLITS_TYPE>();
    output_splits_flat = input_starts_splits_vec;

    // Allocate output & fill output tensors.
    int64 output_values_starts_size = output_values_starts.size();
    Tensor* output_values_starts_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "output_values_starts",
                                TensorShape({output_values_starts_size}),
                                &output_values_starts_tensor));
    auto output_values_starts_data =
        output_values_starts_tensor->flat<int32>().data();
    memcpy(output_values_starts_data, output_values_starts.data(),
           output_values_starts_size * sizeof(int32));

    int64 output_values_limits_size = output_values_limits.size();
    Tensor* output_values_limits_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "output_values_limits",
                                TensorShape({output_values_limits_size}),
                                &output_values_limits_tensor));
    auto output_values_limits_data =
        output_values_limits_tensor->flat<int32>().data();
    memcpy(output_values_limits_data, output_values_limits.data(),
           output_values_limits_size * sizeof(int32));

    LOG(ERROR) << "HTERRY DONE WITH OP!";
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FindSourceOffsetsOp);
};

REGISTER_KERNEL_BUILDER(Name("FindSourceOffsets")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int64>("Tsplits"),
                        FindSourceOffsetsOp<int64>);
REGISTER_KERNEL_BUILDER(Name("FindSourceOffsets")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32>("Tsplits"),
                        FindSourceOffsetsOp<int32>);
}  // namespace text
}  // namespace tensorflow
