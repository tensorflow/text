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

#include <locale>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace text {

namespace {
template <typename SPLITS_TYPE>
class StringNGramsOp : public tensorflow::OpKernel {
 public:
  explicit StringNGramsOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("separator", &separator_));
    OP_REQUIRES_OK(context, context->GetAttr("ngram_width", &ngram_width_));
    OP_REQUIRES_OK(context, context->GetAttr("left_pad", &left_pad_));
    OP_REQUIRES_OK(context, context->GetAttr("right_pad", &right_pad_));
    OP_REQUIRES_OK(context, context->GetAttr("use_pad", &use_pad_));
    OP_REQUIRES_OK(context, context->GetAttr("extend_pad", &extend_pad_));
    pad_size_ = extend_pad_ ? ngram_width_ - 1 : 1;
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor* data;
    OP_REQUIRES_OK(context, context->input("data", &data));
    const auto& input_data = data->flat<string>().data();

    const tensorflow::Tensor* splits;
    OP_REQUIRES_OK(context, context->input("data_splits", &splits));
    const auto& splits_vec = splits->flat<SPLITS_TYPE>();
    int num_batch_items = splits_vec.size() - 1;

    tensorflow::Tensor* ngrams_splits;
    OP_REQUIRES_OK(
        context, context->allocate_output(1, splits->shape(), &ngrams_splits));
    auto ngrams_splits_data = ngrams_splits->flat<SPLITS_TYPE>().data();

    ngrams_splits_data[0] = 0;
    for (int i = 1; i <= num_batch_items; ++i) {
      int length = splits_vec(i) - splits_vec(i - 1);
      int num_tokens;
      if (use_pad_ && extend_pad_) {
        num_tokens = length + pad_size_;
      } else if (use_pad_ && !extend_pad_) {
        num_tokens = max(0, ((length + 2) - ngram_width_) + 1);
      } else {
        num_tokens = max(0, (length - ngram_width_) + 1);
      }
      ngrams_splits_data[i] = ngrams_splits_data[i - 1] + num_tokens;
    }

    tensorflow::Tensor* ngrams;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({ngrams_splits_data[num_batch_items]}), &ngrams));
    auto ngrams_data = ngrams->flat<string>().data();

    for (int i = 1; i <= num_batch_items; ++i) {
      auto data_start = &input_data[splits_vec(i - 1)];
      auto output_start = &ngrams_data[ngrams_splits_data[i - 1]];
      int num_ngrams = ngrams_splits_data[i] - ngrams_splits_data[i - 1];
      create_ngrams(data_start, output_start, num_ngrams);
    }
  }

  void create_ngrams(const string* data, string* output, int num_ngrams) {
    for (int ngram_index = 0; ngram_index < num_ngrams; ++ngram_index) {
      string* ngram_string = &output[ngram_index];
      *ngram_string = "";
      int num_pad_tokens = use_pad_ ? pad_size_ : 0;
      int left_padding = max(0, num_pad_tokens - ngram_index);
      int right_padding =
          max(0, num_pad_tokens - (num_ngrams - (ngram_index + 1)));
      int num_tokens = ngram_width_ - (left_padding + right_padding);

      for (int n = 0; n < left_padding; ++n) {
        *ngram_string += left_pad_;
        *ngram_string += separator_;
      }
      for (int n = 0; n < num_tokens; ++n) {
        int start_index = left_padding > 0 ? 0 : ngram_index - num_pad_tokens;
        *ngram_string += data[start_index + n];
        if (left_padding + n < ngram_width_ - 1) {
          *ngram_string += separator_;
        }
      }
      for (int n = 0; n < right_padding; ++n) {
        *ngram_string += right_pad_;
        if (left_padding + num_tokens + n < ngram_width_ - 1) {
          *ngram_string += separator_;
        }
      }
    }
  }

  string separator_;
  string left_pad_;
  string right_pad_;
  bool use_pad_;
  bool extend_pad_;

  int ngram_width_;
  int pad_size_;
};

}  // namespace
REGISTER_KERNEL_BUILDER(Name("InternalStringNGrams")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32>("Tsplits"),
                        StringNGramsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("InternalStringNGrams")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int64>("Tsplits"),
                        StringNGramsOp<int64>);

}  // namespace text
}  // namespace tensorflow
