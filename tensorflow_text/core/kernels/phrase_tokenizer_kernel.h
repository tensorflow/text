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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_KERNEL_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_KERNEL_H_

#include "tensorflow/lite/kernels/shim/tf_op_shim.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer_kernel_template.h"

namespace tensorflow {
namespace text {

class PhraseTokenizeOpKernel
    : public tflite::shim::TfOpKernel<PhraseTokenizeOp> {
 public:
  using TfOpKernel::TfOpKernel;
};

class PhraseDetokenizeOpKernel
    : public tflite::shim::TfOpKernel<PhraseDetokenizeOp> {
 public:
  using TfOpKernel::TfOpKernel;
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_KERNEL_H_
