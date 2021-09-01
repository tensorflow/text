// Copyright 2021 TF.Text Authors.
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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_OPS_WHITESPACE_TOKENIZER_OP_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_OPS_WHITESPACE_TOKENIZER_OP_H_

#include "tensorflow/lite/kernels/shim/tf_op_shim.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer_kernel.h"

namespace tensorflow {
namespace text {

#define REGISTER_OP_SHIM_IMPL_TMP(ctr, op_kernel_cls)                        \
  static ::tensorflow::InitOnStartupMarker const register_op##ctr            \
      TF_ATTRIBUTE_UNUSED =                                                  \
          TF_INIT_ON_STARTUP_IF(SHOULD_REGISTER_OP(op_kernel_cls::OpName())) \
          << ::tflite::shim::CreateOpDefBuilderWrapper<op_kernel_cls>()

TF_ATTRIBUTE_ANNOTATE("tf:op")
TF_NEW_ID_FOR_INIT(REGISTER_OP_SHIM_IMPL_TMP,
                   WhitespaceTokenizeWithOffsetsV2OpKernel);

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_OPS_WHITESPACE_TOKENIZER_OP_H_
