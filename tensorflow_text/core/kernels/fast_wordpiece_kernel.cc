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

#include "tensorflow_text/core/kernels/fast_wordpiece_kernel.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace text {

REGISTER_KERNEL_BUILDER(Name(FastWordpieceTokenizeWithOffsetsOpKernel::OpName())
                            .Device(tensorflow::DEVICE_CPU),
                        FastWordpieceTokenizeWithOffsetsOpKernel);

REGISTER_KERNEL_BUILDER(Name(FastWordpieceDetokenizeOpKernel::OpName())
                            .Device(tensorflow::DEVICE_CPU),
                        FastWordpieceDetokenizeOpKernel);

}  // namespace text
}  // namespace tensorflow
