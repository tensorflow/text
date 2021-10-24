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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TFLITE_KERNEL_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TFLITE_KERNEL_H_

#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {

extern "C" void AddFastWordpieceTokenizer(
    ::tflite::MutableOpResolver* resolver);

extern "C" void AddFastWordpieceDetokenizer(
    ::tflite::MutableOpResolver* resolver);

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TFLITE_KERNEL_H_
