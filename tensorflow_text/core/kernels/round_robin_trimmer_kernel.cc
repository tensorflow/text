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

#include "tensorflow_text/core/kernels/round_robin_trimmer_kernel.h"

#include <cstdint>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace text {

using RoundRobinGenerateMasksOpKernelInstance =
    RoundRobinGenerateMasksOpKernel<int32_t>;

#define REGISTER_ROUND_ROBIN_GENERATE_MASKS(splits_type)      \
  REGISTER_KERNEL_BUILDER(                                    \
      Name(RoundRobinGenerateMasksOpKernelInstance::OpName()) \
          .Device(tensorflow::DEVICE_CPU)                     \
          .TypeConstraint<splits_type>("Tsplits"),            \
      RoundRobinGenerateMasksOpKernel<splits_type>);

REGISTER_ROUND_ROBIN_GENERATE_MASKS(int32_t)
REGISTER_ROUND_ROBIN_GENERATE_MASKS(int64_t)

#undef REGISTER_ROUND_ROBIN_GENERATE_MASKS

using RoundRobinTrimOpKernelInstance = RoundRobinTrimOpKernel<int32_t>;

#define REGISTER_ROUND_ROBIN_TRIM(splits_type)                           \
  REGISTER_KERNEL_BUILDER(Name(RoundRobinTrimOpKernelInstance::OpName()) \
                              .Device(tensorflow::DEVICE_CPU)            \
                              .TypeConstraint<splits_type>("Tsplits"),   \
                          RoundRobinTrimOpKernel<splits_type>);

REGISTER_ROUND_ROBIN_TRIM(int32_t)
REGISTER_ROUND_ROBIN_TRIM(int64_t)

#undef REGISTER_ROUND_ROBIN_TRIM

}  // namespace text
}  // namespace tensorflow
