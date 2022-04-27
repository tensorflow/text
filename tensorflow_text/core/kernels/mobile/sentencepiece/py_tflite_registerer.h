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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_PY_TFLITE_REGISTERER_H_
#define TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_PY_TFLITE_REGISTERER_H_
#include "tensorflow/lite/mutable_op_resolver.h"

// C-function that is called from the Python Wrapper.

extern "C" void TFLite_SentencepieceTokenizerRegisterer(
    tflite::MutableOpResolver *resolver);

#endif  // TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_PY_TFLITE_REGISTERER_H_
