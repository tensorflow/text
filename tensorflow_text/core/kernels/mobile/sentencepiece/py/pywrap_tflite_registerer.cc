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

#include "tensorflow_text/core/kernels/mobile/sentencepiece/py_tflite_registerer.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"

PYBIND11_MODULE(pywrap_tflite_registerer, m) {
  m.doc() = R"pbdoc(
    pywrap_tflite_registerer
    A module with a wrapper that adds to a Python wrapper for TFLite
    sentencepiece tokenizer.
  )pbdoc";
  m.def(
      "TFLite_SentencepieceTokenizerRegisterer",
      [](uintptr_t resolver) {
        TFLite_SentencepieceTokenizerRegisterer(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds Sentencepiece Tokenizer to the TFLite interpreter.
      )pbdoc");
}
