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

#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_tflite.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer_tflite.h"

PYBIND11_MODULE(pywrap_tflite_registrar, m) {
  m.doc() = R"pbdoc(
    pywrap_tflite_registrar
    A module with a Python wrapper for TFLite TFText ops:
      * WhitespaceTokenizer
  )pbdoc";
  m.def(
      "AddWhitespaceTokenizeWithOffsetsV2",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddWhitespaceTokenizeWithOffsetsV2(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds WhitespaceTokenizeWithOffsetsV2 to the TFLite
      interpreter.
      )pbdoc");
  m.def(
      "AddFastWordpieceTokenizer",
      [](uintptr_t resolver) {
        tflite::ops::custom::AddFastWordpieceTokenizer(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds FastWordpieceTokenizer to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddFastWordpieceDetokenizer",
      [](uintptr_t resolver) {
        tflite::ops::custom::AddFastWordpieceDetokenizer(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
    The function that adds FastWordpieceDetokenizer to the TFLite interpreter.
    )pbdoc");
}
