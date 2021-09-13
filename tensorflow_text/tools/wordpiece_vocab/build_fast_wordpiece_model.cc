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

// Tool to build FastWordpieceTokenizer models.
#include "absl/flags/flag.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_builder.h"

ABSL_FLAG(std::string, vocab_file, "", "The file path to the vocabulary file.");
ABSL_FLAG(int, max_bytes_per_token, 100, "Max bytes per token.");
ABSL_FLAG(std::string, suffix_indicator, "##", "The suffix indicator.");
ABSL_FLAG(std::string, unk_token, "[UNK]", "The unknown token.");
ABSL_FLAG(
    bool, end_to_end, false,
    "Whether to build end-to-end tokenizer for tokenizing general texts.");
ABSL_FLAG(
    bool, support_detokenization, false,
    "Whether the tokenizer to build supports the detokenization function.");
ABSL_FLAG(std::string, output_model_file, "", "The output model file path.");

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // Load the vocababulary.
  std::string vocab_file_content;
  auto status = tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                             absl::GetFlag(FLAGS_vocab_file),
                                             &vocab_file_content);
  TF_CHECK_OK(status) << "Failed to read the vocabulary file.";
  std::vector<std::string> vocab = absl::StrSplit(vocab_file_content, '\n');

  // Build the model.
  auto model_buffer = tensorflow::text::BuildModelAndExportToFlatBuffer(
      vocab, absl::GetFlag(FLAGS_max_bytes_per_token),
      absl::GetFlag(FLAGS_suffix_indicator), absl::GetFlag(FLAGS_unk_token),
      absl::GetFlag(FLAGS_end_to_end),
      absl::GetFlag(FLAGS_support_detokenization));
  CHECK_OK(model_buffer.status())  // Crash OK. An offline tool.
      << "Failed to build the FastWordpieceTokenizer model.";

  // Write to the output.
  status = tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                         absl::GetFlag(FLAGS_output_model_file),
                                         *model_buffer);
  TF_CHECK_OK(status) << "Failed to write to the output model file.";
  return 0;
}
