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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/umachine.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer_model_builder.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer_model_generated.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer.h"
#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {

class PhraseTokenizer {
 public:
  // Creates an instance.
  //
  // Args:
  //  * config_flatbuffer: the pointer to the PhraseTokenizerConfig
  //    flatbuffer, which is not owned by this instance and should be kept alive
  //    through the lifetime of the instance.
  static absl::StatusOr<PhraseTokenizer> Create(const void* config_flatbuffer);

  // Tokenizes a string (or series of character codepoints) by Phrase.
  //
  // Example:
  // input = "Show me the way."
  // output = ["Show me", "the", "way."]
  //
  // The input should be UTF-8 but the tokenization is performed on Unicode
  // codepoints.
  //
  // Args:
  //  * input: The UTF-8 string of an input.
  //  * tokens: The output tokens.
  void Tokenize(const absl::string_view input,
                std::vector<std::string>* result_tokens,
                std::vector<int>* result_token_ids);

  absl::StatusOr<std::vector<std::string>> DetokenizeToTokens(
      const absl::Span<const int> input) const;
  absl::StatusOr<std::string> Detokenize(
      const absl::Span<const int> input) const;

  void FindPhraseTokens(const std::vector<std::string>& tokens,
                        std::vector<std::vector<std::string>>& all_tokens,
                        std::vector<std::vector<int>>& all_token_ids,
                        std::vector<int>& max_ngram_len, int* cur_index);

  LookupStatus PhraseLookup(const absl::string_view& token, bool* in_vocab,
                            int* index);

  std::unique_ptr<StringVocab> vocab_ = nullptr;
  int64_t width_;
  std::unique_ptr<WhitespaceTokenizerConfig> whitespace_config_ = nullptr;
  const PhraseTokenizerConfig* phrase_config_;
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_
