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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_H_

#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/umachine.h"

namespace tensorflow {
namespace text {

// Helper class for working with the WhitespaceaTokenizer config. The
// config is essentially a bit array stored in characters, where each bit in
// the char represents a Unicode character and whether or not it is considered
// as whitespace.
//
// This bit array contains all codepoints up to the largest whitespace
// character. So any codepoint larger than the array is not whitespace, and
// a lookup is simply using the codepoint value as the index. The first 3 bits
// of the codepoint indicate which bit in a character is the value located, and
// using the rest of the bits of the codepoint we can determine which
// character the particular codepoint is located at.
class WhitespaceTokenizerConfig {
 public:
  // This object does not own the config, so make certain it exists for the
  // lifetime of the class.
  WhitespaceTokenizerConfig(const absl::string_view config)
      : config_(config), max_codepoint_(config.length() * 8) {}
  WhitespaceTokenizerConfig(const std::string* config)
      : config_(*config), max_codepoint_(config->length() * 8) {}

  inline bool IsWhitespace(const UChar32 codepoint) const {
    return codepoint <= max_codepoint_ &&
           config_[codepoint >> 3] & (1 << (char)(codepoint & 0x7));
  }

 private:
  const absl::string_view config_;
  const int max_codepoint_;
};
}  // namespace text
}  // namespace tensorflow


#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_H_
