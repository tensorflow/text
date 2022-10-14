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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_MODEL_BUILDER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_MODEL_BUILDER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {

// An implementation of WordpieceVocab, used (1) to store the input vocabulary
// and (2) to call the original implementation of WordPiece tokenization to
// pre-compute the result for the suffix indicator string.
class StringVocab : public WordpieceVocab {
 public:
  explicit StringVocab(const std::vector<std::string>& vocab) : vocab_(vocab) {
    std::cout << "yun zhu!!" << std::endl;
    for (int i = 0; i < vocab.size(); ++i) {
      std::cout << "yun zhu: " << vocab_[i] << std::endl;
      index_map_[vocab_[i]] = i;
    }
  }

  LookupStatus Contains(absl::string_view key, bool* value) const override {
    *value = index_map_.contains(key);
    return LookupStatus();
  }

  absl::optional<int> LookupId(absl::string_view key) const {
    auto it = index_map_.find(key);
    if (it == index_map_.end()) {
      return absl::nullopt;
    } else {
      return it->second;
    }
  }

  // Returns the key of `vocab_id` or empty if `vocab_id` is not valid.
  absl::optional<absl::string_view> LookupWord(int vocab_id) const {
    if (vocab_id >= vocab_.size() || vocab_id < 0) {
      return absl::nullopt;
    }
    return vocab_[vocab_id];
  }

  int Size() const { return index_map_.size(); }

 private:
  std::vector<std::string> vocab_;
  absl::flat_hash_map<absl::string_view, int> index_map_;
};

// Builds a PhraseTokenizer model in flatbuffer format.
//
// Args:
//  * vocab: The phrase vocabulary.
//  * unk_token: The unknown token string.
//. * support_detokenization: Whether to enable the detokenization function.
//    Setting it to true expands the size of the flatbuffer. As a reference,
//    When using 120k multilingual BERT WordPiece vocab, the flatbuffer's size
//    increases from ~5MB to ~6MB.
// Returns:
//  The bytes of the flatbuffer that stores the model.
absl::StatusOr<std::string> BuildPhraseModelAndExportToFlatBuffer(
    const std::vector<std::string>& vocab, absl::string_view unk_token,
    bool support_detokenization = false, int width = 1);
}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_MODEL_BUILDER_H_
