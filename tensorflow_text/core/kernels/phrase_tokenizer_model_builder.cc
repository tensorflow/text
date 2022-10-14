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

#include "tensorflow_text/core/kernels/phrase_tokenizer_model_builder.h"

#include <stdint.h>

#include <memory>
#include <queue>
#include <stack>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "icu4c/source/common/unicode/umachine.h"
#include "icu4c/source/common/unicode/utf8.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_builder.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_wrapper.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_utils.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer_model_generated.h"
#include "tensorflow_text/core/kernels/sentence_fragmenter_v2.h"
#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {
namespace {

// Builds the PhraseTokenizer model.
class PhraseBuilder {
 public:
  // When no_pretokenization is false, we split the input string by punctuation
  // chars (in addition to whitespaces) and then tokenize it to wordpieces.
  absl::Status BuildModel(const std::vector<std::string>& vocab,
                          absl::string_view unk_token,
                          bool support_detokenization, int width);

  absl::StatusOr<std::string> ExportToFlatBuffer() const;

 private:
  absl::optional<StringVocab> vocab_;
  std::string unk_token_;
  int unk_token_id_ = -1;
  // Whether the tokenizer supports the detokenization function.
  bool support_detokenization_;
  int width_;
};

absl::Status PhraseBuilder::BuildModel(const std::vector<std::string>& vocab,
                                       absl::string_view unk_token,
                                       bool support_detokenization, int width) {
  unk_token_ = std::string(unk_token);
  support_detokenization_ = support_detokenization;
  width_ = width;

  vocab_.emplace(vocab);
  if (vocab_->Size() != vocab.size()) {
    return absl::FailedPreconditionError(
        "Tokens in the vocabulary must be unique.");
  }

  // Determine `unk_token_id_`.
  const absl::optional<int> unk_token_id = vocab_->LookupId(unk_token_);
  if (!unk_token_id.has_value()) {
    return absl::FailedPreconditionError("Cannot find unk_token in the vocab!");
  }
  unk_token_id_ = *unk_token_id;

  return absl::OkStatus();
}

absl::StatusOr<std::string> PhraseBuilder::ExportToFlatBuffer() const {
  flatbuffers::FlatBufferBuilder builder;

  const auto unk_token = builder.CreateString(unk_token_);

  std::vector<flatbuffers::Offset<flatbuffers::String>> vocab_fbs_vector;

  if (support_detokenization_) {
    vocab_fbs_vector.reserve(vocab_->Size());
    for (int i = 0; i < vocab_->Size(); ++i) {
      const absl::optional<absl::string_view> word = vocab_->LookupWord(i);
      if (!word.has_value()) {
        return absl::FailedPreconditionError(
            "Impossible. `token_id` is definitely within the range of vocab "
            "token ids; hence LookupWord() should always succeed.");
      }
      absl::string_view token = word.value();
      vocab_fbs_vector.emplace_back(builder.CreateString(token));
    }
  }

  auto vocab_array = builder.CreateVector(vocab_fbs_vector);

  PhraseTokenizerConfigBuilder wtcb(builder);
  wtcb.add_unk_token(unk_token);
  wtcb.add_width(width_);
  wtcb.add_unk_token_id(unk_token_id_);
  wtcb.add_support_detokenization(support_detokenization_);
  wtcb.add_vocab_array(vocab_array);
  FinishPhraseTokenizerConfigBuffer(builder, wtcb.Finish());
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}
}  // namespace

absl::StatusOr<std::string> BuildPhraseModelAndExportToFlatBuffer(
    const std::vector<std::string>& vocab, absl::string_view unk_token,
    bool support_detokenization, int width) {
  PhraseBuilder builder;
  SH_RETURN_IF_ERROR(
      builder.BuildModel(vocab, unk_token, support_detokenization, width));
  SH_ASSIGN_OR_RETURN(std::string flatbuffer, builder.ExportToFlatBuffer());
  return flatbuffer;
}

}  // namespace text
}  // namespace tensorflow
