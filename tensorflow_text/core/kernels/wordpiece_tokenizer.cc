// Copyright 2019 TF.Text Authors.
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

#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/schriter.h"
#include "icu4c/source/common/unicode/unistr.h"
#include "icu4c/source/common/unicode/utf8.h"

namespace tensorflow {
namespace text {

namespace {

LookupStatus Lookup(int byte_start, int byte_end,
                    const absl::string_view& token,
                    const std::string& suffix_indicator,
                    const WordpieceVocab* vocab_map, bool* in_vocab) {
  int byte_len = byte_end - byte_start;
  absl::string_view substr(token.data() + byte_start, byte_len);
  std::string lookup_value;
  if (byte_start > 0) {
    lookup_value = absl::StrCat(suffix_indicator, substr);
  } else {
    // absl::CopyToString
    lookup_value.assign(substr.begin(), substr.end());
  }
  return vocab_map->Contains(lookup_value, in_vocab);
}

// Sets byte_end to the longest byte sequence which:
// 1) is a proper UTF8 sequence
// 2) is in the vocab
// If no match is found, found_match is set to false.
LookupStatus LongestMatchStartingAt(int byte_start,
                                    const absl::string_view& token,
                                    const std::string& suffix_indicator,
                                    const int max_bytes_per_subtoken,
                                    const WordpieceVocab* vocab_map,
                                    int* byte_end, bool* found_match) {
  const char* token_bytes = token.data();
  std::vector<int32_t> byte_ends;
  int upper_limit = token.length();
  if (max_bytes_per_subtoken > 0) {
    upper_limit = std::min(upper_limit, byte_start + max_bytes_per_subtoken);
  }
  for (int32_t i = byte_start; i < upper_limit;) {
    UChar32 c;
    U8_NEXT(token_bytes, i, upper_limit, c);
    byte_ends.push_back(i);
  }
  int n = byte_ends.size();
  for (int i = n - 1; i >= 0; i--) {
    bool in_vocab;
    auto status = Lookup(byte_start, byte_ends[i], token, suffix_indicator,
                         vocab_map, &in_vocab);
    if (!status.success) return status;
    if (in_vocab) {
      *byte_end = byte_ends[i];
      *found_match = true;
      return LookupStatus::OK();
    }
  }
  *found_match = false;
  return LookupStatus::OK();
}

// Sets the outputs 'begin_offset', 'end_offset' and 'num_word_pieces' when no
// token is found.
LookupStatus NoTokenFound(const absl::string_view& token,
                          bool use_unknown_token,
                          const std::string& unknown_token,
                          std::vector<std::string>* subwords,
                          std::vector<int>* begin_offset,
                          std::vector<int>* end_offset, int* num_word_pieces) {
  begin_offset->push_back(0);
  if (use_unknown_token) {
    subwords->push_back(unknown_token);
    end_offset->push_back(token.length());
  } else {
    subwords->emplace_back(token.data(), token.length());
    end_offset->push_back(token.length());
  }
  ++(*num_word_pieces);

  return LookupStatus::OK();
}

// When a subword is found, this helper function will add the outputs to
// 'subwords', 'begin_offset' and 'end_offset'.
void AddWord(const absl::string_view& token, int byte_start, int byte_end,
             const std::string& suffix_indicator,
             std::vector<std::string>* subwords, std::vector<int>* begin_offset,
             std::vector<int>* end_offset) {
  begin_offset->push_back(byte_start);
  int len = byte_end - byte_start;
  if (byte_start > 0) {
    // Prepend suffix_indicator if the token is within a word.
    subwords->push_back(::absl::StrCat(
        suffix_indicator, absl::string_view(token.data() + byte_start, len)));
  } else {
    subwords->emplace_back(token.data(), len);
  }
  end_offset->push_back(byte_end);
}

LookupStatus TokenizeL2RGreedy(
    const absl::string_view& token, const int max_bytes_per_token,
    const int max_bytes_per_subtoken, const std::string& suffix_indicator,
    bool use_unknown_token, const std::string& unknown_token,
    const WordpieceVocab* vocab_map, std::vector<std::string>* subwords,
    std::vector<int>* begin_offset, std::vector<int>* end_offset,
    int* num_word_pieces) {
  std::vector<std::string> candidate_subwords;
  std::vector<int> candidate_begin_offsets;
  std::vector<int> candidate_end_offsets;
  const int token_len = token.length();
  for (int byte_start = 0; byte_start < token_len;) {
    int byte_end;
    bool found_subword;
    auto status = LongestMatchStartingAt(byte_start, token, suffix_indicator,
                                         max_bytes_per_subtoken,
                                         vocab_map, &byte_end, &found_subword);
    if (!status.success) return status;
    if (found_subword) {
      AddWord(token, byte_start, byte_end, suffix_indicator,
              &candidate_subwords, &candidate_begin_offsets,
              &candidate_end_offsets);
      byte_start = byte_end;
    } else {
      return NoTokenFound(token, use_unknown_token, unknown_token, subwords,
                          begin_offset, end_offset, num_word_pieces);
    }
  }

  subwords->insert(subwords->end(), candidate_subwords.begin(),
                   candidate_subwords.end());
  begin_offset->insert(begin_offset->end(), candidate_begin_offsets.begin(),
                       candidate_begin_offsets.end());
  end_offset->insert(end_offset->end(), candidate_end_offsets.begin(),
                     candidate_end_offsets.end());
  *num_word_pieces += candidate_subwords.size();
  return LookupStatus::OK();
}

}  // namespace

LookupStatus WordpieceTokenize(
    const absl::string_view& token, const int max_bytes_per_token,
    const int max_bytes_per_subtoken, const std::string& suffix_indicator,
    bool use_unknown_token, const std::string& unknown_token,
    const WordpieceVocab* vocab_map, std::vector<std::string>* subwords,
    std::vector<int>* begin_offset, std::vector<int>* end_offset,
    int* num_word_pieces) {
  int token_len = token.size();
  if (token_len > max_bytes_per_token) {
    begin_offset->push_back(0);
    *num_word_pieces = 1;
    if (use_unknown_token) {
      end_offset->push_back(unknown_token.size());
      subwords->emplace_back(unknown_token);
    } else {
      subwords->emplace_back(token);
      end_offset->push_back(token.size());
    }
    return LookupStatus::OK();
  }
  return TokenizeL2RGreedy(
    token, max_bytes_per_token, max_bytes_per_subtoken, suffix_indicator,
    use_unknown_token, unknown_token, vocab_map,
    subwords, begin_offset, end_offset, num_word_pieces);
}

}  // namespace text
}  // namespace tensorflow
