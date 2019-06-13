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
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace text {

constexpr int64 kOutOfVocabValue = -1;

LookupTableVocab::LookupTableVocab(lookup::LookupInterface* table,
                                   OpKernelContext* ctx)
    : table_(table), ctx_(ctx), default_value_(DT_INT64, TensorShape({1})) {
  default_value_.flat<int64>()(0) = kOutOfVocabValue;
}

Status LookupTableVocab::Contains(const string& key, bool* value) {
  if (value == nullptr) {
    return errors::InvalidArgument("Bad 'value' param.");
  }
  Tensor keys(DT_STRING, TensorShape({1}));
  keys.flat<string>()(0) = key;
  Tensor values(DT_INT64, TensorShape({1}));
  TF_RETURN_IF_ERROR(table_->Find(ctx_, keys, &values, default_value_));

  if (static_cast<int64>(values.flat<int64>()(0)) != kOutOfVocabValue) {
    *value = true;
    return Status::OK();
  }
  *value = false;
  return Status::OK();
}

namespace {

Status Lookup(int byte_start, int byte_end, const string& token,
              const string& suffix_indicator, LookupTableVocab* vocab_map,
              bool* in_vocab) {
  int byte_len = byte_end - byte_start;
  absl::string_view substr(token.data() + byte_start, byte_len);
  string lookup_value;
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
Status LongestMatchStartingAt(int byte_start, const string& token,
                              const string& suffix_indicator,
                              LookupTableVocab* vocab_map, int* byte_end,
                              bool* found_match) {
  const char* token_bytes = token.data();
  const int token_len = token.length();
  std::vector<int32_t> byte_ends;
  for (int32_t i = byte_start; i < token_len;) {
    UChar32 c;
    U8_NEXT(token_bytes, i, token_len, c);
    byte_ends.push_back(i);
  }
  int n = byte_ends.size();
  for (int i = n - 1; i >= 0; i--) {
    bool in_vocab;
    TF_RETURN_IF_ERROR(Lookup(byte_start, byte_ends[i], token, suffix_indicator,
                              vocab_map, &in_vocab));
    if (in_vocab) {
      *byte_end = byte_ends[i];
      *found_match = true;
      return Status::OK();
    }
  }
  *found_match = false;
  return Status::OK();
}

// Sets the outputs 'begin_offset', 'end_offset' and 'num_word_pieces' when no
// token is found.
Status NoTokenFound(const string& token, bool use_unknown_token,
                    const string& unknown_token, std::vector<string>* subwords,
                    std::vector<int>* begin_offset,
                    std::vector<int>* end_offset, int* num_word_pieces) {
  begin_offset->push_back(0);
  if (use_unknown_token) {
    subwords->push_back(unknown_token);
    end_offset->push_back(token.length());
  } else {
    subwords->push_back(token);
    end_offset->push_back(token.length());
  }
  ++(*num_word_pieces);

  return Status::OK();
}

// When a subword is found, this helper function will add the outputs to
// 'subwords', 'begin_offset' and 'end_offset'.
void AddWord(const string& token, int byte_start, int byte_end,
             const string& suffix_indicator, std::vector<string>* subwords,
             std::vector<int>* begin_offset, std::vector<int>* end_offset) {
  begin_offset->push_back(byte_start);
  int len = byte_end - byte_start;
  if (byte_start > 0) {
    // Prepend suffix_indicator if the token is within a word.
    subwords->push_back(
        ::absl::StrCat(suffix_indicator, token.substr(byte_start, len)));
  } else {
    subwords->push_back(token.substr(byte_start, len));
  }
  end_offset->push_back(byte_end);
}

Status TokenizeL2RGreedy(const string& token, const int64 max_bytes_per_token,
                         const string& suffix_indicator, bool use_unknown_token,
                         const string& unknown_token,
                         LookupTableVocab* vocab_map,
                         std::vector<string>* subwords,
                         std::vector<int>* begin_offset,
                         std::vector<int>* end_offset, int* num_word_pieces) {
  std::vector<string> candidate_subwords;
  std::vector<int> candidate_begin_offsets;
  std::vector<int> candidate_end_offsets;
  const int token_len = token.length();
  for (int byte_start = 0; byte_start < token_len;) {
    int byte_end;
    bool found_subword;
    TF_RETURN_IF_ERROR(LongestMatchStartingAt(byte_start, token,
                                              suffix_indicator, vocab_map,
                                              &byte_end, &found_subword));
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
  return Status::OK();
}

}  // namespace

Status WordpieceTokenize(const string& token, const int64 max_bytes_per_token,
                         const string& suffix_indicator, bool use_unknown_token,
                         const string& unknown_token,
                         LookupTableVocab* vocab_map,
                         std::vector<string>* subwords,
                         std::vector<int>* begin_offset,
                         std::vector<int>* end_offset, int* num_word_pieces) {
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
    return Status::OK();
  }
  return TokenizeL2RGreedy(token, max_bytes_per_token, suffix_indicator,
                           use_unknown_token, unknown_token, vocab_map,
                           subwords, begin_offset, end_offset, num_word_pieces);
}

}  // namespace text
}  // namespace tensorflow
