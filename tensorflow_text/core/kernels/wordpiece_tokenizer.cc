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

Status WordpieceTokenize(const string& token, const int64 max_bytes_per_token,
                         const string& suffix_indicator, bool use_unknown_token,
                         const string& unknown_token,
                         LookupTableVocab* vocab_map,
                         std::vector<string>* subwords,
                         std::vector<int>* begin_offset,
                         std::vector<int>* end_offset, int* num_word_pieces) {
  if (token.size() > max_bytes_per_token) {
    if (use_unknown_token) {
      subwords->emplace_back(unknown_token);
      end_offset->push_back(unknown_token.size());
    } else {
      subwords->emplace_back(token);
      end_offset->push_back(token.size());
    }
    begin_offset->push_back(0);
    *num_word_pieces = 1;
    return Status::OK();
  }

  icu::UnicodeString token_unicode = icu::UnicodeString::fromUTF8(token);
  bool is_bad = false;
  int start = 0;
  int byte_offset_start = 0;
  std::vector<string> sub_tokens;
  std::vector<int> sub_tokens_begin_offset;
  std::vector<int> sub_tokens_end_offset;
  while (start < token_unicode.length()) {
    string cur_substr;
    int end = token_unicode.length();
    int num_subword_bytes = token.size() - byte_offset_start;
    icu::StringCharacterIterator backward_iter(token_unicode, start, end,
                                               start);
    backward_iter.last32();

    while (num_subword_bytes > 0) {
      absl::string_view substr(token.data() + byte_offset_start,
                               num_subword_bytes);
      string lookup_value;
      if (byte_offset_start > 0) {
        lookup_value = absl::StrCat(suffix_indicator, substr);
      } else {
        // absl::CopyToString
        lookup_value.assign(substr.begin(), substr.end());
      }

      bool found_in_vocab;
      TF_RETURN_IF_ERROR(vocab_map->Contains(lookup_value, &found_in_vocab));
      if (found_in_vocab) {
        cur_substr.swap(lookup_value);
        break;
      }
      --end;
      num_subword_bytes -= U8_LENGTH(backward_iter.current32());
      backward_iter.previous32();
    }
    if (cur_substr.empty()) {
      is_bad = true;
      break;
    }

    sub_tokens.emplace_back(cur_substr);
    sub_tokens_begin_offset.emplace_back(byte_offset_start);
    sub_tokens_end_offset.emplace_back(byte_offset_start + num_subword_bytes);
    start = end;
    byte_offset_start += num_subword_bytes;
  }
  if (is_bad) {
    if (use_unknown_token) {
      subwords->emplace_back(unknown_token);
    } else {
      subwords->emplace_back(token);
    }
    begin_offset->emplace_back(0);
    end_offset->emplace_back(token.size());
    *num_word_pieces = 1;
  } else {
    subwords->insert(subwords->end(), sub_tokens.begin(), sub_tokens.end());
    begin_offset->insert(begin_offset->end(), sub_tokens_begin_offset.begin(),
                         sub_tokens_begin_offset.end());
    end_offset->insert(end_offset->end(), sub_tokens_end_offset.begin(),
                       sub_tokens_end_offset.end());
    *num_word_pieces = sub_tokens.size();
  }
  return Status::OK();
}

}  // namespace text
}  // namespace tensorflow
