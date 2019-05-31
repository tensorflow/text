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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_WORDPIECE_TOKENIZER_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_WORDPIECE_TOKENIZER_H_

#include <map>
#include "third_party/tensorflow/core/framework/lookup_interface.h"
#include "third_party/tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

class WordpieceVocab {
 public:
  virtual ~WordpieceVocab() {}
  virtual Status Contains(const string& key, bool* value) = 0;
};

class LookupTableVocab : public WordpieceVocab {
 public:
  LookupTableVocab(lookup::LookupInterface* table, OpKernelContext* ctx);

  virtual Status Contains(const string& key, bool* value);

 private:
  // not owned
  lookup::LookupInterface* table_;
  OpKernelContext* ctx_;
  Tensor default_value_;
};

Status WordpieceTokenize(const string& token, const int64 max_bytes_per_token,
                         const string& suffix_indicator, bool use_unknown_token,
                         const string& unknown_token,
                         LookupTableVocab* vocab_map,
                         std::vector<string>* subwords,
                         std::vector<int>* begin_offset,
                         std::vector<int>* end_offset, int* num_word_pieces);

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_WORDPIECE_TOKENIZER_H_
