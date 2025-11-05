// Copyright 2025 TF.Text Authors.
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

#include "tensorflow_text/core/kernels/whitespace_tokenizer_config_builder.h"

#include <string>
#include <cassert>

#include "icu4c/source/common/unicode/umachine.h"
#include "icu4c/source/common/unicode/uniset.h"
#include "icu4c/source/common/unicode/unistr.h"
#include "icu4c/source/common/unicode/utypes.h"

namespace tensorflow {
namespace text {

namespace {

const icu::UnicodeSet& WhiteSpaceSet() {
  // Use a C++11 static lambda to safely initialize the UnicodeSet.
  static const icu::UnicodeSet white_space_set = []() {
    UErrorCode status = U_ZERO_ERROR;
    // The pattern "[:White_Space:]" selects all whitespace characters.
    icu::UnicodeSet set(u"[:White_Space:]", status);
    // This should never fail as the pattern is hardcoded and valid.
    assert(U_SUCCESS(status));
    return set;
  }();
  return white_space_set;
}

}  // namespace

std::string BuildWhitespaceString() {
  icu::UnicodeString ustr;
  for (auto cp : WhiteSpaceSet().codePoints()) {
    ustr.append(cp);
  }
  std::string str;
  ustr.toUTF8String(str);
  return str;
}

std::string BuildWhitespaceTokenizerConfig() {
  const icu::UnicodeSet& set = WhiteSpaceSet();
  int range_count = set.getRangeCount();
  UChar32 largest_whitespace = set.getRangeEnd(range_count - 1);
  // The string will hold our bit array
  std::string bitset((largest_whitespace >> 3) + 1, 0);
  for (auto cp : set.codePoints()) {
    int index = cp >> 3;
    bitset[index] |= 1 << (cp & 7);
  }
  return bitset;
}

}  // namespace text
}  // namespace tensorflow
