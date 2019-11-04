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

#include "tensorflow_text/core/kernels/regex_split.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "re2/re2.h"

namespace tensorflow {
namespace text {
namespace {

std::vector<absl::string_view> RunTest(const string& input, const string& regex,
                                       const string& delim_regex) {
  RE2 re2(regex);
  RE2 include_delim_re2(delim_regex);

  std::vector<int64> begin_offsets;
  std::vector<int64> end_offsets;
  std::vector<absl::string_view> tokens;

  RegexSplit(input, re2, true, include_delim_re2, &tokens, &begin_offsets,
             &end_offsets);
  return tokens;
}

TEST(RegexSplitTest, JapaneseAndWhitespace) {
  string regex = "(\\p{Hiragana}+|\\p{Katakana}+|\\s)";
  string delim_regex = "(\\p{Hiragana}+|\\p{Katakana}+)";
  string input = "He said フランスです";
  auto extracted_tokens = RunTest(input, regex, delim_regex);
  EXPECT_THAT(extracted_tokens, testing::ElementsAreArray({
                                    "He",
                                    "said",
                                    "フランス",
                                    "です",
                                }));
}

TEST(RegexSplitTest, Japanese) {
  string regex = "(\\p{Hiragana}+|\\p{Katakana}+)";
  string input = "He said フランスです";
  auto extracted_tokens = RunTest(input, regex, regex);
  EXPECT_THAT(extracted_tokens, testing::ElementsAreArray({
                                    "He said ",
                                    "フランス",
                                    "です",
                                }));
}

TEST(RegexSplitTest, ChineseHan) {
  string regex = "(\\p{Han})";
  string input = "敵人變盟友背後盤算";
  auto extracted_tokens = RunTest(input, regex, regex);
  EXPECT_THAT(extracted_tokens,
              testing::ElementsAreArray(
                  {"敵", "人", "變", "盟", "友", "背", "後", "盤", "算"}));
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
