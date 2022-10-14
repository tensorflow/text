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

#include "tensorflow_text/core/kernels/phrase_tokenizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace text {
namespace {

using ::testing::ElementsAre;

constexpr char kTestConfigPath[] =
    "third_party/tensorflow_text/python/ops/test_data/"
    "phrase_tokenizer_model.fb";

// TEST(PhraseTokenizerTest, Tokenize) {
//   absl::string_view input("I heard the news      today");
//   std::vector<std::string> output_tokens;
//   std::vector<int> output_token_ids;
//   // std::string config = BuildPhraseTokenizerConfig();
//   // StringVocab vocab({"heard news today", "the news today"});

//   std::string config_flatbuffer;
//   auto status = tensorflow::ReadFileToString(
//       tensorflow::Env::Default(), kTestConfigPath, &config_flatbuffer);
//   ASSERT_TRUE(status.ok());

//   ASSERT_OK_AND_ASSIGN(auto tokenizer,
//                        PhraseTokenizer::Create(config_flatbuffer.data()));

//   tokenizer.Tokenize(input, &output_tokens, &output_token_ids);
//   EXPECT_THAT(output_tokens, ElementsAre("I", "<UNK>", "the news today"));
//   EXPECT_THAT(output_token_ids, ElementsAre(1, 0, 4));
// }

TEST(PhraseTokenizerTest, TokenizeLonger) {
  absl::string_view input("I heard the news      today I heard");
  std::vector<std::string> output_tokens;
  std::vector<int> output_token_ids;
  // std::string config = BuildPhraseTokenizerConfig();
  // StringVocab vocab({"heard news today", "the news today"});

  std::string config_flatbuffer;
  auto status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(), kTestConfigPath, &config_flatbuffer);
  ASSERT_TRUE(status.ok());

  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       PhraseTokenizer::Create(config_flatbuffer.data()));

  tokenizer.Tokenize(input, &output_tokens, &output_token_ids);
  EXPECT_THAT(output_tokens,
              ElementsAre("I", "<UNK>", "the news today", "I", "<UNK>"));
  EXPECT_THAT(output_token_ids, ElementsAre(1, 0, 4, 1, 0));
}

// TEST(PhraseTokenizerTest, DeTokenize) {
//   std::vector<int> input({1, 0, 4});

//   std::string config_flatbuffer;
//   auto status = tensorflow::ReadFileToString(
//       tensorflow::Env::Default(), kTestConfigPath, &config_flatbuffer);
//   ASSERT_TRUE(status.ok());

//   ASSERT_OK_AND_ASSIGN(auto tokenizer,
//                        PhraseTokenizer::Create(config_flatbuffer.data()));

//   auto output_string = tokenizer.Detokenize(input);
//   EXPECT_EQ(output_string.value(), "I <UNK> the news today");
// }

}  // namespace
}  // namespace text
}  // namespace tensorflow
