// Copyright 2026 TF.Text Authors.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_builder.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_wrapper.h"

namespace tensorflow {
namespace text {
namespace trie_utils {

using ::testing::status::StatusIs;

TEST(DartsCloneTrieTest, CreateCursorPointToRootAndTryTraverseOneStep) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  cursor = trie.CreateTraversalCursorPointToRoot();  // Create a cursor to point
                                                     // to the root.
  EXPECT_TRUE(trie.TryTraverseOneStep(cursor, 'A'));
  EXPECT_FALSE(trie.TryGetData(cursor, data));
  EXPECT_TRUE(trie.TryTraverseOneStep(cursor, 'b'));
  EXPECT_FALSE(trie.TryGetData(cursor, data));
  EXPECT_TRUE(trie.TryTraverseOneStep(cursor, 'c'));
  EXPECT_TRUE(trie.TryGetData(cursor, data));
  EXPECT_THAT(data, 2);
  EXPECT_FALSE(trie.TryTraverseOneStep(cursor, 'c'));
}

TEST(DartsCloneTrieTest, CreateCursorAndTryTraverseSeveralSteps) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  cursor = trie.CreateTraversalCursor(trie.kRootNodeId);  // Create a cursor to
                                                          // point to the root.
  EXPECT_TRUE(trie.TryTraverseSeveralSteps(cursor, "def"));
  EXPECT_TRUE(trie.TryGetData(cursor, data));
  EXPECT_THAT(data, 0);
}

TEST(DartsCloneTrieTest, TraversePathNotExisted) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array));

  DartsCloneTrieWrapper::TraversalCursor cursor;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, "dez"));
}

TEST(DartsCloneTrieTest, TraverseOnUtf8Path) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_TRUE(trie.TryTraverseSeveralSteps(cursor, "\xe1\xb8\x8aZZ"));
  EXPECT_TRUE(trie.TryGetData(cursor, data));
  EXPECT_THAT(data, 1);
}

TEST(DartsCloneTrieTest, TraverseOnPartialUtf8Path) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_TRUE(trie.TryTraverseSeveralSteps(cursor, "\xe1\xb8"));
  EXPECT_FALSE(trie.TryGetData(cursor, data));
}

TEST(DartsCloneTrieTest, TraverseOnUtf8PathNotExisted) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array));

  DartsCloneTrieWrapper::TraversalCursor cursor;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, "\xe1\xb8\x84"));
}

TEST(DartsCloneTrieBuildError, KeysValuesSizeDifferent) {
  // The test vocabulary.
  std::vector<std::string> keys{"def", "\xe1\xb8\x8aZZ", "Abc"};
  std::vector<int> values{1, 2, 3, 4};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(keys, values),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST(DartsCloneTrieBuildError, DuplicatedKeys) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc", "def"};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(vocab_tokens),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST(DartsCloneTrieBuildError, EmptyStringsInKeys) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc", ""};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(vocab_tokens),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST(DartsCloneTrieBuildError, NegativeValues) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};
  std::vector<int> vocab_values{0, -1, 1};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(vocab_tokens, vocab_values),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST(DartsCloneTrieTest, OutOfBoundsTraverseOneStepRejected) {
  // A malicious 1-element trie formatted to yield a large internal offset.
  // 0x4E2000 is 5120000; right-shifted by 10 yields offset 5000.
  // When traversing with 'a' (97), next_node_id = 0 ^ 5000 ^ 97 = 4905,
  // which is far beyond the 1-element vector.
  std::vector<uint32_t> malicious_trie = {0x4E2000};
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(malicious_trie));

  EXPECT_EQ(trie.size(), 1);
  auto cursor = trie.CreateTraversalCursorPointToRoot();
  EXPECT_FALSE(trie.TryTraverseOneStep(cursor, 'a'));
  // Cursor should not have changed.
  EXPECT_EQ(cursor.node_id, DartsCloneTrieWrapper::kRootNodeId);
}

TEST(DartsCloneTrieTest, OutOfBoundsTraverseSeveralStepsRejected) {
  std::vector<uint32_t> malicious_trie = {0x4E2000};
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(malicious_trie));

  auto cursor = trie.CreateTraversalCursorPointToRoot();
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, "abc"));
  EXPECT_EQ(cursor.node_id, DartsCloneTrieWrapper::kRootNodeId);
}

TEST(DartsCloneTrieTest, OutOfBoundsGetDataRejected) {
  // 0x4E2100: offset 5000 and has_leaf bit 0x100 set.
  std::vector<uint32_t> malicious_trie = {0x4E2100};
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(malicious_trie));

  auto cursor = trie.CreateTraversalCursorPointToRoot();
  int data = 0;
  EXPECT_FALSE(trie.TryGetData(cursor, data));
}

TEST(DartsCloneTrieTest, CreateWithInvalidSizeOrNullFails) {
  std::vector<uint32_t> empty_trie;
  EXPECT_THAT(DartsCloneTrieWrapper::Create(empty_trie),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      DartsCloneTrieWrapper::Create(absl::Span<const uint32_t>(nullptr, 10)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  uint32_t dummy = 0;
  EXPECT_THAT(
      DartsCloneTrieWrapper::Create(absl::Span<const uint32_t>(&dummy, 0)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(DartsCloneTrieWrapper::Create(absl::Span<const uint32_t>()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(DartsCloneTrieTest, CursorOutOfBoundsSafe) {
  std::vector<uint32_t> malicious_trie = {0x4E2000};
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(malicious_trie));

  auto cursor = trie.CreateTraversalCursor(100);
  EXPECT_EQ(cursor.node_id, 100);
  EXPECT_EQ(cursor.unit, 0);

  trie.SetTraversalCursor(cursor, 200);
  EXPECT_EQ(cursor.node_id, 200);
  EXPECT_EQ(cursor.unit, 0);

  EXPECT_FALSE(trie.TryTraverseOneStep(cursor, 'a'));
}

TEST(DartsCloneTrieTest, InvalidCursorCannotTraverseEvenIfNextIdLandsInBounds) {
  // A 2-element trie where element 0 has label 100.
  // If cursor has node_id = 100 (out of bounds for size 2) and unit = 0:
  // 100 ^ offset(0) ^ 100 = 0 (which is in bounds < 2, and label matches 100).
  // Traversal must be rejected because the source cursor is out of bounds.
  std::vector<uint32_t> trie_data = {100, 0};
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_data));

  auto cursor = trie.CreateTraversalCursor(100);
  EXPECT_EQ(cursor.node_id, 100);
  EXPECT_FALSE(trie.TryTraverseOneStep(cursor, 100));
  // Cursor must not have been modified.
  EXPECT_EQ(cursor.node_id, 100);

  // Several steps with empty path on invalid cursor must also be rejected.
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, ""));
  EXPECT_EQ(cursor.node_id, 100);

  // Several steps with non-empty path on invalid cursor must be rejected.
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, "d"));
  EXPECT_EQ(cursor.node_id, 100);
}

TEST(DartsCloneTrieTest, InvalidCursorGetDataRejectedEvenIfLeafOffsetInBounds) {
  // A 2-element trie. Element 0 is a leaf with value 42 (0x80000000 | 42).
  std::vector<uint32_t> trie_data = {0x8000002A, 0};
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_data));

  // Craft a cursor with out-of-bounds node_id = 100, but has_leaf bit (0x100)
  // and offset = 100 ((100 >> 0) << 10 = 100 << 10 = 0x19000).
  // value_node_id = 100 ^ offset(unit) = 100 ^ 100 = 0 (< 2).
  DartsCloneTrieWrapper::TraversalCursor cursor;
  cursor.node_id = 100;
  cursor.unit = 0x100 | (100 << 10);
  int data = 0;
  EXPECT_FALSE(trie.TryGetData(cursor, data));
}

TEST(DartsCloneTrieTest,
     SeveralStepsPartialMatchFailureLeavesCursorUnmodified) {
  std::vector<std::string> vocab_tokens{"abc", "def"};
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array));

  auto cursor = trie.CreateTraversalCursorPointToRoot();
  EXPECT_TRUE(trie.TryTraverseOneStep(cursor, 'a'));
  const uint32_t original_node_id = cursor.node_id;

  // "bz" matches 'b' but fails at 'z'.
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, "bz"));
  // Cursor should still point to 'a', not moved to 'b'.
  EXPECT_EQ(cursor.node_id, original_node_id);
}

}  // namespace trie_utils
}  // namespace text
}  // namespace tensorflow
