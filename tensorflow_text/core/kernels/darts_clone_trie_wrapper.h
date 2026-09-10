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

// Access utils for Darts-clone tries.
//
// Darts-clone is a compact and efficient implementation of Darts (Double-ARray
// Trie System). For more info, see https://github.com/s-yata/darts-clone.
//
// This header file contains utils that access a darts-clone trie. To build such
// a darts-clone trie, use the utils from the companion header file
// darts_clone_trie_builder.h.
//
// Note that although there is a 'traverse()' function in the original source
// (see https://github.com/s-yata/darts-clone/blob/master/include/darts.h), the
// utils in this header file are more efficient and the APIs are more flexible.
#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_DARTS_CLONE_TRIE_WRAPPER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_DARTS_CLONE_TRIE_WRAPPER_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tensorflow {
namespace text {
namespace trie_utils {

// A wrapper class of darts_clone trie for traversing and getting data on the
// trie. It does not own the actual 'trie_array'.
class DartsCloneTrieWrapper {
 public:
  // Represents the root node id.
  static constexpr uint32_t kRootNodeId = 0;

  // A struct serving as the trie traversal cursor. It holds 'node_id' and
  // 'unit' (which is 'trie_array_[node_id]'). The reason is to save and reuse
  // the 'trie_array_[node_id]'.
  struct TraversalCursor {
    uint32_t node_id = 0;
    uint32_t unit = 0;
  };

  // Constructs an instance from an absl::Span.
  // The caller needs to make sure that 'trie_array' points to a valid structure
  // returned by darts_clone trie builder. The caller also needs to maintain the
  // availability of 'trie_array' throughout the lifetime of this instance.
  static absl::StatusOr<DartsCloneTrieWrapper> Create(
      absl::Span<const uint32_t> trie_array) {
    if (trie_array.empty()) {
      return absl::InvalidArgumentError("trie_array must not be empty.");
    }
    if (trie_array.data() == nullptr) {
      return absl::InvalidArgumentError("trie_array is nullptr.");
    }
    return DartsCloneTrieWrapper(trie_array.data(), trie_array.size());
  }

  // Creates a cursor pointing to the root.
  TraversalCursor CreateTraversalCursorPointToRoot() const {
    return CreateTraversalCursor(kRootNodeId);
  }

  // Creates a cursor pointing to the 'node_id'.
  TraversalCursor CreateTraversalCursor(uint32_t node_id) const {
    if (node_id >= trie_array_size_) {
      return {node_id, 0};
    }
    return {node_id, trie_array_[node_id]};
  }

  // Sets the cursor to point to 'node_id'.
  void SetTraversalCursor(TraversalCursor& cursor, uint32_t node_id) const {
    cursor.node_id = node_id;
    if (node_id >= trie_array_size_) {
      cursor.unit = 0;
      return;
    }
    cursor.unit = trie_array_[node_id];
  }

  // Traverses one step from 'cursor' following 'ch'. If successful (i.e., there
  // exists such an edge), moves 'cursor' to the new node and returns true.
  // Otherwise, does nothing (i.e., 'cursor' is not changed) and returns false.
  bool TryTraverseOneStep(TraversalCursor& cursor, unsigned char ch) const {
    if (cursor.node_id >= trie_array_size_) {
      return false;
    }
    const uint32_t next_node_id = cursor.node_id ^ offset(cursor.unit) ^ ch;
    if (next_node_id >= trie_array_size_) {
      return false;
    }
    const uint32_t next_node_unit = trie_array_[next_node_id];
    if (label(next_node_unit) != ch) {
      return false;
    }
    cursor.node_id = next_node_id;
    cursor.unit = next_node_unit;
    return true;
  }

  // Traverses several steps from 'cursor' following the characters on 'path'.
  // If *all* steps are successful, moves 'cursor' to the new node and returns
  // true. Otherwise, does nothing (i.e., 'cursor' is not changed) and returns
  // false.
  bool TryTraverseSeveralSteps(TraversalCursor& cursor,
                               absl::string_view path) const {
    if (cursor.node_id >= trie_array_size_) {
      return false;
    }
    uint32_t cur_id = cursor.node_id;
    uint32_t cur_unit = cursor.unit;
    for (const char c : path) {
      const unsigned char ch = static_cast<unsigned char>(c);
      cur_id ^= offset(cur_unit) ^ ch;
      if (cur_id >= trie_array_size_) {
        return false;
      }
      cur_unit = trie_array_[cur_id];
      if (label(cur_unit) != ch) {
        return false;
      }
    }
    cursor.node_id = cur_id;
    cursor.unit = cur_unit;
    return true;
  }

  // If the node pointed by 'cursor' has data, read into 'out_data' and returns
  // true; otherwise, does nothing and returns false.
  bool TryGetData(const TraversalCursor& cursor, int& out_data) const {
    if (cursor.node_id >= trie_array_size_) {
      return false;
    }
    if (!has_leaf(cursor.unit)) {
      return false;
    }
    const uint32_t value_node_id = cursor.node_id ^ offset(cursor.unit);
    if (value_node_id >= trie_array_size_) {
      return false;
    }
    const uint32_t value_unit = trie_array_[value_node_id];
    out_data = value(value_unit);
    return true;
  }

  // Returns the number of elements in the trie array.
  size_t size() const { return trie_array_size_; }

 private:
  // Use Create() instead of the constructor.
  explicit DartsCloneTrieWrapper(const uint32_t* trie_array,
                                 size_t trie_array_size)
      : trie_array_(trie_array), trie_array_size_(trie_array_size) {}

  // The helper functions below are based on
  // https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/custom_ops/kernel/sentencepiece/double_array_trie.h

  // Returns offset to children.
  static uint32_t offset(uint32_t unit) {
    return (unit >> 10) << ((unit & 0x200) >> 6);
  }

  // Returns a label associated with a node.
  // A leaf node will have the MSB set and thus return an invalid label.
  static uint32_t label(uint32_t unit) { return unit & 0x800000ff; }

  // Returns whether a node has a leaf as a child.
  static bool has_leaf(uint32_t unit) { return unit & 0x100; }

  // Returns a value associated with a node. Available when a node is a leaf.
  static int value(uint32_t unit) {
    return static_cast<int>(unit & 0x7fffffff);
  }

  // The pointer to the darts trie array.
  const uint32_t* trie_array_;
  // The number of elements in the trie array.
  size_t trie_array_size_;
};

}  // namespace trie_utils
}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_DARTS_CLONE_TRIE_WRAPPER_H_
