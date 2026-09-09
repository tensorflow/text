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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROW_SPLITS_VALIDATOR_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROW_SPLITS_VALIDATOR_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace tensorflow {
namespace text {

template <typename Tsplits>
inline absl::Status ValidateRowSplits(absl::Span<const Tsplits> row_splits,
                                      int64_t max_values_size = -1) {
  if (row_splits.empty()) {
    return absl::InvalidArgumentError("row_splits cannot be empty.");
  }
  if (row_splits[0] != 0) {
    return absl::InvalidArgumentError("row_splits must start with 0.");
  }
  for (size_t i = 0; i < row_splits.size() - 1; ++i) {
    if (row_splits[i + 1] < row_splits[i]) {
      return absl::InvalidArgumentError(
          "row_splits must be monotonically increasing.");
    }
  }
  if (max_values_size >= 0 && row_splits.back() > max_values_size) {
    return absl::InvalidArgumentError(
        "row_splits values exceed the size of the values array.");
  }
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROW_SPLITS_VALIDATOR_H_
