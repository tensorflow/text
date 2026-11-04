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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TEST_STATUS_MACROS_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TEST_STATUS_MACROS_H_

#include <utility>  // IWYU pragma: keep

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "absl/status/status.h"  // IWYU pragma: keep
#include "absl/status/statusor.h"  // IWYU pragma: keep

// google3's gmock.h already defines ASSERT_OK_AND_ASSIGN. OSS GoogleTest
// and Abseil do not, so provide a minimal fallback for OSS builds only.
// Unlike the google3 macro, `lhs` must not contain unparenthesized commas.
#ifndef ASSERT_OK_AND_ASSIGN
#define TFTEXT_CONCAT_INNER_(x, y) x##y
#define TFTEXT_CONCAT_(x, y) TFTEXT_CONCAT_INNER_(x, y)
#define TFTEXT_ASSERT_OK_AND_ASSIGN_IMPL_(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                      \
  ASSERT_TRUE(statusor.ok()) << statusor.status();              \
  lhs = std::move(statusor).value()
#define ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  TFTEXT_ASSERT_OK_AND_ASSIGN_IMPL_(     \
      TFTEXT_CONCAT_(status_or_, __LINE__), lhs, rexpr)
#endif  // ASSERT_OK_AND_ASSIGN

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TEST_STATUS_MACROS_H_
