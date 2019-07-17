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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_text/core/kernels/text_kernels_test_util.h"

namespace tensorflow {
namespace text {

using tensorflow::FakeInput;
using tensorflow::NodeDefBuilder;
using tensorflow::Status;
using tensorflow::TensorShape;
using tensorflow::text_kernels_test_util::VectorEq;

class NgramKernelTest : public tensorflow::OpsTestBase {
 public:
  void MakeOp(string separator, int ngram_width, string left_pad,
              string right_pad, bool use_pad, bool extend_pad) {
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "InternalStringNGrams")
                     .Attr("separator", separator)
                     .Attr("ngram_width", ngram_width)
                     .Attr("left_pad", left_pad)
                     .Attr("right_pad", right_pad)
                     .Attr("use_pad", use_pad)
                     .Attr("extend_pad", extend_pad)
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(NgramKernelTest, TestPaddedTrigrams) {
  MakeOp("|", 3, "LP", "RP", true, true);
  // Batch items are:
  // 0: "a", "b", "c", "d"
  // 1: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({"LP|LP|a", "LP|a|b", "a|b|c", "b|c|d",
                                       "c|d|RP", "d|RP|RP", "LP|LP|e", "LP|e|f",
                                       "e|f|RP", "f|RP|RP"});
  std::vector<int64> expected_splits({0, 6, 10});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestPaddedBigrams) {
  MakeOp("|", 2, "LP", "RP", true, true);
  // Batch items are:
  // 0: "a", "b", "c", "d"
  // 1: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values(
      {"LP|a", "a|b", "b|c", "c|d", "d|RP", "LP|e", "e|f", "f|RP"});
  std::vector<int64> expected_splits({0, 5, 8});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestOverlappingPaddedNGrams) {
  MakeOp("|", 3, "LP", "RP", true, true);
  // Batch items are:
  // 0: "a"
  // 1: "b", "c", "d"
  // 2: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({4}), {0, 1, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values(
      {"LP|LP|a", "LP|a|RP", "a|RP|RP",                    // ngrams for elem. 0
       "LP|LP|b", "LP|b|c", "b|c|d", "c|d|RP", "d|RP|RP",  // ngrams for elem. 1
       "LP|LP|e", "LP|e|f", "e|f|RP", "f|RP|RP"});         // ngrams for elem. 2
  std::vector<int64> expected_splits({0, 3, 8, 12});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestOverlappingPaddedMultiCharNGrams) {
  MakeOp("|", 3, "LP", "RP", true, true);
  // Batch items are:
  // 0: "a"
  // 1: "b", "c", "d"
  // 2: "e", "f"
  AddInputFromArray<string>(TensorShape({6}),
                            {"aa", "bb", "cc", "dd", "ee", "ff"});
  AddInputFromArray<int64>(TensorShape({4}), {0, 1, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values(
      {"LP|LP|aa", "LP|aa|RP", "aa|RP|RP",                          //
       "LP|LP|bb", "LP|bb|cc", "bb|cc|dd", "cc|dd|RP", "dd|RP|RP",  //
       "LP|LP|ee", "LP|ee|ff", "ee|ff|RP", "ff|RP|RP"});            //
  std::vector<int64> expected_splits({0, 3, 8, 12});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestMultiOverlappingPaddedNGrams) {
  MakeOp("|", 5, "LP", "RP", true, true);
  // Batch items are:
  // 0: "a"
  AddInputFromArray<string>(TensorShape({1}), {"a"});
  AddInputFromArray<int64>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({"LP|LP|LP|LP|a", "LP|LP|LP|a|RP",
                                       "LP|LP|a|RP|RP", "LP|a|RP|RP|RP",
                                       "a|RP|RP|RP|RP"});
  std::vector<int64> expected_splits({0, 5});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestUnpaddedTrigrams) {
  MakeOp("|", 3, "", "", false, false);
  // Batch items are:
  // 0: "a", "b", "c", "d"
  // 1: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({"a|b|c", "b|c|d"});
  std::vector<int64> expected_splits({0, 2, 2});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestUnpaddedBigrams) {
  MakeOp("|", 2, "", "", false, false);
  // Batch items are:
  // 0: "a", "b", "c", "d"
  // 1: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({"a|b", "b|c", "c|d", "e|f"});
  std::vector<int64> expected_splits({0, 3, 4});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestOverlappingUnpaddedNGrams) {
  MakeOp("|", 3, "", "", false, false);
  // Batch items are:
  // 0: "a"
  // 1: "b", "c", "d"
  // 2: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({4}), {0, 1, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({"b|c|d"});
  std::vector<int64> expected_splits({0, 0, 1, 1});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestOverlappingUnpaddedNGramsNoOutput) {
  MakeOp("|", 5, "", "", false, false);
  // Batch items are:
  // 0: "a"
  // 1: "b", "c", "d"
  // 2: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({4}), {0, 1, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({});
  std::vector<int64> expected_splits({0, 0, 0, 0});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestSinglyPaddedTrigrams) {
  MakeOp("|", 3, "LP", "RP", true, false);
  // Batch items are:
  // 0: "a", "b", "c", "d"
  // 1: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({"LP|a|b", "a|b|c", "b|c|d", "c|d|RP",  //
                                       "LP|e|f", "e|f|RP"});
  std::vector<int64> expected_splits({0, 4, 6});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestSinglyPaddedBigrams) {
  MakeOp("|", 2, "LP", "RP", true, false);
  // Batch items are:
  // 0: "a", "b", "c", "d"
  // 1: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values(
      {"LP|a", "a|b", "b|c", "c|d", "d|RP", "LP|e", "e|f", "f|RP"});
  std::vector<int64> expected_splits({0, 5, 8});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestOverlappingSinglyPaddedNGrams) {
  MakeOp("|", 3, "LP", "RP", true, false);
  // Batch items are:
  // 0: "a"
  // 1: "b", "c", "d"
  // 2: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({4}), {0, 1, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values(
      {"LP|a|RP",                    // ngrams for elem. 0
       "LP|b|c", "b|c|d", "c|d|RP",  // ngrams for elem. 1
       "LP|e|f", "e|f|RP"});         // ngrams for elem. 2
  std::vector<int64> expected_splits({0, 1, 4, 6});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}

TEST_F(NgramKernelTest, TestOverlappingSinglyPaddedNGramsNoOutput) {
  MakeOp("|", 5, "LP", "RP", true, false);
  // Batch items are:
  // 0: "a"
  // 1: "b", "c", "d"
  // 2: "e", "f"
  AddInputFromArray<string>(TensorShape({6}), {"a", "b", "c", "d", "e", "f"});
  AddInputFromArray<int64>(TensorShape({4}), {0, 1, 4, 6});
  TF_ASSERT_OK(RunOpKernel());

  std::vector<string> expected_values({"LP|b|c|d|RP"});
  std::vector<int64> expected_splits({0, 0, 1, 1});

  EXPECT_THAT(*GetOutput(0), VectorEq(expected_values));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_splits));
}
}  // namespace text
}  // namespace tensorflow
