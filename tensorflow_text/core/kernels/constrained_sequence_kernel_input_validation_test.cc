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

#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_text/core/kernels/kernel_test_base.h"
#include "tensorflow_text/core/kernels/text_kernels_test_util.h"

namespace tensorflow {

using tensorflow::DT_INT32;
using tensorflow::FakeInput;
using tensorflow::NodeDefBuilder;
using tensorflow::TensorShape;
using tensorflow::text_kernels_test_util::VectorEq;

class ConstrainedSequenceInputValidationTest
    : public tensorflow::text::KernelTestBase {
 public:
  void SetUpOp(bool use_viterbi, bool use_log_space, bool use_start_end,
               tensorflow::DataType input_datatype,
               tensorflow::DataType splits_datatype = DT_INT64) {
    // Prepare graph.
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "ConstrainedSequence")
                     .Attr("Tin", input_datatype)
                     .Attr("Tsplits", splits_datatype)
                     .Attr("use_viterbi", use_viterbi)
                     .Attr("use_log_space", use_log_space)
                     .Attr("use_start_and_end_states", use_start_end)
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void SetUpOpWithDefaults(bool use_start_end,
                           tensorflow::DataType input_datatype) {
    SetUpOp(/*use_viterbi=*/true, /*use_log_space=*/true, use_start_end,
            input_datatype);
  }

  void SetUpOpWithStartEnd() { SetUpOpWithDefaults(true, DT_INT32); }

  void SetUpOpWithNoStartEnd() { SetUpOpWithDefaults(false, DT_INT32); }
};
// TODO(b/122968457): There are a bunch of tests that only validate !ok instead
// of looking for specific error messages; fix that.

// This test examines evaluations with only a permissions matrix.
TEST_F(ConstrainedSequenceInputValidationTest, WorksWithInt64InputLengths) {
  // Prepare graph.
  SetUpOpWithDefaults(true, DT_INT64);
  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  std::vector<int64_t> input_lengths({1, 1, 1});
  AddInputFromArray<int64_t>(TensorShape({3}), input_lengths);

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO OUT
                              true, true, true,  true, true,   // FROM 0
                              true, true, true,  true, true,   // FROM 1
                              true, true, true,  true, true,   // FROM 2
                              true, true, true,  true, true,   // FROM 3
                              true, true, false, true, false,  // FROM 'OUTSIDE'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  // The first sequence's highest score is 2, but OUT->2 is not ok, so it's 1.
  // The second sequence's highest score is 3, which is ok.
  // The third sequence's highest score is 0, which is ok.

  // Validate the output.
  std::vector<int32_t> expected_transitions({1, 3, 0});
  std::vector<int64_t> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizePermissionMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({4, 5}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizePermissionMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 4}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnWrongRankPermissionMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({25}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizeWeightMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({4, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizeWeightMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 4}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest, FailsOnWrongRankWeightMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({25}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.1, 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       PassesWithCorrectSizedWeightAndPermissionsMatrix) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({4, 4}), {
                                                   true, true, true, true,  //
                                                   true, true, true, true,  //
                                                   true, true, true, true,  //
                                                   true, true, true, true,  //
                                               });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({4, 4}), {0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_TRUE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizePermissionMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({4, 5}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizePermissionMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 4}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnWrongRankPermissionMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({16}), {
                                                 true, true, true, true,  //
                                                 true, true, true, true,  //
                                                 true, true, true, true,  //
                                                 true, true, true, true,  //
                                             });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizeWeightMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({4, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizeWeightMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 4}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnWrongRankWeightMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();
  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({16}), {0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnNegativeInt64SequenceLengthGreedy) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT64);
  AddInputFromArray<float>(TensorShape({3, 5, 2}), std::vector<float>(30, 0.0));
  AddInputFromArray<int64_t>(TensorShape({3}), {1, -5, 5});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnNegativeInt64SequenceLengthViterbi) {
  SetUpOp(/*use_viterbi=*/true, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT64);
  AddInputFromArray<float>(TensorShape({3, 5, 2}), std::vector<float>(30, 0.0));
  AddInputFromArray<int64_t>(TensorShape({3}), {1, -5, 5});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnNegativeInt32SequenceLength) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT32, DT_INT32);
  AddInputFromArray<float>(TensorShape({3, 2, 2}), std::vector<float>(12, 0.0));
  AddInputFromArray<int32_t>(TensorShape({3}), {-1, 1, 1});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInt64MinSequenceLength) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT64);
  AddInputFromArray<float>(TensorShape({3, 1, 2}), std::vector<float>(6, 0.0));
  AddInputFromArray<int64_t>(TensorShape({3}),
                             {std::numeric_limits<int64_t>::min(), 0, 0});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInt64SequenceLengthExceedingNumStepsWith32BitTruncation) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT64);
  AddInputFromArray<float>(TensorShape({1, 2, 2}), std::vector<float>(4, 0.0));
  AddInputFromArray<int64_t>(TensorShape({1}), {int64_t{1} << 32});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnTotalLengthOverflowingInt32Splits) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT32);
  AddInputFromArray<float>(TensorShape({3, 1073741824, 0}), {});
  AddInputFromArray<int64_t>(TensorShape({3}), {1073741824, 1073741823, 1});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnTotalLengthWrappingInt32ToZero) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT32);
  AddInputFromArray<float>(TensorShape({4, 1073741824, 0}), {});
  AddInputFromArray<int64_t>(TensorShape({4}),
                             {1073741824, 1073741824, 1073741824, 1073741824});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       WorksWithNumStepsExceedingInt32Max) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT64);
  constexpr int64_t kLargeNumSteps = (int64_t{1} << 31);
  AddInputFromArray<float>(TensorShape({1, kLargeNumSteps, 0}), {});
  AddInputFromArray<int64_t>(TensorShape({1}), {2});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  std::vector<int32_t> expected_transitions({-1, -1});
  std::vector<int64_t> expected_offsets({0, 2});
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsWhenLengthExceedsLargeNumSteps) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT64);
  constexpr int64_t kLargeNumSteps = (int64_t{1} << 31);
  AddInputFromArray<float>(TensorShape({1, kLargeNumSteps, 0}), {});
  AddInputFromArray<int64_t>(TensorShape({1}), {kLargeNumSteps + 1});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnNumScoresAtInt32Max) {
  SetUpOp(/*use_viterbi=*/true, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT32, DT_INT64);
  AddInputFromArray<float>(
      TensorShape({1, 0, std::numeric_limits<int32_t>::max()}), {});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ConstrainedSequenceInputValidationTest, WorksWithZeroScoresExpViterbi) {
  SetUpOp(/*use_viterbi=*/true, /*use_log_space=*/false,
          /*use_start_end=*/false, DT_INT32, DT_INT64);
  AddInputFromArray<float>(TensorShape({1, 2, 0}), {});
  AddInputFromArray<int32_t>(TensorShape({1}), {2});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  std::vector<int32_t> expected_transitions({-1, -1});
  std::vector<int64_t> expected_offsets({0, 2});
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

TEST_F(ConstrainedSequenceInputValidationTest,
       WorksWithInt32SplitsAndEmptyMiddleRowGreedy) {
  SetUpOp(/*use_viterbi=*/false, /*use_log_space=*/true,
          /*use_start_end=*/false, DT_INT64, DT_INT32);
  AddInputFromArray<float>(TensorShape({3, 2, 3}),
                           {{
                               1.0, 5.0, 2.0,  // batch 0, step 0 -> 1
                               4.0, 1.0, 9.0,  // batch 0, step 1 -> 2
                               9.0, 9.0, 9.0,  // batch 1 (empty, ignored)
                               9.0, 9.0, 9.0,  // batch 1 (empty, ignored)
                               8.0, 3.0, 1.0,  // batch 2, step 0 -> 0
                               1.0, 9.0, 1.0,  // batch 2, step 1 (ignored)
                           }});
  AddInputFromArray<int64_t>(TensorShape({3}), {2, 0, 1});
  AddInputFromArray<bool>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  std::vector<int32_t> expected_transitions({1, 2, 0});
  std::vector<int32_t> expected_offsets({0, 2, 2, 3});
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

}  // namespace tensorflow
