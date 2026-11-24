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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_CONSTRAINED_SEQUENCE_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_CONSTRAINED_SEQUENCE_H_

#include <cstdint>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace text {

class ScoreAccessor {
 public:
  explicit ScoreAccessor(const Tensor &score_tensor,
                         const Tensor &lengths_tensor);

  // Get a score out of the data tensor.
  float GetScore(int64_t batch_idx, int64_t step_idx, int64_t score_idx) const;

  int64_t GetLength(int64_t batch_idx) const;

  int64_t batch_size() const;
  int64_t num_steps() const;
  int64_t num_scores() const;
  bool has_explicit_batch() const;

 private:
  // A pointer into the underlying data of the score tensor. Not owned.
  const float *data_;

  // A pointer into the underlying data of the lengths tensor. Not owned.
  const int32_t *lengths_ = nullptr;
  const int64_t *long_lengths_ = nullptr;

  // Whether the passed lengths tensor is int32 or int64.
  bool use_long_lengths_;

  // The batch size associated with the data tensor.
  int64_t batch_size_;

  // The number of steps in the data tensor.
  int64_t num_steps_;

  // The number of scores in the data tensor.
  int64_t num_scores_;

  // The amount to increase the offset within the flat data array if the batch
  // index increases by 1.
  int64_t batch_offset_;

  // The amount to increase the offset within the flat data array if the step
  // index increases by 1.
  int64_t step_offset_;

  // True if the original tensor had an explicit batch dimension (that is,
  // it was of rank 3).
  bool has_explicit_batch_;
};

// Perform Viterbi analysis on a single batch item.
void ViterbiAnalysis(
    const ScoreAccessor &scores,
    const tensorflow::TTypes<const float>::Matrix &transition_weights,
    const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
    int64_t batch, bool use_log_space, bool use_start_end_states,
    int32_t *output_data);

// Perform a greedy analysis on a single batch item.
void GreedyAnalysis(
    const ScoreAccessor &scores,
    const tensorflow::TTypes<const float>::Matrix &transition_weights,
    const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
    int64_t batch, bool use_log_space, bool use_start_end_states,
    int32_t *output_data);

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_CONSTRAINED_SEQUENCE_H_
