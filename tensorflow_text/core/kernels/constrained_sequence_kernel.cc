// Copyright 2020 TF.Text Authors.
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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using ::tensorflow::DataType;
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::DT_BOOL;
using ::tensorflow::DT_FLOAT;
using ::tensorflow::DT_INT32;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::InvalidArgument;

// State index to use if the sequence in question requires an impossible
// transition.
constexpr int kErrorState = -1;

// State index to use when outputting a padded tensor and the sequence in
// question does not have a token for a given step.
constexpr int kPaddingState = -2;

namespace {

// Validate that a given constraint tensor is the proper shape (dimension
// 2, with shape [num_states + 1, num_states + 1].
tensorflow::Status ValidateConstraintTensor(const Tensor &tensor,
                                            const int num_states,
                                            const bool use_start_end_states,
                                            const string &name) {
  if (tensor.shape().dims() != 2) {
    return InvalidArgument(
        tensorflow::strings::StrCat(name, " must be of rank 2"));
  }
  int expected_size = use_start_end_states ? num_states + 1 : num_states;
  if (tensor.shape().dim_size(0) != expected_size) {
    return InvalidArgument(tensorflow::strings::StrCat(
        name, " must have a zeroth dimension of size ", expected_size,
        " when num_states is ", num_states, " and use_start_and_end_states is ",
        use_start_end_states));
  }
  if (tensor.shape().dim_size(1) != expected_size) {
    return InvalidArgument(tensorflow::strings::StrCat(
        name, " must have a first dimension of size ", expected_size,
        " when num_states is ", num_states, " and use_start_and_end_states is ",
        use_start_end_states));
  }
  return tensorflow::Status::OK();
}

// Helper class to handle cases where the score tensor has rank of 2 or 3.
class ScoreAccessor {
 public:
  explicit ScoreAccessor(const Tensor &score_tensor,
                         const Tensor &lengths_tensor) {
    data_ = score_tensor.flat<float>().data();
    if (lengths_tensor.dtype() == DT_INT64) {
      use_long_lengths_ = true;
      long_lengths_ = lengths_tensor.flat<int64>().data();
    } else {
      use_long_lengths_ = false;
      lengths_ = lengths_tensor.flat<int>().data();
    }
    has_explicit_batch_ = (score_tensor.shape().dims() == 3);
    if (has_explicit_batch_) {
      batch_size_ = score_tensor.shape().dim_size(0);
      num_steps_ = score_tensor.shape().dim_size(1);
      num_scores_ = score_tensor.shape().dim_size(2);
    } else {
      batch_size_ = 1;
      num_steps_ = score_tensor.shape().dim_size(0);
      num_scores_ = score_tensor.shape().dim_size(1);
    }
    batch_offset_ = num_scores_ * num_steps_;
    step_offset_ = num_scores_;
  }

  // Get a score out of the data tensor.
  float GetScore(int batch_idx, int step_idx, int score_idx) const {
    DCHECK_LE(batch_idx, batch_size_);
    DCHECK_LE(step_idx, num_steps_);
    DCHECK_LE(score_idx, num_scores_);
    return data_[batch_offset_ * batch_idx + step_offset_ * step_idx +
                 score_idx];
  }

  int64 GetLength(int batch_idx) const {
    DCHECK_LE(batch_idx, batch_size_);
    if (use_long_lengths_) {
      return long_lengths_[batch_idx];
    } else {
      return lengths_[batch_idx];
    }
  }

  int batch_size() const { return batch_size_; }
  int num_steps() const { return num_steps_; }
  int num_scores() const { return num_scores_; }
  bool has_explicit_batch() const { return has_explicit_batch_; }

 private:
  // A pointer into the underlying data of the score tensor. Not owned.
  const float *data_;

  // A pointer into the underlying data of the lengths tensor. Not owned.
  const int *lengths_;
  const int64 *long_lengths_;

  // Whether the passed lengths tensor is int32 or int64.
  bool use_long_lengths_;

  // The batch size associated with the data tensor.
  int batch_size_;

  // The number of steps in the data tensor.
  int num_steps_;

  // The number of scores in the data tensor.
  int num_scores_;

  // The amount to increase the offset within the flat data array if the batch
  // index increases by 1.
  int batch_offset_;

  // The amount to increase the offset within the flat data array if the step
  // index increases by 1.
  int step_offset_;

  // True if the original tensor had an explicit batch dimension (that is,
  // it was of rank 3).
  bool has_explicit_batch_;
};

}  // namespace

template <typename Tin, typename Tsplits>
class ConstrainedSequence : public OpKernel {
 public:
  explicit ConstrainedSequence(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_viterbi", &use_viterbi_));
    OP_REQUIRES_OK(context, context->GetAttr("use_log_space", &use_log_space_));
    OP_REQUIRES_OK(context, context->GetAttr("use_start_and_end_states",
                                             &use_start_end_states_));
  }

  void Compute(OpKernelContext *context) override {
    const auto &score_tensor = context->input(0);
    OP_REQUIRES(context,
                (score_tensor.shape().dims() == 2) ||
                    (score_tensor.shape().dims() == 3),
                InvalidArgument("The score tensor must be of rank 2 or 3."));
    const auto &lengths_tensor = context->input(1);

    ScoreAccessor scores(score_tensor, lengths_tensor);

    // The scores tensor should be [batch, step, scores].
    const int batch_size = scores.batch_size();
    const int num_steps = scores.num_steps();
    const int num_scores = scores.num_scores();

    OP_REQUIRES(context, lengths_tensor.NumElements() == batch_size,
                InvalidArgument(tensorflow::strings::StrCat(
                    "There should be exactly one length for every batch "
                    "element. Found ",
                    lengths_tensor.NumElements(),
                    " length elements for a batch size of ", batch_size)));

    VLOG(2) << "batch: " << batch_size;
    VLOG(2) << "steps: " << num_steps;
    VLOG(2) << "score: " << num_scores;

    // Make sure there's enough data to advance every sequence.
    int max_length = 0;
    int total_length = 0;
    for (int i = 0; i < batch_size; ++i) {
      int64 length = scores.GetLength(i);
      total_length += length;
      if (length > max_length) {
        max_length = length;
      }
    }

    OP_REQUIRES(
        context, num_steps >= max_length,
        InvalidArgument(
            "The scores tensor is too short for the longest sequence length."));

    // Validate the constraint tensors.
    const auto &allowed_transitions_tensor = context->input(2);
    bool has_allowed_transitions =
        allowed_transitions_tensor.NumElements() != 0;
    VLOG(4) << allowed_transitions_tensor.NumElements();
    if (has_allowed_transitions) {
      OP_REQUIRES_OK(context,
                     ValidateConstraintTensor(allowed_transitions_tensor,
                                              num_scores, use_start_end_states_,
                                              "allowed_transitions"));
    }

    const auto &transition_weights_tensor = context->input(3);

    VLOG(4) << transition_weights_tensor.NumElements();
    bool has_transition_weights = transition_weights_tensor.NumElements() != 0;
    if (has_transition_weights) {
      OP_REQUIRES_OK(context, ValidateConstraintTensor(
                                  transition_weights_tensor, num_scores,
                                  use_start_end_states_, "transition_weights"));

      // If we have transition weights in exp-space, all values must be non-
      // negative.
      if (!use_log_space_) {
        for (int i = 0; i < transition_weights_tensor.NumElements(); ++i) {
          OP_REQUIRES(context, transition_weights_tensor.flat<float>()(i) >= 0,
                      InvalidArgument("The transition weights tensor must not "
                                      "contain negative values."));
        }
      }
    }

    const tensorflow::Tensor empty_float(DT_FLOAT, TensorShape({0, 0}));
    const tensorflow::Tensor empty_bool(DT_BOOL, TensorShape({0, 0}));

    const auto &transition_weights =
        has_transition_weights ? transition_weights_tensor.matrix<float>()
                               : empty_float.matrix<float>();

    const auto &allowed_transitions =
        has_allowed_transitions ? allowed_transitions_tensor.matrix<bool>()
                                : empty_bool.matrix<bool>();

    Tensor *output;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({total_length}), &output));
    int32 *output_data = output->flat<int32>().data();

    Tensor *offsets;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({batch_size + 1}), &offsets));
    Tsplits *offset_data = offsets->flat<Tsplits>().data();
    offset_data[0] = 0;

    for (int batch = 0; batch < batch_size; ++batch) {
      int step_offset = offset_data[batch];
      int64 num_steps = scores.GetLength(batch);
      offset_data[batch + 1] = step_offset + num_steps;
      if (use_viterbi_) {
        DoViterbiAnalysis(transition_weights, allowed_transitions, batch,
                          scores, &output_data[step_offset]);
      } else {
        DoGreedyAnalysis(transition_weights, allowed_transitions, batch, scores,
                         &output_data[step_offset]);
      }
    }
  }

 private:
  // Perform Viterbi analysis on a single batch item.
  void DoViterbiAnalysis(
      const tensorflow::TTypes<const float>::Matrix &transition_weights,
      const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
      const int batch, const ScoreAccessor &scores, int32 *output_data) {
    VLOG(2) << "Analyzing batch " << batch;
    const bool has_transition_weights = transition_weights.size() != 0;
    const bool has_allowed_transitions = allowed_transitions.size() != 0;
    const int num_states = scores.num_scores();
    const int out_of_bounds_index = num_states;

    int64 num_steps = scores.GetLength(batch);

    // Create two vectors to hold scores. These will be bound to referents later
    // so the names here are somewhat irrelevant.
    std::vector<double> scores_a(num_states,
                                 std::numeric_limits<float>::lowest());
    std::vector<double> scores_b(num_states,
                                 std::numeric_limits<float>::lowest());

    // Create a chart of backpointers. Include rows for [start] and [end]
    // transitions. By initializing this to kErrorState, we ensure unreachable
    // transitions get marked as errors.
    std::vector<std::vector<int>> backpointers(
        num_steps, std::vector<int>(num_states, kErrorState));

    // Set current and previous references for step 0
    std::vector<double> *previous_scores = &scores_a;
    std::vector<double> *current_scores = &scores_b;

    for (int curr_state = 0; curr_state < num_states; ++curr_state) {
      std::vector<int> &current_bps = backpointers[0];
      if (use_start_end_states_) {
        // Initialize the zeroth step BPs to kOutOfBoundsIndex for all states
        // where the OOB->state transition is valid, and set scores as needed.
        if (!has_allowed_transitions ||
            allowed_transitions(out_of_bounds_index, curr_state)) {
          // Because the backpointer vectors are initialized to kErrorState, we
          // need only to set the valid transition paths to have come from the
          // padding state.
          current_bps[curr_state] = out_of_bounds_index;

          // For valid transitions, get the score (and adjust as appropriate).
          const int step = 0;
          float current_score = scores.GetScore(batch, step, curr_state);
          if (has_transition_weights) {
            if (use_log_space_) {
              current_score +=
                  transition_weights(out_of_bounds_index, curr_state);
            } else {
              current_score *=
                  transition_weights(out_of_bounds_index, curr_state);
            }
          }
          current_scores->at(curr_state) = current_score;
        }
      } else {
        // If we don't have specific start and end states, all bp's are valid
        // and all starting scores are the unadjusted step 0 scores.
        current_bps[curr_state] = out_of_bounds_index;
        const int step = 0;
        current_scores->at(curr_state) =
            scores.GetScore(batch, step, curr_state);
      }
    }

    // Update the current scores (and normalize if we're not in log space).
    if (!use_log_space_) {
      const double max_score =
          *std::max_element(current_scores->begin(), current_scores->end());
      if (max_score > 0) {
        for (double &score : *current_scores) score /= max_score;
      }
    }
    // Swap current and previous score arrays, as we are advancing a step.
    std::vector<double> *tmp = previous_scores;
    previous_scores = current_scores;
    current_scores = tmp;

    // Handle all steps save for the first and last in this loop.
    for (int step = 1; step < num_steps; ++step) {
      const std::vector<int> &previous_bps = backpointers[step - 1];
      std::vector<int> &current_bps = backpointers[step];

      for (int curr_state = 0; curr_state < num_states; ++curr_state) {
        int best_source_state = kErrorState;
        float best_score = std::numeric_limits<float>::lowest();
        for (int prev_state = 0; prev_state < num_states; ++prev_state) {
          // If the previous state was an error state, pass to the next state.
          if (previous_bps[prev_state] == kErrorState) {
            continue;
          }

          // If this is not a permitted transition, continue.
          if (has_allowed_transitions &&
              !allowed_transitions(prev_state, curr_state)) {
            continue;
          }

          float current_score = scores.GetScore(batch, step, curr_state);
          if (has_transition_weights) {
            if (use_log_space_) {
              current_score += transition_weights(prev_state, curr_state);
              current_score += previous_scores->at(prev_state);
            } else {
              current_score *= transition_weights(prev_state, curr_state);
              current_score *= previous_scores->at(prev_state);
            }
          }

          if (has_transition_weights) {
            VLOG(3) << "Total score (" << batch << ", " << step << ", "
                    << prev_state << "->" << curr_state
                    << "): " << current_score
                    << " (raw: " << scores.GetScore(batch, step, curr_state)
                    << ", tw: " << transition_weights(prev_state, curr_state)
                    << ")";
          } else {
            VLOG(3) << "Total score (" << batch << ", " << step << ", "
                    << prev_state << "->" << curr_state
                    << "): " << current_score
                    << " (raw: " << scores.GetScore(batch, step, curr_state)
                    << ")";
          }

          if (current_score >= best_score) {
            best_source_state = prev_state;
            best_score = current_score;
          }
        }
        current_bps[curr_state] = best_source_state;
        current_scores->at(curr_state) = best_score;
      }

      // Normalize if we're not in log space.
      if (!use_log_space_) {
        const double max_score =
            *std::max_element(current_scores->begin(), current_scores->end());
        if (max_score > 0) {
          for (double &score : *current_scores) score /= max_score;
        }
      }

      // After each step, switch the current scores to the previous scores and
      // use the previous previous scores as the current scores.
      std::vector<double> *tmp = previous_scores;
      previous_scores = current_scores;
      current_scores = tmp;
    }

    // Handle the final transition out of the sequence.
    int final_state = out_of_bounds_index;
    const std::vector<int> &previous_bps = backpointers[num_steps - 1];
    int best_source_state = kErrorState;
    float final_score = std::numeric_limits<float>::lowest();

    for (int prev_state = 0; prev_state < num_states; ++prev_state) {
      // If the previous state was an error state, pass to the next state.
      if (previous_bps[prev_state] == kErrorState) {
        current_scores->at(prev_state) = std::numeric_limits<float>::lowest();
        continue;
      }

      // If this is not a permitted transition, continue.
      if (has_allowed_transitions && use_start_end_states_ &&
          !allowed_transitions(prev_state, final_state)) {
        current_scores->at(prev_state) = std::numeric_limits<float>::lowest();
        continue;
      }

      // Weight the final transition score by the probability of exiting the
      // sequence as well.
      float current_score = previous_scores->at(prev_state);
      if (has_transition_weights && use_start_end_states_) {
        if (use_log_space_) {
          current_score += transition_weights(prev_state, final_state);
        } else {
          current_score *= transition_weights(prev_state, final_state);
        }
      }

      current_scores->at(prev_state) = current_score;
      if (current_score >= final_score) {
        best_source_state = prev_state;
        final_score = current_score;
      }
    }

    VLOG(3) << "Final score: " << final_score;

    // Calculate the path.
    if (best_source_state == kErrorState) {
      // If the best source is an error state, the path is unknowable. Report
      // error states for the whole sequence.
      for (int64 i = 0; i < scores.GetLength(batch); ++i) {
        output_data[i] = kErrorState;
      }
    } else {
      // If the best source is a 'real' state, report the state path.
      int steps_to_report = scores.GetLength(batch);
      int previous_state = best_source_state;
      for (int64 i = steps_to_report - 1; i >= 0; --i) {
        output_data[i] = previous_state;
        previous_state = backpointers[i][previous_state];
      }
    }
  }

  // Perform a greedy analysis on a single batch item.
  void DoGreedyAnalysis(
      const tensorflow::TTypes<const float>::Matrix &transition_weights,
      const tensorflow::TTypes<const bool>::Matrix &allowed_transitions,
      int batch, const ScoreAccessor &scores, int32 *output_data) {
    const bool has_transition_weights = transition_weights.size() != 0;
    const bool has_allowed_transitions = allowed_transitions.size() != 0;
    const int num_states = scores.num_scores();
    const int out_of_bounds_index = num_states;
    int64 num_steps = scores.GetLength(batch);

    for (int step = 0; step < num_steps; ++step) {
      // Do final step calculations if this is the final step in the sequence
      // and we are calculating based on implicit start and end states.
      bool do_final_step =
          (step == scores.GetLength(batch) - 1) && use_start_end_states_;
      VLOG(2) << "is last step: " << do_final_step;

      const int previous_state =
          (step == 0) ? (out_of_bounds_index) : (output_data[step - 1]);

      if (previous_state == kErrorState) {
        // If the previous state is the error state, the current state must
        // also be the error state.
        output_data[step] = kErrorState;
        continue;
      }

      // If no transition is possible, this will stay the error state.
      int best_new_state = kErrorState;
      float best_new_score = std::numeric_limits<float>::lowest();

      for (int state = 0; state < num_states; ++state) {
        float current_score = scores.GetScore(batch, step, state);

        // If we are not using start/end states AND step is 0, then
        // current_score will not be altered.
        if (use_start_end_states_ || step > 0) {
          if (has_allowed_transitions) {
            // If either the transition from the previous state to this state
            // is disallowed, or we need to analyze the final step and the
            // transition from this state to the final step is not allowed,
            // disallow this transition.
            if (!allowed_transitions(previous_state, state) ||
                (do_final_step &&
                 !allowed_transitions(state, out_of_bounds_index))) {
              continue;
            }
          }

          if (has_transition_weights) {
            if (use_log_space_) {
              current_score += transition_weights(previous_state, state);
            } else {
              current_score *= transition_weights(previous_state, state);
            }
            // On the last step, also analyze by the weight value of
            // transitioning from this state to the out-of-bounds state.
            if (do_final_step) {
              if (use_log_space_) {
                current_score += transition_weights(state, out_of_bounds_index);
              } else {
                current_score *= transition_weights(state, out_of_bounds_index);
              }
            }
          }
        }
        if (current_score >= best_new_score) {
          best_new_state = state;
          best_new_score = current_score;
        }
      }
      output_data[step] = best_new_state;
      VLOG(2) << "Best state for step " << step << " is " << output_data[step]
              << " with score " << best_new_score;
    }
  }

  // True if this op should perform calculations in log-space (using addition).
  // If false, will perform calculations in normalized exp-space (using
  // multiplication).
  bool use_log_space_;

  // True if this op should calculate scores using the Viterbi algorithm. If
  // false, will use a greedy algorithm.
  bool use_viterbi_;

  // True if this op should calculate sequences based on an implicit start
  // and end state.
  bool use_start_end_states_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConstrainedSequence);
};

#define REGISTER_KERNELS(Tin)                                    \
  REGISTER_KERNEL_BUILDER(Name("ConstrainedSequence")            \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<Tin>("Tin")        \
                              .TypeConstraint<int32>("Tsplits"), \
                          ConstrainedSequence<Tin, int32>);      \
  REGISTER_KERNEL_BUILDER(Name("ConstrainedSequence")            \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<Tin>("Tin")        \
                              .TypeConstraint<int64>("Tsplits"), \
                          ConstrainedSequence<Tin, int64>)

REGISTER_KERNELS(int32);
REGISTER_KERNELS(int64);

#undef REGISTER_KERNELS

}  // namespace tensorflow
