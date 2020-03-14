<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.greedy_constrained_sequence" />
<meta itemprop="path" content="Stable" />
</div>

# text.greedy_constrained_sequence

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/greedy_constrained_sequence_op.py">View
source</a>

Performs greedy constrained sequence on a batch of examples.

```python
text.greedy_constrained_sequence(
    scores, sequence_length=None, allowed_transitions=None, transition_weights=None,
    use_log_space=False, use_start_and_end_states=False, name=None
)
```

<!-- Placeholder for "Used in" -->

Constrains a set of predictions based on a set of legal transitions and/or a set
of transition weights, returning the legal sequence that maximizes the product
or sum of the state scores and the transition weights at each step. If
`use_log_space` is true, the sum is used; if false, the product is used.

This op also takes a parameter `use_start_and_end_states`, which when true will
add an implicit start and end state to each sequence. These implicit states
allow the user to specify additional weights and permitted transitions to start
and end a sequence (so, for instance, if you wanted to forbid your output from
ending in a certain set of states you could do so).

Inputs to this op can take one of three forms: a single TensorFlow tensor of
scores with no sequence lengths, a TensorFlow tensor of scores along with a
TensorFlow tensor of sequence lengths, or a RaggedTensor. If only the scores
tensor is passed, this op will assume that the sequence lengths are equal to the
size of the tensor (and so use all the data provided). If a scores tensor and
sequence_lengths tensor is provided, the op will only use the data in the scores
tensor as specified by the sequence_lengths tensor. Finally, if a RaggedTensor
is provided, the sequence_lengths will be ignored and the variable length
sequences in the RaggedTensor will be used.

#### Args:

*   <b>`scores`</b>: `<float32> [batch_size, num_steps, |num_states|]` A tensor
    of scores, where `scores[b, t, s]` is the predicted score for transitioning
    to state `s` at step `t` for batch `b`. The |num_states| dimension must
    correspond to the num_states attribute for this op. This input may be
    ragged; if it is ragged, the ragged tensor should have the same structure
    [b, t, s] and only axis 1 should be ragged.

*   <b>`sequence_length`</b>: `<{int32, int64}>[batch_size]` A rank-1 tensor
    representing the length of the output sequence. If None, and the 'scores'
    input is not ragged, sequence lengths will be assumed to be the length of
    the score tensor.

*   <b>`allowed_transitions`</b>: if use_start_and_end_states is TRUE:
    `<bool>[num_states+1, num_states+1]` if use_start_and_end_states is FALSE:
    `<bool>[num_states, num_states]` A rank-2 tensor representing allowed
    transitions.

    -   allowed_transitions[i][j] is true if the transition from state i to
        state j is allowed for i and j in 0...(num_states).
    -   allowed_transitions[num_states][num_states] is ignored. If
        use_start_and_end_states is TRUE:
    -   allowed_transitions[num_states][j] is true if the sequence is allowed to
        start from state j.
    -   allowed_transitions[i][num_states] is true if the sequence is allowed to
        end on state i. Default - An empty tensor. This allows all sequence
        states to transition to all other sequence states.

*   <b>`transition_weights`</b>: if use_start_and_end_states is TRUE:
    `<float32>[num_states+1, num_states+1]` if use_start_and_end_states is
    FALSE: `<float32>[num_states, num_states]` A rank-2 tensor representing
    transition weights.

    -   transition_weights[i][j] is the coefficient that a candidate transition
        score will be multiplied by if that transition is from state i to state
        j.
    -   transition_weights[num_states][num_states] is ignored. If
        use_start_and_end_states is TRUE:
    -   transition_weights[num_states][j] is the coefficient that will be used
        if the transition starts with state j.
    -   transition_weights[i][num_states] is the coefficient that will be used
        if the final state in the sequence is state i. Default - An empty
        tensor. This assigns a wieght of 1.0 all transitions

*   <b>`use_log_space`</b>: Whether to use log space for the calculation. If
    false, calculations will be done in exp-space.

*   <b>`use_start_and_end_states`</b>: If True, sequences will have an implicit
    start and end state added.

*   <b>`name`</b>: The name scope within which this op should be constructed.

#### Returns:

An <int32>[batch_size, (num_steps)] ragged tensor containing the appropriate
sequence of transitions. If a sequence is impossible, the value of the
RaggedTensor for that and all following transitions in that sequence shall be
'-1'.
