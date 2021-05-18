# coding=utf-8
# Copyright 2021 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TensorFlow ops for maximum spanning tree problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_mst_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_mst_ops.so'))


def max_spanning_tree(num_nodes, scores, forest, name=None):
  r"""Finds the maximum directed spanning tree of a digraph.

  Given a batch of directed graphs with scored arcs and root selections, solves
  for the maximum spanning tree of each digraph, where the score of a tree is
  defined as the sum of the scores of the arcs and roots making up the tree.

  Returns the score of the maximum spanning tree of each digraph, as well as the
  arcs and roots in that tree.  Each digraph in a batch may contain a different
  number of nodes, so the sizes of the digraphs must be provided as an input.

  Note that this operation is only differentiable w.r.t. its |scores| input and
  its |max_scores| output.

  The code here is intended for NLP applications, but attempts to remain
  agnostic to particular NLP tasks (such as dependency parsing).

  >>> num_nodes = np.array([4, 3], np.int32)
  >>> scores = np.array([[[0, 0, 0, 0],
  ...                     [1, 0, 0, 0],
  ...                     [1, 2, 0, 0],
  ...                     [1, 2, 3, 4]],
  ...                    [[4, 3, 2, 9],
  ...                     [0, 0, 2, 9],
  ...                     [0, 0, 0, 9],
  ...                     [9, 9, 9, 9]]], np.int32)
  >>> max_scores, argmax_sources = max_spanning_tree(num_nodes=num_nodes,
  ...                                                scores=scores,
  ...                                                forest=False)
  >>> print('max_scores: %s, \nargmax_sources: %s' % (max_scores,
  ...        argmax_sources))
  max_scores: tf.Tensor([7 6], shape=(2,), dtype=int32),
  argmax_sources: tf.Tensor(
   [[ 3  0  1  3]
    [ 0  2  0 -1]], shape=(2, 4), dtype=int32)

  You can also quickly find the roots of the tree using `argmax_sources`:
  >>> num_nodes = np.array([4, 3], np.int32)
  >>> scores = np.array([[[0, 0, 0, 0],
  ...                     [1, 0, 0, 0],
  ...                     [1, 2, 0, 0],
  ...                     [1, 2, 3, 4]],
  ...                    [[4, 3, 2, 9],
  ...                     [0, 0, 2, 9],
  ...                     [0, 0, 0, 9],
  ...                     [9, 9, 9, 9]]], np.int32)
  >>> _, argmax_sources = max_spanning_tree(num_nodes=num_nodes,
  ...                                       scores=scores,
  ...                                       forest=False)
  >>> tf.equal(tf.map_fn(
  ...          lambda x: tf.range(tf.size(x)), argmax_sources), argmax_sources)
  <tf.Tensor: shape=(2, 4), dtype=bool, numpy=
  array([[False, False, False,  True],
         [ True, False, False, False]])>

  Args:
    num_nodes: [B] vector where entry b is number of nodes in the b'th digraph.
    scores: [B,M,M] tensor where entry b,t,s is the score of the arc from node s
      to node t in the b'th directed graph if s!=t, or the score of selecting
      node t as a root in the b'th digraph if s==t. This uniform tensor requires
      that M is >= num_nodes[b] for all b (ie. all graphs in the batch), and
      ignores entries b,s,t where s or t is >= num_nodes[b]. Arcs or root
      selections with non-finite score are treated as nonexistent.
    forest: If true, solves for a maximum spanning forest instead of a maximum
      spanning tree, where a spanning forest is a set of disjoint trees that
      span the nodes of the digraph.
    name: An optional name for this op.

  Returns:
    A tuple (max_scores, argmax_sources):
    max_scores is a [B] vector where entry b is the score of the maximum
      spanning tree of the b'th digraph.
    argmax_sources: [B,M] matrix where entry b,t is the source of the arc
      inbound to t in the maximum spanning tree of the b'th digraph, or t if t
      is a root. Entries b,t where t is >= num_nodes[b] are set to -1; quickly
      finding the roots can be done as: tf.equal(tf.map_fn(lambda x:
      tf.range(tf.size(x)), argmax_sources), argmax_sources)

  """
  return gen_mst_ops.max_spanning_tree(
      num_nodes=num_nodes, scores=scores, forest=forest, name=name)


@ops.RegisterGradient("MaxSpanningTree")
def max_spanning_tree_gradient(mst_op, d_loss_d_max_scores, *_):
  """Returns a subgradient of the MaximumSpanningTree op.

  Note that MaximumSpanningTree is only differentiable w.r.t. its |scores| input
  and its |max_scores| output.

  Args:
    mst_op: The MaximumSpanningTree op being differentiated.
    d_loss_d_max_scores: [B] vector where entry b is the gradient of the network
      loss w.r.t. entry b of the |max_scores| output of the |mst_op|.
    *_: The gradients w.r.t. the other outputs; ignored.

  Returns:
    1. None, since the op is not differentiable w.r.t. its |num_nodes| input.
    2. [B,M,M] tensor where entry b,t,s is a subgradient of the network loss
       w.r.t. entry b,t,s of the |scores| input, with the same dtype as
       |d_loss_d_max_scores|.
  """
  dtype = d_loss_d_max_scores.dtype.base_dtype
  if dtype is None:
    raise errors.InvalidArgumentError("Expected (%s) is not None" % dtype)

  argmax_sources_bxm = mst_op.outputs[1]
  input_dim = array_ops.shape(argmax_sources_bxm)[1]  # M in the docstring

  # The one-hot argmax is a subgradient of max.  Convert the batch of maximal
  # spanning trees into 0/1 indicators, then scale them by the relevant output
  # gradients from |d_loss_d_max_scores|.  Note that |d_loss_d_max_scores| must
  # be reshaped in order for it to broadcast across the batch dimension.
  indicators_bxmxm = standard_ops.one_hot(
      argmax_sources_bxm, input_dim, dtype=dtype)
  d_loss_d_max_scores_bx1 = array_ops.expand_dims(d_loss_d_max_scores, -1)
  d_loss_d_max_scores_bx1x1 = array_ops.expand_dims(d_loss_d_max_scores_bx1, -1)
  d_loss_d_scores_bxmxm = indicators_bxmxm * d_loss_d_max_scores_bx1x1
  return None, d_loss_d_scores_bxmxm
