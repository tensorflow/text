<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.max_spanning_tree_gradient" />
<meta itemprop="path" content="Stable" />
</div>

# text.max_spanning_tree_gradient

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/mst_ops.py">View
source</a>

Returns a subgradient of the MaximumSpanningTree op.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.max_spanning_tree_gradient(
    mst_op, d_loss_d_max_scores, *_
)
</code></pre>

<!-- Placeholder for "Used in" -->

Note that MaximumSpanningTree is only differentiable w.r.t. its |scores| input
and its |max_scores| output.

#### Args:

*   <b>`mst_op`</b>: The MaximumSpanningTree op being differentiated.
*   <b>`d_loss_d_max_scores`</b>: [B] vector where entry b is the gradient of
    the network loss w.r.t. entry b of the |max_scores| output of the |mst_op|.
*   <b>`*_`</b>: The gradients w.r.t. the other outputs; ignored.

#### Returns:

1.  None, since the op is not differentiable w.r.t. its |num_nodes| input.
2.  [B,M,M] tensor where entry b,t,s is a subgradient of the network loss w.r.t.
    entry b,t,s of the |scores| input, with the same dtype as
    |d_loss_d_max_scores|.
