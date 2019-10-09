<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text" />
<meta itemprop="path" content="Stable" />
</div>

# Module: text

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/__init__.py">View
source</a>

Various tensorflow ops related to text-processing.

<!-- Placeholder for "Used in" -->

## Modules

[`keras`](./text/keras.md) module: Tensorflow Text Layers for Keras API.

[`metrics`](./text/metrics.md) module: Tensorflow text-processing metrics.

## Classes

[`class BertTokenizer`](./text/BertTokenizer.md): Tokenizer used for BERT.

[`class Detokenizer`](./text/Detokenizer.md): Base class for detokenizer
implementations.

[`class Reduction`](./text/Reduction.md): Type of reduction to be done by the
n-gram op.

[`class SentencepieceTokenizer`](./text/SentencepieceTokenizer.md): Tokenizes a
tensor of UTF-8 strings.

[`class Tokenizer`](./text/Tokenizer.md): Base class for tokenizer
implementations.

[`class TokenizerWithOffsets`](./text/TokenizerWithOffsets.md): Base class for
tokenizer implementations that return offsets.

[`class UnicodeScriptTokenizer`](./text/UnicodeScriptTokenizer.md): Tokenizes a
tensor of UTF-8 strings on Unicode script boundaries.

[`class WhitespaceTokenizer`](./text/WhitespaceTokenizer.md): Tokenizes a tensor
of UTF-8 strings on whitespaces.

[`class WordShape`](./text/WordShape.md): Values for the 'pattern' arg of the
wordshape op.

[`class WordpieceTokenizer`](./text/WordpieceTokenizer.md): Tokenizes a tensor
of UTF-8 string tokens into subword pieces.

## Functions

[`case_fold_utf8(...)`](./text/case_fold_utf8.md): Applies case folding to every
UTF-8 string in the input.

[`coerce_to_structurally_valid_utf8(...)`](./text/coerce_to_structurally_valid_utf8.md): Coerce UTF-8 input strings to structurally valid UTF-8.

[`gather_with_default(...)`](./text/gather_with_default.md): Gather slices with `indices=-1` mapped to `default`.

[`greedy_constrained_sequence(...)`](./text/greedy_constrained_sequence.md): Performs greedy constrained sequence on a batch of examples.

[`ngrams(...)`](./text/ngrams.md): Create a tensor of n-grams based on the input data `data`.

[`normalize_utf8(...)`](./text/normalize_utf8.md): Normalizes each UTF-8 string
in the input tensor using the specified rule.

[`pad_along_dimension(...)`](./text/pad_along_dimension.md): Add padding to the beginning and end of data in a specific dimension.

[`sentence_fragments(...)`](./text/sentence_fragments.md): Find the sentence fragments in a given text.

[`sliding_window(...)`](./text/sliding_window.md): Builds a sliding window for `data` with a specified width.

[`span_alignment(...)`](./text/span_alignment.md): Return an alignment from a set of source spans to a set of target spans.

[`span_overlaps(...)`](./text/span_overlaps.md): Returns a boolean tensor indicating which source and target spans overlap.

[`viterbi_constrained_sequence(...)`](./text/viterbi_constrained_sequence.md): Performs greedy constrained sequence on a batch of examples.

[`wordshape(...)`](./text/wordshape.md): Determine wordshape features for each input string.

