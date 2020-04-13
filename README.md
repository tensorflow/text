# TensorFlow Text - Text processing in Tensorflow

IMPORTANT: When installing TF Text with `pip install`, please note the version
of TensorFlow you are running, as you should specify the corresponding version
of TF Text.

[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/text.svg)](https://github.com/tensorflow/text/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

TensorFlow Text provides a collection of text related classes and ops ready to
use with TensorFlow 2.0. The library can perform the preprocessing regularly
required by text-based models, and includes other features useful for sequence
modeling not provided by core TensorFlow.

The benefit of using these ops in your text preprocessing is that they are done
in the TensorFlow graph. You do not need to worry about tokenization in
training being different than the tokenization at inference, or managing
preprocessing scripts.

## Eager Execution

TensorFlow Text is compatible with both TensorFlow eager mode and graph mode.

```python
import tensorflow as tf
import tensorflow_text as text
tf.enable_eager_execution()
```

## Unicode

Most ops expect that the strings are in UTF-8. If you're using a different
encoding, you can use the core tensorflow transcode op to transcode into UTF-8.
You can also use the same op to coerce your string to structurally valid UTF-8
if your input could be invalid.

```python
docs = tf.constant([u'Everything not saved will be lost.'.encode('UTF-16-BE'),
                    u'Sad☹'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE',
                                         output_encoding='UTF-8')
```

## Normalization

When dealing with different sources of text, it's important that the same words
are recognized to be identical. A common technique for case-insensitive matching
in Unicode is case folding (similar to lower-casing). (Note that case folding
internally applies NFKC normalization.)

We also provide Unicode normalization ops for transforming strings into a
canonical representation of characters, with Normalization Form KC being the
default ([NFKC](http://unicode.org/reports/tr15/)).

```python
print(text.case_fold_utf8(['Everything not saved will be lost.']))
print(text.normalize_utf8(['Äffin']))
print(text.normalize_utf8(['Äffin'], 'nfkd'))
```

```sh
tf.Tensor(['everything not saved will be lost.'], shape=(1,), dtype=string)
tf.Tensor(['\xc3\x84ffin'], shape=(1,), dtype=string)
tf.Tensor(['A\xcc\x88ffin'], shape=(1,), dtype=string)
```

## Tokenization

Tokenization is the process of breaking up a string into tokens. Commonly, these
tokens are words, numbers, and/or punctuation.

The main interfaces are `Tokenizer` and `TokenizerWithOffsets` which each have a
single method `tokenize` and `tokenizeWithOffsets` respectively. There are
multiple implementing tokenizers available now. Each of these implement
`TokenizerWithOffsets` (which extends `Tokenizer`) which includes an option for
getting byte offsets into the original string. This allows the caller to know
the bytes in the original string the token was created from.

All of the tokenizers return RaggedTensors with the inner-most dimension of
tokens mapping to the original individual strings. As a result, the resulting
shape's rank is increased by one. Please review the ragged tensor guide if you
are unfamiliar with them. https://www.tensorflow.org/guide/ragged_tensors

### WhitespaceTokenizer

This is a basic tokenizer that splits UTF-8 strings on ICU defined whitespace
characters (eg. space, tab, new line).

```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
```

```sh
[['everything', 'not', 'saved', 'will', 'be', 'lost.'], ['Sad\xe2\x98\xb9']]
```

### UnicodeScriptTokenizer

This tokenizer splits UTF-8 strings based on Unicode script boundaries. The
script codes used correspond to International Components for Unicode (ICU)
UScriptCode values. See: http://icu-project.org/apiref/icu4c/uscript_8h.html

In practice, this is similar to the `WhitespaceTokenizer` with the most apparent
difference being that it will split punctuation (USCRIPT_COMMON) from language
texts (eg. USCRIPT_LATIN, USCRIPT_CYRILLIC, etc) while also separating language
texts from each other.

```python
tokenizer = text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.',
                             u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
```

```sh
[['everything', 'not', 'saved', 'will', 'be', 'lost', '.'],
 ['Sad', '\xe2\x98\xb9']]
```

### Unicode split

When tokenizing languages without whitespace to segment words, it is common to
just split by character, which can be accomplished using the
[unicode_split](https://www.tensorflow.org/api_docs/python/tf/strings/unicode_split)
op found in core.

```python
tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
print(tokens.to_list())
```

```sh
[['\xe4\xbb\x85', '\xe4\xbb\x8a', '\xe5\xb9\xb4', '\xe5\x89\x8d']]
```

### Offsets

When tokenizing strings, it is often desired to know where in the original
string the token originated from. For this reason, each tokenizer which
implements `TokenizerWithOffsets` has a *tokenize_with_offsets* method that will
return the byte offsets along with the tokens. The offset_starts lists the bytes
in the original string each token starts at, and the offset_limits lists the
bytes where each token ends at.

```python
tokenizer = text.UnicodeScriptTokenizer()
(tokens, offset_starts, offset_limits) = tokenizer.tokenize_with_offsets(
    ['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
print(offset_starts.to_list())
print(offset_limits.to_list())
```

```sh
[['everything', 'not', 'saved', 'will', 'be', 'lost', '.'],
 ['Sad', '\xe2\x98\xb9']]
[[0, 11, 15, 21, 26, 29, 33], [0, 3]]
[[10, 14, 20, 25, 28, 33, 34], [3, 6]]
```

### TF.Data Example

Tokenizers work as expected with the tf.data API. A simple example is provided
below.

```python
docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'],
                                           ["It's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = tokenized_docs.make_one_shot_iterator()
print(iterator.get_next().to_list())
print(iterator.get_next().to_list())
```

```sh
[['Never', 'tell', 'me', 'the', 'odds.']]
[["It's", 'a', 'trap!']]
```

### Keras API

When you use different tokenizers and ops to preprocess your data, the resulting
outputs are Ragged Tensors. The Keras API makes it easy now to train a model
using Ragged Tensors without having to worry about padding or masking the data,
by either using the ToDense layer which handles all of these for you or relying
on Keras built-in layers support for natively working on ragged data.

```python
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(None,), dtype='int32', ragged=True),
  text.keras.layers.ToDense(pad_value=0, mask=True),
  tf.keras.layers.Embedding(100, 16),
  tf.keras.layers.LSTM(32),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Other Text Ops

TF.Text packages other useful preprocessing ops. We will review a couple below.

### Wordshape

A common feature used in some natural language understanding models is to see
if the text string has a certain property. For example, a sentence breaking
model might contain features which check for word capitalization or if a
punctuation character is at the end of a string.

Wordshape defines a variety of useful regular expression based helper functions
for matching various relevant patterns in your input text. Here are a few
examples.

```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.',
                             u'Sad☹'.encode('UTF-8')])

# Is capitalized?
f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
# Are all letters uppercased?
f2 = text.wordshape(tokens, text.WordShape.IS_UPPERCASE)
# Does the token contain punctuation?
f3 = text.wordshape(tokens, text.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
# Is the token a number?
f4 = text.wordshape(tokens, text.WordShape.IS_NUMERIC_VALUE)

print(f1.to_list())
print(f2.to_list())
print(f3.to_list())
print(f4.to_list())
```

```sh
[[True, False, False, False, False, False], [True]]
[[False, False, False, False, False, False], [False]]
[[False, False, False, False, False, True], [True]]
[[False, False, False, False, False, False], [False]]
```

### N-grams & Sliding Window

N-grams are sequential words given a sliding window size of *n*. When combining
the tokens, there are three reduction mechanisms supported. For text, you would
want to use `Reduction.STRING_JOIN` which appends the strings to each other.
The default separator character is a space, but this can be changed with the
string_separater argument.

The other two reduction methods are most often used with numerical values, and
these are `Reduction.SUM` and `Reduction.MEAN`.

```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.',
                             u'Sad☹'.encode('UTF-8')])

# Ngrams, in this case bi-gram (n = 2)
bigrams = text.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

print(bigrams.to_list())
```

```sh
[['Everything not', 'not saved', 'saved will', 'will be', 'be lost.'], []]
```

## Installation

### Install using PIP

When installing TF Text with `pip install`, please note the version
of TensorFlow you are running, as you should specify the corresponding version
of TF Text. For example, if you're using TF 2.0, install the 2.0 version of TF
Text, and if you're using TF 1.15, install the 1.15 version of TF Text.

```bash
pip install -U tensorflow-text==<version>
```

### Build from source steps:

Note that TF Text needs to be built in the same environment as TensorFlow. Thus,
if you manually build TF Text, it is highly recommended that you also build
TensorFlow.

1. [build and install TensorFlow](https://www.tensorflow.org/install/source).
1. Clone the TF Text repo:
   `git clone https://github.com/tensorflow/text.git`
1. Run the build script to create a pip package:
   `./oss_scripts/run_build.sh`
