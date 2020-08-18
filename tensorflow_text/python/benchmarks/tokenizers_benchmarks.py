# coding=utf-8
# Copyright 2020 TF.Text Authors.
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

"""Microbenchmarks for tokenizers on IMDB dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import six

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow_text.python import ops as text_ops
from tensorflow_text.python.benchmarks import benchmark_utils
from tensorflow_text.python.ops.bert_tokenizer import BasicTokenizer

_BERT_VOCAB_PATH = "third_party/tensorflow_text/python/benchmarks/test_data/uncased_L-12_H-768_A-12/vocab.txt"
_HUB_MODULE_HANDLE = "third_party/tensorflow_text/python/ops/test_data/segmenter_hub_module"
_SENTENCEPIECE_MODEL_FILE = "third_party/tensorflow_text/python/ops/test_data/test_oss_model.model"
_RUN_ITERS = 1000
_BURN_ITERS = 10
_BATCH_SIZE = 32
_EAGER_EXECUTION = True
_USE_TF_FUNCTION = True
_XPROF_TRACING = False


class TokenizationBenchmark(
    six.with_metaclass(benchmark.ParameterizedBenchmark,
                       benchmark_utils.OpBenchmark)):
  """Benchmarks for tokenizers."""

  def __init__(self):
    self.load_input_data(_BATCH_SIZE)

  # Tokenizers to benchmark which do not require a special/extra input can be
  # added here as parameters to "_benchmark_parameters".
  # This method assumes the tokenizers given implement the Tokenizer class and
  # will run benchmarks for the "tokenize" and "tokenize_with_offsets" methods.

  # The parameters for each tokenizers are:
  #   - The tokenizer name
  #   - The tokenizer class to instantiate
  #   - The kwargs used in instantiating and initialization of the tokenizer
  _benchmark_parameters = [
      ("whitespace_tokenizer", text_ops.WhitespaceTokenizer),
      ("unicode_script_tokenizer", text_ops.UnicodeScriptTokenizer),
      ("unicode_char_tokenizer", text_ops.UnicodeCharTokenizer),
      ("bert_tokenizer", text_ops.BertTokenizer, {
          "vocab_lookup_table": _BERT_VOCAB_PATH,
          "token_out_type": dtypes.int32,
          "lower_case": False
      }),
      ("hub_module_tokenizer", text_ops.HubModuleTokenizer, {
          "hub_module_handle": _HUB_MODULE_HANDLE
      }),
      ("basic_tokenizer", BasicTokenizer),
  ]

  def benchmark_op(self, tokenizer, kwargs=None):

    tokenizer = tokenizer(**(kwargs or {}))

    for op in [tokenizer.tokenize, tokenizer.tokenize_with_offsets]:
      benchmark_name = self._get_name() + "_" + op.__name__
      self.run_and_report(
          op,
          _RUN_ITERS,
          _BURN_ITERS,
          benchmark_name,
          use_tf_function=_USE_TF_FUNCTION,
          xprof_enabled=_XPROF_TRACING)


class CustomInputTokenizationBenchmark(benchmark_utils.OpBenchmark):
  """Benchmarks for tokenizers that require extra preprocessing or inputs."""

  def __init__(self):
    self.load_input_data(_BATCH_SIZE)

  def _create_table(self, vocab, num_oov=100):
    init = lookup_ops.TextFileIdTableInitializer(vocab)
    return lookup_ops.StaticVocabularyTableV1(init, num_oov)

  def _run(self, tokenizer, kwargs=None):
    for op in [tokenizer.tokenize, tokenizer.tokenize_with_offsets]:
      benchmark_name = self._get_name() + "_" + op.__name__
      self.run_and_report(
          op,
          _RUN_ITERS,
          _BURN_ITERS,
          benchmark_name,
          use_tf_function=_USE_TF_FUNCTION,
          xprof_enabled=_XPROF_TRACING,
          **(kwargs or {}))

  def benchmark_op_wordpiece_tokenizer(self):
    self.input_data = text_ops.WhitespaceTokenizer().tokenize(self.input_data)

    tokenizer = text_ops.WordpieceTokenizer(
        vocab_lookup_table=self._create_table((_BERT_VOCAB_PATH)),
        unknown_token=None,
        token_out_type=dtypes.int64)
    self._run(tokenizer)

  def benchmark_op_sentencepiece_tokenizer(self):
    model = gfile.GFile((_SENTENCEPIECE_MODEL_FILE), "rb").read()
    tokenizer = text_ops.SentencepieceTokenizer(model)
    self._run(tokenizer)
    # TODO(irinabejan): Add benchmark for detokenization

  def _get_char_level_splits(self):
    """Get splits that match inputs char level."""
    char_tokenizer = text_ops.UnicodeCharTokenizer()
    char_splits = array_ops.zeros_like(char_tokenizer.tokenize(self.input_data))

    return char_splits

  def benchmark_op_split_merge_tokenizer(self):
    random_seed.set_seed(5)

    char_splits = self._get_char_level_splits()
    if not context.executing_eagerly():
      # Evaluate splits as their shape cannot be infered in graph mode
      # and are needed for mapping
      with session.Session() as sess:
        sess.run(self.iterator.initializer)
        char_splits = sess.run(char_splits)

    def randomize_splits(inputs):
      return random_ops.random_uniform(
          inputs.shape, maxval=2, dtype=dtypes.int32)

    labels = ragged_functional_ops.map_flat_values(randomize_splits,
                                                   char_splits)

    if not context.executing_eagerly():
      # Evaluate labels computation to exclude these steps from op benchmarking
      with session.Session() as sess:
        labels = sess.run(labels)

    tokenizer = text_ops.SplitMergeTokenizer()
    self._run(tokenizer, {"labels": labels})

  def benchmark_op_split_merge_from_logits_tokenizer(self):
    random_seed.set_seed(5)

    char_splits = self._get_char_level_splits().to_tensor()
    if not context.executing_eagerly():
      with session.Session() as sess:
        sess.run(self.iterator.initializer)
        char_splits = sess.run(char_splits)

    logits = random_ops.random_uniform(
        char_splits.shape + (2,), minval=-6, maxval=6, dtype=dtypes.float32)

    if not context.executing_eagerly():
      # Evaluate logits computation to exclude these steps from op benchmarking
      with session.Session() as sess:
        logits = sess.run(logits)

    tokenizer = text_ops.SplitMergeFromLogitsTokenizer()
    self._run(tokenizer, {"logits": logits})


if __name__ == "__main__":
  if not _EAGER_EXECUTION:
    ops.disable_eager_execution()

  app.run(test.main())
