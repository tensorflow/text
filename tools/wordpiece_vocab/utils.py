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

# Lint as: python2, python3
"""PTransforms used for wordpiece vocabulary generation pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

# Dependency imports

import apache_beam as beam
from language.bert import tokenization
import numpy as np
import six
import tensorflow as tf

from google3.learning.brain.contrib.text.python.ops import lang_id_ops
from google3.nlp.neon.dual_encoder.proto import example_pair_pb2
from google3.nlp.nlx.infrastructure.python.clif import ctl_unicode
from google3.sstable.python import sstable
from tensorflow_text.google.tools import wordpiece_tokenizer_learner_lib as learner
from tensorflow_text.python.ops import bert_tokenizer


def _get_lang_from_file_path(path):
  filedir, _ = os.path.split(path)
  _, lang = os.path.split(filedir)
  return lang


LangQueryText = collections.namedtuple('LangQueryText', ['lang', 'text'])


def convert_batched_data_to_numpy(elements):
  batched_elems = collections.defaultdict(list)
  for e in elements:
    for k, v in e.items():
      batched_elems[k].append(v)
  return {k: np.array(v) for k, v in batched_elems.items()}


class ConvertEagerTensorsIntoUnbatchedData(beam.DoFn):
  """DoFn that converts EagerTensors into unbatched data."""

  def process(self, element):
    for i in range((list(element.values())[0]).shape[0]):
      rval = {}
      for k, v in element.items():
        val = v[i].numpy()
        if isinstance(val, np.ndarray):
          val = val.tolist()
          if isinstance(val[0], bytes):
            val = [i.decode('utf-8') for i in val]
        else:
          if isinstance(val, bytes):
            val = val.decode('utf-8')
        rval[k] = val
      yield rval


class ReadAndExtractLangQueryTextFromGFV(beam.DoFn):
  """Given a filepath, read sstable and extract (lang, query_text)."""

  def process(self, filename):
    lang = _get_lang_from_file_path(filename)
    query_sstable = sstable.SSTable(filename)
    for query in six.iterkeys(query_sstable):
      yield LangQueryText(lang=lang, text=query)


class ReadAndExtractLangQueryTextFromNavBoost(beam.DoFn):
  """Given a filepath, read sstable and extract (lang, query_text)."""

  def __init__(self):
    self.queries_processed_counter = beam.metrics.Metrics.counter(
        '_ReadAndExtractLangQueryTextFromNavBoost', 'queries-processed')
    self.queries_in_sstable_counter = beam.metrics.Metrics.counter(
        '_ReadAndExtractLangQueryTextFromNavBoost', 'queries-in-sstables')

  def process(self, filename):
    lang = _get_lang_from_file_path(filename)
    query_sstable = sstable.SSTable(filename)
    self.queries_in_sstable_counter.inc(len(query_sstable))
    for query_pair_str in six.itervalues(query_sstable):
      # Decode into a nlp_neon.dual_encoder.ExamplePair proto.
      example_pair_proto = example_pair_pb2.ExamplePair().FromString(
          query_pair_str)
      left_query = example_pair_proto.left
      right_query = example_pair_proto.right
      yield LangQueryText(
          lang=lang,
          text=left_query.features.feature['text'].bytes_list.value[0])
      self.queries_processed_counter.inc()
      yield LangQueryText(
          lang=lang,
          text=right_query.features.feature['text'].bytes_list.value[0])
      self.queries_processed_counter.inc()


class ExtractTokensCTL(beam.DoFn):
  """Tokenizes text using Core Text Library tokenizer."""

  def __init__(self, text_key, language_code_key):
    self._text_key = text_key
    self._language_code_key = language_code_key

  def setup(self):
    self._tokenizer = ctl_unicode.CTLText()

  def process(self, record):
    content = getattr(record, self._text_key)
    lang = getattr(record, self._language_code_key)

    tokenized = {}
    tokenized['lang'] = lang

    text = tokenization.convert_to_unicode(content)
    tokenized['tokens'] = [
        text[token.start:token.start + token.length]
        for token in self._tokenizer.tokenize(text, six.ensure_str(
            lang, 'utf-8'))
    ]
    yield tokenized


class ExtractTokensBasicBERT(beam.DoFn):
  """Tokenizes text using basic BERT tokenizer."""

  def __init__(self, text_key, language_code_key, do_lower_case=False):
    self._tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    self._text_key = text_key
    self._language_code_key = language_code_key

  def process(self, record):
    content = getattr(record, self._text_key)
    lang = getattr(record, self._language_code_key)

    tokenized = {}
    tokenized['lang'] = lang
    tokenized['tokens'] = self._tokenizer.tokenize(content)
    yield tokenized


class FilterTokensByLang(beam.DoFn):
  """Filters out languages if necessary and yields each (token, lang) pair."""

  def __init__(self, lang_set, include_other_languages=False):
    self._lang_set = lang_set
    self._include_other_languages = include_other_languages

  def process(self, element):
    lang = element['lang']

    if lang in self._lang_set or self._include_other_languages:
      returned_lang = lang if lang in self._lang_set else 'other'

      for token in element['tokens']:
        yield token, returned_lang


class CalculateCoefficients(beam.CombineFn):
  """Calculates smoothing coefficient for each language."""

  def __init__(self, smoothing_exponent):
    self._smoothing_exponent = smoothing_exponent

  def create_accumulator(self):
    return {'total_count': 0, 'lang_count': collections.Counter()}

  def add_input(self, accumulator, element):
    _, lang = element
    accumulator['total_count'] += 1
    accumulator['lang_count'].update([lang])
    return accumulator

  def merge_accumulators(self, accumulators):
    merged = self.create_accumulator()
    for acc in accumulators:
      for key in merged:
        merged[key] += acc[key]
    return merged

  def extract_output(self, accumulator):
    lang_count = accumulator['lang_count']
    total = accumulator['total_count']
    probs, exp = {}, {}
    for lang in lang_count:
      probs[lang] = lang_count[lang] / total
      exp[lang] = pow(probs[lang], self._smoothing_exponent)
    total_weight = sum(exp.values())
    for lang in exp:
      exp[lang] = exp[lang] / (total_weight * probs[lang])
    return exp


class ExponentialSmoothing(beam.DoFn):
  """Applies exponential smoothing coefficients to the counts."""

  def __init__(self, corpus_multiplier=1):
    self._corpus_multiplier = corpus_multiplier

  def process(self, word_and_lang, coeffs):
    word, lang = word_and_lang
    count = coeffs[lang] * self._corpus_multiplier
    yield word, count


def create_filter_by_count_fn(max_word_length, min_token_frequency=2):
  """Filters words with counts below some threshold."""

  def filter_fn(word_and_count):
    word, count = word_and_count
    return count > min_token_frequency and len(word) <= max_word_length

  return filter_fn


class SortByCount(beam.CombineFn):
  """Sorts words by count."""

  def create_accumulator(self):
    return []

  def add_input(self, accumulator, element):
    if not accumulator:
      accumulator = self.create_accumulator()

    word, count = element
    accumulator.append((word, int(count)))
    return accumulator

  def merge_accumulators(self, accumulators):
    merged = self.create_accumulator()
    for accumulator in accumulators:
      if accumulator:
        merged.extend(accumulator)
    return merged

  def extract_output(self, accumulator):
    return sorted(sorted(accumulator, key=lambda x: x[0]), key=lambda x: x[1],
                  reverse=True)


class CompileTokenizationInfo(beam.DoFn):
  """Expands list of tokens and computes intermediate metrics."""

  def process(self, record):
    dropped = record['num_dropped_chars']
    preserved = record['num_preserved_chars']
    non_unk = record['num_non_unk_wordpieces']
    preserved_ratio_sum = preserved / non_unk if non_unk else 0.0
    dropped_ratio_sum = (
        dropped / (dropped + preserved)) if (dropped + preserved) > 0 else 0.0
    tokenization_info = {
        'lang': record['lang'],
        'count': 1,
        'num_preserved_chars': preserved,
        'num_dropped_chars': dropped,
        'num_non_unk_wordpieces': non_unk,
        'preserved_ratio_sum': preserved_ratio_sum,
        'dropped_ratio_sum': dropped_ratio_sum,
    }
    yield tokenization_info


def default():
  return {
      'count': 0,
      'num_preserved_chars': 0,
      'num_dropped_chars': 0,
      'num_non_unk_wordpieces': 0,
      'preserved_ratio_sum': 0.0,
      'dropped_ratio_sum': 0.0,
  }


class AggregateLang(beam.CombineFn):
  """Aggregates intermediate metrics for each language."""

  def create_accumulator(self):
    return collections.defaultdict(default)

  def add_input(self, accumulator, element):
    lang = element['lang']
    for key in accumulator[lang].keys():
      accumulator[lang][key] += element[key]
    return accumulator

  def merge_accumulators(self, accumulators):
    merged = self.create_accumulator()
    for acc in accumulators:
      for lang in acc.keys():
        for key in acc[lang].keys():
          merged[lang][key] += acc[lang][key]
    return merged

  def extract_output(self, accumulator):
    return accumulator


class LearnVocab(beam.DoFn):
  """Learns vocabulary from word count list."""

  def __init__(self, params):
    self._params = params

  def process(self, wordcounts):
    wordcounts_unicode = []
    for word, count in wordcounts:
      wordcounts_unicode.append((six.ensure_text(word, 'utf-8', 'ignore'),
                                 count))
    return [learner.learn(wordcounts_unicode, self._params)]


class CalculateMetrics(beam.DoFn):
  """Calculates metrics for each language given tokenization info."""

  def process(self, info_dict):
    for lang in info_dict.keys():
      infos = info_dict[lang]
      yield {
          'lang':
              lang,
          'sample_count':
              infos['count'],
          'micro_drop_char_percent':
              self._format_float_or_none(
                  self._get_micro_dropped_char_percent(infos)),
          'macro_drop_char_percent':
              self._format_float_or_none(
                  self._get_macro_dropped_char_percent(infos)),
          'micro_compress_ratio':
              self._format_float_or_none(
                  self._get_micro_compression_ratio(infos)),
          'macro_compress_ratio':
              self._format_float_or_none(
                  self._get_macro_compression_ratio(infos)),
      }

  def _get_list_mean(self, l):
    return sum(l) / len(l) if l else None

  def _get_micro_compression_ratio(self, infos):
    if infos['num_non_unk_wordpieces']:
      return infos['num_preserved_chars'] / infos['num_non_unk_wordpieces']
    else:
      return None

  def _get_macro_compression_ratio(self, infos):
    return infos['preserved_ratio_sum'] / infos['count']

  def _get_micro_dropped_char_percent(self, infos):
    if infos['num_preserved_chars'] + infos['num_dropped_chars']:
      return 100.0 * infos['num_dropped_chars'] / (
          infos['num_preserved_chars'] + infos['num_dropped_chars'])
    else:
      return None

  def _get_macro_dropped_char_percent(self, infos):
    return 100.0 * infos['dropped_ratio_sum'] / infos['count']

  def _format_float_or_none(self, value):
    if isinstance(value, float):
      return '{:.3f}'.format(value)
    else:
      return None


def count_tokenizing_fn(text_key, language_code_key, do_lower_case=False):
  """Generates a tokenizing function to be used in generating word counts.

  Args:
    text_key: feature key in tf.Example for text
    language_code_key: feature key in tf.Example for language_code
    do_lower_case: If true, a step is added to lowercase the
      text, apply NFD normalization, and strip accents characters.

  Returns:
    a tokenizing function
  """

  def tokenizing_fn(inputs):
    """Tokenizing function used for word counts.

       Tokenizes input and detects language if there is no associated
       language_code.

    Args:
       inputs: dataset of tf.Examples containing text samples

    Returns:
       transformed outputs
    """

    outputs = {}

    basic_tokenizer = bert_tokenizer.BasicTokenizer(lower_case=do_lower_case)
    outputs['tokens'] = basic_tokenizer.tokenize(inputs[text_key])
    if language_code_key:
      outputs['lang'] = tf.convert_to_tensor(inputs[language_code_key])
    else:
      langs = lang_id_ops.detect_language(inputs[text_key])[0]
      outputs['lang'] = tf.gather(langs.to_tensor(), 0, axis=1)
    return outputs

  return tokenizing_fn


def metrics_tokenizing_fn(vocab_file,
                          text_key,
                          language_code_key,
                          do_lower_case=False):
  """Generates a tokenizing function to be used in generating word counts.

  Args:
    vocab_file: path to file containing wordpiece vocabulary
    text_key: feature key in tf.Example for text
    language_code_key: feature key in tf.Example for language_code
    do_lower_case: If true, a step is added to lowercase the
      text, apply NFD normalization, and strip accents characters.

  Returns:
    a tokenizing function
  """

  def tokenizing_fn(inputs):
    """Tokenizing function used for metrics calculation.

    Args:
       inputs: the input dataset of tf.Examples

    Returns:
       tokenized outputs
    """
    vocab_file_init = tf.lookup.TextFileInitializer(
        vocab_file, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER)
    vocab_table = tf.lookup.StaticHashTable(vocab_file_init, -1)

    basic_tokenizer = bert_tokenizer.BasicTokenizer(lower_case=do_lower_case)
    full_bert_tokenizer = bert_tokenizer.BertTokenizer(
        vocab_lookup_table=vocab_table,
        token_out_type=tf.string,
        lower_case=do_lower_case,
    )
    tokens = basic_tokenizer.tokenize(inputs[text_key])
    wordpieces = full_bert_tokenizer.tokenize(inputs[text_key])

    # Compute the total segment length
    wordpieces_merged = wordpieces.merge_dims(-2, -1)
    wordpieces_row_lengths = wordpieces_merged.row_lengths()
    total_sequence_lengths = wordpieces_row_lengths

    # Compute the segment length of only non-unk wordpieces in the batch
    ones = wordpieces_merged.with_flat_values(
        tf.repeat(1, tf.size(wordpieces_merged.flat_values)))
    zeros = wordpieces_merged.with_flat_values(
        tf.repeat(0, tf.size(wordpieces_merged.flat_values)))
    non_unk_wordpieces = tf.where(
        tf.logical_not(tf.equal(wordpieces_merged, '[UNK]')), x=ones, y=zeros)
    total_non_unk_sequence_lengths = tf.reduce_sum(non_unk_wordpieces, axis=1)

    wordpieces_flat = wordpieces.flat_values
    wordpieces_flat.set_shape([None])
    wordpieces = tf.RaggedTensor.from_nested_row_splits(
        wordpieces_flat, wordpieces.nested_row_splits)

    known_mask = tf.cast(tf.not_equal(wordpieces, '[UNK]'), tf.int32)
    num_non_unk_wordpieces = tf.reduce_sum(known_mask, axis=[1, 2])

    wordpiece_is_unknown = tf.equal(wordpieces, '[UNK]')
    token_has_unknown = tf.reduce_any(wordpiece_is_unknown, axis=-1)
    unknown_tokens = tf.ragged.boolean_mask(tokens, token_has_unknown)
    unknown_lengths = tf.strings.length(unknown_tokens)
    num_dropped_chars = tf.math.reduce_sum(unknown_lengths, axis=1)

    token_lengths = tf.strings.length(tokens)
    total_chars = tf.reduce_sum(token_lengths, axis=-1)
    num_preserved_chars = total_chars - num_dropped_chars

    flattened = tf.RaggedTensor.from_row_splits(
        wordpieces.flat_values, tf.gather(wordpieces.values.row_splits,
                                          wordpieces.row_splits))

    outputs = {}
    outputs['num_non_unk_wordpieces'] = tf.cast(num_non_unk_wordpieces,
                                                tf.int64)
    outputs['num_dropped_chars'] = tf.cast(num_dropped_chars, tf.int64)
    outputs['num_preserved_chars'] = tf.cast(num_preserved_chars, tf.int64)
    outputs['wordpieces'] = flattened

    outputs['sequence_length'] = tf.cast(total_sequence_lengths, tf.int64)
    outputs['non_unk_sequence_length'] = tf.cast(total_non_unk_sequence_lengths,
                                                 tf.int64)

    if language_code_key:
      outputs['lang'] = inputs[language_code_key]
    else:
      langs = lang_id_ops.detect_language(inputs[text_key])[0]
      outputs['lang'] = tf.gather(langs.to_tensor(), 0, axis=1)
    return outputs

  return tokenizing_fn
