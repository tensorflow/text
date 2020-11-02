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

"""Tests for google3.third_party.tensorflow_text.google.tools.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tempfile

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import six
import tensorflow as tf  # tf
from google3.testing.pybase import googletest
from tensorflow_text.google.tools import utils


class Document(object):

  def __init__(self, content, language_code):
    self.content = content
    self.language_code = language_code


class ConvertBatchedDataToNumpyTest(googletest.TestCase):

  def setUp(self):
    super(ConvertBatchedDataToNumpyTest, self).setUp()
    self.sample_input = [{
        'a': ['a', 'b', 'c'],
        'b': 'en'
    }, {
        'a': ['d', 'e', 'f'],
        'b': 'fr'
    }]
    self.expected_output = {
        'a': [['a', 'b', 'c'], ['d', 'e', 'f']],
        'b': ['en', 'fr']
    }

  def testBatchedDataConversion(self):
    result = utils.convert_batched_data_to_numpy(self.sample_input)
    self.assertSameElements(result.keys(), ['a', 'b'])
    for i in ['a', 'b']:
      self.assertListEqual(self.expected_output[i], result[i].tolist())


class ConvertEagerTensorsIntoUnbatchedDataTest(googletest.TestCase):

  def setUp(self):
    super(ConvertEagerTensorsIntoUnbatchedDataTest, self).setUp()
    self.sample_input = [{
        'a': tf.constant([['a', 'b', 'c'], ['d', 'e', 'f']]),
        'b': tf.constant(['en', 'fr'])
    }]
    self.expected_output = [{
        'a': ['a', 'b', 'c'],
        'b': 'en'
    }, {
        'a': ['d', 'e', 'f'],
        'b': 'fr'
    }]

  def testTensorConversion(self):
    with TestPipeline() as p:
      result = (
          p | beam.Create(self.sample_input)
          | beam.ParDo(utils.ConvertEagerTensorsIntoUnbatchedData()))
      assert_that(result, equal_to(self.expected_output))


class ExtractTokensCTLTest(googletest.TestCase):

  def setUp(self):
    super(ExtractTokensCTLTest, self).setUp()
    my_proto = Document(content='I like pie.', language_code='en')
    self.sample_input = [my_proto]
    self.expected_output = [{'lang': 'en',
                             'tokens': ['I', 'like', 'pie', '.']}]

  def testSimpleTokenization(self):
    with TestPipeline() as p:
      text = p | beam.Create(self.sample_input)
      result = text | beam.ParDo(utils.ExtractTokensCTL('content',
                                                        'language_code'))
      assert_that(result, equal_to(self.expected_output))


class ExtractTokensBasicBERTTest(googletest.TestCase):

  def setUp(self):
    super(ExtractTokensBasicBERTTest, self).setUp()
    my_proto = Document(content='I like pie.', language_code='en')
    self.sample_input = [my_proto]
    self.expected_output = [{'lang': 'en',
                             'tokens': ['I', 'like', 'pie', '.']}]

  def testSimpleTokenization(self):
    with TestPipeline() as p:
      text = p | beam.Create(self.sample_input)
      result = text | beam.ParDo(utils.ExtractTokensBasicBERT('content',
                                                              'language_code'))
      assert_that(result, equal_to(self.expected_output))


class FilterTokensByLangTest(googletest.TestCase):

  def setUp(self):
    super(FilterTokensByLangTest, self).setUp()
    self.sample_input = [{'lang': 'en',
                          'tokens': ['I', 'like', 'pie', '.']}]

  def testLangInLangSet(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.FilterTokensByLang({'en'}))
      assert_that(result, equal_to([('I', 'en'),
                                    ('like', 'en'),
                                    ('pie', 'en'),
                                    ('.', 'en')]))

  def testLangNotInLangSet(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.FilterTokensByLang({'fr'}))
      assert_that(result, equal_to([]))

  def testLangNotInLangSetIncludeOthers(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.FilterTokensByLang({'fr'}, True))
      assert_that(result, equal_to([('I', 'other'),
                                    ('like', 'other'),
                                    ('pie', 'other'),
                                    ('.', 'other')]))


class CompareValues(beam.DoFn):

  def process(self, element):
    return [element['en'] < element['fr']]


class CalculateCoefficientsTest(googletest.TestCase):

  def setUp(self):
    super(CalculateCoefficientsTest, self).setUp()
    self.sample_input = [('I', 'en'), ('really', 'en'),
                         ('like', 'en'), ('pie', 'en'),
                         ('.', 'en'), ('Je', 'fr'),
                         ('suis', 'fr'), ('une', 'fr'),
                         ('fille', 'fr'), ('.', 'fr')]

  def testEqual(self):
    with TestPipeline() as p:
      tokens = p | beam.Create(self.sample_input)
      result = tokens | beam.CombineGlobally(utils.CalculateCoefficients(0.5))
      assert_that(result, equal_to([{'en': 1.0, 'fr': 1.0}]))

  def testNotEqual(self):
    with TestPipeline() as p:
      sample_input = [('I', 'en'), ('kind', 'en'), ('of', 'en'), ('like', 'en'),
                      ('to', 'en'), ('eat', 'en'), ('pie', 'en'), ('!', 'en'),
                      ('Je', 'fr'), ('suis', 'fr'), ('une', 'fr'),
                      ('fille', 'fr'), ('.', 'fr')]
      tokens = p | beam.Create(sample_input)
      result = (tokens
                | beam.CombineGlobally(utils.CalculateCoefficients(0.5))
                | beam.ParDo(CompareValues()))
      assert_that(result, equal_to([True]))


class ExponentialSmoothingTest(googletest.TestCase):

  def setUp(self):
    super(ExponentialSmoothingTest, self).setUp()
    self.sample_input = [('Hello', 'en'), (',', 'en'),
                         ('world', 'en'), ('!', 'en'),
                         ('Bonjour', 'fr'), ('.', 'fr')]
    self.coeffs = [{'en': 0.75, 'fr': 1.5}]

  def testBasic(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      coeffs = p | 'CreateCoeffs' >> beam.Create(self.coeffs)
      result = tokens | beam.ParDo(
          utils.ExponentialSmoothing(), beam.pvalue.AsSingleton(coeffs))
      assert_that(result, equal_to([('Hello', 0.75), (',', 0.75),
                                    ('world', 0.75), ('!', 0.75),
                                    ('Bonjour', 1.5), ('.', 1.5)]))


class FilterByCountTest(googletest.TestCase):

  def setUp(self):
    super(FilterByCountTest, self).setUp()
    self.sample_input = [('one', 1), ('two', 2), ('three', 3), ('four', 4)]
    self.max_token_length = 50

  def testBelowThreshold(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      result = tokens | beam.Filter(
          utils.create_filter_by_count_fn(
              self.max_token_length, min_token_frequency=2))
      assert_that(result, equal_to([('three', 3), ('four', 4)]))

  def testTokenTooLong(self):
    sample_input = [('one', 1), ('two', 2), ('three', 3), ('four', 4),
                    ('qwertyuiopasdfghjklzxcvbnmqwertyuiopasdfghjklzxcvbnm', 5),
                    ('blah', 20)]

    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(sample_input)
      result = tokens | beam.Filter(
          utils.create_filter_by_count_fn(
              self.max_token_length, min_token_frequency=2))
      assert_that(result, equal_to([('three', 3), ('four', 4), ('blah', 20)]))


class SortByCountTest(googletest.TestCase):

  def setUp(self):
    super(SortByCountTest, self).setUp()
    self.sample_input = [('a', 5), ('b', 2), ('c', 9), ('d', 4)]

  def testUnsorted(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      result = tokens | beam.CombineGlobally(utils.SortByCount())
      assert_that(result, equal_to([[('c', 9), ('a', 5), ('d', 4), ('b', 2)]]))


class CompileTokenizationInfoTest(googletest.TestCase):

  def setUp(self):
    super(CompileTokenizationInfoTest, self).setUp()
    self.sample_input = [{
        'lang': 'en',
        'num_non_unk_wordpieces': 4,
        'num_dropped_chars': 2,
        'num_preserved_chars': 13,
    }, {
        'lang': 'fr',
        'num_non_unk_wordpieces': 5,
        'num_dropped_chars': 0,
        'num_preserved_chars': 14,
    }]

  def testTwoLangs(self):
    with TestPipeline() as p:
      tokens = p | 'CreateInput' >> beam.Create(self.sample_input)
      result = tokens | beam.ParDo(utils.CompileTokenizationInfo())
      assert_that(
          result,
          equal_to([{
              'lang': 'en',
              'count': 1,
              'num_preserved_chars': 13,
              'num_dropped_chars': 2,
              'num_non_unk_wordpieces': 4,
              'preserved_ratio_sum': 13 / 4.0,
              'dropped_ratio_sum': 2.0 / 15,
          }, {
              'lang': 'fr',
              'count': 1,
              'num_preserved_chars': 14,
              'num_dropped_chars': 0,
              'num_non_unk_wordpieces': 5,
              'preserved_ratio_sum': 14 / 5.0,
              'dropped_ratio_sum': 0.0,
          }]))


class AggregateLangTest(googletest.TestCase):

  def setUp(self):
    super(AggregateLangTest, self).setUp()
    self.aggregator = utils.AggregateLang()
    self.sample_input = [{
        'lang': 'en',
        'count': 1,
        'num_preserved_chars': 13,
        'num_dropped_chars': 2,
        'num_non_unk_wordpieces': 4,
        'preserved_ratio_sum': 13 / 4.0,
        'dropped_ratio_sum': 2 / 15.0,
    }, {
        'lang': 'en',
        'count': 1,
        'num_preserved_chars': 11,
        'num_dropped_chars': 0,
        'num_non_unk_wordpieces': 4,
        'preserved_ratio_sum': 11.0 / 4,
        'dropped_ratio_sum': 0.0,
    }]

  def testMultiEntryOneLang(self):
    expected_output = self.aggregator.create_accumulator()
    expected_output['en'] = {
        'count': 2,
        'num_preserved_chars': 24,
        'num_dropped_chars': 2,
        'num_non_unk_wordpieces': 8,
        'preserved_ratio_sum': 13 / 4.0 + 11.0 / 4,
        'dropped_ratio_sum': 2 / 15.0,
    }
    # Test create_accumulator.
    accumulator = self.aggregator.create_accumulator()
    # Test add_input.
    for element in self.sample_input:
      accumulator = self.aggregator.add_input(accumulator, element)
    # Test merge_accumulators.
    merged = self.aggregator.merge_accumulators([
        accumulator, self.aggregator.create_accumulator()])
    # Test extract_output.
    output = self.aggregator.extract_output(merged)
    self.assertEqual(output, expected_output)


class CalculateMetricsTest(googletest.TestCase):

  def setUp(self):
    super(CalculateMetricsTest, self).setUp()
    self.info_dict = {
        'en': {
            'count': 2,
            'num_preserved_chars': 24,
            'num_dropped_chars': 2,
            'num_non_unk_wordpieces': 8,
            'preserved_ratio_sum': 2 + 3,
            'dropped_ratio_sum': 0.5 + 0,
        },
        'fr': {
            'count': 2,
            'num_preserved_chars': 24,
            'num_dropped_chars': 2,
            'num_non_unk_wordpieces': 8,
            'preserved_ratio_sum': 5 + 7,
            'dropped_ratio_sum': 0.4 + 0.6,
        }
    }
    self.metrics = utils.CalculateMetrics()

  def testListMean(self):
    test_list = [1, 2, 3, 4, 5]
    mean = self.metrics._get_list_mean(test_list)
    self.assertEqual(mean, 3)

  def testMicroCompressionRatio(self):
    fr_micro_compression = self.metrics._get_micro_compression_ratio(
        self.info_dict['fr'])
    self.assertEqual(fr_micro_compression, 3)

  def testMacroCompressionRatio(self):
    en_macro_compression = self.metrics._get_macro_compression_ratio(
        self.info_dict['en'])
    self.assertEqual(en_macro_compression, 2.5)

  def testMicroDroppedCharPercent(self):
    en_micro_dropped_char = self.metrics._get_micro_dropped_char_percent(
        self.info_dict['en'])
    self.assertEqual(en_micro_dropped_char, 100/13)

  def testMacroDroppedCharPercent(self):
    fr_macro_dropped_char = self.metrics._get_macro_dropped_char_percent(
        self.info_dict['fr'])
    self.assertEqual(fr_macro_dropped_char, 50.0)

  def testFormatFloatOrNone(self):
    extra_digits = 0.12345
    self.assertEqual(self.metrics._format_float_or_none(extra_digits), '0.123')
    fewer_digits = 0.1
    self.assertEqual(self.metrics._format_float_or_none(fewer_digits), '0.100')
    non_float = ''
    self.assertIsNone(self.metrics._format_float_or_none(non_float))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _encode_strings(items):
  """Encode all the strings in items with six.ensure_binary."""
  return [six.ensure_binary(x) for x in items]


class CountPreprocessingFnTest(googletest.TestCase):

  def setUp(self):
    super(CountPreprocessingFnTest, self).setUp()
    self.raw_data = {
        'label': ['1'],
        'text_a': ['The boy jumped into the air.'],
        'guid': ['dev-0'],
    }

  def testDetectLang(self):
    preprocessing_fn = utils.count_tokenizing_fn('text_a', None)
    expected_tokens = ['The', 'boy', 'jumped', 'into', 'the', 'air', '.']
    outputs = preprocessing_fn(self.raw_data)

    self.assertEqual(outputs['lang'].numpy(), six.ensure_binary('en'))
    self.assertSequenceAlmostEqual(outputs['tokens'].values,
                                   _encode_strings(expected_tokens))

  def testUseGivenLang(self):
    preprocessing_fn = utils.count_tokenizing_fn('text', 'language_code')
    raw_data = {
        'text': ['Let\'s make this Chinese even though it\'s English.'],
        'language_code': ['zh'],
    }
    expected_tokens = [
        'Let', '\'', 's', 'make', 'this', 'Chinese', 'even', 'though', 'it',
        '\'', 's', 'English', '.'
    ]

    outputs = preprocessing_fn(raw_data)
    self.assertEqual(outputs['lang'].numpy(), six.ensure_binary('zh'))
    self.assertSequenceAlmostEqual(outputs['tokens'].values.numpy(),
                                   _encode_strings(expected_tokens))


class MetricsPreprocessingFnTest(googletest.TestCase):

  def setUp(self):
    super(MetricsPreprocessingFnTest, self).setUp()
    self.raw_data = {
        'label': ['1'],
        'text_a': ['The boy jumped into the air.'],
        'guid': ['dev-0'],
    }
    self.vocab = ['The', 'jump', '##ed', 'in', '##to', 'the', 'air', '.', 'bo',
                  'jumped', 'to', 'cat', 'sat', 'on', 'a', 'h', '##at', 'c']
    self.expected_wordpieces = ['The', '[UNK]', 'jumped', 'in', '##to', 'the',
                                'air', '.']

  def testSingleElement(self):
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as vocab:
      vocab.writelines([word + '\n' for word in self.vocab])
      vocab.flush()
      preprocessing_fn = utils.metrics_tokenizing_fn(vocab.name, 'text_a', None)
      outputs = preprocessing_fn(self.raw_data)

      self.assertEqual(outputs['lang'].numpy(), six.ensure_binary('en'))
      self.assertEqual(outputs['num_non_unk_wordpieces'].numpy(), 7)
      self.assertEqual(outputs['num_preserved_chars'].numpy(), 20)
      self.assertEqual(outputs['num_dropped_chars'].numpy(), 3)
      self.assertSequenceAlmostEqual(outputs['wordpieces'].values,
                                     _encode_strings(self.expected_wordpieces))

  def testLargerBatchSize(self):
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as vocab:
      raw_data = {
          'label': ['1', '2'],
          'text_a': ['The boy jumped into the air.', 'The cat sat on a hat.'],
          'guid': ['dev-0', 'dev-0'],
      }
      expected_wordpieces = [
          'The', '[UNK]', 'jumped', 'in', '##to', 'the', 'air', '.', 'The',
          'cat', 'sat', 'on', 'a', 'h', '##at', '.'
      ]
      vocab.writelines([word + '\n' for word in self.vocab])
      vocab.flush()
      preprocessing_fn = utils.metrics_tokenizing_fn(vocab.name, 'text_a', None)
      outputs = preprocessing_fn(raw_data)

      self.assertSequenceAlmostEqual(outputs['lang'].numpy(),
                                     _encode_strings(['en', 'en']))
      self.assertSequenceAlmostEqual(outputs['num_preserved_chars'].numpy(),
                                     [20, 16])
      self.assertSequenceAlmostEqual(outputs['num_dropped_chars'].numpy(),
                                     [3, 0])
      self.assertSequenceAlmostEqual(outputs['wordpieces'].values.numpy(),
                                     _encode_strings(expected_wordpieces))
      self.assertSequenceAlmostEqual(outputs['num_non_unk_wordpieces'].numpy(),
                                     [7, 8])


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  googletest.main()
