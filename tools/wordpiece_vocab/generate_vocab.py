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

"""Generate wordpiece vocab and compute metrics over dataset of tf.Examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
from absl import app
from absl import flags
import apache_beam as beam
from six.moves import range
import tensorflow as tf  # tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from google3.pipeline.flume.py import runner
from google3.pipeline.flume.py.io import recordio
from google3.pipeline.flume.py.io import sstableio
from tensorflow_text.google.tools import utils
from tensorflow_text.google.tools import vocab_lib
from tensorflow_text.google.tools import wordpiece_tokenizer_learner_lib as learner


FLAGS = flags.FLAGS

flags.DEFINE_string('data_file', None, 'The input data file path.')
flags.DEFINE_enum('input_format', 'recordio', ['recordio', 'sstable'],
                  'Format of the input data file.')
flags.DEFINE_string('vocab_file', None, 'The output vocab file path.')
flags.DEFINE_string('metrics_file', None, 'The output metrics file path.')
flags.DEFINE_string('temp_dir', None, 'The path to store TFT temp files. If '
                    'running pipeline in borg, be sure to use cns path')
flags.DEFINE_string(
    'lang_set', 'en,es,ru,ar,de,fr,it,pt,ja,pl,fa,zh',
    'Set of languages used to build wordpiece model, '
    'given as a comma-separated list.')
flags.DEFINE_string('text_key', 'text_a', 'Text feature key in input examples.')
flags.DEFINE_string(
    'language_code_key', None,
    'Optional language code feature key. If not set, language of each '
    'sample will be automatically detected.')
flags.DEFINE_float(
    'smoothing_exponent', 0.5,
    'Exponent used in calculating exponential smoothing coefficients.')
flags.DEFINE_integer('max_word_length', 50,
                     'Keep only words shorter than max_word_length.')
flags.DEFINE_integer('upper_thresh', 10000000,
                     'Upper threshold for binary search.')
flags.DEFINE_integer('lower_thresh', 10, 'Lower threshold for binary search.')
flags.DEFINE_integer('num_iterations', 4,
                     'Number of iterations in wordpiece learning algorithm.')
flags.DEFINE_integer('num_pad_tokens', 100, 'Number of padding tokens to '
                     'include in vocab.')
flags.DEFINE_integer('max_input_tokens', 5000000,
                     'Maximum number of input tokens, where -1 means no max.')
flags.DEFINE_integer('max_token_length', 50, 'Maximum length of a token.')
flags.DEFINE_integer('max_unique_chars', 1000,
                     'Maximum number of unique characters as tokens.')
flags.DEFINE_integer('vocab_size', 110000, 'Target size of generated vocab, '
                     'where vocab_size is an upper bound and the size of vocab '
                     'can be within slack_ratio less than the vocab_size.')
flags.DEFINE_float('slack_ratio', 0.05,
                   'Difference permitted between target and actual vocab size.')
flags.DEFINE_bool('include_joiner_token', True,
                  'Whether to include joiner token in word suffixes.')
flags.DEFINE_string('joiner', '##', 'Joiner token in word suffixes.')
flags.DEFINE_list('reserved_tokens',
                  ['<unk>', '<s>', '</s>', '<mask>',
                   '<cls>', '<sep>', '<S>', '<T>'],
                  'Reserved tokens to be included in vocab.')
flags.DEFINE_integer('min_token_frequency', 2,
                     'The min frequency for a token to be included.')
flags.DEFINE_bool('lower_case', False,
                  'If true, a preprocessing step is added to lowercase the '
                  'text, apply NFD normalization, and strip accents '
                  'characters.')


def generate_vocab(data_file,
                   vocab_file,
                   metrics_file,
                   temp_dir,
                   raw_metadata,
                   params,
                   min_token_frequency=2,
                   input_format='recordio'):
  """Returns a pipeline generating a vocab and writing the output.

  Args:
    data_file: recordio file to read
    vocab_file: path in which to write the vocab
    metrics_file: path in which to write the metrics
    temp_dir: path to directory in which to store TFT temp files
    raw_metadata: schema for dataset
    params: parameters for wordpiece vocab learning algorithm
    min_token_frequency: the min frequency for a token to be included
    input_format: the format of the input data file.
  """
  lang_set = set(FLAGS.lang_set.encode('utf-8').split(b','))

  # Schema to format metrics as CSV.
  csv_schema = dataset_schema.from_feature_spec({
      'lang': tf.FixedLenFeature([], tf.string),
      'sample_count': tf.FixedLenFeature([], tf.int64),
      'micro_drop_char_percent': tf.FixedLenFeature([], tf.string),
      'macro_drop_char_percent': tf.FixedLenFeature([], tf.string),
      'micro_compress_ratio': tf.FixedLenFeature([], tf.string),
      'macro_compress_ratio': tf.FixedLenFeature([], tf.string),
  })

  columns = [
      'lang',
      'sample_count',
      'micro_drop_char_percent',
      'macro_drop_char_percent',
      'micro_compress_ratio',
      'macro_compress_ratio',
  ]

  example_converter = tft.coders.ExampleProtoCoder(raw_metadata.schema,
                                                   serialized=False)

  def vocab_pipeline(root):
    """Creates a pipeline to generate wordpiece vocab over a corpus."""

    with tft_beam.Context(temp_dir=temp_dir):
      # Read raw data and convert to TF Transform encoded dict.
      if input_format == 'recordio':
        raw_data = (
            root
            | 'ReadInputDataFromRecordIO' >> recordio.ReadFromRecordIO(
                data_file, coder=beam.coders.ProtoCoder(tf.train.Example)))
      elif input_format == 'sstable':
        raw_data = (
            root
            | 'ReadInputDataFromSSTable' >> sstableio.ReadFromSSTable(
                data_file, value_coder=beam.coders.ProtoCoder(tf.train.Example))
        )
        raw_data |= 'DropKeys' >> beam.Values()
      else:
        raise ValueError(
            'Input data format must be one of recordio or sstable.')

      raw_data |= 'DecodeInputData' >> beam.Map(example_converter.decode)

      vocab_transform = vocab_lib.ConstructVocab(
          FLAGS.text_key, FLAGS.language_code_key, lang_set,
          FLAGS.smoothing_exponent, FLAGS.max_word_length, min_token_frequency,
          FLAGS.max_input_tokens, params, FLAGS.lower_case)
      _ = (
          raw_data
          | 'BatchInputData' >> beam.BatchElements()
          | 'ConvertToNumPy' >> beam.Map(utils.convert_batched_data_to_numpy)
          | vocab_transform
          | 'ConvertToList' >> beam.Map(lambda x: x.tolist())
          | 'Flatten' >> beam.FlatMap(lambda x: [i + '\n' for i in x])
          | 'WriteVocab' >> beam.io.WriteToText(
              vocab_file,
              shard_name_template='',
              append_trailing_newlines=False))

  def metrics_pipeline(root):
    """Creates a pipeline to measure wordpiece vocab metrics over a corpus."""

    with tft_beam.Context(temp_dir=temp_dir):
      # Read raw data and convert to TF Transform encoded dict.
      if input_format == 'recordio':
        raw_data = (
            root
            | 'ReadInputDataFromRecordIO' >> recordio.ReadFromRecordIO(
                data_file, coder=beam.coders.ProtoCoder(tf.train.Example)))
      elif input_format == 'sstable':
        raw_data = (
            root
            | 'ReadInputDataFromSSTable' >> sstableio.ReadFromSSTable(
                data_file, value_coder=beam.coders.ProtoCoder(tf.train.Example))
        )
        raw_data |= 'DropKeys' >> beam.Values()
      else:
        raise ValueError(
            'Input data format must be one of recordio or sstable.')

      raw_data |= 'DecodeInputData' >> beam.Map(example_converter.decode)

      metrics_transform = vocab_lib.ComputeVocabMetrics(FLAGS.vocab_file,
                                                        FLAGS.text_key,
                                                        FLAGS.language_code_key,
                                                        FLAGS.lower_case)
      # Initialize CSV coder. Aggregate values for each lang, calculate metrics,
      # and write to output to a CSV file.
      csv_converter = tft.coders.CsvCoder(columns, csv_schema)
      _ = (
          raw_data
          | 'BatchInputData' >> beam.BatchElements()
          | 'ConvertToNumPy' >> beam.Map(utils.convert_batched_data_to_numpy)
          | metrics_transform
          | 'ConvertToList' >>
          beam.Map(lambda x: {k: v.tolist() for k, v in x.items()})
          | 'EncodeMetrics' >> beam.Map(csv_converter.encode)
          | 'WriteMetrics' >> beam.io.WriteToText(
              metrics_file, shard_name_template='', header=','.join(columns)))

  runner.FlumeRunner().run(vocab_pipeline).wait_until_finish()
  runner.FlumeRunner().run(metrics_pipeline).wait_until_finish()


def main(_):
  # Define schema.
  feature_spec = {FLAGS.text_key: tf.FixedLenFeature([], tf.string)}
  if FLAGS.language_code_key:
    feature_spec[FLAGS.language_code_key] = tf.FixedLenFeature([], tf.string)
  raw_metadata = dataset_metadata.DatasetMetadata(
      dataset_schema.from_feature_spec(feature_spec))

  # Add in padding tokens.
  reserved_tokens = FLAGS.reserved_tokens
  if FLAGS.num_pad_tokens:
    padded_tokens = ['[PAD]']
    padded_tokens += ['[unused%d]' % i for i in range(1, FLAGS.num_pad_tokens)]
    reserved_tokens = padded_tokens + reserved_tokens

  params = learner.Params(FLAGS.upper_thresh, FLAGS.lower_thresh,
                          FLAGS.num_iterations, FLAGS.max_input_tokens,
                          FLAGS.max_token_length, FLAGS.max_unique_chars,
                          FLAGS.vocab_size, FLAGS.slack_ratio,
                          FLAGS.include_joiner_token, FLAGS.joiner,
                          reserved_tokens)

  temp_dir = FLAGS.temp_dir if FLAGS.temp_dir else tempfile.mkdtemp()

  generate_vocab(FLAGS.data_file, FLAGS.vocab_file, FLAGS.metrics_file,
                 temp_dir, raw_metadata, params, FLAGS.min_token_frequency,
                 FLAGS.input_format)


if __name__ == '__main__':
  app.run(main)
