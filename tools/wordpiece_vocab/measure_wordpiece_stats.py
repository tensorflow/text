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

"""Measure wordpiece model stats on a corpus sample.

Stats:
1) Dropped character percent
   How many non-control, non-whitespace characters are getting dropping during
   wordpiece tokenization?

   To reduce the number of characters getting dropped, you will need to
   increase --max_uniques_chars value in learn_wordpiece_tokenizer_main.cc.

2) Compression ratio
   Number of characters / Number of wordpieces.

   To increase compression ratio for a particular language, you will either
   need to increase the overall vocab size (--vocab_size in
   learn_wordpiece_tokenizer_main.c) or oversample that language.

3) Wordpiece overlap with English:
   (Wordpieces present in both en and xx samples /
    Wordpieces present in xx samples)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
from absl import app
from absl import flags
import apache_beam as beam
import tensorflow as tf  # tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from google3.pipeline.flume.py import runner
from google3.pipeline.flume.py.io import recordio
from tensorflow_text.google.tools import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', None, 'Input RecordIO file.')
flags.DEFINE_string('output_file', None,
                    'File in which to store calculated statistics.')
flags.DEFINE_string('temp_dir', None, 'The path to store TFT temp files.')
flags.DEFINE_string('text_key', 'text_a', 'Text feature key in input examples.')
flags.DEFINE_string(
    'language_code_key', None,
    'Optional language code feature key. If not set, language of each '
    'sample will be automatically detected.')
flags.DEFINE_string('vocab_file', None, 'Wordpiece vocab file.')


def calculate_metrics():
  """Returns a pipeline to compute wordpiece model stats given a vocab and corpus."""

  # Schema of input dataset.
  raw_metadata = dataset_metadata.DatasetMetadata(
      dataset_schema.from_feature_spec({
          'label': tf.FixedLenFeature([], tf.string),
          'text_a': tf.FixedLenFeature([], tf.string),
          'guid': tf.FixedLenFeature([], tf.string),
      }))

  # Schema to format metrics as CSV.
  csv_schema = dataset_schema.from_feature_spec({
      'lang': tf.FixedLenFeature([], tf.string),
      'sample_count': tf.FixedLenFeature([], tf.int64),
      'micro_drop_char_percent': tf.FixedLenFeature([], tf.string),
      'macro_drop_char_percent': tf.FixedLenFeature([], tf.string),
      'micro_compress_ratio': tf.FixedLenFeature([], tf.string),
      'macro_compress_ratio': tf.FixedLenFeature([], tf.string),
      'unweighted_en_wp_overlap_percent': tf.FixedLenFeature([], tf.string),
      'weighted_en_wp_overlap_percent': tf.FixedLenFeature([], tf.string),
  })

  columns = ['lang',
             'sample_count',
             'micro_drop_char_percent',
             'macro_drop_char_percent',
             'micro_compress_ratio',
             'macro_compress_ratio',
             'unweighted_en_wp_overlap_percent',
             'weighted_en_wp_overlap_percent']

  temp_dir = FLAGS.temp_dir if FLAGS.temp_dir else tempfile.mkdtemp()

  def pipeline(root):
    """Creates a pipeline to measure wordpiece vocab metrics over a corpus."""

    with tft_beam.Context(temp_dir=temp_dir):
      example_converter = tft.coders.ExampleProtoCoder(
          raw_metadata.schema, serialized=False)
      csv_converter = tft.coders.CsvCoder(columns, csv_schema)

      # Read raw data and convert to TF Transform encoded dict.
      raw_data = (
          root
          | 'ReadInputData' >> recordio.ReadFromRecordIO(
              FLAGS.input_file, coder=beam.coders.ProtoCoder(tf.train.Example))
          | 'DecodeInputData' >> beam.Map(example_converter.decode))

      # Apply transform to wordpiece-tokenize input.
      (transformed_data, _), _ = (
          (raw_data, raw_metadata)
          | 'WordpieceTokenizeInput' >> tft_beam.AnalyzeAndTransformDataset(
              utils.metrics_preprocessing_fn(FLAGS.vocab_file,
                                             FLAGS.text_key,
                                             FLAGS.language_code_key)))

      # Aggregate values for each lang, calculate metrics, and write to output.
      _ = (
          transformed_data
          | 'CompileTokenInfo' >> beam.ParDo(utils.CompileTokenizationInfo())
          | 'CombineStatsForLang' >> beam.CombineGlobally(utils.AggregateLang())
          | 'CalculateMetrics' >> beam.ParDo(utils.CalculateMetrics())
          | 'EncodeMetrics' >> beam.Map(csv_converter.encode)
          | 'WriteMetrics' >> beam.io.WriteToText(FLAGS.output_file,
                                                  shard_name_template='',
                                                  header=','.join(columns)))

  return pipeline


def main(_):
  pipeline = calculate_metrics()
  runner.FlumeRunner().run(pipeline).wait_until_finish()


if __name__ == '__main__':
  app.run(main)
