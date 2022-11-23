# coding=utf-8
# Copyright 2022 TF.Text Authors.
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

"""Tests for byte_splitter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def _split(s):
  return list(s.encode())


@test_util.run_all_in_graph_and_eager_modes
class ByteSplitterTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(ByteSplitterTest, self).setUp()
    self.byte_splitter = tf_text.ByteSplitter()

  def testScalar(self):
    test_value = tf.constant('hello')
    expected_bytes = _split('hello')
    expected_start_offsets = range(5)
    expected_end_offsets = range(1, 6)
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytes)
    (bytez, start_offsets, end_offsets) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytes)
    self.assertAllEqual(start_offsets, expected_start_offsets)
    self.assertAllEqual(end_offsets, expected_end_offsets)

  def testVectorSingleValue(self):
    test_value = tf.constant(['hello'])
    expected_bytez = [_split('hello')]
    expected_offset_starts = [range(5)]
    expected_offset_ends = [range(1, 6)]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testVector(self):
    test_value = tf.constant(['hello', 'muñdʓ'])
    expected_bytez = [_split('hello'), _split('muñdʓ')]
    expected_offset_starts = [[*range(5)], [*range(7)]]
    expected_offset_ends = [[*range(1, 6)], [*range(1, 8)]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrix(self):
    test_value = tf.constant([['hello', 'hola'],
                              ['goodbye', 'muñdʓ']])
    expected_bytez = [[_split('hello'), _split('hola')],
                      [_split('goodbye'), _split('muñdʓ')]]
    expected_offset_starts = [[[*range(5)], [*range(4)]],
                              [[*range(7)], [*range(7)]]]
    expected_offset_ends = [[[*range(1, 6)], [*range(1, 5)]],
                            [[*range(1, 8)], [*range(1, 8)]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testMatrixRagged(self):
    test_value = tf.ragged.constant([['hello', 'hola'], ['muñdʓ']])
    expected_bytez = [[_split('hello'), _split('hola')], [_split('muñdʓ')]]
    expected_offset_starts = [[[*range(5)], [*range(4)]], [[*range(7)]]]
    expected_offset_ends = [[[*range(1, 6)], [*range(1, 5)]], [[*range(1, 8)]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrix(self):
    test_value = tf.constant([[['hello', 'hola'],
                               ['lol', 'ha']],
                              [['goodbye', 'muñdʓ'],
                               ['bye', 'mudʓ']]])
    expected_bytez = [[[_split('hello'), _split('hola')],
                       [_split('lol'), _split('ha')]],
                      [[_split('goodbye'), _split('muñdʓ')],
                       [_split('bye'), _split('mudʓ')]]]
    expected_offset_starts = [[[[*range(5)], [*range(4)]],
                               [[*range(3)], [*range(2)]]],
                              [[[*range(7)], [*range(7)]],
                               [[*range(3)], [*range(5)]]]]
    expected_offset_ends = [[[[*range(1, 6)], [*range(1, 5)]],
                             [[*range(1, 4)], [*range(1, 3)]]],
                            [[[*range(1, 8)], [*range(1, 8)]],
                             [[*range(1, 4)], [*range(1, 6)]]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def test3DimMatrixRagged(self):
    test_value = tf.ragged.constant([[['hello'], ['lol', 'ha']],
                                     [['bye', 'mudʓ']]])
    expected_bytez = [[[_split('hello')], [_split('lol'), _split('ha')]],
                      [[_split('bye'), _split('mudʓ')]]]
    expected_offset_starts = [[[[*range(5)]],
                               [[*range(3)], [*range(2)]]],
                              [[[*range(3)], [*range(5)]]]]
    expected_offset_ends = [[[[*range(1, 6)]],
                             [[*range(1, 4)], [*range(1, 3)]]],
                            [[[*range(1, 4)], [*range(1, 6)]]]]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyStringSingle(self):
    test_value = tf.constant('')
    expected_bytez = []
    expected_offset_starts = []
    expected_offset_ends = []
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyStrings(self):
    test_value = tf.constant(['', 'hello', '', 'muñdʓ', ''])
    expected_bytez = [[], _split('hello'), [], _split('muñdʓ'), []]
    expected_offset_starts = [[], [*range(5)], [], [*range(7)], []]
    expected_offset_ends = [[], [*range(1, 6)], [], [*range(1, 8)], []]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testEmptyDimensions(self):
    test_value = tf.ragged.constant([[[], ['lol', 'ha']], []])
    expected_bytez = [[[], [_split('lol'), _split('ha')]], []]
    expected_offset_starts = [[[], [[*range(3)], [*range(2)]]], []]
    expected_offset_ends = [[[], [[*range(1, 4)], [*range(1, 3)]]], []]
    bytez = self.byte_splitter.split(test_value)
    self.assertAllEqual(bytez, expected_bytez)
    (bytez, starts, ends) = (
        self.byte_splitter.split_with_offsets(test_value))
    self.assertAllEqual(bytez, expected_bytez)
    self.assertAllEqual(starts, expected_offset_starts)
    self.assertAllEqual(ends, expected_offset_ends)

  def testTfLite(self):
    """Checks TFLite conversion and inference."""

    class Model(tf.keras.Model):

      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bs = tf_text.ByteSplitter()

      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name='input')
      ])
      def call(self, input_tensor):
        return {'result': self.bs.split(input_tensor).flat_values}

    # Test input data.
    input_data = np.array(['Some minds are better kept apart'])

    # Define a model.
    model = Model()
    # Do TF inference.
    tf_result = model(tf.constant(input_data))['result']

    # Convert to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Do TFLite inference.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
    print(interp.get_signature_list())
    split = interp.get_signature_runner('serving_default')
    output = split(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output['result']
    else:
      tflite_result = output['output_1']

    # Assert the results are identical.
    self.assertAllEqual(tflite_result, tf_result)

    input_data = np.array(['Some minds are better kept apart oh'])
    tf_result = model(tf.constant(input_data))['result']

    output = split(input=input_data)
    if tf.executing_eagerly():
      tflite_result = output['result']
    else:
      tflite_result = output['output_1']
    self.assertAllEqual(tflite_result, tf_result)


if __name__ == '__main__':
  test.main()
