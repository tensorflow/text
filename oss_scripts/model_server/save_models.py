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

"""Integration tests for TF.Text ops in model server."""

import os
import tensorflow as tf
import tensorflow_text as text

tf.compat.v1.enable_eager_execution()
op_dir = os.path.join(os.path.dirname(text.__file__), 'python/ops')

# Constrained sequence
module = tf.load_op_library(os.path.join(op_dir,
                                         '_constrained_sequence_op.so'))
export_path = os.path.join('/tmp/mdl_seq',
                           '01')
root = tf.train.Checkpoint()
root.f = tf.function(module.constrained_sequence)
to_save = root.f.get_concrete_function(tf.TensorSpec(None, tf.float32),
                                       tf.TensorSpec(None, tf.int32),
                                       tf.TensorSpec(None, tf.bool),
                                       tf.TensorSpec(None, tf.float32),
                                       False,
                                       False,
                                       False)
tf.saved_model.save(root, export_path, to_save)

# Sentence fragments
module = tf.load_op_library(os.path.join(op_dir,
                                         '_sentence_breaking_ops.so'))
export_path = os.path.join('/tmp/mdl_sent',
                           '01')
root = tf.train.Checkpoint()
root.f = tf.function(module.sentence_fragments)
to_save = root.f.get_concrete_function(tf.TensorSpec(None, tf.int64),
                                       tf.TensorSpec(None, tf.int64),
                                       tf.TensorSpec(None, tf.int64),
                                       tf.TensorSpec(None, tf.string),
                                       tf.TensorSpec(None, tf.int64),
                                       'UTF-8')
tf.saved_model.save(root, export_path, to_save)

# Unicode script tokenizer
module = tf.load_op_library(os.path.join(op_dir,
                                         '_unicode_script_tokenizer.so'))
export_path = os.path.join('/tmp/mdl_uni',
                           '01')
root = tf.train.Checkpoint()
root.f = tf.function(module.unicode_script_tokenize_with_offsets)
to_save = root.f.get_concrete_function(tf.TensorSpec(None, tf.int32),
                                       tf.TensorSpec(None, tf.int32))
tf.saved_model.save(root, export_path, to_save)

# Whitespace tokenizer
module = tf.load_op_library(os.path.join(op_dir,
                                         '_whitespace_tokenizer.so'))
export_path = os.path.join('/tmp/mdl_ws',
                           '01')
root = tf.train.Checkpoint()
root.f = tf.function(module.whitespace_tokenize_with_offsets)
to_save = root.f.get_concrete_function(tf.TensorSpec(None, tf.int32),
                                       tf.TensorSpec(None, tf.int32))
tf.saved_model.save(root, export_path, to_save)

# Wordpiece tokenizer
module = tf.load_op_library(os.path.join(op_dir,
                                         '_wordpiece_tokenizer.so'))
export_path = os.path.join('/tmp/mdl_wp',
                           '01')
root = tf.train.Checkpoint()
vocab = tf.constant(['hell', '##o', 'world'])
init = tf.lookup.KeyValueTensorInitializer(
    vocab, tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
    key_dtype=tf.string, value_dtype=tf.int64)
root.table = tf.lookup.StaticVocabularyTable(init, 1,
                                             lookup_key_dtype=tf.string)


def fn(self, tokens):
  return module.wordpiece_tokenize_with_offsets(
      tokens, self.table.resource_handle, '##', 100, True, '<UNK>')
root.f = tf.function(fn)
to_save = root.f.get_concrete_function(root, tf.TensorSpec(None, tf.string))
tf.saved_model.save(root, export_path, to_save)

