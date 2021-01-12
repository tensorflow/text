# coding=utf-8
# Copyright 2021 TF.Text Authors.
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

# Lint as: python3
"""Integration tests for TF.Text ops in model server."""

import os
import shutil
import tempfile

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_text as text

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dest',
    ('third_party/tensorflow_serving/servables/tensorflow/testdata/'
     'tf_text_regression/01'),
    'Destination directory for the model.')


class TfTextOps(tf.Module):
  """Module for saving TF Text concrete function."""

  @tf.function
  def __call__(self, x):
    # Constrained sequence
    cs_scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    cs_input = np.array([cs_scores, cs_scores, cs_scores], dtype=np.float32)
    cs_transition_weights = np.array([[-1.0, 1.0, -2.0, 2.0, 0.0],
                                      [3.0, -3.0, 4.0, -4.0, 0.0],
                                      [5.0, 1.0, 10.0, 1.0, 1.0],
                                      [-7.0, 7.0, -8.0, 8.0, 0.0],
                                      [0.0, 1.0, 2.0, 3.0, 0.0]],
                                     dtype=np.float32)
    cs_allowed_transitions = np.array([[True, True, True, True, True],
                                       [True, True, True, True, True],
                                       [True, False, True, False, False],
                                       [True, True, True, True, True],
                                       [True, False, True, True, True]])
    constrained_sequence = text.viterbi_constrained_sequence(
        cs_input, [2, 2, 2], allowed_transitions=cs_allowed_transitions,
        transition_weights=cs_transition_weights, use_log_space=True,
        use_start_and_end_states=True)
    # Max Spanning Tree
    mst_num_nodes = tf.constant([4, 3], tf.int32)
    mst_scores = tf.constant([[[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [1, 2, 0, 0],
                               [1, 2, 3, 4]],
                              [[4, 3, 2, 9],
                               [0, 0, 2, 9],
                               [0, 0, 0, 9],
                               [9, 9, 9, 9]]],
                             tf.int32)  # pyformat: disable
    (max_spanning_tree, _) = text.max_spanning_tree(mst_num_nodes, mst_scores)
    # Normalize
    normalized = text.case_fold_utf8(['A String'])
    normalized = text.normalize_utf8(normalized)
    # Regex split
    regex_split = text.regex_split(input=['Yo dawg!'],
                                   delim_regex_pattern=r'\s')
    # Rouge-L
    rl_hypotheses = tf.ragged.constant(
        [['captain', 'of', 'the', 'delta', 'flight'],
         ['the', '1990', 'transcript']])
    rl_references = tf.ragged.constant(
        [['delta', 'air', 'lines', 'flight'],
         ['this', 'concludes', 'the', 'transcript']])
    (rouge_l, _, _) = text.metrics.rouge_l(rl_hypotheses, rl_references)
    # Sentence breaking version 1 (token dependent)
    sb_token_word = [['Welcome', 'to', 'the', 'U.S.', '!', 'Harry'],
                     ['Wu', 'Tang', 'Clan', ';', 'ain\'t', 'nothing']]
    sb_token_properties = [[0, 0, 0, 256, 0, 0], [0, 0, 0, 0, 0, 0]]
    sb_token_starts = []
    sb_token_ends = []
    for sentence in sb_token_word:
      sentence_string = ''
      sentence_start = []
      sentence_end = []
      for word in sentence:
        sentence_start.append(len(sentence_string))
        sentence_string = sentence_string.join([word, ' '])
        sentence_end.append(len(sentence_string))
      sb_token_starts.append(sentence_start)
      sb_token_ends.append(sentence_end)
    sb_token_starts = tf.constant(sb_token_starts, dtype=tf.int64)
    sb_token_ends = tf.constant(sb_token_ends, dtype=tf.int64)
    sb_token_properties = tf.ragged.constant(sb_token_properties,
                                             dtype=tf.int64)
    (sentence_breaking, _, _, _) = text.sentence_fragments(
        sb_token_word, sb_token_starts, sb_token_ends, sb_token_properties)
    # Sentence breaking version 2 (StateBasedSentenceBreaker)
    sbv2_text_input = [['Welcome to the U.S.! Harry'],
                       ['Wu Tang Clan; ain\'t nothing']]
    sentence_breaker_v2 = text.StateBasedSentenceBreaker()
    sbv2_fragment_text, _, _ = (
        sentence_breaker_v2.break_sentences_with_offsets(sbv2_text_input))
    # Sentencepiece tokenizer
    sp_model_file = (
        'third_party/tensorflow_text/python/ops/test_data/test_oss_model.model')
    sp_model = open(sp_model_file, 'rb').read()
    sp_tokenizer = text.SentencepieceTokenizer(sp_model)
    sentencepiece = sp_tokenizer.tokenize(['A sentence of things.'])
    sentencepiece = sp_tokenizer.detokenize(sentencepiece)
    (sentencepiece, _, _) = sp_tokenizer.tokenize_with_offsets(sentencepiece)
    sentencepiece_size = sp_tokenizer.vocab_size()
    sentencepiece_id = sp_tokenizer.id_to_string(1)
    sentencepiece_str = sp_tokenizer.string_to_id('<s>')
    # Split merge tokenizer
    sm_tokenizer = text.SplitMergeTokenizer()
    split_merge = sm_tokenizer.tokenize(b'IloveFlume!',
                                        [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    # Split merge from logits tokenizer
    smfl_tokenizer = text.SplitMergeFromLogitsTokenizer()
    split_merge_from_logits = smfl_tokenizer.tokenize(
        b'IloveFlume!',
        # One pair of logits for each Unicode character from the text.  Each
        # pair indicates a "split" action if the first component is greater than
        # the second one, and a "merge" otherwise.
        [
            [2.7, -0.3],  # I: split
            [4.1, 0.82],  # l: split
            [-2.3, 4.3],  # o: merge
            [3.1, 12.2],  # v: merge
            [-3.0, 4.7],  # e: merge
            [2.7, -0.7],  # F: split
            [0.7, 15.0],  # l: merge
            [1.6, 23.0],  # u: merge
            [2.1, 11.0],  # m: merge
            [0.0, 20.0],  # e: merge
            [18.0, 0.7],  # !: split
        ])
    # Confirm TF unicode_script op that requires ICU works
    tf_unicode_script = tf.strings.unicode_script(
        [ord('a'), 0x0411, 0x82b8, ord(',')])
    # Unicode script tokenizer
    us_tokenizer = text.UnicodeScriptTokenizer()
    unicode_script = us_tokenizer.tokenize(['a string'])
    # Whitespace tokenizer
    ws_tokenizer = text.WhitespaceTokenizer()
    whitespace = ws_tokenizer.tokenize(['a string'])
    # Wordpiece tokenizer
    wp_initializer = tf.lookup.KeyValueTensorInitializer(
        ['i'], [1], key_dtype=tf.string, value_dtype=tf.int64)
    self.wp_vocab_table = tf.lookup.StaticHashTable(wp_initializer,
                                                    default_value=-1)
    wp_tokenizer = text.WordpieceTokenizer(self.wp_vocab_table)
    wordpiece = wp_tokenizer.tokenize(['i am'])
    # Wordshape
    wordshapes = text.wordshape([u'a-b', u'a\u2010b'.encode('utf-8')],
                                text.WordShape.HAS_PUNCTUATION_DASH)

    # Assertion method
    def assert_check(tensor):
      return tf.assert_equal(tensor, tf.identity(tensor))

    # Assertions
    constrained_sequence_assert = assert_check(constrained_sequence.to_tensor())
    max_spanning_tree_assert = assert_check(max_spanning_tree)
    normalized_assert = assert_check(normalized)
    regex_split_assert = assert_check(regex_split.to_tensor())
    rouge_l_assert = assert_check(rouge_l)
    sentence_breaking_assert = assert_check(sentence_breaking.to_tensor())
    sentence_breaking_v2_assert = assert_check(sbv2_fragment_text.to_tensor())
    sentencepiece_assert = assert_check(sentencepiece.to_tensor())
    sentencepiece_id_assert = assert_check(sentencepiece_id)
    sentencepiece_size_assert = assert_check(sentencepiece_size)
    sentencepiece_str_assert = assert_check(sentencepiece_str)
    split_merge_assert = assert_check(split_merge)
    split_merge_from_logits_assert = assert_check(split_merge_from_logits)
    tf_unicode_script_assert = assert_check(tf_unicode_script)
    unicode_script_assert = assert_check(unicode_script.to_tensor())
    whitespace_assert = assert_check(whitespace.to_tensor())
    wordpiece_assert = assert_check(wordpiece.to_tensor())
    wordshapes_assert = assert_check(wordshapes)

    with tf.control_dependencies([constrained_sequence_assert,
                                  max_spanning_tree_assert,
                                  normalized_assert,
                                  regex_split_assert,
                                  rouge_l_assert,
                                  sentence_breaking_assert,
                                  sentence_breaking_v2_assert,
                                  sentencepiece_assert,
                                  sentencepiece_id_assert,
                                  sentencepiece_size_assert,
                                  sentencepiece_str_assert,
                                  split_merge_assert,
                                  split_merge_from_logits_assert,
                                  tf_unicode_script_assert,
                                  unicode_script_assert,
                                  whitespace_assert,
                                  wordpiece_assert,
                                  wordshapes_assert]):
      y = tf.add(x, [1])
    return {'y': y}


module = TfTextOps()
export_path = tempfile.TemporaryDirectory()
print('Exporting saved model to ', export_path)
call = module.__call__.get_concrete_function(
    tf.TensorSpec([1], tf.float32, 'x'))
tf.saved_model.save(module, export_path.name, call)

# Copy files from temp directory
print('Moving files:')
for src_dir, dirs, files in os.walk(export_path.name):
  dst_dir = src_dir.replace(export_path.name, FLAGS.dest, 1)
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
  for file_ in files:
    print(file_)
    src_file = os.path.join(src_dir, file_)
    dst_file = os.path.join(dst_dir, file_)
    if os.path.exists(dst_file):
      # in case of the src and dst are the same file
      if os.path.samefile(src_file, dst_file):
        continue
      os.remove(dst_file)
    shutil.move(src_file, dst_dir)
