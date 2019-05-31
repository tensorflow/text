# coding=utf-8
# Copyright 2019 TF.Text Authors.
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

# coding=utf-8
"""Tests for normalization ops in tensorflow_text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # tf
import tensorflow_text as text

from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_test_util


@test_util.run_all_in_graph_and_eager_modes
class NormalizeOpsTest(ragged_test_util.RaggedTensorTestCase):

  def test_lowercase_one_string(self):
    txt = [
        " TExt to loWERcase! ",
    ]
    expected = [
        " text to lowercase! ",
    ]
    self.assertAllEqual(expected, text.case_fold_utf8(txt))

  def test_lowercase_text(self):
    txt = [
        "Punctuation and digits: -*/+$#%@%$123456789#^$*%&",
        "Non-latin UTF8 chars: ΘͽʦȺЩ",
        "Accented chars: ĎÔPQRŔSŠoóôpqrŕsštťuúvwxyý",
        "Non-UTF8-letters: e.g. ◆, ♥, and the emoji symbol ( ͡° ͜ʖ ͡°)",
        "Folded: ßς",
        ""
    ]
    expected = [
        "punctuation and digits: -*/+$#%@%$123456789#^$*%&",
        "non-latin utf8 chars: θͽʦⱥщ",
        "accented chars: ďôpqrŕsšoóôpqrŕsštťuúvwxyý",
        "non-utf8-letters: e.g. ◆, ♥, and the emoji symbol ( ͡° ͜ʖ ͡°)",
        "folded: ssσ",
        ""
    ]
    self.assertAllEqual(expected, text.case_fold_utf8(txt))

  def test_lowercase_one_string_ragged(self):
    txt = ragged_factory_ops.constant([[" TExt ", "to", " loWERcase! "],
                                       [" TExt to loWERcase! "]])
    expected = [[" text ", "to", " lowercase! "], [" text to lowercase! "]]
    self.assertRaggedEqual(expected, text.case_fold_utf8(txt))

  def test_lowercase_empty_string(self):
    txt = [
        "",
    ]
    expected = [
        "",
    ]
    self.assertAllEqual(expected, text.case_fold_utf8(txt))

  def test_normalize_nfkc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"ṩ".encode("utf-8"),
    ]
    self.assertAllEqual(expected, text.normalize_utf8(txt, "NFKC"))
    self.assertAllEqual(expected, text.normalize_utf8(txt, "nfkc"))

  def test_normalize_nfkc_batch(self):
    txt = [
        u"\u1e9b\u0323",
        u"\ufb01",
    ]
    expected = [
        u"ṩ".encode("utf-8"),
        "fi",
    ]
    self.assertAllEqual(expected, text.normalize_utf8(txt, "NFKC"))
    self.assertAllEqual(expected, text.normalize_utf8(txt, "nfkc"))

  def test_normalize_nfkc_ragged(self):
    txt = ragged_factory_ops.constant([[[u"\u1e9b\u0323 \ufb01"], []],
                                       [[u"\u1e9b\u0323", u"\ufb01"]]])
    expected = [[[u"ṩ fi".encode("utf-8")], []], [[u"ṩ".encode("utf-8"), "fi"]]]
    self.assertRaggedEqual(expected, text.normalize_utf8(txt, "NFKC"))
    self.assertRaggedEqual(expected, text.normalize_utf8(txt, "nfkc"))

  def test_normalize_nfc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"\u1e9b\u0323".encode("utf-8"),
    ]
    self.assertAllEqual(expected, text.normalize_utf8(txt, "NFC"))
    self.assertAllEqual(expected, text.normalize_utf8(txt, "nfc"))

  def test_normalize_nfd(self):
    txt = [u"\u1e9b\u0323"]
    expected = [
        u"\u017f\u0323\u0307".encode("utf-8"),
    ]
    self.assertAllEqual(expected, text.normalize_utf8(txt, "NFD"))
    self.assertAllEqual(expected, text.normalize_utf8(txt, "nfd"))

  def test_normalize_nfkd(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"\u0073\u0323\u0307".encode("utf-8"),
    ]
    self.assertAllEqual(expected, text.normalize_utf8(txt, "NFKD"))
    self.assertAllEqual(expected, text.normalize_utf8(txt, "nfkd"))

  def test_unknown_normalization_form(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      bomb = text.normalize_utf8(["cant readme", "wont read me"], "cantfindme")
      self.evaluate(bomb)


if __name__ == "__main__":
  tf.test.main()
