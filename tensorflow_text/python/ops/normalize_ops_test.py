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

# coding=utf-8
"""Tests for normalization ops in tensorflow_text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import normalize_ops


def _Utf8(char):
  return char.encode("utf-8")


@test_util.run_all_in_graph_and_eager_modes
class NormalizeOpsTest(test.TestCase):

  def test_lowercase_one_string(self):
    txt = [
        " TExt to loWERcase! ",
    ]
    expected = [
        b" text to lowercase! ",
    ]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_lowercase_text(self):
    txt = [
        "Punctuation and digits: -*/+$#%@%$123456789#^$*%&",
        "Non-latin UTF8 chars: ΘͽʦȺЩ",
        "Accented chars: ĎÔPQRŔSŠoóôpqrŕsštťuúvwxyý",
        "Non-UTF8-letters: e.g. ◆, ♥, and the emoji symbol ( ͡° ͜ʖ ͡°)",
        "Folded: ßς", ""
    ]
    expected = [
        _Utf8(u"punctuation and digits: -*/+$#%@%$123456789#^$*%&"),
        _Utf8(u"non-latin utf8 chars: θͽʦⱥщ"),
        _Utf8(u"accented chars: ďôpqrŕsšoóôpqrŕsštťuúvwxyý"),
        _Utf8(
            u"non-utf8-letters: e.g. ◆, ♥, and the emoji symbol ( ͡° ͜ʖ ͡°)"
        ),
        _Utf8(u"folded: ssσ"), b""
    ]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_lowercase_one_string_ragged(self):
    txt = ragged_factory_ops.constant([[" TExt ", "to", " loWERcase! "],
                                       [" TExt to loWERcase! "]])
    expected = [[b" text ", b"to", b" lowercase! "], [b" text to lowercase! "]]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_lowercase_empty_string(self):
    txt = [
        "",
    ]
    expected = [
        b"",
    ]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_normalize_nfkc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"ṩ".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFKC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfkc"))

  def test_normalize_nfkc_batch(self):
    txt = [
        u"\u1e9b\u0323",
        u"\ufb01",
    ]
    expected = [
        b"\xe1\xb9\xa9",
        b"fi",
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, u"NFKC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, u"nfkc"))

  def test_normalize_nfkc_ragged(self):
    txt = ragged_factory_ops.constant([[[u"\u1e9b\u0323 \ufb01"], []],
                                       [[u"\u1e9b\u0323", u"\ufb01"]]])
    expected = [[[u"ṩ fi".encode("utf-8")], []],
                [[u"ṩ".encode("utf-8"), b"fi"]]]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFKC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfkc"))

  def test_normalize_nfc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"\u1e9b\u0323".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfc"))

  def test_normalize_nfd(self):
    txt = [u"\u1e9b\u0323"]
    expected = [
        u"\u017f\u0323\u0307".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFD"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfd"))

  def test_normalize_nfkd(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"\u0073\u0323\u0307".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFKD"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfkd"))

  def test_unknown_normalization_form(self):
    with self.assertRaises(errors.InvalidArgumentError):
      bomb = normalize_ops.normalize_utf8(["cant readme", "wont read me"],
                                          "cantfindme")
      self.evaluate(bomb)


@test_util.run_all_in_graph_and_eager_modes
class NormalizeWithOffsetsOpsTest(test.TestCase):

  def test_multiple_edits_can_be_traced(self):
    pass

  def test_normalize_nfkc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"ṩ".encode("utf-8"),
    ]
    actual, _ = normalize_ops.normalize_utf8_with_offsets(txt, "NFKC")
    self.assertAllEqual(expected, actual)
    actual, _ = normalize_ops.normalize_utf8_with_offsets(txt, "nfkc")
    self.assertAllEqual(expected, actual)

  def test_normalize_nfkc_batch(self):
    txt = [
        u"\u1e9b\u0323",
        u"\ufb01",
    ]
    expected = [
        b"\xe1\xb9\xa9",
        b"fi",
    ]
    actual, _ = normalize_ops.normalize_utf8_with_offsets(txt, u"NFKC")
    self.assertAllEqual(expected, actual)
    actual, _ = normalize_ops.normalize_utf8_with_offsets(txt, u"nfkc")
    self.assertAllEqual(expected, actual)

  def test_unaccepted_normalization_form(self):
    with self.assertRaises(errors.InvalidArgumentError):
      bomb = normalize_ops.normalize_utf8_with_offsets(
          ["cant readme", "wont read me"], "NFKD")
      self.evaluate(bomb)


@test_util.run_all_in_graph_and_eager_modes
class FindSourceOffsetsTest(test.TestCase):

  def test_one_string(self):
    txt = [
        "株式会社ＫＡＤＯＫＡＷＡ",
    ]
    actual, handle = normalize_ops.normalize_utf8_with_offsets(txt, u"NFKC")
    self.assertAllEqual(
        actual, [b"\xe6\xa0\xaa\xe5\xbc\x8f\xe4\xbc\x9a\xe7\xa4\xbeKADOKAWA"])

    input_starts = [[0, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]]
    input_limits = [[6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
    starts, limits = normalize_ops.find_source_offsets(handle, input_starts,
                                                       input_limits)
    expected_starts = [[0, 6, 12, 15, 18, 21, 24, 27, 30, 33, 36, 36]]
    expected_limits = [[6, 12, 15, 18, 21, 24, 27, 30, 33, 36, 36, 36]]
    self.assertAllEqual(expected_starts, starts)
    self.assertAllEqual(expected_limits, limits)

  def test_multiple_strings_ragged(self):
    txt = [
        "株式会社ＫＡＤＯＫＡＷＡ",
        "ＫＡＤＯＫＡＷＡ",
    ]
    actual, handle = normalize_ops.normalize_utf8_with_offsets(txt, u"NFKC")
    self.assertAllEqual(actual, [
        b"\xe6\xa0\xaa\xe5\xbc\x8f\xe4\xbc\x9a\xe7\xa4\xbeKADOKAWA",
        b"KADOKAWA",
    ])

    input_starts = ragged_factory_ops.constant([[0, 12], [0, 2, 4]])
    input_limits = ragged_factory_ops.constant([[12, 20], [2, 4, 8]])
    starts, limits = normalize_ops.find_source_offsets(handle, input_starts,
                                                       input_limits)
    expected_starts = [[0, 12], [0, 6, 12]]
    expected_limits = [[12, 36], [6, 12, 24]]
    self.assertAllEqual(expected_starts, starts)
    self.assertAllEqual(expected_limits, limits)


if __name__ == "__main__":
  test.main()
