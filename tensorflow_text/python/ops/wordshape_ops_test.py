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

"""Tests for wordshape ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow_text.python.ops import wordshape_ops


@test_util.run_all_in_graph_and_eager_modes
class Utf8CharsOpTest(test.TestCase):

  def testDashShape(self):
    test_string = [u"a-b", u"a\u2010b".encode("utf-8"),
                   u"a\u2013b".encode("utf-8"), u"a\u2e3ab".encode("utf-8"),
                   u"abc".encode("utf-8")]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.HAS_PUNCTUATION_DASH)
    self.assertAllEqual(shapes, [True, True, True, True, False])

  def testNoDigits(self):
    test_string = [u"abc", u"a\u06f3m".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_NO_DIGITS)
    self.assertAllEqual(shapes, [True, False])

  def testSomeDigits(self):
    test_string = [u"abc", u"a\u06f3m".encode("utf-8"),
                   u"90\u06f3".encode("utf-8"),
                   u"a9b8c7", u"9ab87c", u"\u06f3m\u06f3"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_SOME_DIGITS)
    self.assertAllEqual(shapes, [False, True, False, True, True, True])

  def testSomeDigitAndCurrency(self):
    test_string = [u"abc", u"a\u06f3m".encode("utf-8"),
                   u"90\u06f3".encode("utf-8"),
                   u"a9b8c7", u"$9ab87c$", u"\u06f3m\u06f3"]
    pattern_list = [wordshape_ops.WordShape.HAS_SOME_DIGITS,
                    wordshape_ops.WordShape.HAS_CURRENCY_SYMBOL]
    shapes = wordshape_ops.wordshape(test_string,
                                     pattern=pattern_list)
    self.assertAllEqual(shapes, [[False, False],
                                 [True, False],
                                 [False, False],
                                 [True, False],
                                 [True, True],
                                 [True, False]])

  def testOnlyDigits(self):
    test_string = [u"abc", u"a9b".encode("utf-8"), u"90\u06f3".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_ONLY_DIGITS)
    self.assertAllEqual(shapes, [False, False, True])

  def testNumericValue(self):
    test_string = [u"98.6", u"-0.3", u"2.783E4", u"e4", u"1e10"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.IS_NUMERIC_VALUE)
    self.assertAllEqual(shapes, [True, True, True, False, True])

  def SKIP_testWhitespace(self):
    test_string = [u" ", u"\v", u"\r\n", u"\u3000".encode("utf-8"),
                   u" a", u"abc", u"a\nb", u"\u3000 \n".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.IS_WHITESPACE)
    self.assertAllEqual(shapes,
                        [True, True, True, True, False, False, False, True])

  def testNoPunct(self):
    test_string = [u"abc", u"a;m".encode("utf-8")]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.HAS_NO_PUNCT_OR_SYMBOL)
    self.assertAllEqual(shapes, [True, False])

  def testSomePunct(self):
    test_string = [u"abc", u"a;m".encode("utf-8"),
                   u".,!".encode("utf-8"),
                   u"a@b.c,", u".ab8;c", u"\u0f08m\u0f08"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
    self.assertAllEqual(shapes, [False, True, False, True, True, True])

  def testAllPunct(self):
    test_string = [u"abc", u"a;b".encode("utf-8"), u";,\u0f08".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.IS_PUNCT_OR_SYMBOL)
    self.assertAllEqual(shapes, [False, False, True])

  def testLeadingPunct(self):
    test_string = [u"abc", u";b", u"b;", u";,\u0f08".encode("utf-8")]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.BEGINS_WITH_PUNCT_OR_SYMBOL)
    self.assertAllEqual(shapes, [False, True, False, True])

  def testTrailingPunct(self):
    test_string = [u"abc", u";b", u"b;", u";,\u0f08".encode("utf-8")]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.ENDS_WITH_PUNCT_OR_SYMBOL)
    self.assertAllEqual(shapes, [False, False, True, True])

  def SKIP_testSentenceTerminal(self):
    test_string = [u"abc", u".b", u"b.", u"b,", u"b!!!", u"abc?!"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.ENDS_WITH_SENTENCE_TERMINAL)
    self.assertAllEqual(shapes, [False, False, True, False, True, True])

  def SKIP_testMultipleSentenceTerminal(self):
    test_string = [u"abc", u".b", u"b.", u"b,", u"b!!!", u"abc?!"]
    shapes = wordshape_ops.wordshape(
        test_string,
        wordshape_ops.WordShape.ENDS_WITH_MULTIPLE_SENTENCE_TERMINAL)
    self.assertAllEqual(shapes, [False, False, False, False, True, True])

  def SKIP_testTerminalPunct(self):
    test_string = [u"abc", u".b", u"b.", u"b,", u"b!!!", u"abc?!"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.ENDS_WITH_TERMINAL_PUNCT)
    self.assertAllEqual(shapes, [False, False, True, True, True, True])

  def SKIP_testMultipleTerminalPunct(self):
    test_string = [u"abc", u".b", u"b.", u"b,,", u"b!!!", u"abc?!"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.ENDS_WITH_MULTIPLE_TERMINAL_PUNCT)
    self.assertAllEqual(shapes, [False, False, False, True, True, True])

  def testEllipsis(self):
    test_string = [u"abc", u"abc...", u"...abc", u"abc\u2026".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.ENDS_WITH_ELLIPSIS)
    self.assertAllEqual(shapes, [False, True, False, True])

  def testEndsWithEmoticon(self):
    test_string = [u"abc", u":-)", u"O:)", u"8)x", u":\u3063C", u"abc:-)"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.ENDS_WITH_EMOTICON)
    self.assertAllEqual(shapes, [False, True, True, False, True, True])

  def testIsEmoticon(self):
    test_string = [u"abc", u":-)", u"O:)", u"8)x", u":\u3063C", u"abc:-)"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.IS_EMOTICON)
    self.assertAllEqual(shapes, [False, True, False, False, True, False])

  def testEmoji(self):
    test_string = [u"\U0001f604m".encode("utf-8"), u"m\u2605m".encode("utf-8"),
                   u"O:)", u"m\U0001f604".encode("utf-8"),
                   u"\u2105k".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_EMOJI)
    self.assertAllEqual(shapes, [True, True, False, True, False])

  def testAcronym(self):
    test_string = [u"abc", u"A.B.", u"A.B.C.)", u"ABC"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.IS_ACRONYM_WITH_PERIODS)
    self.assertAllEqual(shapes, [False, True, False, False])

  def testAllUppercase(self):
    test_string = [u"abc", u"ABc", u"ABC"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.IS_UPPERCASE)
    self.assertAllEqual(shapes, [False, False, True])

  def testAllLowercase(self):
    test_string = [u"abc", u"ABc", u"ABC"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.IS_LOWERCASE)
    self.assertAllEqual(shapes, [True, False, False])

  def testMixedCase(self):
    test_string = [u"abc", u"ABc", u"ABC", u"abC"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_MIXED_CASE)
    self.assertAllEqual(shapes, [False, True, False, True])

  def testMixedCaseLetters(self):
    test_string = [u"abc", u"ABc", u"ABC", u"abC", u"abC."]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.IS_MIXED_CASE_LETTERS)
    self.assertAllEqual(shapes, [False, True, False, True, False])

  def testTitleCase(self):
    test_string = [u"abc", u"ABc", u"ABC", u"Abc", u"aBcd",
                   u"\u01c8bc".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_TITLE_CASE)
    self.assertAllEqual(shapes, [False, False, False, True, False, True])

  def SKIP_testNoQuotes(self):
    test_string = [u"abc", u"\"ABc", u"ABC'", u"Abc\u201c".encode("utf-8"),
                   u"aBcd"]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_NO_QUOTES)
    self.assertAllEqual(shapes, [True, False, False, False, True])

  def testOpenQuote(self):
    test_string = [u"''", u"ABc\"", u"\uff07".encode("utf-8"),
                   u"\u2018".encode("utf-8"), u"aBcd", u"``"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.BEGINS_WITH_OPEN_QUOTE)
    self.assertAllEqual(shapes, [False, False, True, True, False, True])

  def testCloseQuote(self):
    test_string = [u"''", u"ABc\"", u"\u300f".encode("utf-8"),
                   u"\u2018".encode("utf-8"), u"aBcd", u"``"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.ENDS_WITH_CLOSE_QUOTE)
    self.assertAllEqual(shapes, [True, True, True, False, False, False])

  def SKIP_testQuote(self):
    test_string = [u"''", u"ABc\"", u"\uff07".encode("utf-8"),
                   u"\u2018".encode("utf-8"), u"aBcd", u"``",
                   u"\u300d".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_QUOTE)
    self.assertAllEqual(shapes, [True, True, True, True, False, True, True])

  def testMathSymbol(self):
    test_string = [u"''", u"\u003c", u"\uff07".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_MATH_SYMBOL)
    self.assertAllEqual(shapes, [False, True, False])

  def testCurrencySymbol(self):
    test_string = [u"''", u"ABc$", u"$\uff07".encode("utf-8")]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.HAS_CURRENCY_SYMBOL)
    self.assertAllEqual(shapes, [False, True, True])

  def testCurrencySymbolAtBeginning(self):
    test_string = [u"''", u"ABc$", u"$ABc", u"A$Bc"]
    shapes = wordshape_ops.wordshape(
        test_string, wordshape_ops.WordShape.HAS_CURRENCY_SYMBOL)
    self.assertAllEqual(shapes, [False, True, True, True])

  def testNonLetters(self):
    test_string = [u"''", u"ABc", u"\uff07".encode("utf-8"),
                   u"\u2018".encode("utf-8"), u"aBcd", u"`#ab",
                   u"\u300d".encode("utf-8")]
    shapes = wordshape_ops.wordshape(test_string,
                                     wordshape_ops.WordShape.HAS_NON_LETTER)
    self.assertAllEqual(shapes, [True, False, True, True, False, True, True])

  def testMultipleShapes(self):
    test_string = [u"abc", u"ABc", u"ABC"]
    shapes = wordshape_ops.wordshape(test_string, [
        wordshape_ops.WordShape.IS_UPPERCASE,
        wordshape_ops.WordShape.IS_LOWERCASE
    ])
    self.assertAllEqual(shapes, [[False, True], [False, False], [True, False]])

  def testNonShapePassedToShapeArg(self):
    test_string = [u"abc", u"ABc", u"ABC"]
    with self.assertRaises(TypeError):
      wordshape_ops.wordshape(test_string, "This is not a Shape")


if __name__ == "__main__":
  test.main()
