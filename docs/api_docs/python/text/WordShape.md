<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.WordShape" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="BEGINS_WITH_OPEN_QUOTE"/>
<meta itemprop="property" content="BEGINS_WITH_PUNCT_OR_SYMBOL"/>
<meta itemprop="property" content="ENDS_WITH_CLOSE_QUOTE"/>
<meta itemprop="property" content="ENDS_WITH_ELLIPSIS"/>
<meta itemprop="property" content="ENDS_WITH_EMOTICON"/>
<meta itemprop="property" content="ENDS_WITH_MULTIPLE_SENTENCE_TERMINAL"/>
<meta itemprop="property" content="ENDS_WITH_MULTIPLE_TERMINAL_PUNCT"/>
<meta itemprop="property" content="ENDS_WITH_PUNCT_OR_SYMBOL"/>
<meta itemprop="property" content="ENDS_WITH_SENTENCE_TERMINAL"/>
<meta itemprop="property" content="ENDS_WITH_TERMINAL_PUNCT"/>
<meta itemprop="property" content="HAS_CURRENCY_SYMBOL"/>
<meta itemprop="property" content="HAS_EMOJI"/>
<meta itemprop="property" content="HAS_MATH_SYMBOL"/>
<meta itemprop="property" content="HAS_MIXED_CASE"/>
<meta itemprop="property" content="HAS_NON_LETTER"/>
<meta itemprop="property" content="HAS_NO_DIGITS"/>
<meta itemprop="property" content="HAS_NO_PUNCT_OR_SYMBOL"/>
<meta itemprop="property" content="HAS_NO_QUOTES"/>
<meta itemprop="property" content="HAS_ONLY_DIGITS"/>
<meta itemprop="property" content="HAS_PUNCTUATION_DASH"/>
<meta itemprop="property" content="HAS_QUOTE"/>
<meta itemprop="property" content="HAS_SOME_DIGITS"/>
<meta itemprop="property" content="HAS_SOME_PUNCT_OR_SYMBOL"/>
<meta itemprop="property" content="HAS_TITLE_CASE"/>
<meta itemprop="property" content="IS_ACRONYM_WITH_PERIODS"/>
<meta itemprop="property" content="IS_EMOTICON"/>
<meta itemprop="property" content="IS_LOWERCASE"/>
<meta itemprop="property" content="IS_MIXED_CASE_LETTERS"/>
<meta itemprop="property" content="IS_NUMERIC_VALUE"/>
<meta itemprop="property" content="IS_PUNCT_OR_SYMBOL"/>
<meta itemprop="property" content="IS_UPPERCASE"/>
<meta itemprop="property" content="IS_WHITESPACE"/>
</div>

# text.WordShape

## Class `WordShape`

Values for the 'pattern' arg of the WordShape op.

Defined in
[`python/ops/wordshape_ops.py`](https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordshape_ops.py).

<!-- Placeholder for "Used in" -->

The supported wordshape identifiers are:

   * `WordShape.BEGINS_WITH_OPEN_QUOTE`:
     The input begins with an open quote.

     The following strings are considered open quotes:

     ```
          "  QUOTATION MARK
          '  APOSTROPHE
          `  GRAVE ACCENT
         ``  Pair of GRAVE ACCENTs
     \uFF02  FULLWIDTH QUOTATION MARK
     \uFF07  FULLWIDTH APOSTROPHE
     \u00AB  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
     \u2018  LEFT SINGLE QUOTATION MARK
     \u201A  SINGLE LOW-9 QUOTATION MARK
     \u201B  SINGLE HIGH-REVERSED-9 QUOTATION MARK
     \u201C  LEFT DOUBLE QUOTATION MARK
     \u201E  DOUBLE LOW-9 QUOTATION MARK
     \u201F  DOUBLE HIGH-REVERSED-9 QUOTATION MARK
     \u2039  SINGLE LEFT-POINTING ANGLE QUOTATION MARK
     \u300C  LEFT CORNER BRACKET
     \u300E  LEFT WHITE CORNER BRACKET
     \u301D  REVERSED DOUBLE PRIME QUOTATION MARK
     \u2E42  DOUBLE LOW-REVERSED-9 QUOTATION MARK
     \uFF62  HALFWIDTH LEFT CORNER BRACKET
     \uFE41  PRESENTATION FORM FOR VERTICAL LEFT CORNER BRACKET
     \uFE43  PRESENTATION FORM FOR VERTICAL LEFT WHITE CORNER BRACKET
     ```

     Note: U+B4 (acute accent) not included.
     

   * `WordShape.BEGINS_WITH_PUNCT_OR_SYMBOL`:
     The input starts with a punctuation or symbol character.
     

   * `WordShape.ENDS_WITH_CLOSE_QUOTE`:
     The input ends witha closing quote character.

     The following strings are considered close quotes:

     ```
          "  QUOTATION MARK
          '  APOSTROPHE
          `  GRAVE ACCENT
         ''  Pair of APOSTROPHEs
     \uFF02  FULLWIDTH QUOTATION MARK
     \uFF07  FULLWIDTH APOSTROPHE
     \u00BB  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
     \u2019  RIGHT SINGLE QUOTATION MARK
     \u201D  RIGHT DOUBLE QUOTATION MARK
     \u203A  SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
     \u300D  RIGHT CORNER BRACKET
     \u300F  RIGHT WHITE CORNER BRACKET
     \u301E  DOUBLE PRIME QUOTATION MARK
     \u301F  LOW DOUBLE PRIME QUOTATION MARK
     \uFE42  PRESENTATION FORM FOR VERTICAL RIGHT CORNER BRACKET
     \uFE44  PRESENTATION FORM FOR VERTICAL RIGHT WHITE CORNER BRACKET
     \uFF63  HALFWIDTH RIGHT CORNER BRACKET
     ```

     Note: U+B4 (ACUTE ACCENT) is not included.
     

   * `WordShape.ENDS_WITH_ELLIPSIS`:
     The input ends with an ellipsis: i.e., with three or more
     periods, or with a unicode ellipsis character.

   * `WordShape.ENDS_WITH_EMOTICON`:
     The input ends with an emoticon.
     

   * `WordShape.ENDS_WITH_MULTIPLE_SENTENCE_TERMINAL`:
     The input ends with multiple sentence-terminal characters.
     

   * `WordShape.ENDS_WITH_MULTIPLE_TERMINAL_PUNCT`:
     The input ends with multiple terminal-punctuation characters.
     

   * `WordShape.ENDS_WITH_PUNCT_OR_SYMBOL`:
     The input ends with a punctuation or symbol character.
     

   * `WordShape.ENDS_WITH_SENTENCE_TERMINAL`:
     The input ends with a sentence-terminal character.
     

   * `WordShape.ENDS_WITH_TERMINAL_PUNCT`:
     The input ends with a terminal-punctuation character.
     

   * `WordShape.HAS_CURRENCY_SYMBOL`:
     The input contains a currency symbol.
     

   * `WordShape.HAS_EMOJI`:
     The input contains an emoji character.

     See http://www.unicode.org/Public/emoji/1.0//emoji-data.txt.
     Emojis are in unicode ranges `2600-26FF`, `1F300-1F6FF`, `1F900-1F9FF`
     

   * `WordShape.HAS_MATH_SYMBOL`:
     The input contains a mathematical symbol.
     

   * `WordShape.HAS_MIXED_CASE`:
     The input contains both uppercase and lowercase letterforms.
     

   * `WordShape.HAS_NON_LETTER`:
     The input contains a non-letter character.
     

   * `WordShape.HAS_NO_DIGITS`:
     The input contains no digit characters.
     

   * `WordShape.HAS_NO_PUNCT_OR_SYMBOL`:
     The input contains no unicode punctuation or symbol characters.
     

   * `WordShape.HAS_NO_QUOTES`:
     The input string contains no quote characters.
     

   * `WordShape.HAS_ONLY_DIGITS`:
     The input consists entirely of unicode digit characters.
     

   * `WordShape.HAS_PUNCTUATION_DASH`:
     The input contains at least one unicode dash character.

     Note that this is similar to HAS_ANY_HYPHEN, but uses the Pd (Dash)
     unicode property. (This property will not match to soft-hyphens and
     katakana middle dot characters.)
     

   * `WordShape.HAS_QUOTE`:
     The input starts or ends with a unicode quotation mark.
     

   * `WordShape.HAS_SOME_DIGITS`:
     The input contains a mix of digit characters and non-digit
     characters.
     

   * `WordShape.HAS_SOME_PUNCT_OR_SYMBOL`:
     The input contains a mix of punctuation or symbol characters,
     and non-punctuation non-symbol characters.
     

   * `WordShape.HAS_TITLE_CASE`:
     The input has title case.  I.e., the first character is upper case
     or title case, and the remaining characters are lowercase.
     

   * `WordShape.IS_ACRONYM_WITH_PERIODS`:
     The input is a period-separated acronym.
     This matches for strings of the form "I.B.M." but not "IBM".
     

   * `WordShape.IS_EMOTICON`:
     The input is a single emoticon.
     

   * `WordShape.IS_LOWERCASE`:
     The input contains only lowercase letterforms.
     

   * `WordShape.IS_MIXED_CASE_LETTERS`:
     The input contains only uppercase and lowercase letterforms.
     

   * `WordShape.IS_NUMERIC_VALUE`:
     The input is parseable as a numeric value.  This will match a
     fairly broad set of floating point and integer representations (but
     not Nan or Inf).
     

   * `WordShape.IS_PUNCT_OR_SYMBOL`:
     The input contains only punctuation and symbol characters.
     

   * `WordShape.IS_UPPERCASE`:
     The input contains only uppercase letterforms.
     

   * `WordShape.IS_WHITESPACE`:
     The input consists entirely of whitespace.
     

## Class Members

<h3 id="BEGINS_WITH_OPEN_QUOTE"><code>BEGINS_WITH_OPEN_QUOTE</code></h3>

<h3 id="BEGINS_WITH_PUNCT_OR_SYMBOL"><code>BEGINS_WITH_PUNCT_OR_SYMBOL</code></h3>

<h3 id="ENDS_WITH_CLOSE_QUOTE"><code>ENDS_WITH_CLOSE_QUOTE</code></h3>

<h3 id="ENDS_WITH_ELLIPSIS"><code>ENDS_WITH_ELLIPSIS</code></h3>

<h3 id="ENDS_WITH_EMOTICON"><code>ENDS_WITH_EMOTICON</code></h3>

<h3 id="ENDS_WITH_MULTIPLE_SENTENCE_TERMINAL"><code>ENDS_WITH_MULTIPLE_SENTENCE_TERMINAL</code></h3>

<h3 id="ENDS_WITH_MULTIPLE_TERMINAL_PUNCT"><code>ENDS_WITH_MULTIPLE_TERMINAL_PUNCT</code></h3>

<h3 id="ENDS_WITH_PUNCT_OR_SYMBOL"><code>ENDS_WITH_PUNCT_OR_SYMBOL</code></h3>

<h3 id="ENDS_WITH_SENTENCE_TERMINAL"><code>ENDS_WITH_SENTENCE_TERMINAL</code></h3>

<h3 id="ENDS_WITH_TERMINAL_PUNCT"><code>ENDS_WITH_TERMINAL_PUNCT</code></h3>

<h3 id="HAS_CURRENCY_SYMBOL"><code>HAS_CURRENCY_SYMBOL</code></h3>

<h3 id="HAS_EMOJI"><code>HAS_EMOJI</code></h3>

<h3 id="HAS_MATH_SYMBOL"><code>HAS_MATH_SYMBOL</code></h3>

<h3 id="HAS_MIXED_CASE"><code>HAS_MIXED_CASE</code></h3>

<h3 id="HAS_NON_LETTER"><code>HAS_NON_LETTER</code></h3>

<h3 id="HAS_NO_DIGITS"><code>HAS_NO_DIGITS</code></h3>

<h3 id="HAS_NO_PUNCT_OR_SYMBOL"><code>HAS_NO_PUNCT_OR_SYMBOL</code></h3>

<h3 id="HAS_NO_QUOTES"><code>HAS_NO_QUOTES</code></h3>

<h3 id="HAS_ONLY_DIGITS"><code>HAS_ONLY_DIGITS</code></h3>

<h3 id="HAS_PUNCTUATION_DASH"><code>HAS_PUNCTUATION_DASH</code></h3>

<h3 id="HAS_QUOTE"><code>HAS_QUOTE</code></h3>

<h3 id="HAS_SOME_DIGITS"><code>HAS_SOME_DIGITS</code></h3>

<h3 id="HAS_SOME_PUNCT_OR_SYMBOL"><code>HAS_SOME_PUNCT_OR_SYMBOL</code></h3>

<h3 id="HAS_TITLE_CASE"><code>HAS_TITLE_CASE</code></h3>

<h3 id="IS_ACRONYM_WITH_PERIODS"><code>IS_ACRONYM_WITH_PERIODS</code></h3>

<h3 id="IS_EMOTICON"><code>IS_EMOTICON</code></h3>

<h3 id="IS_LOWERCASE"><code>IS_LOWERCASE</code></h3>

<h3 id="IS_MIXED_CASE_LETTERS"><code>IS_MIXED_CASE_LETTERS</code></h3>

<h3 id="IS_NUMERIC_VALUE"><code>IS_NUMERIC_VALUE</code></h3>

<h3 id="IS_PUNCT_OR_SYMBOL"><code>IS_PUNCT_OR_SYMBOL</code></h3>

<h3 id="IS_UPPERCASE"><code>IS_UPPERCASE</code></h3>

<h3 id="IS_WHITESPACE"><code>IS_WHITESPACE</code></h3>

