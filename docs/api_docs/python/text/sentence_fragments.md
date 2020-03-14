<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.sentence_fragments" />
<meta itemprop="path" content="Stable" />
</div>

# text.sentence_fragments

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentence_breaking_ops.py">View
source</a>

Find the sentence fragments in a given text.

```python
text.sentence_fragments(
    token_word, token_starts, token_ends, token_properties, input_encoding='UTF-8',
    errors='replace', replacement_char=65533, replace_control_characters=False
)
```

<!-- Placeholder for "Used in" -->

A sentence fragment is a potential next sentence determined using
deterministic heuristics based on punctuation, capitalization, and similar
text attributes.

#### Args:

*   <b>`token_word`</b>: A Tensor (w/ rank=2) or a RaggedTensor (w/
    ragged_rank=1) containing the token strings.
*   <b>`token_starts`</b>: A Tensor (w/ rank=2) or a RaggedTensor (w/
    ragged_rank=1) containing offsets where the token starts.
*   <b>`token_ends`</b>: A Tensor (w/ rank=2) or a RaggedTensor (w/
    ragged_rank=1) containing offsets where the token ends.
*   <b>`token_properties`</b>: A Tensor (w/ rank=2) or a RaggedTensor (w/
    ragged_rank=1) containing a bitmask.

    The values of the bitmask are:

    *   0x01 (ILL_FORMED) - Text is ill-formed according to TextExtractor;
        typically applies to all tokens of a paragraph that is too short or
        lacks terminal punctuation. 0x40 (TITLE)
    *   0x02 (HEADING)
    *   0x04 (BOLD)
    *   0x10 (UNDERLINED)
    *   0x20 (LIST)
    *   0x80 (EMOTICON)
    *   0x100 (ACRONYM) - Token was identified by Lexer as an acronym. Lexer
        identifies period-, hyphen-, and space-separated acronyms: "U.S.",
        "U-S", and "U S". Lexer normalizes all three to "US", but the token word
        field normalizes only space-separated acronyms.
    *   0x200 (HYPERLINK) - Indicates that the token (or part of the token) is a
        covered by at least one hyperlink. More information of the hyperlink is
        stored in the first token covered by the hyperlink.

*   <b>`input_encoding`</b>: String name for the unicode encoding that should be
    used to decode each string.

*   <b>`errors`</b>: Specifies the response when an input string can't be
    converted using the indicated encoding. One of:

    *   `'strict'`: Raise an exception for any illegal substrings.
    *   `'replace'`: Replace illegal substrings with `replacement_char`.
    *   `'ignore'`: Skip illegal substrings.

*   <b>`replacement_char`</b>: The replacement codepoint to be used in place of
    invalid substrings in `input` when `errors='replace'`; and in place of C0
    control characters in `input` when `replace_control_characters=True`.

*   <b>`replace_control_characters`</b>: Whether to replace the C0 control
    characters `(U+0000 - U+001F)` with the `replacement_char`.

#### Returns:

A RaggedTensor of `fragment_start`, `fragment_end`, `fragment_properties`
and `terminal_punc_token`.

`fragment_properties` is an int32 bitmask whose values may contain:

*   1 = fragment ends with terminal punctuation
*   2 = fragment ends with multiple terminal punctuations (e.g. "She said
    what?!")
*   3 = Has close parenthesis (e.g. "Mushrooms (they're fungi).")
*   4 = Has sentential close parenthesis (e.g. "(Mushrooms are fungi!)")

    `terminal_punc_token` is a RaggedTensor containing the index of terminal
    punctuation token immediately following the last word in the fragment -- or
    index of the last word itself, if it's an acronym (since acronyms include
    the terminal punctuation). index of the terminal punctuation token.
