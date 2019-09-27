<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.coerce_to_structurally_valid_utf8" />
<meta itemprop="path" content="Stable" />
</div>

# text.coerce_to_structurally_valid_utf8

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/string_ops.py">View
source</a>

Coerce UTF-8 input strings to structurally valid UTF-8.

``` python
text.coerce_to_structurally_valid_utf8(
    input,
    replacement_char=_unichr(65533),
    name=None
)
```

<!-- Placeholder for "Used in" -->

Any bytes which cause the input string to be invalid UTF-8 are substituted with
the provided replacement character codepoint (default 65533). If you plan on
overriding the default, use a single byte replacement character codepoint to
preserve alignment to the source input string.

#### Args:

*   <b>`input`</b>: UTF-8 string tensor to coerce to valid UTF-8.
*   <b>`replacement_char`</b>: The replacement character to be used in place of
    any invalid byte in the input. Any valid Unicode character may be used. The
    default value is the default Unicode replacement character which is 0xFFFD
    (or U+65533). Note that passing a replacement character expressible in 1
    byte, such as ' ' or '?', will preserve string alignment to the source since
    individual invalid bytes will be replaced with a 1-byte replacement.
    (optional)
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A tensor of type string with the same shape as the input.
