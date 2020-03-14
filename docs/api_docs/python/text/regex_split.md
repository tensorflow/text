<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.regex_split" />
<meta itemprop="path" content="Stable" />
</div>

# text.regex_split

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/regex_split_ops.py">View source</a>



Split `input` by delimiters that match a regex pattern.

```python
text.regex_split(
    input, delim_regex_pattern, keep_delim_regex_pattern='', name=None
)
```

<!-- Placeholder for "Used in" -->

`regex_split` will split `input` using delimiters that match a
regex pattern in `delim_regex_pattern`. Here is an example:

```
text_input=["hello there"]
# split by whitespace
result, begin, end = regex_split_with_offsets(text_input, "\s")
# result = [["hello", "there"]]

```

By default, delimiters are not included in the split string results.
Delimiters may be included by specifying a regex pattern
`keep_delim_regex_pattern`. For example:

```
text_input=["hello there"]
# split by whitespace
result, begin, end = regex_split_with_offsets(text_input, "\s", "\s")
# result = [["hello", " ", "there"]]
```

If there are multiple delimiters in a row, there are no empty splits emitted.
For example:

```
text_input=["hello  there"]  # two continuous whitespace characters
# split by whitespace
result, begin, end = regex_split_with_offsets(text_input, "\s")
# result = [["hello", "there"]]
```

See https://github.com/google/re2/wiki/Syntax for the full list of supported
expressions.

#### Args:


* <b>`input`</b>: A Tensor or RaggedTensor of string input.
* <b>`delim_regex_pattern`</b>: A string containing the regex pattern of a delimiter.
* <b>`keep_delim_regex_pattern`</b>: (optional) Regex pattern of delimiters that should
  be kept in the result.
* <b>`name`</b>: (optional) Name of the op.


#### Returns:

A RaggedTensors containing of type string containing the split string
pieces.
