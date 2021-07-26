# Code that exposes C++ libraries to Python via pybind11.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

licenses(["notice"])

package(default_visibility = [
    "//tensorflow_text:__subpackages__",
])

pybind_extension(
    name = "pywrap_whitespace_tokenizer_config_builder",
    srcs = ["pywrap_whitespace_tokenizer_config_builder.cc"],
    additional_exported_symbols = [
        "BuildWhitespaceTokenizerConfig",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pywrap_whitespace_tokenizer_config_builder",
    srcs_version = "PY3ONLY",
    deps = [
        "//tensorflow_text/core/kernels:whitespace_tokenizer_config_builder",
        "@pybind11",
    ],
)

py_test(
    name = "pywrap_whitespace_tokenizer_config_builder_test",
    srcs = ["pywrap_whitespace_tokenizer_config_builder_test.py"],
    python_version = "PY3",
    deps = [
        ":pywrap_whitespace_tokenizer_config_builder",
        # tensorflow package dep,
    ],
)
