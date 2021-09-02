# Code that exposes C++ libraries to Python via pybind11.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

licenses(["notice"])

package(default_visibility = [
    "//tensorflow_text:__subpackages__",
])

pybind_extension(
    name = "pywrap_tflite_registrar",
    srcs = [
        "pywrap_tflite_registrar.cc",
    ],
    hdrs = [
        "//tensorflow_text/core/kernels:tflite_headers",
    ],
    additional_exported_symbols = ["AddWhitespaceTokenizeWithOffsetsV2"],
    module_name = "pywrap_tflite_registrar",
    deps = [
        "@pybind11",
        # lite:framework tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:builtin_ops tensorflow dep,
        "//tensorflow_text/core/kernels:whitespace_tokenizer_tflite",
    ],
)

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

pybind_extension(
    name = "pywrap_fast_wordpiece_builder",
    srcs = ["pywrap_fast_wordpiece_builder.cc"],
    additional_exported_symbols = [
        "BuildFastWordpieceModel",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pywrap_fast_wordpiece_builder",
    srcs_version = "PY3ONLY",
    deps = [
        "//tensorflow_text/core/kernels:fast_wordpiece_builder",
        "@pybind11",
    ],
)

py_test(
    name = "pywrap_fast_wordpiece_builder_test",
    srcs = ["pywrap_fast_wordpiece_builder_test.py"],
    data = [
        "//tensorflow_text/core/kernels:fast_wordpiece_tokenizer_testdata",
    ],
    python_version = "PY3",
    deps = [
        ":pywrap_fast_wordpiece_builder",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

pybind_extension(
    name = "pywrap_fast_wordpiece_tflite_registrar",
    srcs = [
        "pywrap_fast_wordpiece_tflite_registrar.cc",
    ],
    hdrs = [
        "//tensorflow_text/core/kernels:fast_wordpiece_tflite_headers",
    ],
    additional_exported_symbols = ["AddFastWordpieceTokenizer"],
    module_name = "pywrap_fast_wordpiece_tflite_registrar",
    deps = [
        "@pybind11",
        # lite:framework tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:builtin_ops tensorflow dep,
        "//tensorflow_text/core/kernels:fast_wordpiece_tflite",
    ],
)
