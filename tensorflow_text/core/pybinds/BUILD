# Code that exposes C++ libraries to Python via pybind11.

load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//tensorflow_text:tftext.bzl", "if_pywrap", "pybind_extension", "pywrap_binaries", "pywrap_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_text:license"],
    default_visibility = [
        "//nlp/sage/nlu/features/python:__pkg__",
        "//nlp/semantic_parsing/learning/neural/portable/tools/release:__pkg__",
        "//tensorflow_text:__subpackages__",
    ],
)

pybind_extension(
    name = "tflite_registrar",
    srcs = [
        "tflite_registrar.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels/text:byte_splitter_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:ngrams_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:ragged_tensor_to_tensor_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:round_robin_trimmer_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:sentence_fragmenter_v2_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:utf8_binarize_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_tflite",
        "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:py_tflite_registerer",
        # lite:framework tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:builtin_ops tensorflow dep,
    ],
)

pybind_extension(
    name = "pywrap_fast_bert_normalizer_model_builder",
    srcs = ["pywrap_fast_bert_normalizer_model_builder.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels/text:fast_bert_normalizer_model_builder",
    ],
)

py_test(
    name = "pywrap_fast_bert_normalizer_model_builder_test",
    srcs = ["pywrap_fast_bert_normalizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/fast_bert_normalizer_model.fb",
        "//tensorflow_text:python/ops/test_data/fast_bert_normalizer_model_lower_case_nfd_strip_accents.fb",
    ],
    strict_deps = False,
    tags = [
        "-sickbay",  # Depends on ICU version.
    ],
    deps = [
        ":pywrap_fast_bert_normalizer_model_builder",
        "@absl_py//absl/flags",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pybind_extension(
    name = "pywrap_fast_wordpiece_tokenizer_model_builder",
    srcs = ["pywrap_fast_wordpiece_tokenizer_model_builder.cc"],
    copts = ["-fexceptions"],
    data = [
        "pywrap_fast_wordpiece_tokenizer_model_builder.pyi",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels/text:fast_wordpiece_tokenizer_model_builder",
    ],
)

py_test(
    name = "pywrap_fast_wordpiece_tokenizer_model_builder_test",
    srcs = ["pywrap_fast_wordpiece_tokenizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/fast_wordpiece_tokenizer_model.fb",
    ],
    strict_deps = False,
    deps = [
        ":pywrap_fast_wordpiece_tokenizer_model_builder",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pybind_extension(
    name = "pywrap_phrase_tokenizer_model_builder",
    srcs = ["pywrap_phrase_tokenizer_model_builder.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = [
        "//knowledge/cerebra/nlu/models/smartv2/input:__pkg__",
        "//tensorflow_text:__subpackages__",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels/text:phrase_tokenizer_model_builder",
    ],
)

py_test(
    name = "pywrap_phrase_tokenizer_model_builder_test",
    srcs = ["pywrap_phrase_tokenizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/phrase_tokenizer_model_test.fb",
    ],
    strict_deps = False,
    deps = [
        ":pywrap_phrase_tokenizer_model_builder",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pybind_extension(
    name = "pywrap_model_converter",
    srcs = ["pywrap_model_converter.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:model_converter",
    ],
)

pybind_extension(
    name = "pywrap_whitespace_tokenizer_config_builder",
    srcs = ["pywrap_whitespace_tokenizer_config_builder.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels/text:whitespace_tokenizer_config_builder",
    ],
)

py_test(
    name = "pywrap_whitespace_tokenizer_config_builder_test",
    srcs = ["pywrap_whitespace_tokenizer_config_builder_test.py"],
    strict_deps = False,
    deps = [
        ":pywrap_whitespace_tokenizer_config_builder",
        "@release_or_nightly//:tensorflow_pkg",  # tensorflow package dep
    ] + if_pywrap([":pybinds_library"]),
)

pywrap_library(
    name = "pybinds_library",
    deps = [
        ":pywrap_fast_bert_normalizer_model_builder",
        ":pywrap_fast_wordpiece_tokenizer_model_builder",
        ":pywrap_model_converter",
        ":pywrap_phrase_tokenizer_model_builder",
        ":pywrap_whitespace_tokenizer_config_builder",
        ":tflite_registrar",
    ],
)

pywrap_binaries(
    name = "pybinds_binaries",
    dep = ":pybinds_library",
    visibility = ["//visibility:public"],
)
