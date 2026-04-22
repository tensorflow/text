"""Sentencepiece kernels for tf.text ops.
All implementation files moved to //third_party/tensorflow/core/kernels/text/sentencepiece.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//tensorflow_text:license"],
    default_compatible_with = ["//buildenv/target:non_prod"],
    default_visibility = ["//visibility:public"],
)

# Aliases to relocated targets

ALIAS_NAMES = [
    "testdata",
    "config_fbs",
    "sp_headers",
    "config",
    "encoder_config",
    "decoder_config",
    "double_array_trie_test",
    "sentencepiece_tokenizer_kernel",
    "sentencepiece_detokenizer_kernel",
    "sentencepiece_tokenizer_tflite",
    "sentencepiece_detokenizer_tflite",
    "optimized_encoder_test",
    "optimized_decoder_test",
    "macos",
    "apple",
]

alias(
    name = "testdata",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:testdata",
)

alias(
    name = "config_fbs",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:config_fbs",
)

alias(
    name = "sp_headers",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:sp_headers",
)

alias(
    name = "config",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:config",
)

alias(
    name = "encoder_config",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:encoder_config",
)

alias(
    name = "decoder_config",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:decoder_config",
)

alias(
    name = "double_array_trie_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:double_array_trie_test",
)

alias(
    name = "sentencepiece_tokenizer_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:sentencepiece_tokenizer_kernel",
)

alias(
    name = "sentencepiece_detokenizer_kernel",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:sentencepiece_detokenizer_kernel",
)

alias(
    name = "sentencepiece_tokenizer_tflite",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:sentencepiece_tokenizer_tflite",
)

alias(
    name = "sentencepiece_detokenizer_tflite",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:sentencepiece_detokenizer_tflite",
)

alias(
    name = "optimized_encoder_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:optimized_encoder_test",
)

alias(
    name = "optimized_decoder_test",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:optimized_decoder_test",
)

alias(
    name = "macos",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:macos",
)

alias(
    name = "apple",
    actual = "@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:apple",
)

LIBRARY_HEADERS = {
    "utils": "utils.h",
    "double_array_trie": "double_array_trie.h",
    "double_array_trie_builder": "double_array_trie_builder.h",
    "sentencepiece_constants": "sentencepiece_constants.h",
    "model_converter": "model_converter.h",
    "optimized_encoder": "optimized_encoder.h",
    "optimized_decoder": "optimized_decoder.h",
    "sentencepiece_tokenizer_h": "sentencepiece_tokenizer.h",
    "sentencepiece_detokenizer_h": "sentencepiece_detokenizer.h",
    "py_tflite_registerer": "py_tflite_registerer.h",
}

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:utils"],
)

cc_library(
    name = "double_array_trie",
    hdrs = ["double_array_trie.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:double_array_trie"],
)

cc_library(
    name = "double_array_trie_builder",
    hdrs = ["double_array_trie_builder.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:double_array_trie_builder"],
)

cc_library(
    name = "sentencepiece_constants",
    hdrs = ["sentencepiece_constants.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:sentencepiece_constants"],
)

cc_library(
    name = "model_converter",
    hdrs = ["model_converter.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:model_converter"],
)

cc_library(
    name = "optimized_encoder",
    srcs = [
        "optimized_encoder.cc",
    ],
    hdrs = [
        "optimized_encoder.h",
    ],
    deps = [
        ":double_array_trie",
        ":encoder_config",
    ],
)

cc_library(
    name = "optimized_decoder",
    hdrs = ["optimized_decoder.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:optimized_decoder"],
)

cc_library(
    name = "sentencepiece_tokenizer_h",
    hdrs = ["sentencepiece_tokenizer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:sentencepiece_tokenizer_h"],
)

cc_library(
    name = "sentencepiece_detokenizer_h",
    hdrs = [
        "sentencepiece_detokenizer.h",
    ],
)

cc_library(
    name = "py_tflite_registerer",
    hdrs = ["py_tflite_registerer.h"],
    deps = ["@org_tensorflow//tensorflow/core/kernels/text/sentencepiece:py_tflite_registerer"],
)
