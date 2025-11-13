load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load(":tftext.bzl", "extra_py_deps", "py_tf_text_library")

# Visibility rules
package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

exports_files([
    "python/ops/test_data/fast_sentencepiece.model",
    "python/ops/test_data/phrase.model",
])

####################################
# Public TF.Text library to import #
####################################

# Most TF users will want to add this as their dependency.
alias(
    name = "tensorflow_text",
    actual = ":ops",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ops_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":boise_offset_converter_cc",
        ":byte_splitter_cc",
        ":constrained_sequence_op_cc",
        ":fast_bert_normalizer_cc",
        ":fast_sentencepiece_tokenizer_cc",
        ":fast_wordpiece_tokenizer_cc",
        ":mst_ops_cc",
        ":ngrams_op_cc",
        ":normalize_ops_cc",
        ":phrase_tokenizer_cc",
        ":regex_split_ops_cc",
        ":sentence_breaking_ops_cc",
        ":sentencepiece_tokenizer_cc",
        ":split_merge_from_logits_tokenizer_cc",
        ":split_merge_tokenizer_cc",
        ":state_based_sentence_breaker_op_cc",
        ":text_similarity_metric_ops_cc",
        ":trimmer_ops_cc",
        ":unicode_script_tokenizer_cc",
        ":utf8_binarize_op_cc",
        ":whitespace_tokenizer_cc",
        ":whitespace_tokenizer_v2_cc",
        ":wordpiece_tokenizer_cc",
    ],
)

tflite_cc_shared_object(
    name = "libtftextops_for_testing.so",
    deps = [":ops_lib"],
)

py_library(
    name = "ops",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/keras/__init__.py",
        "python/keras/layers/__init__.py",
        "python/metrics/__init__.py",
        "python/numpy/__init__.py",
        "python/ops/__init__.py",
        "tools/__init__.py",
    ],
    visibility = [
        "//tensorflow_text:__pkg__",
        "//tensorflow_text:__subpackages__",
    ],
    deps = [
        ":bert_tokenizer",
        ":boise_offset_converter",
        ":byte_splitter",
        ":create_feature_bitmask_op",
        ":fast_bert_normalizer",
        ":fast_bert_tokenizer",
        ":fast_sentencepiece_tokenizer",
        ":fast_wordpiece_tokenizer",
        ":greedy_constrained_sequence_op",
        ":hub_module_splitter",
        ":hub_module_tokenizer",
        ":item_selector_ops",
        ":masking_ops",
        ":mst_ops",
        ":ngrams_op",
        ":normalize_ops",
        ":pad_along_dimension_op",
        ":pad_model_inputs_ops",
        ":phrase_tokenizer",
        ":pointer_ops",
        ":regex_split_ops",
        ":segment_combiner_ops",
        ":sentence_breaking_ops",
        ":sentencepiece_tokenizer",
        ":sliding_window_op",
        ":split_merge_from_logits_tokenizer",
        ":split_merge_tokenizer",
        ":splitter",
        ":state_based_sentence_breaker_op",
        ":string_ops",
        ":text_similarity_metric_ops",
        ":todense_layer",
        ":tokenization",
        ":tokenization_layers",
        ":trimmer_ops",
        ":unicode_char_tokenizer",
        ":unicode_script_tokenizer",
        ":utf8_binarize_op",
        ":viterbi_constrained_sequence_op",
        ":viterbi_decode",
        ":whitespace_tokenizer",
        ":whitespace_tokenizer_v2",
        ":wordpiece_tokenizer",
        ":wordshape_ops",
        "//tensorflow_text/tools/wordpiece_vocab",
        "@org_tensorflow//tensorflow/python/util:all_util",
        "@org_tensorflow_text//tensorflow_text/core/pybinds:tflite_registrar_library",
    ] + extra_py_deps(),
)

build_test(
    name = "tf-text-build-test",
    targets = [
        ":ops",
        ":ops_lib",
    ],
)

# This is required for other external users to build tests using these models.
exports_files(
    glob(["python/ops/test_data/**"]),
)

##########################
# Individual tf.text ops #
##########################
# The py libraries are ordered alphabetically and are grouped with their corresponding tests.

py_library(
    name = "bert_tokenizer",
    srcs = ["python/ops/bert_tokenizer.py"],
    tags = ["ignore_srcs"],
    deps = [
        ":normalize_ops",
        ":regex_split_ops",
        ":tokenization",
        ":wordpiece_tokenizer",
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
    ] + extra_py_deps(),
)

py_test(
    name = "bert_tokenizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/ops/bert_tokenizer_test.py"],
    deps = [
        ":bert_tokenizer",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "boise_offset_converter",
    srcs = ["python/ops/boise_offset_converter.py"],
    cc_op_defs = [
        "@org_tensorflow_text//tensorflow_text/core/ops:boise_offset_converter_op.cc",
    ],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:boise_offset_converter_kernel",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_conversion_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "boise_offset_converter_test",
    size = "medium",
    srcs = ["python/ops/boise_offset_converter_test.py"],
    deps = [
        ":tensorflow_text",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "byte_splitter",
    srcs = ["python/ops/byte_splitter.py"],
    cc_op_defs = [
        "@org_tensorflow_text//tensorflow_text/core/ops:byte_splitter_op.cc",
    ],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:byte_splitter_kernel",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops/ragged",
    ],
)

py_test(
    name = "byte_splitter_test",
    size = "medium",
    srcs = ["python/ops/byte_splitter_test.py"],
    deps = [
        ":tensorflow_text",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/keras/api",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "constrained_sequence_op",
    srcs = [],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:constrained_sequence_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:constrained_sequence_kernel",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops/ragged",
    ],
)

py_library(
    name = "create_feature_bitmask_op",
    srcs = ["python/ops/create_feature_bitmask_op.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:check_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
    ] + extra_py_deps(),
)

py_test(
    name = "create_feature_bitmask_op_test",
    size = "small",
    srcs = ["python/ops/create_feature_bitmask_op_test.py"],
    deps = [
        ":create_feature_bitmask_op",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "fast_bert_normalizer",
    srcs = ["python/ops/fast_bert_normalizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:fast_bert_normalizer_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:fast_bert_normalizer_tf_kernel",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/module",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow_text//tensorflow_text/core/pybinds:pywrap_fast_bert_normalizer_model_builder",
    ],
)

py_test(
    name = "fast_bert_normalizer_test",
    size = "small",
    srcs = ["python/ops/fast_bert_normalizer_test.py"],
    deps = [
        ":tensorflow_text",
        "//tensorflow_text:normalize_ops",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/keras/api",
        "@org_tensorflow//tensorflow/python/compat",
        "@org_tensorflow//tensorflow/python/data/kernel_tests:test_base",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow_text//tensorflow_text/core/pybinds:pywrap_fast_bert_normalizer_model_builder",
    ],
)

py_library(
    name = "fast_bert_tokenizer",
    srcs = ["python/ops/fast_bert_tokenizer.py"],
    tags = ["ignore_srcs"],
    deps = [
        ":fast_bert_normalizer",
        ":fast_wordpiece_tokenizer",
        ":tokenization",
        "//tensorflow_text:normalize_ops",
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
    ],
)

py_test(
    name = "fast_bert_tokenizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/ops/fast_bert_tokenizer_test.py"],
    deps = [
        ":tensorflow_text",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "fast_sentencepiece_tokenizer",
    srcs = ["python/ops/fast_sentencepiece_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:fast_sentencepiece_ops.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels/sentencepiece:sentencepiece_tokenizer_kernel",
        "@org_tensorflow_text//tensorflow_text/core/kernels/sentencepiece:sentencepiece_detokenizer_kernel",
    ],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow_text//tensorflow_text/core/pybinds:pywrap_model_converter",
    ],
)

py_test(
    name = "fast_sentencepiece_tokenizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/ops/fast_sentencepiece_tokenizer_test.py"],
    data = [
        ":python/ops/test_data/fast_sentencepiece.model",
    ],
    deps = [
        ":fast_sentencepiece_tokenizer",
        ":tensorflow_text",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/keras/api",
    ],
)

py_tf_text_library(
    name = "fast_wordpiece_tokenizer",
    srcs = ["python/ops/fast_wordpiece_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:fast_wordpiece_tokenizer_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:fast_wordpiece_tokenizer_kernel",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow_text//tensorflow_text/core/pybinds:pywrap_fast_wordpiece_tokenizer_model_builder",
    ],
)

py_test(
    name = "fast_wordpiece_tokenizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/ops/fast_wordpiece_tokenizer_test.py"],
    data = [
        ":python/ops/test_data/fast_wordpiece_tokenizer_model.fb",
    ],
    deps = [
        ":tensorflow_text",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/compat",
        "@org_tensorflow//tensorflow/python/data/kernel_tests:test_base",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "phrase_tokenizer",
    srcs = ["python/ops/phrase_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:phrase_tokenizer_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:phrase_tokenizer_kernel",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow_text//tensorflow_text/core/pybinds:pywrap_phrase_tokenizer_model_builder",
    ],
)

py_test(
    name = "phrase_tokenizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/ops/phrase_tokenizer_test.py"],
    data = [
        ":python/ops/test_data/phrase_tokenizer_model.fb",
    ],
    deps = [
        ":tensorflow_text",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow/python/compat",
        "@org_tensorflow//tensorflow/python/data/kernel_tests:test_base",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "greedy_constrained_sequence_op",
    srcs = ["python/ops/greedy_constrained_sequence_op.py"],
    cc_op_kernels = [
        ":constrained_sequence_op",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "greedy_constrained_sequence_op_test",
    size = "small",
    srcs = ["python/ops/greedy_constrained_sequence_op_test.py"],
    deps = [
        ":greedy_constrained_sequence_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "hub_module_splitter",
    srcs = ["python/ops/hub_module_splitter.py"],
    deps = [
        ":splitter",
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/saved_model:load",
    ],
)

py_test(
    name = "hub_module_splitter_test",
    size = "large",
    srcs = ["python/ops/hub_module_splitter_test.py"],
    data = [
        ":test_data_segmenter_hub_module",
    ],
    deps = [
        ":ops",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:variables",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "hub_module_tokenizer",
    srcs = ["python/ops/hub_module_tokenizer.py"],
    deps = [
        ":hub_module_splitter",
        ":tokenization",
    ],
)

py_test(
    name = "hub_module_tokenizer_test",
    size = "large",
    srcs = ["python/ops/hub_module_tokenizer_test.py"],
    data = [
        ":test_data_segmenter_hub_module",
    ],
    deps = [
        ":ops",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:variables",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "item_selector_ops",
    srcs = ["python/ops/item_selector_ops.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:control_flow_ops",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:map_fn",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:random_ops",
        "@org_tensorflow//tensorflow/python/ops:sort_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_batch_gather_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_functional_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor_shape",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_where_op",
    ] + extra_py_deps(),
)

py_test(
    name = "item_selector_ops_test",
    size = "medium",
    srcs = ["python/ops/item_selector_ops_test.py"],
    deps = [
        ":item_selector_ops",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "masking_ops",
    srcs = ["python/ops/masking_ops.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:control_flow_assert",
        "@org_tensorflow//tensorflow/python/ops:map_fn",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:random_ops",
        "@org_tensorflow//tensorflow/python/ops:sort_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_batch_gather_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_functional_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_where_op",
    ],
)

py_test(
    name = "masking_ops_test",
    size = "medium",
    srcs = ["python/ops/masking_ops_test.py"],
    deps = [
        ":item_selector_ops",
        ":masking_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "mst_ops",
    srcs = ["python/ops/mst_ops.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:mst_ops.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:mst_op_kernels",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:standard_ops",
    ] + extra_py_deps(),
)

py_test(
    name = "mst_ops_test",
    srcs = ["python/ops/mst_ops_test.py"],
    deps = [
        ":mst_ops",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "ngrams_op",
    srcs = ["python/ops/ngrams_op.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:ngrams_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:ngrams_kernel",
    ],
    deps = [
        ":sliding_window_op",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_functional_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "ngrams_op_test",
    size = "medium",
    srcs = ["python/ops/ngrams_op_test.py"],
    deps = [
        ":tensorflow_text",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "normalize_ops",
    srcs = ["python/ops/normalize_ops.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:normalize_ops.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:normalize_kernels",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_conversion_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "normalize_ops_test",
    size = "small",
    srcs = ["python/ops/normalize_ops_test.py"],
    deps = [
        ":normalize_ops",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "pad_along_dimension_op",
    srcs = ["python/ops/pad_along_dimension_op.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "pad_along_dimension_op_test",
    size = "medium",
    srcs = ["python/ops/pad_along_dimension_op_test.py"],
    deps = [
        ":pad_along_dimension_op",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/eager:context",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "pad_model_inputs_ops",
    srcs = ["python/ops/pad_model_inputs_ops.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:control_flow_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:tensor_array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "pad_model_inputs_ops_test",
    srcs = ["python/ops/pad_model_inputs_ops_test.py"],
    deps = [
        ":pad_model_inputs_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "pointer_ops",
    srcs = ["python/ops/pointer_ops.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:check_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_functional_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_gather_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_where_op",
        "@org_tensorflow//tensorflow/python/ops/ragged:segment_id_ops",
    ] + extra_py_deps(),
)

py_test(
    name = "gather_with_default_op_test",
    srcs = ["python/ops/gather_with_default_op_test.py"],
    deps = [
        ":pointer_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "span_alignment_op_test",
    srcs = ["python/ops/span_alignment_op_test.py"],
    deps = [
        ":pointer_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "span_overlaps_op_test",
    srcs = ["python/ops/span_overlaps_op_test.py"],
    deps = [
        ":pointer_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/eager:context",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "ragged_tensor_to_tensor_test",
    srcs = ["//third_party/tensorflow_text/python/ragged:ragged_tensor_to_tensor_test.py"],
    deps = [
        ":tensorflow_text",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "regex_split_ops",
    srcs = ["python/ops/regex_split_ops.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:regex_split_ops.cc"],
    cc_op_kernels = ["@org_tensorflow_text//tensorflow_text/core/kernels:regex_split_kernels"],
    deps = [
        ":splitter",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "regex_split_ops_test",
    size = "medium",
    srcs = ["python/ops/regex_split_ops_test.py"],
    deps = [
        ":regex_split_ops",
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
    ],
)

py_library(
    name = "segment_combiner_ops",
    srcs = ["python/ops/segment_combiner_ops.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
    ] + extra_py_deps(),
)

py_test(
    name = "segment_combiner_ops_test",
    srcs = ["python/ops/segment_combiner_ops_test.py"],
    deps = [
        ":segment_combiner_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "sentence_breaking_ops",
    srcs = ["python/ops/sentence_breaking_ops.py"],
    cc_op_defs = [
        "@org_tensorflow_text//tensorflow_text/core/ops:sentence_breaking_ops.cc",
    ],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:sentence_breaking_kernels",
    ],
    deps = [
        ":regex_split_ops",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/util:deprecation",
    ],
)

py_test(
    name = "sentence_breaking_ops_test",
    size = "small",
    srcs = ["python/ops/sentence_breaking_ops_test.py"],
    deps = [
        ":sentence_breaking_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "sentencepiece_tokenizer",
    srcs = ["python/ops/sentencepiece_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:sentencepiece_ops.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:sentencepiece_kernels",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_conversion_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "sentencepiece_tokenizer_test",
    size = "large",
    srcs = ["python/ops/sentencepiece_tokenizer_test.py"],
    data = [
        ":python/ops/test_data/test_oss_model.model",
    ],
    deps = [
        ":sentencepiece_tokenizer",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_gather_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/saved_model",
    ],
)

py_library(
    name = "sliding_window_op",
    srcs = ["python/ops/sliding_window_op.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "sliding_window_op_test",
    size = "small",
    srcs = ["python/ops/sliding_window_op_test.py"],
    deps = [
        ":sliding_window_op",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "split_merge_from_logits_tokenizer",
    srcs = ["python/ops/split_merge_from_logits_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:tokenizer_from_logits_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:tokenizer_from_logits_kernel",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops/ragged",
    ],
)

py_test(
    name = "split_merge_from_logits_tokenizer_test",
    size = "small",
    srcs = ["python/ops/split_merge_from_logits_tokenizer_test.py"],
    deps = [
        ":split_merge_from_logits_tokenizer",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "split_merge_tokenizer",
    srcs = ["python/ops/split_merge_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:split_merge_tokenize_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:split_merge_tokenize_kernel",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops/ragged",
    ],
)

py_test(
    name = "split_merge_tokenizer_test",
    size = "small",
    srcs = ["python/ops/split_merge_tokenizer_test.py"],
    deps = [
        ":split_merge_tokenizer",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "splitter",
    srcs = ["python/ops/splitter.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/module",
    ],
)

py_tf_text_library(
    name = "state_based_sentence_breaker_op",
    srcs = ["python/ops/state_based_sentence_breaker_op.py"],
    cc_op_defs = [
        "@org_tensorflow_text//tensorflow_text/core/ops:sentence_fragmenter_v2_op.cc",
    ],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:sentence_fragmenter_v2_kernel",
    ],
    deps = [
        ":byte_splitter",
        ":sentence_breaking_ops",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "state_based_sentence_breaker_op_test",
    size = "medium",
    srcs = ["python/ops/state_based_sentence_breaker_op_test.py"],
    deps = [
        ":state_based_sentence_breaker_op",
        ":tensorflow_text",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:errors",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "string_ops",
    srcs = ["python/ops/string_ops.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/ops:string_ops",
    ] + extra_py_deps(),
)

py_test(
    name = "coerce_to_valid_utf8_op_test",
    size = "small",
    srcs = ["python/ops/coerce_to_valid_utf8_op_test.py"],
    deps = [
        ":string_ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "text_similarity_metric_ops",
    srcs = ["python/metrics/text_similarity_metric_ops.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:rouge_l_op.cc"],
    cc_op_kernels = ["@org_tensorflow_text//tensorflow_text/core/kernels:rouge_l_kernel"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "text_similarity_metric_ops_test",
    size = "small",
    srcs = ["python/metrics/text_similarity_metric_ops_test.py"],
    deps = [
        ":text_similarity_metric_ops",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "todense_layer",
    srcs = ["python/keras/layers/todense.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "todense_test",
    size = "large",
    srcs = ["python/keras/layers/todense_test.py"],
    shard_count = 2,
    deps = [
        ":todense_layer",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/keras/api",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "tokenization",
    srcs = ["python/ops/tokenization.py"],
    deps = [
        ":splitter",
        "@org_tensorflow//tensorflow/python/module",
    ] + extra_py_deps(),
)

py_library(
    name = "tokenization_layers",
    srcs = ["python/keras/layers/tokenization_layers.py"],
    deps = [
        ":unicode_script_tokenizer",
        ":whitespace_tokenizer_v2",
        ":wordpiece_tokenizer",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_conversion_ops",
    ],
)

py_test(
    name = "tokenization_layers_test",
    size = "large",
    srcs = ["python/keras/layers/tokenization_layers_test.py"],
    shard_count = 20,
    deps = [
        ":tokenization_layers",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/keras/api",
        "@pypi//numpy:pkg",
    ],
)

py_tf_text_library(
    name = "trimmer_ops",
    srcs = ["python/ops/trimmer_ops.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:trimmer_ops.cc"],
    cc_op_kernels = ["@org_tensorflow_text//tensorflow_text/core/kernels:round_robin_trimmer_kernel"],
    deps = [
        ":item_selector_ops",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:functional_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:while_loop",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_map_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "trimmer_ops_test",
    srcs = ["python/ops/trimmer_ops_test.py"],
    deps = [
        ":tensorflow_text",
        ":trimmer_ops",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

filegroup(
    name = "test_data_segmenter_hub_module",
    srcs = glob([
        "python/ops/test_data/segmenter_hub_module/**",
    ]),
    visibility = ["//learning/tfx/users/nlx:__subpackages__"],
)

py_library(
    name = "unicode_char_tokenizer",
    srcs = ["python/ops/unicode_char_tokenizer.py"],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "unicode_char_tokenizer_test",
    size = "large",
    srcs = ["python/ops/unicode_char_tokenizer_test.py"],
    shard_count = 5,
    deps = [
        ":unicode_char_tokenizer",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "unicode_script_tokenizer",
    srcs = ["python/ops/unicode_script_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:unicode_script_tokenize_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:unicode_script_tokenize_kernel",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_conversion_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "unicode_script_tokenizer_test",
    size = "large",
    srcs = ["python/ops/unicode_script_tokenizer_test.py"],
    shard_count = 5,
    deps = [
        ":unicode_script_tokenizer",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "utf8_binarize_op",
    srcs = ["python/ops/utf8_binarize_op.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:utf8_binarize_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:utf8_binarize_kernel",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
    ],
)

py_test(
    name = "utf8_binarize_op_test",
    size = "small",
    srcs = ["python/ops/utf8_binarize_op_test.py"],
    deps = [
        ":tensorflow_text",
        ":utf8_binarize_op",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "viterbi_constrained_sequence_op",
    srcs = ["python/ops/viterbi_constrained_sequence_op.py"],
    cc_op_kernels = [
        ":constrained_sequence_op",
    ],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ] + extra_py_deps(),
)

py_test(
    name = "viterbi_constrained_sequence_op_test",
    size = "small",
    srcs = ["python/ops/viterbi_constrained_sequence_op_test.py"],
    deps = [
        ":viterbi_constrained_sequence_op",
        ":viterbi_decode",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow_text//tensorflow_text/python/numpy",
    ],
)

py_library(
    name = "viterbi_decode",
    srcs = ["python/numpy/viterbi_decode.py"],
    deps = [
        "@pypi//numpy:pkg",
    ] + extra_py_deps(),
)

py_test(
    name = "viterbi_decode_test",
    size = "small",
    srcs = ["python/numpy/viterbi_decode_test.py"],
    deps = [
        ":viterbi_decode",
        "//third_party/py/absl/testing:absltest",
        "@pypi//numpy:pkg",
    ],
)

py_tf_text_library(
    name = "whitespace_tokenizer",
    srcs = [],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:whitespace_tokenize_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:whitespace_tokenize_kernel",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_conversion_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_tf_text_library(
    name = "whitespace_tokenizer_v2",
    srcs = ["python/ops/whitespace_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:whitespace_tokenizer_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:whitespace_tokenizer_kernel",
    ],
    deps = [
        ":tokenization",
        ":whitespace_tokenizer",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_conversion_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
        "@org_tensorflow_text//tensorflow_text/core/pybinds:pywrap_whitespace_tokenizer_config_builder",
    ],
)

py_test(
    name = "whitespace_tokenizer_test",
    size = "large",
    srcs = ["python/ops/whitespace_tokenizer_test.py"],
    shard_count = 4,
    deps = [
        ":tensorflow_text",
        "//third_party/py/tensorflow",
        "//third_party/py/tf_keras/api:tf_keras_api",
        "@org_tensorflow//tensorflow/python/framework:constant_op",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_tf_text_library(
    name = "wordpiece_tokenizer",
    srcs = ["python/ops/wordpiece_tokenizer.py"],
    cc_op_defs = ["@org_tensorflow_text//tensorflow_text/core/ops:wordpiece_op.cc"],
    cc_op_kernels = [
        "@org_tensorflow_text//tensorflow_text/core/kernels:wordpiece_kernel",
    ],
    deps = [
        ":tokenization",
        "@org_tensorflow//tensorflow/python/compat",
        "@org_tensorflow//tensorflow/python/eager:monitoring",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/framework:tensor",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:check_ops",
        "@org_tensorflow//tensorflow/python/ops:control_flow_assert",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops:sort_ops",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_functional_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_string_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "wordpiece_tokenizer_test",
    size = "small",
    srcs = ["python/ops/wordpiece_tokenizer_test.py"],
    deps = [
        ":wordpiece_tokenizer",
        "//third_party/py/absl/testing:parameterized",
        "@org_tensorflow//tensorflow/python/compat",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/ops:array_ops",
        "@org_tensorflow//tensorflow/python/ops:lookup_ops",
        "@org_tensorflow//tensorflow/python/ops:math_ops",
        "@org_tensorflow//tensorflow/python/ops/ragged:ragged_factory_ops",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "wordshape_ops",
    srcs = ["python/ops/wordshape_ops.py"],
    deps = [
        "@org_tensorflow//tensorflow/python/framework:ops",
        "@org_tensorflow//tensorflow/python/ops:array_ops_stack",
        "@org_tensorflow//tensorflow/python/ops:string_ops",
    ] + extra_py_deps(),
)

py_test(
    name = "wordshape_ops_test",
    size = "small",
    srcs = ["python/ops/wordshape_ops_test.py"],
    deps = [
        ":wordshape_ops",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
    ],
)

##############
# Benchmarks #
##############

# To run the benchmarks:
#   bazel run -c opt benchmark_test_name -- --benchmark_filter=.

# The --benchmark_filter flag specifies the list of benchmarks to run using a
# regular expression that is matched against the benchmark methods' name.
# e.g. --benchmark_filter="wordpiece" will run the Wordpiece Tokenizer benchmarks.

py_library(
    name = "profiler_lib",
    deps = [
        "//perftools/gputools/profiler:dsprof_lib",
        "//perftools/gputools/profiler:gpuprof_lib",
        "//perftools/gputools/profiler:xprof_lib",
        "//perftools/gputools/profiler:xprofilez",
        "//perftools/gputools/profiler:xprofilez_with_server",
    ],
)

py_library(
    name = "benchmark_utils",
    srcs = ["python/benchmarks/benchmark_utils.py"],
    deps = [
        ":profiler_lib",
        "//perftools/accelerators/xprof/api/python:xprof_session",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets",
        "@org_tensorflow//tensorflow/python/client:session",
        "@org_tensorflow//tensorflow/python/data/ops:dataset_ops",
        "@org_tensorflow//tensorflow/python/eager:context",
        "@org_tensorflow//tensorflow/python/eager:def_function",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/util:tf_inspect",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "tokenizers_benchmarks",
    size = "medium",
    srcs = ["python/benchmarks/tokenizers_benchmarks.py"],
    data = [
        ":python/benchmarks/test_data/uncased_L-12_H-768_A-12/vocab.txt",
        ":python/ops/test_data/fast_sentencepiece.model",
        ":python/ops/test_data/test_oss_model.model",
        ":test_data_segmenter_hub_module",
    ],
    deps = [
        ":benchmark_utils",
        ":ops",
        "//third_party/py/absl:app",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/user_ops:ops",
    ],
)

py_test(
    name = "ops_benchmarks",
    size = "medium",
    srcs = ["python/benchmarks/ops_benchmarks.py"],
    deps = [
        ":benchmark_utils",
        ":ops",
        "//third_party/py/absl:app",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow/python/user_ops:ops",
    ],
)
