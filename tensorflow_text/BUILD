# Placeholder: load py_library
# Placeholder: load py_test
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object")
load("//tensorflow_text:tftext.bzl", "py_tf_text_library")
# [internal] load build_test.bzl

# Visibility rules
package(
    default_visibility = ["//visibility:public"],
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
    srcs_version = "PY3",
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
        # python/util:all_util tensorflow dep,
        "//tensorflow_text/core/pybinds:tflite_registrar",
        "//tensorflow_text/tools/wordpiece_vocab",
    ],
)

# build_test

# public_names_test

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
        # python/eager:monitoring tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
    ],
)

py_test(
    name = "bert_tokenizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/ops/bert_tokenizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bert_tokenizer",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "boise_offset_converter",
    srcs = ["python/ops/boise_offset_converter.py"],
    cc_op_defs = [
        "//tensorflow_text/core/ops:boise_offset_converter_op.cc",
    ],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:boise_offset_converter_kernel",
    ],
    deps = [
        # python/framework:ops tensorflow dep,
        # python/ops/ragged:ragged_conversion_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "boise_offset_converter_test",
    size = "medium",
    srcs = ["python/ops/boise_offset_converter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        # tensorflow package dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "byte_splitter",
    srcs = ["python/ops/byte_splitter.py"],
    cc_op_defs = [
        "//tensorflow_text/core/ops:byte_splitter_op.cc",
    ],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:byte_splitter_kernel",
    ],
    deps = [
        # python/framework:ops tensorflow dep,
        # python/ops/ragged tensorflow dep,
    ],
)

py_test(
    name = "byte_splitter_test",
    size = "medium",
    srcs = ["python/ops/byte_splitter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        # tensorflow package dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "constrained_sequence_op",
    srcs = [],
    cc_op_defs = ["//tensorflow_text/core/ops:constrained_sequence_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:constrained_sequence_kernel",
    ],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops/ragged tensorflow dep,
    ],
)

py_library(
    name = "create_feature_bitmask_op",
    srcs = ["python/ops/create_feature_bitmask_op.py"],
    deps = [
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:check_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
    ],
)

py_test(
    name = "create_feature_bitmask_op_test",
    size = "small",
    srcs = ["python/ops/create_feature_bitmask_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":create_feature_bitmask_op",
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "fast_bert_normalizer",
    srcs = ["python/ops/fast_bert_normalizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:fast_bert_normalizer_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:fast_bert_normalizer_tf_kernel",
    ],
    deps = [
        # python/eager:monitoring tensorflow dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/module tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        "//tensorflow_text/core/pybinds:pywrap_fast_bert_normalizer_model_builder",
    ],
)

py_test(
    name = "fast_bert_normalizer_test",
    size = "small",
    srcs = ["python/ops/fast_bert_normalizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/compat tensorflow dep,
        # python/data/kernel_tests:test_base tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
        "//tensorflow_text:normalize_ops",
        "//tensorflow_text/core/pybinds:pywrap_fast_bert_normalizer_model_builder",
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
        # python/eager:monitoring tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        "//tensorflow_text:normalize_ops",
    ],
)

py_test(
    name = "fast_bert_tokenizer_test",
    size = "small",
    timeout = "moderate",
    srcs = ["python/ops/fast_bert_tokenizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "fast_sentencepiece_tokenizer",
    srcs = ["python/ops/fast_sentencepiece_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:fast_sentencepiece_ops.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels/sentencepiece:sentencepiece_tokenizer_kernel",
        "//tensorflow_text/core/kernels/sentencepiece:sentencepiece_detokenizer_kernel",
    ],
    deps = [
        # tensorflow package dep,
        "//tensorflow_text/core/pybinds:pywrap_model_converter",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fast_sentencepiece_tokenizer",
        ":tensorflow_text",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
    ],
)

py_tf_text_library(
    name = "fast_wordpiece_tokenizer",
    srcs = ["python/ops/fast_wordpiece_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:fast_wordpiece_tokenizer_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:fast_wordpiece_tokenizer_kernel",
    ],
    deps = [
        ":tokenization",
        # python/eager:monitoring tensorflow dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        "//tensorflow_text/core/pybinds:pywrap_fast_wordpiece_tokenizer_model_builder",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/compat tensorflow dep,
        # python/data/kernel_tests:test_base tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "phrase_tokenizer",
    srcs = ["python/ops/phrase_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:phrase_tokenizer_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:phrase_tokenizer_kernel",
    ],
    deps = [
        ":tokenization",
        # python/eager:monitoring tensorflow dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        "//tensorflow_text/core/pybinds:pywrap_phrase_tokenizer_model_builder",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/compat tensorflow dep,
        # python/data/kernel_tests:test_base tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "greedy_constrained_sequence_op",
    srcs = ["python/ops/greedy_constrained_sequence_op.py"],
    cc_op_kernels = [
        ":constrained_sequence_op",
    ],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "greedy_constrained_sequence_op_test",
    size = "small",
    srcs = ["python/ops/greedy_constrained_sequence_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":greedy_constrained_sequence_op",
        #  numpy dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "hub_module_splitter",
    srcs = ["python/ops/hub_module_splitter.py"],
    deps = [
        ":splitter",
        # python/eager:monitoring tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/saved_model:load tensorflow dep,
    ],
)

py_test(
    name = "hub_module_splitter_test",
    size = "large",
    srcs = ["python/ops/hub_module_splitter_test.py"],
    data = [
        ":test_data_segmenter_hub_module",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        # python/framework:ops tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:variables tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        # python/framework:ops tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:variables tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "item_selector_ops",
    srcs = ["python/ops/item_selector_ops.py"],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:control_flow_ops tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:map_fn tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:random_ops tensorflow dep,
        # python/ops:sort_ops tensorflow dep,
        # python/ops/ragged:ragged_array_ops tensorflow dep,
        # python/ops/ragged:ragged_batch_gather_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_functional_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_math_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/ops/ragged:ragged_tensor_shape tensorflow dep,
        # python/ops/ragged:ragged_where_op tensorflow dep,
    ],
)

py_test(
    name = "item_selector_ops_test",
    size = "medium",
    srcs = ["python/ops/item_selector_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":item_selector_ops",
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "masking_ops",
    srcs = ["python/ops/masking_ops.py"],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:control_flow_assert tensorflow dep,
        # python/ops:map_fn tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:random_ops tensorflow dep,
        # python/ops:sort_ops tensorflow dep,
        # python/ops/ragged:ragged_batch_gather_ops tensorflow dep,
        # python/ops/ragged:ragged_functional_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_math_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/ops/ragged:ragged_where_op tensorflow dep,
    ],
)

py_test(
    name = "masking_ops_test",
    size = "medium",
    srcs = ["python/ops/masking_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":item_selector_ops",
        ":masking_ops",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "mst_ops",
    srcs = ["python/ops/mst_ops.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:mst_ops.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:mst_op_kernels",
    ],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:standard_ops tensorflow dep,
    ],
)

py_test(
    name = "mst_ops_test",
    srcs = ["python/ops/mst_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mst_ops",
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "ngrams_op",
    srcs = ["python/ops/ngrams_op.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:ngrams_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:ngrams_kernel",
    ],
    deps = [
        ":sliding_window_op",
        # python/framework:errors tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_functional_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "ngrams_op_test",
    size = "medium",
    srcs = ["python/ops/ngrams_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        # tensorflow package dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "normalize_ops",
    srcs = ["python/ops/normalize_ops.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:normalize_ops.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:normalize_kernels",
    ],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/ops/ragged:ragged_conversion_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "normalize_ops_test",
    size = "small",
    srcs = ["python/ops/normalize_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":normalize_ops",
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "pad_along_dimension_op",
    srcs = ["python/ops/pad_along_dimension_op.py"],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "pad_along_dimension_op_test",
    size = "medium",
    srcs = ["python/ops/pad_along_dimension_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pad_along_dimension_op",
        "@absl_py//absl/testing:parameterized",
        # python/eager:context tensorflow dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "pad_model_inputs_ops",
    srcs = ["python/ops/pad_model_inputs_ops.py"],
    deps = [
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:control_flow_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:tensor_array_ops tensorflow dep,
        # python/ops/ragged:ragged_array_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "pad_model_inputs_ops_test",
    srcs = ["python/ops/pad_model_inputs_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pad_model_inputs_ops",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "pointer_ops",
    srcs = ["python/ops/pointer_ops.py"],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:check_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops/ragged:ragged_functional_ops tensorflow dep,
        # python/ops/ragged:ragged_gather_ops tensorflow dep,
        # python/ops/ragged:ragged_math_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/ops/ragged:ragged_where_op tensorflow dep,
        # python/ops/ragged:segment_id_ops tensorflow dep,
    ],
)

py_test(
    name = "gather_with_default_op_test",
    srcs = ["python/ops/gather_with_default_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pointer_ops",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_test(
    name = "span_alignment_op_test",
    srcs = ["python/ops/span_alignment_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pointer_ops",
        "@absl_py//absl/testing:parameterized",
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_test(
    name = "span_overlaps_op_test",
    srcs = ["python/ops/span_overlaps_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pointer_ops",
        "@absl_py//absl/testing:parameterized",
        # python/eager:context tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_test(
    name = "ragged_tensor_to_tensor_test",
    srcs = ["python/ragged/ragged_tensor_to_tensor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        # tensorflow package dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "regex_split_ops",
    srcs = ["python/ops/regex_split_ops.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:regex_split_ops.cc"],
    cc_op_kernels = ["//tensorflow_text/core/kernels:regex_split_kernels"],
    deps = [
        ":splitter",
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "regex_split_ops_test",
    size = "medium",
    srcs = ["python/ops/regex_split_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":regex_split_ops",
        # tensorflow package dep,
        # python/framework:test_lib tensorflow dep,
    ],
)

py_library(
    name = "segment_combiner_ops",
    srcs = ["python/ops/segment_combiner_ops.py"],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
    ],
)

py_test(
    name = "segment_combiner_ops_test",
    srcs = ["python/ops/segment_combiner_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":segment_combiner_ops",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "sentence_breaking_ops",
    srcs = ["python/ops/sentence_breaking_ops.py"],
    cc_op_defs = [
        "//tensorflow_text/core/ops:sentence_breaking_ops.cc",
    ],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:sentence_breaking_kernels",
    ],
    deps = [
        ":regex_split_ops",
        # tf:lib tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/util:deprecation tensorflow dep,
    ],
)

py_test(
    name = "sentence_breaking_ops_test",
    size = "small",
    srcs = ["python/ops/sentence_breaking_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sentence_breaking_ops",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "sentencepiece_tokenizer",
    srcs = ["python/ops/sentencepiece_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:sentencepiece_ops.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:sentencepiece_kernels",
    ],
    deps = [
        ":tokenization",
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_conversion_ops tensorflow dep,
        # python/ops/ragged:ragged_string_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "sentencepiece_tokenizer_test",
    size = "large",
    srcs = ["python/ops/sentencepiece_tokenizer_test.py"],
    data = [
        ":python/ops/test_data/test_oss_model.model",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sentencepiece_tokenizer",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_gather_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
        # python/saved_model tensorflow dep,
    ],
)

py_library(
    name = "sliding_window_op",
    srcs = ["python/ops/sliding_window_op.py"],
    deps = [
        # python/framework:errors tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "sliding_window_op_test",
    size = "small",
    srcs = ["python/ops/sliding_window_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sliding_window_op",
        "@absl_py//absl/testing:parameterized",
        # python/framework:constant_op tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "split_merge_from_logits_tokenizer",
    srcs = ["python/ops/split_merge_from_logits_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:tokenizer_from_logits_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:tokenizer_from_logits_kernel",
    ],
    deps = [
        ":tokenization",
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops/ragged tensorflow dep,
    ],
)

py_test(
    name = "split_merge_from_logits_tokenizer_test",
    size = "small",
    srcs = ["python/ops/split_merge_from_logits_tokenizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":split_merge_from_logits_tokenizer",
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "split_merge_tokenizer",
    srcs = ["python/ops/split_merge_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:split_merge_tokenize_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:split_merge_tokenize_kernel",
    ],
    deps = [
        ":tokenization",
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops/ragged tensorflow dep,
    ],
)

py_test(
    name = "split_merge_tokenizer_test",
    size = "small",
    srcs = ["python/ops/split_merge_tokenizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":split_merge_tokenizer",
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "splitter",
    srcs = ["python/ops/splitter.py"],
    deps = [
        # python/module tensorflow dep,
    ],
)

py_tf_text_library(
    name = "state_based_sentence_breaker_op",
    srcs = ["python/ops/state_based_sentence_breaker_op.py"],
    cc_op_defs = [
        "//tensorflow_text/core/ops:sentence_fragmenter_v2_op.cc",
    ],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:sentence_fragmenter_v2_kernel",
    ],
    deps = [
        ":byte_splitter",
        ":sentence_breaking_ops",
        # tf:lib tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "state_based_sentence_breaker_op_test",
    size = "medium",
    srcs = ["python/ops/state_based_sentence_breaker_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":state_based_sentence_breaker_op",
        ":tensorflow_text",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:errors tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "string_ops",
    srcs = ["python/ops/string_ops.py"],
    deps = [
        # python/ops:string_ops tensorflow dep,
    ],
)

py_test(
    name = "coerce_to_valid_utf8_op_test",
    size = "small",
    srcs = ["python/ops/coerce_to_valid_utf8_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":string_ops",
        # python/framework:test_lib tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "text_similarity_metric_ops",
    srcs = ["python/metrics/text_similarity_metric_ops.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:rouge_l_op.cc"],
    cc_op_kernels = ["//tensorflow_text/core/kernels:rouge_l_kernel"],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "text_similarity_metric_ops_test",
    size = "small",
    srcs = ["python/metrics/text_similarity_metric_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":text_similarity_metric_ops",
        "@absl_py//absl/testing:parameterized",
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "todense_layer",
    srcs = ["python/keras/layers/todense.py"],
    deps = [
        # tensorflow package dep,
    ],
)

# todense_py_test

py_library(
    name = "tokenization",
    srcs = ["python/ops/tokenization.py"],
    deps = [
        ":splitter",
        # python/module tensorflow dep,
    ],
)

py_library(
    name = "tokenization_layers",
    srcs = ["python/keras/layers/tokenization_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":unicode_script_tokenizer",
        ":whitespace_tokenizer_v2",
        ":wordpiece_tokenizer",
        # tensorflow package dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops/ragged:ragged_conversion_ops tensorflow dep,
    ],
)

# tokenization_layers_py_test

py_tf_text_library(
    name = "trimmer_ops",
    srcs = ["python/ops/trimmer_ops.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:trimmer_ops.cc"],
    cc_op_kernels = ["//tensorflow_text/core/kernels:round_robin_trimmer_kernel"],
    deps = [
        ":item_selector_ops",
        # python/framework:constant_op tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:functional_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:while_loop tensorflow dep,
        # python/ops/ragged:ragged_array_ops tensorflow dep,
        # python/ops/ragged:ragged_map_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "trimmer_ops_test",
    srcs = ["python/ops/trimmer_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        ":trimmer_ops",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

filegroup(
    name = "test_data_segmenter_hub_module",
    srcs = glob([
        "python/ops/test_data/segmenter_hub_module/**",
    ]),
)

py_library(
    name = "unicode_char_tokenizer",
    srcs = ["python/ops/unicode_char_tokenizer.py"],
    deps = [
        ":tokenization",
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_string_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "unicode_char_tokenizer_test",
    size = "large",
    srcs = ["python/ops/unicode_char_tokenizer_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        ":unicode_char_tokenizer",
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "unicode_script_tokenizer",
    srcs = ["python/ops/unicode_script_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:unicode_script_tokenize_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:unicode_script_tokenize_kernel",
    ],
    deps = [
        ":tokenization",
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_conversion_ops tensorflow dep,
        # python/ops/ragged:ragged_string_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "unicode_script_tokenizer_test",
    size = "large",
    srcs = ["python/ops/unicode_script_tokenizer_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        ":unicode_script_tokenizer",
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "utf8_binarize_op",
    srcs = ["python/ops/utf8_binarize_op.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:utf8_binarize_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:utf8_binarize_kernel",
    ],
    visibility = [
        "//visibility:private",  # Only private by automation, not intent. Owner may accept CLs adding visibility. See go/scheuklappen#explicit-private.
    ],
    deps = [
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
    ],
)

py_test(
    name = "utf8_binarize_op_test",
    size = "small",
    srcs = ["python/ops/utf8_binarize_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        ":utf8_binarize_op",
        # tensorflow package dep,
        # python/framework:test_lib tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "viterbi_constrained_sequence_op",
    srcs = ["python/ops/viterbi_constrained_sequence_op.py"],
    cc_op_kernels = [
        ":constrained_sequence_op",
    ],
    deps = [
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "viterbi_constrained_sequence_op_test",
    size = "small",
    srcs = ["python/ops/viterbi_constrained_sequence_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":viterbi_constrained_sequence_op",
        ":viterbi_decode",
        #  numpy dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "viterbi_decode",
    srcs = ["python/numpy/viterbi_decode.py"],
    deps = [
        #  numpy dep,
    ],
)

py_test(
    name = "viterbi_decode_test",
    size = "small",
    srcs = ["python/numpy/viterbi_decode_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":viterbi_decode",
        "@absl_py//absl/testing:absltest",
        #  numpy dep,
    ],
)

py_tf_text_library(
    name = "whitespace_tokenizer",
    srcs = [],
    cc_op_defs = ["//tensorflow_text/core/ops:whitespace_tokenize_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:whitespace_tokenize_kernel",
    ],
    deps = [
        ":tokenization",
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_conversion_ops tensorflow dep,
        # python/ops/ragged:ragged_string_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_tf_text_library(
    name = "whitespace_tokenizer_v2",
    srcs = ["python/ops/whitespace_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:whitespace_tokenizer_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:whitespace_tokenizer_kernel",
    ],
    deps = [
        ":tokenization",
        ":whitespace_tokenizer",
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_conversion_ops tensorflow dep,
        # python/ops/ragged:ragged_string_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
        "//tensorflow_text/core/pybinds:pywrap_whitespace_tokenizer_config_builder",
    ],
)

py_test(
    name = "whitespace_tokenizer_test",
    size = "large",
    srcs = ["python/ops/whitespace_tokenizer_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":tensorflow_text",
        # tensorflow package dep,
        # python/framework:constant_op tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_tf_text_library(
    name = "wordpiece_tokenizer",
    srcs = ["python/ops/wordpiece_tokenizer.py"],
    cc_op_defs = ["//tensorflow_text/core/ops:wordpiece_op.cc"],
    cc_op_kernels = [
        "//tensorflow_text/core/kernels:wordpiece_kernel",
    ],
    deps = [
        ":tokenization",
        # python/compat tensorflow dep,
        # python/eager:monitoring tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:ops tensorflow dep,
        # python/framework:tensor tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:check_ops tensorflow dep,
        # python/ops:control_flow_assert tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops:sort_ops tensorflow dep,
        # python/ops:string_ops tensorflow dep,
        # python/ops/ragged:ragged_functional_ops tensorflow dep,
        # python/ops/ragged:ragged_string_ops tensorflow dep,
        # python/ops/ragged:ragged_tensor tensorflow dep,
    ],
)

py_test(
    name = "wordpiece_tokenizer_test",
    size = "small",
    srcs = ["python/ops/wordpiece_tokenizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":wordpiece_tokenizer",
        "@absl_py//absl/testing:parameterized",
        # python/compat tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/framework:test_lib tensorflow dep,
        # python/ops:array_ops tensorflow dep,
        # python/ops:lookup_ops tensorflow dep,
        # python/ops:math_ops tensorflow dep,
        # python/ops/ragged:ragged_factory_ops tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
    ],
)

py_library(
    name = "wordshape_ops",
    srcs = ["python/ops/wordshape_ops.py"],
    deps = [
        # python/framework:ops tensorflow dep,
        # python/ops:array_ops_stack tensorflow dep,
        # python/ops:string_ops tensorflow dep,
    ],
)

py_test(
    name = "wordshape_ops_test",
    size = "small",
    srcs = ["python/ops/wordshape_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":wordshape_ops",
        # python/framework:test_lib tensorflow dep,
        # python/platform:client_testlib tensorflow dep,
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

py_library(name = "profiler_lib")

py_library(
    name = "benchmark_utils",
    srcs = ["python/benchmarks/benchmark_utils.py"],
    deps = [
        #  numpy dep,
        # tensorflow package dep,
        #  tensorflow datasets dep,
        # python/client:session tensorflow dep,
        # python/data/ops:dataset_ops tensorflow dep,
        # python/eager:context tensorflow dep,
        # python/eager:def_function tensorflow dep,
        # python/framework:dtypes tensorflow dep,
        # python/util:tf_inspect tensorflow dep,
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":benchmark_utils",
        ":ops",
        "@absl_py//absl:app",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/framework:dtypes tensorflow dep,
        # python/user_ops:ops tensorflow dep,
    ],
)

py_test(
    name = "ops_benchmarks",
    size = "medium",
    srcs = ["python/benchmarks/ops_benchmarks.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":benchmark_utils",
        ":ops",
        "@absl_py//absl:app",
        "@absl_py//absl/testing:parameterized",
        # tensorflow package dep,
        # python/user_ops:ops tensorflow dep,
    ],
)
