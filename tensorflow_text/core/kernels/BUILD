# Kernels for tf.text ops.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_kernel_library")

licenses(["notice"])  # Apache 2.0

# Visibility rules
package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

# These are "random" deps likely needed by each library (http://b/142433427).
# It is meant to be empty and filled by copybara.
OSS_DEPS = [
    "@com_google_absl//absl/container:inlined_vector",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/types:span",
    "@local_config_tf//:libtensorflow_framework",
    "@local_config_tf//:tf_header_lib",
]

tf_kernel_library(
    name = "constrained_sequence_kernel",
    srcs = ["constrained_sequence_kernel.cc"],
    deps = OSS_DEPS + [
        "@com_google_absl//absl/base:core_headers",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
    ],
)

cc_test(
    name = "constrained_sequence_kernel_input_validation_test",
    srcs = ["constrained_sequence_kernel_input_validation_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

cc_library(
    name = "regex_split",
    srcs = ["regex_split.cc"],
    hdrs = ["regex_split.h"],
    deps = OSS_DEPS + [
        # absl/strings dep
        "@com_google_re2//:re2",
        # tf:framework tensorflow dep,
    ],
)

tf_kernel_library(
    name = "regex_split_kernels",
    srcs = ["regex_split_kernels.cc"],
    deps = OSS_DEPS + [
        ":regex_split",
        "@com_google_absl//absl/memory",
        # tf:framework tensorflow dep,
    ],
)

cc_test(
    name = "regex_split_test",
    srcs = ["regex_split_test.cc"],
    deps = [
        ":regex_split",
        "@com_google_googletest//:gtest_main",
        # absl/strings dep
        "@com_google_re2//:re2",
    ],
)

cc_test(
    name = "exp_greedy_constrained_sequence_kernel_test",
    srcs = ["exp_greedy_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

cc_test(
    name = "exp_viterbi_constrained_sequence_kernel_test",
    srcs = ["exp_viterbi_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

cc_test(
    name = "log_greedy_constrained_sequence_kernel_test",
    srcs = ["log_greedy_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

cc_test(
    name = "log_viterbi_constrained_sequence_kernel_test",
    srcs = ["log_viterbi_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:constrained_sequence_op_cc",
    ],
)

tf_kernel_library(
    name = "normalize_kernels",
    srcs = ["normalize_kernels.cc"],
    deps = OSS_DEPS + [
        "//third_party/icu/data:icu_normalization_data",
        # absl/strings dep
        "@icu//:common",
        # tf:framework tensorflow dep,
    ],
)

tf_kernel_library(
    name = "rouge_l_kernel",
    srcs = ["rouge_l_kernel.cc"],
    deps = OSS_DEPS + [
        # absl/strings dep
        "@icu//:common",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
)

cc_test(
    name = "rouge_l_kernel_test",
    size = "small",
    srcs = ["rouge_l_kernel_test.cc"],
    deps = [
        ":rouge_l_kernel",
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:text_similarity_metric_ops_cc",
    ],
)

tf_kernel_library(
    name = "sentence_breaking_kernels",
    srcs = ["sentence_breaking_kernels.cc"],
    deps = OSS_DEPS + [
        ":sentence_breaking_utils",
        ":sentence_fragmenter",
        # absl/strings dep
        "@icu//:common",
        # tf:framework tensorflow dep,
    ],
)

cc_library(
    name = "sentence_breaking_utils",
    srcs = ["sentence_breaking_utils.cc"],
    hdrs = ["sentence_breaking_utils.h"],
    deps = OSS_DEPS + [
        # absl/strings dep
        "@icu//:common",
        # tf:lib tensorflow dep,
    ],
)

cc_test(
    name = "sentence_breaking_utils_test",
    size = "small",
    srcs = ["sentence_breaking_utils_test.cc"],
    deps = [
        ":sentence_breaking_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@icu//:common",
    ],
)

cc_library(
    name = "sentence_fragmenter",
    srcs = ["sentence_fragmenter.cc"],
    hdrs = ["sentence_fragmenter.h"],
    deps = OSS_DEPS + [
        ":sentence_breaking_utils",
        # tf:lib tensorflow dep,
    ],
)

tf_kernel_library(
    name = "sentencepiece_kernels",
    srcs = ["sentencepiece_kernels.cc"],
    deps = OSS_DEPS + [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        # absl/strings dep
        # absl/types:span dep
        "@icu//:common",
        "@com_google_sentencepiece//src:sentencepiece_cc_proto",
        "@com_google_sentencepiece//src:sentencepiece_processor",
        # tf:framework tensorflow dep,
        # tf:framework_headers_lib tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:protos_all_cc tensorflow dep,
        # tf/kernels:tensor_flag_utils tensorflow dep,
    ],
)

cc_library(
    name = "text_kernels_test_util",
    testonly = 1,
    srcs = ["text_kernels_test_util.cc"],
    hdrs = ["text_kernels_test_util.h"],
    deps = OSS_DEPS + [
        "@com_google_googletest//:gtest",
        # tf:framework tensorflow dep,
        # tf:testlib tensorflow dep,
    ],
)

tf_kernel_library(
    name = "unicode_script_tokenize_kernel",
    srcs = ["unicode_script_tokenize_kernel.cc"],
    deps = OSS_DEPS + [
        "@icu//:common",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
)

cc_test(
    name = "unicode_script_tokenize_kernel_test",
    srcs = ["unicode_script_tokenize_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:unicode_script_tokenizer_cc",
    ],
)

tf_kernel_library(
    name = "whitespace_tokenize_kernel",
    srcs = ["whitespace_tokenize_kernel.cc"],
    deps = OSS_DEPS + [
        "@icu//:common",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
)

cc_test(
    name = "whitespace_tokenize_kernel_test",
    srcs = ["whitespace_tokenize_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "@com_google_googletest//:gtest_main",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
        # tf:test tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:whitespace_tokenizer_cc",
    ],
)

tf_kernel_library(
    name = "wordpiece_kernel",
    srcs = ["wordpiece_kernel.cc"],
    deps = OSS_DEPS + [
        ":wordpiece_tokenizer",
        "@com_google_absl//absl/base:core_headers",
        # tf:framework tensorflow dep,
        # tf:lib tensorflow dep,
    ],
)

cc_library(
    name = "wordpiece_tokenizer",
    srcs = ["wordpiece_tokenizer.cc"],
    hdrs = ["wordpiece_tokenizer.h"],
    deps = OSS_DEPS + [
        # absl/strings dep
        "@icu//:common",
    ],
)

cc_test(
    name = "wordpiece_kernel_test",
    size = "small",
    srcs = ["wordpiece_kernel_test.cc"],
    deps = [
        ":wordpiece_kernel",
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf:testlib tensorflow dep,
        # tf/kernels:ops_testutil tensorflow dep,
        "//tensorflow_text:wordpiece_tokenizer_cc",
    ],
)
