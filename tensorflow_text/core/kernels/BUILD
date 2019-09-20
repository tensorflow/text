# Kernels for tf.text ops.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_kernel_library")

licenses(["notice"])  # Apache 2.0

# Visibility rules
package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

tf_kernel_library(
    name = "constrained_sequence_kernel",
    srcs = ["constrained_sequence_kernel.cc"],
    deps = [
        "//third_party/tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/base:core_headers",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "constrained_sequence_kernel_input_validation_test",
    srcs = ["constrained_sequence_kernel_input_validation_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "//tensorflow_text:constrained_sequence_op_cc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

cc_test(
    name = "exp_greedy_constrained_sequence_kernel_test",
    srcs = ["exp_greedy_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "//tensorflow_text:constrained_sequence_op_cc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

cc_test(
    name = "exp_viterbi_constrained_sequence_kernel_test",
    srcs = ["exp_viterbi_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "//tensorflow_text:constrained_sequence_op_cc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

cc_test(
    name = "log_greedy_constrained_sequence_kernel_test",
    srcs = ["log_greedy_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "//tensorflow_text:constrained_sequence_op_cc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

cc_test(
    name = "log_viterbi_constrained_sequence_kernel_test",
    srcs = ["log_viterbi_constrained_sequence_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "//tensorflow_text:constrained_sequence_op_cc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "normalize_kernels",
    srcs = ["normalize_kernels.cc"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "rouge_l_kernel",
    srcs = ["rouge_l_kernel.cc"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "rouge_l_kernel_test",
    size = "small",
    srcs = ["rouge_l_kernel_test.cc"],
    deps = [
        ":rouge_l_kernel",
        "//tensorflow_text:text_similarity_metric_ops_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "sentence_breaking_kernels",
    srcs = ["sentence_breaking_kernels.cc"],
    deps = [
        ":sentence_breaking_utils",
        ":sentence_fragmenter",
        "@com_google_absl//absl/strings",
        "@icu//:common",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "sentence_breaking_utils",
    srcs = ["sentence_breaking_utils.cc"],
    hdrs = ["sentence_breaking_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
        "@org_tensorflow//tensorflow/core:lib",
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
    deps = [
        ":sentence_breaking_utils",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "text_kernels_test_util",
    testonly = 1,
    srcs = ["text_kernels_test_util.cc"],
    hdrs = ["text_kernels_test_util.h"],
    deps = [
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "unicode_script_tokenize_kernel",
    srcs = ["unicode_script_tokenize_kernel.cc"],
    deps = [
        "@icu//:common",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "unicode_script_tokenize_kernel_test",
    srcs = ["unicode_script_tokenize_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "//tensorflow_text:unicode_script_tokenizer_cc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "whitespace_tokenize_kernel",
    srcs = ["whitespace_tokenize_kernel.cc"],
    deps = [
        "@icu//:common",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "whitespace_tokenize_kernel_test",
    srcs = ["whitespace_tokenize_kernel_test.cc"],
    deps = [
        ":text_kernels_test_util",
        "//tensorflow_text:whitespace_tokenizer_cc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "wordpiece_kernel",
    srcs = ["wordpiece_kernel.cc"],
    deps = [
        ":wordpiece_tokenizer",
        "@com_google_absl//absl/base:core_headers",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "wordpiece_tokenizer",
    srcs = ["wordpiece_tokenizer.cc"],
    hdrs = ["wordpiece_tokenizer.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
    ],
)

cc_test(
    name = "wordpiece_kernel_test",
    size = "small",
    srcs = ["wordpiece_kernel_test.cc"],
    deps = [
        ":wordpiece_kernel",
        "//tensorflow_text:wordpiece_tokenizer_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:test_main",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
    ],
)
