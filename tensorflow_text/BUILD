licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# main target to build
py_library(
    name = "tf-text",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    srcs_version = "PY2AND3",
    deps = [
        ":create_feature_bitmask_op",
        ":greedy_constrained_sequence_op",
        ":ngrams_op",
        ":normalize_ops",
        ":pad_along_dimension_op",
        ":pointer_ops",
        ":sentence_breaking_ops",
        ":sliding_window_op",
        ":string_ops",
        ":tokenization",
        ":unicode_script_tokenizer",
        ":viterbi_constrained_sequence_op",
        ":viterbi_decode",
        ":whitespace_tokenizer",
        ":wordpiece_tokenizer",
        ":wordshape_ops",
    ],
)

cc_binary(
    name = "python/ops/_constrained_sequence_op.so",
    srcs = [
        "core/kernels/constrained_sequence_kernel.cc",
        "core/ops/constrained_sequence_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
    ],
    linkshared = 1,
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@icu//:common",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "create_feature_bitmask_op",
    srcs = ["python/ops/create_feature_bitmask_op.py"],
)

py_test(
    name = "create_feature_bitmask_op_test",
    size = "small",
    srcs = ["python/ops/create_feature_bitmask_op_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":create_feature_bitmask_op",
    ],
)

py_library(
    name = "greedy_constrained_sequence_op",
    srcs = ["python/ops/greedy_constrained_sequence_op.py"],
    data = [
        ":python/ops/_constrained_sequence_op.so",
    ],
)

py_test(
    name = "greedy_constrained_sequence_op_test",
    size = "small",
    srcs = ["python/ops/greedy_constrained_sequence_op_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":greedy_constrained_sequence_op",
    ],
)

py_library(
    name = "ngrams_op",
    srcs = ["python/ops/ngrams_op.py"],
    deps = [
        ":sliding_window_op",
    ],
)

py_test(
    name = "ngrams_op_test",
    size = "small",
    srcs = ["python/ops/ngrams_op_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":ngrams_op",
    ],
)

cc_binary(
    name = "python/ops/_normalize_ops.so",
    srcs = [
        "core/kernels/normalize_kernels.cc",
        "core/ops/normalize_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
    ],
    linkshared = 1,
    deps = [
        "//third_party/icu/data:icu_normalization_data",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@icu//:common",
        "@icu//:headers",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "normalize_ops",
    srcs = ([
        "python/ops/normalize_ops.py",
    ]),
    data = [
        ":python/ops/_normalize_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "normalize_ops_test",
    srcs = [
        "python/ops/normalize_ops_test.py",
    ],
    main = "python/ops/normalize_ops_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":normalize_ops",
    ],
)

py_library(
    name = "pad_along_dimension_op",
    srcs = ["python/ops/pad_along_dimension_op.py"],
)

py_test(
    name = "pad_along_dimension_op_test",
    size = "medium",
    srcs = ["python/ops/pad_along_dimension_op_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":pad_along_dimension_op",
    ],
)

py_library(
    name = "pointer_ops",
    srcs = ["python/ops/pointer_ops.py"],
)

py_test(
    name = "gather_with_default_op_test",
    srcs = ["python/ops/gather_with_default_op_test.py"],
    python_version = "PY2",
    deps = [
        ":pointer_ops",
    ],
)

py_test(
    name = "span_alignment_op_test",
    srcs = ["python/ops/span_alignment_op_test.py"],
    python_version = "PY2",
    deps = [
        ":pointer_ops",
    ],
)

py_test(
    name = "span_overlaps_op_test",
    srcs = ["python/ops/span_overlaps_op_test.py"],
    python_version = "PY2",
    deps = [
        ":pointer_ops",
    ],
)

py_library(
    name = "tokenization",
    srcs = ([
        "python/ops/tokenization.py",
    ]),
    srcs_version = "PY2AND3",
)

cc_binary(
    name = "python/ops/_sentence_breaking_ops.so",
    srcs = [
        "core/kernels/sentence_breaking_kernels.cc",
        "core/ops/sentence_breaking_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
    ],
    linkshared = 1,
    deps = [
        ":sentence_breaking_utils",
        ":sentence_fragmenter",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@icu//:common",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "sentence_breaking_utils",
    srcs = ["core/kernels/sentence_breaking_utils.cc"],
    hdrs = ["core/kernels/sentence_breaking_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu//:common",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "sentence_fragmenter",
    srcs = ["core/kernels/sentence_fragmenter.cc"],
    hdrs = ["core/kernels/sentence_fragmenter.h"],
    deps = [
        ":sentence_breaking_utils",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "sentence_breaking_ops",
    srcs = ([
        "python/ops/sentence_breaking_ops.py",
    ]),
    data = [
        ":python/ops/_sentence_breaking_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "sentence_breaking_ops_test",
    size = "small",
    srcs = ["python/ops/sentence_breaking_ops_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":sentence_breaking_ops",
    ],
)

py_library(
    name = "sliding_window_op",
    srcs = ["python/ops/sliding_window_op.py"],
)

py_test(
    name = "sliding_window_op_test",
    size = "small",
    srcs = ["python/ops/sliding_window_op_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":sliding_window_op",
    ],
)

py_library(
    name = "string_ops",
    srcs = ["python/ops/string_ops.py"],
)

py_test(
    name = "coerce_to_valid_utf8_op_test",
    size = "small",
    srcs = ["python/ops/coerce_to_valid_utf8_op_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":string_ops",
    ],
)

cc_binary(
    name = "python/ops/_unicode_script_tokenizer.so",
    srcs = [
        "core/kernels/unicode_script_tokenize_kernel.cc",
        "core/ops/unicode_script_tokenize_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
    ],
    linkshared = 1,
    deps = [
        "@icu//:common",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "unicode_script_tokenizer",
    srcs = ([
        "python/ops/unicode_script_tokenizer.py",
    ]),
    data = [
        ":python/ops/_unicode_script_tokenizer.so",
        ":tokenization",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "unicode_script_tokenizer_test",
    size = "large",
    srcs = ["python/ops/unicode_script_tokenizer_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":unicode_script_tokenizer",
    ],
)

py_library(
    name = "viterbi_constrained_sequence_op",
    srcs = ["python/ops/viterbi_constrained_sequence_op.py"],
    data = [
        ":python/ops/_constrained_sequence_op.so",
    ],
)

py_test(
    name = "viterbi_constrained_sequence_op_test",
    size = "small",
    srcs = ["python/ops/viterbi_constrained_sequence_op_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":viterbi_constrained_sequence_op",
        ":viterbi_decode",
    ],
)

py_library(
    name = "viterbi_decode",
    srcs = ["python/numpy/viterbi_decode.py"],
)

py_test(
    name = "viterbi_decode_test",
    size = "small",
    srcs = ["python/numpy/viterbi_decode_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":viterbi_decode",
    ],
)

cc_binary(
    name = "python/ops/_whitespace_tokenizer.so",
    srcs = [
        "core/kernels/whitespace_tokenize_kernel.cc",
        "core/ops/whitespace_tokenize_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
    ],
    linkshared = 1,
    deps = [
        "@icu//:common",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "whitespace_tokenizer",
    srcs = ([
        "python/ops/whitespace_tokenizer.py",
    ]),
    data = [
        ":python/ops/_whitespace_tokenizer.so",
        ":tokenization",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "whitespace_tokenizer_test",
    srcs = [
        "python/ops/whitespace_tokenizer_test.py",
    ],
    main = "python/ops/whitespace_tokenizer_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_test_util",
        ":whitespace_tokenizer",
    ],
)

cc_binary(
    name = "python/ops/_wordpiece_tokenizer.so",
    srcs = [
        "core/kernels/wordpiece_kernel.cc",
        "core/kernels/wordpiece_tokenizer.cc",
        "core/kernels/wordpiece_tokenizer.h",
        "core/ops/wordpiece_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
    ],
    linkshared = 1,
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@icu//:common",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "wordpiece_tokenizer",
    srcs = ([
        "python/ops/wordpiece_tokenizer.py",
    ]),
    data = [
        ":python/ops/_wordpiece_tokenizer.so",
        ":tokenization",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "wordpiece_tokenizer_test",
    size = "small",
    srcs = ["python/ops/wordpiece_tokenizer_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":wordpiece_tokenizer",
    ],
)

py_library(
    name = "wordshape_ops",
    srcs = ["python/ops/wordshape_ops.py"],
)
