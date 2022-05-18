# Tests for op kernel shape functions

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

exports_files(glob(["*.cc"]))

cc_test(
    name = "byte_splitter_op_test",
    size = "small",
    srcs = [
        "byte_splitter_op.cc",
        "byte_splitter_op_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:byte_splitter_kernel",
    ],
)

cc_test(
    name = "fast_wordpiece_tokenizer_op_test",
    size = "small",
    srcs = [
        "fast_wordpiece_tokenizer_op.cc",
        "fast_wordpiece_tokenizer_op_test.cc",
    ],
    deps = [
        # tf:framework tensorflow dep,
        # tf:test tensorflow dep,
        # tf:test_main tensorflow dep,
        # tf/framework:shape_inference_testutil tensorflow dep,
        "//tensorflow_text/core/kernels:fast_wordpiece_tokenizer_kernel",
    ],
)
