# Numpy-based code for text processing.
package(
    default_visibility = [
        "//tensorflow_text:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "viterbi_decode",
    srcs = ["viterbi_decode.py"],
    deps = [
        #  numpy dep,
    ],
)

py_test(
    name = "viterbi_decode_test",
    size = "small",
    srcs = ["viterbi_decode_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":viterbi_decode",
        "@com_google_absl//absl/testing:absltest",
    ],
)
