# Opensource scripts

licenses(["notice"])

py_binary(
    name = "build_docs",
    testonly = 1,
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "//third_party/py/absl/flags",
        # tensorflow package dep,
        "//third_party/py/tensorflow_docs/api_generator:doc_controls",
        "//third_party/py/tensorflow_docs/api_generator:generate_lib",
        "//third_party/py/tensorflow_docs/api_generator:public_api",
        "//third_party/py/tensorflow_text",
    ],
)
