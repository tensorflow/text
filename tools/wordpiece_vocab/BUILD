licenses(["notice"])

py_library(
    name = "wordpiece_tokenizer_learner_lib",
    srcs = ["wordpiece_tokenizer_learner_lib.py"],
    srcs_version = "PY3",
    deps = [
        #  numpy dep,
        "//tensorflow_text:ops",
    ],
)

py_test(
    name = "wordpiece_tokenizer_learner_test",
    srcs = ["wordpiece_tokenizer_learner_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":wordpiece_tokenizer_learner_lib",
        #  numpy dep,
        # python/data/ops:dataset_ops tensorflow dep,
        "//tensorflow_text:ops",
    ],
)
