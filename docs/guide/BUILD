load("//third_party/py/tensorflow_docs/google:tf_org.bzl", "tf_org_notebook_test")

licenses(["notice"])

tf_org_notebook_test(
    name = "tokenizers",
    execute = False,
    ipynb = "tokenizers.ipynb",
    deps = [
        "//third_party/py/requests",
        "//third_party/py/tensorflow_text",
    ],
)

tf_org_notebook_test(
    name = "unicode",
    ipynb = "unicode.ipynb",
    deps = [],
)

tf_org_notebook_test(
    name = "word_embeddings",
    execute = False,
    ipynb = "word_embeddings.ipynb",
    deps = [],
)

tf_org_notebook_test(
    name = "subwords_tokenizer",
    execute = False,
    ipynb = "subwords_tokenizer.ipynb",
    deps = [
        "//third_party/py/matplotlib",
        #  numpy dep,
        #  tensorflow datasets dep,
        "//third_party/py/tensorflow_text",
        "//tensorflow_text/tools/wordpiece_vocab:bert_vocab_from_dataset",
    ],
)

tf_org_notebook_test(
    name = "decoding_api",
    execute = False,
    ipynb = "decoding_api.ipynb",
    deps = [
        "//third_party/py/matplotlib",
        #  numpy dep,
    ],
)

tf_org_notebook_test(
    name = "bert_preprocessing_guide",
    execute = True,
    ipynb = "bert_preprocessing_guide.ipynb",
    deps = [
        "//third_party/py/tensorflow_text",
    ],
)
