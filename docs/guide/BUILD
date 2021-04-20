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
