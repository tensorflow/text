load("//third_party/py/tensorflow_docs/google:tf_org.bzl", "tf_org_notebook_test")

licenses(["notice"])

tf_org_notebook_test(
    name = "bert_glue",
    execute = False,
    ipynb = "bert_glue.ipynb",
    tpu = True,
    deps = [],
)

tf_org_notebook_test(
    name = "classify_text_with_bert",
    execute = False,
    ipynb = "classify_text_with_bert.ipynb",
    deps = [
        "//third_party/py/matplotlib",
        #  numpy dep,
        #  tensorflow datasets dep,
    ],
)

tf_org_notebook_test(
    name = "nmt_with_attention",
    ipynb = "nmt_with_attention.ipynb",
    deps = [
        "//third_party/py/matplotlib",
        #  numpy dep,
        "//third_party/py/sklearn",
    ],
)

tf_org_notebook_test(
    name = "text_generation",
    ipynb = "text_generation.ipynb",
    deps = [
        "//third_party/py/matplotlib",
        #  numpy dep,
        #  tensorflow datasets dep,
    ],
)

tf_org_notebook_test(
    name = "text_similarity",
    ipynb = "text_similarity.ipynb",
    deps = [
        "//third_party/py/tensorflow_text",
    ],
)

tf_org_notebook_test(
    name = "transformer",
    ipynb = "transformer.ipynb",
    deps = [
        "//third_party/py/matplotlib",
        #  numpy dep,
        #  tensorflow datasets dep,
        "//third_party/py/tensorflow_text",
    ],
)
