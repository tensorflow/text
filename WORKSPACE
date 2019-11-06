workspace(name = "org_tensorflow_text")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "absl_py",
    sha256 = "280c76ec0c9ab7a1dff550cdc37b7c7cd28551103dc3955202760ea8e381aa9d",
    strip_prefix = "abseil-py-pypi-v0.8.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-py/archive/pypi-v0.8.0.tar.gz",
        "https://github.com/abseil/abseil-py/archive/pypi-v0.8.0.tar.gz",
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)

http_archive(
    name = "com_google_absl",
    sha256 = "acd93f6baaedc4414ebd08b33bebca7c7a46888916101d8c0b8083573526d070",
    strip_prefix = "abseil-cpp-43ef2148c0936ebf7cb4be6b19927a9d9d145b8f",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
    ],
)

http_archive(
    name = "com_google_glog",
    sha256 = "1ee310e5d0a19b9d584a855000434bb724aa744745d5b8ab1855c85bff8a8e21",
    strip_prefix = "glog-028d37889a1e80e8a07da1b8945ac706259e5fd8",
    urls = [
        "https://mirror.bazel.build/github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
        "https://github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
    ],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    urls = [
        "http://mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
    ],
)

http_archive(
    name = "com_google_re2",
    sha256 = "7f33d878666aa1a1f034ec7b06cc192820117aecdf5ee0b5107cc93c935aa9d4",
    strip_prefix = "re2-6921af4d12664c5a0e7317e0cfeb4ca7ba9d5bfa",
    urls = [
        "https://github.com/google/re2/archive/6921af4d12664c5a0e7317e0cfeb4ca7ba9d5bfa.zip",
    ],
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-master",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/master.zip"],
)

http_archive(
    name = "com_google_sentencepiece",
    strip_prefix = "sentencepiece-1.0.0",
    sha256 = "c05901f30a1d0ed64cbcf40eba08e48894e1b0e985777217b7c9036cac631346",
    urls = [
        "https://github.com/google/sentencepiece/archive/1.0.0.zip"
    ],
)

http_archive(
    name = "icu",
    strip_prefix = "icu-release-64-2",
    sha256 = "dfc62618aa4bd3ca14a3df548cd65fe393155edd213e49c39f3a30ccd618fc27",
    urls = [
        "https://github.com/unicode-org/icu/archive/release-64-2.zip",
    ],
    build_file = "//third_party/icu:BUILD.bzl",
    patches = ["//third_party/icu:udata.patch"],
    patch_args = ["-p1"],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.0.0",
    sha256 = "4c13e99a2e5fdddc491e003304141f9d9060e11584499061b1b4224e500dc49a",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.0.0.zip"
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(tf_repo_name="@org_tensorflow")

load("//third_party/tensorflow:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")
