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

# https://github.com/bazelbuild/bazel-skylib/releases
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl",
    sha256 = "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a",  # SHARED_ABSL_SHA
    strip_prefix = "abseil-cpp-df3ea785d8c30a9503321a3d35ee7d35808f190d",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
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
    name = "com_google_protobuf",
    sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
    strip_prefix = "protobuf-3.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
    ],
)

http_archive(
    name = "com_google_re2",
    sha256 = "d070e2ffc5476c496a6a872a6f246bfddce8e7797d6ba605a7c8d72866743bf9",
    strip_prefix = "re2-506cfa4bffd060c06ec338ce50ea3468daa6c814",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
        "https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
    ],
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
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/unicode-org/icu/archive/release-64-2.zip",
        "https://github.com/unicode-org/icu/archive/release-64-2.zip",
    ],
    build_file = "//third_party/icu:BUILD.bzl",
    patches = ["//third_party/icu:udata.patch"],
    patch_args = ["-p1", "-s"],
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

# NOTE: according to
# https://docs.bazel.build/versions/master/external.html#transitive-dependencies
# we should list the transitive dependencies of @org_tensorflow_hub in this
# WORKSPACE file.  Still, all of them are already listed by tf_workspace() which
# is called later in this file.
http_archive(
    name = "org_tensorflow_hub",
    strip_prefix = "hub-0.8.0",
    sha256 = "968af30c448d51c36501b68df2c916fb4a61007db3240adc9248fa3a9be2da6f",
    urls = [
        "https://github.com/tensorflow/hub/archive/v0.8.0.zip"
    ],
)

http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.3.0",
    sha256 = "1a6f24d9e3b1cf5cc55ecfe076d3a61516701bc045925915b26a9d39f4084c34",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.3.0.zip"
    ],
)

http_archive(
    name = "org_tensorflow_datasets",
    sha256 = "c6ff4e2306387f0ca45d4f616d9a1c5e79e02ef16d0a8958230a8049ea07fc98",
    strip_prefix = "datasets-3.2.0",
    urls = [
        "https://github.com/tensorflow/datasets/archive/v3.2.0.zip",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(tf_repo_name="@org_tensorflow")

load("//third_party/tensorflow:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")
