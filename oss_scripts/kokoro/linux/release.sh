#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

source "${KOKORO_GFILE_DIR}/common.sh"

install_ubuntu_16_pip_deps pip${1}
setup_pypi_credentials

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# Checkout the release branch if specified.
git checkout "${RELEASE_BRANCH:-master}"

# Run configure.
./oss_scripts/configure.sh

# Build the pip package
bazel build \
  --crosstool_top=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010:toolchain \
  oss_scripts/pip_package:build_pip_package

./bazel-bin/oss_scripts/pip_package/build_pip_package $HOME/wheels
