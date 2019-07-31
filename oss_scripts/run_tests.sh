#!/bin/bash

set -x  # print commands as they are executed
set -e  # fail and exit on any command erroring

export BAZEL_VERSION=0.24.1
export TF_VERSION=1.14.0

install_bazel() {
  # Install Bazel for tests. Based on instructions at
  # https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu
  # (We skip the openjdk8 install step, since travis lets us have that by
  # default).

  # Update apt and install bazel (use -qq to minimize log cruft)
  sudo apt-get update
  sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3
  curl -SsL -O https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
}

install_python_packages() {
  # TensorFlow pulls in other deps, like numpy, absl, and six, transitively.
  pip install -U pip
  pip install tensorflow==$TF_VERSION
}

link_lib() {
  # Make certain we link to the appropriate shared object file
  TMP=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
  cd ${TMP:2}
  TMP=$(ls libtensorflow_framework.so*)
  if [[ $TMP != 'libtensorflow_framework.so.1' ]]; then
    ln -s $TMP libtensorflow_framework.so.1
  fi
  cd -
}

cd /workspace
install_bazel
install_python_packages
link_lib
./oss_scripts/configure.sh
bazel test --test_output=errors tensorflow_text:all
