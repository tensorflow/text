#!/bin/bash

set -x  # print commands as they are executed
set -e  # fail and exit on any command erroring

export BAZEL_VERSION=0.24.1
export TF_VERSION=2.0.0b1

install_bazel() {
  # Install Bazel for tests. Based on instructions at
  # https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu
  # (We skip the openjdk8 install step, since travis lets us have that by
  # default).

  # Add Bazel distribution URI as a package source
  #echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" \
  # | sudo tee /etc/apt/sources.list.d/bazel.list
  #curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

  # Update apt and install bazel (use -qq to minimize log cruft)
  sudo apt-get update
  sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3
  curl -SsL -O https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
  export PATH=$PATH:$HOME/bin
}

install_python_packages() {
  #export PYTHON_VERSION=`python --version 2>&1 | sed -e 's/Python \(.\).*$/\1/'`
  #pip install -U pip six numpy wheel setuptools mock
  #if (($PYTHON_VERSION < 3)); then
  #   pip install -U future enum34
  #fi
  #pip install -U keras_applications==1.0.6 --no-deps
  #pip install -U keras_preprocessing==1.0.5 --no-deps

  # TensorFlow pulls in other deps, like numpy, absl, and six, transitively.
  pip install tensorflow==$TF_VERSION

  # Upgrade numpy to the latest to address issues that happen when testing with
  # Python 3 (https://github.com/tensorflow/tensorflow/issues/16488).
  #pip install -U numpy
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

install_bazel
install_python_packages
link_lib
./oss_scripts/configure.sh
bazel test --test_output=errors '//tensorflow_text:all'
