#!/bin/bash

set -x  # print commands as they are executed
set -e  # fail and exit on any command erroring

install_bazel() {
  # Install Bazel for tests. Based on instructions at
  # https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu
  # (We skip the openjdk8 install step, since travis lets us have that by
  # default).

  # Add Bazel distribution URI as a package source
  echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | sudo tee /etc/apt/sources.list.d/bazel.list
  curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

  # Update apt and install bazel (use -qq to minimize log cruft)
  sudo apt-get update
  sudo apt-get install bazel
}

install_python_packages() {
  # TensorFlow pulls in other deps, like numpy, absl, and six, transitively.
  pip install tensorflow=2.0.0b0

  # Upgrade numpy to the latest to address issues that happen when testing with
  # Python 3 (https://github.com/tensorflow/tensorflow/issues/16488).
  pip install -U numpy
}

install_bazel
install_python_packages
bazel test '//tensorflow_text:all'
