#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# install deps
"pip${PY_VERSION}" install --user --upgrade pip
"pip${PY_VERSION}" install --user --upgrade attrs
"pip${PY_VERSION}" install keras_applications==1.0.8 --no-deps --user
"pip${PY_VERSION}" install keras_preprocessing==1.0.2 --no-deps --user
"pip${PY_VERSION}" install numpy==1.14.5 --user
"pip${PY_VERSION}" install --user --upgrade "future>=0.17.1"
"pip${PY_VERSION}" install gast==0.2.2 --user
"pip${PY_VERSION}" install h5py==2.8.0 --user
"pip${PY_VERSION}" install grpcio --user
"pip${PY_VERSION}" install portpicker --user
"pip${PY_VERSION}" install scipy --user
"pip${PY_VERSION}" install scikit-learn --user

# setup_pypi_credentials

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# Checkout the release branch if specified.
git checkout "${RELEASE_BRANCH:-master}"

# Run configure.
./oss_scripts/configure.sh python${PY_VERSION}

# Build the pip package
bazel build \
  --crosstool_top=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010:toolchain \
  oss_scripts/pip_package:build_pip_package

./bazel-bin/oss_scripts/pip_package/build_pip_package $HOME/wheels
