#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# create virtual env
"pip${PY_VERSION}" install --user --upgrade pip
"pip${PY_VERSION}" install --user --upgrade virtualenv
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate

# install deps
pip install --upgrade attrs
pip install keras_applications==1.0.8 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps
pip install numpy==1.14.5
pip install --upgrade "future>=0.17.1"
pip install gast==0.2.2
pip install h5py==2.8.0
pip install grpcio
pip install portpicker
pip install scipy
pip install scikit-learn

# setup_pypi_credentials

# Checkout the release branch if specified.
git checkout "${RELEASE_BRANCH:-master}"

# Run configure.
./oss_scripts/configure.sh

# Build the pip package
bazel build \
  --crosstool_top=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010:toolchain \
  oss_scripts/pip_package:build_pip_package

./bazel-bin/oss_scripts/pip_package/build_pip_package $HOME/wheels
