#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# cd into the base directory
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# Checkout a particular branch if specified.
if [[ ! -z "$BRANCH" ]]; then
  git checkout -f "${BRANCH}"
fi

source ${KOKORO_GFILE_DIR}/install_bazel.sh

# Create virtual Python environment.
"python${PY_VERSION}" -m venv env
source env/bin/activate
python -m ensurepip --upgrade

if [[ -z "$TF_VERSION" ]]; then
  pip install tf-nightly
else
  tf_command="tensorflow=="
  tf_command+=$TF_VERSION
  echo $tf_command
  pip install $tf_command
fi

# Get commit sha of tf-nightly
short_commit_sha=$(python -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1 | grep -oP '(?<=-g)[0-9a-f]*$')
commit_sha=$(curl -SsL https://github.com/tensorflow/tensorflow/commit/${short_commit_sha} | grep sha-block | grep commit | sed -e 's/.*\([a-f0-9]\{40\}\).*/\1/')

# Update TF dependency to current nightly
sed -i "s/strip_prefix = \"tensorflow-2\.[0-9]\+\.[0-9]\+\(-rc[0-9]\+\)\?\",/strip_prefix = \"tensorflow-${commit_sha}\",/" WORKSPACE
sed -i "s|\"https://github.com/tensorflow/tensorflow/archive/v.\+\.zip\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_sha}.zip\"|" WORKSPACE
prev_shasum=$(grep -A 1 -e "strip_prefix.*tensorflow-" WORKSPACE | tail -1 | awk -F '"' '{print $2}')
sed -i "s/sha256 = \"${prev_shasum}\",//" WORKSPACE

# Run configure.
./oss_scripts/configure.sh

# Install dependecies
pip install tensorflow_datasets
pip install keras-nightly

# Test
bazel test --config=manylinux2010 --test_output=errors tensorflow_text:all

deactivate
