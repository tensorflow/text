#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# Export PATH for Python & Bazel
export PATH=$PATH:/c/tools:/c/Python${PY_VERSION//.}:/c/Python${PY_VERSION//.}/Scripts
python -V
bazel version

# Test against nightly
pip install tf-nightly

# Get commit sha of tf-nightly
short_commit_sha=$(python -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1 | grep -o '.\{10\}$')
commit_sha=$(curl -SsL https://github.com/tensorflow/tensorflow/commit/${short_commit_sha} | grep sha-block | grep commit | sed -e 's/.*\([a-f0-9]\{40\}\).*/\1/')

# Update TF dependency to current nightly
sed -i "s/strip_prefix = \"tensorflow-2\.[0-9]\+\.[0-9]\+\(-rc[0-9]\+\)\?\",/strip_prefix = \"tensorflow-${commit_sha}\",/" WORKSPACE
sed -i "s|\"https://github.com/tensorflow/tensorflow/archive/v.\+\.zip\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_sha}.zip\"|" WORKSPACE
prev_shasum=$(grep -A 1 -e "strip_prefix.*tensorflow-" WORKSPACE | tail -1 | awk -F '"' '{print $2}')
sed -i "s/sha256 = \"${prev_shasum}\",//" WORKSPACE

# Run configure.
bash oss_scripts/configure.sh

# Install dependecy
pip install tensorflow_datasets

# Test
bazel test --test_output=errors tensorflow_text:all
