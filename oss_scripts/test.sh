#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# cd into the base directory
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# create virtual env
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate
bazel version

# Test against nightly
# local tfnightly=$(curl -s https://pypi.org/pypi/tf-nightly/json | jq -r '.releases | keys[]' | sort -t. -k 1,1n -k 2,2n -k 3,3n -k 4,4 | tail -1)
pip install tf-nightly

# Get commit
short_commit_sha=$(python -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1 | grep -o '.\{10\}$')
commit_sha=$(curl -SsL https://github.com/tensorflow/tensorflow/commit/${short_commit_sha} | grep sha-block | grep commit | sed -e 's/.*\([a-f0-9]\{40\}\).*/\1/')
git_url="https://github.com/tensorflow/tensorflow/archive/${commit_sha}.zip"

# Update TF dependency
sed -i "s/strip_prefix = \"tensorflow-2\.[0-9]\+\.[0-9]\+\(-rc[0-9]\+\)\?\",/strip_prefix = \"tensorflow-${commit_sha}\",/" WORKSPACE
sed -i "s|\"https://github.com/tensorflow/tensorflow/archive/v.\+\.zip\"|\"${git_url}\"|" WORKSPACE
prev_shasum=$(grep -A 1 -e "strip_prefix.*tensorflow-" WORKSPACE | tail -1 | awk -F '"' '{print $2}')
sed -i "s/sha256 = \"${prev_shasum}\",//" WORKSPACE

# Run configure.
./oss_scripts/configure.sh

# Test
bazel test --config=manylinux2010 --test_output=errors tensorflow_text:all

deactivate
