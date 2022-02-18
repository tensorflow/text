#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}
TF_VERSION=${2}
TF_BRANCH="latest"


# Pull docker image specific to the python verison and tensorflow version
if [[ ! -z "$BRANCH" ]]; then
  TF_BRANCH="${BRANCH}"
fi

TF_BRANCH="2.8"

docker pull tensorflow/build:${TF_BRANCH}-python${PY_VERSION}

# Run docker container. Container name: tf_text_container.
docker run --privileged --name tf_text_container -w /github -itd --rm \
  -v "$KOKORO_GFILE_DIR:/tf_text" \
  -v "$KOKORO_ARTIFACTS_DIR/github/tensorflow_text/:/github" \
  tensorflow/build:${TF_BRANCH}-python${PY_VERSION} \
  bash

sudo chmod u+x install_bazel.sh
# Intalling bazel inside running container
docker exec tf_text_container /tf_text/install_bazel.sh


if [[ -z "$TF_VERSION" ]]; then
  docker exec tf_text_container pip install tf-nightly
else
  docker exec tf_text_container pip install $TF_VERSION
fi

docker exec tf_text_container pip list

# Get commit sha of tf-nightly
short_commit_sha=$(docker exec tf_text_container python -c 'import tensorflow as tf; print(tf.__git_version__)' | tail -1 | grep -oP '(?<=-g)[0-9a-f]*$')
commit_sha=$(docker exec tf_text_container curl -SsL https://github.com/tensorflow/tensorflow/commit/${short_commit_sha} | grep sha-block | grep commit | sed -e 's/.*\([a-f0-9]\{40\}\).*/\1/')

# Update TF dependency to current nightly
docker exec tf_text_container sed -i "s/strip_prefix = \"tensorflow-2\.[0-9]\+\.[0-9]\+\(-rc[0-9]\+\)\?\",/strip_prefix = \"tensorflow-${commit_sha}\",/" WORKSPACE
docker exec tf_text_container sed -i "s|\"https://github.com/tensorflow/tensorflow/archive/v.\+\.zip\"|\"https://github.com/tensorflow/tensorflow/archive/${commit_sha}.zip\"|" WORKSPACE
prev_shasum=$(docker exec tf_text_container grep -A 1 -e "strip_prefix.*tensorflow-" WORKSPACE | tail -1 | awk -F '"' '{print $2}')
docker exec tf_text_container sed -i "s/sha256 = \"${prev_shasum}\",//" WORKSPACE

# Run configure.
docker exec tf_text_container oss_scripts/configure.sh

# Install dependecies
docker exec tf_text_container pip install tensorflow_datasets

# Test
docker exec tf_text_container -w bazel test --config=manylinux2010 --test_output=errors tensorflow_text:all

docker stop tf_text_container
