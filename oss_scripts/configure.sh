#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

osname="$(uname -s | tr 'A-Z' 'a-z')"
echo $osname

function is_macos() {
  [[ "${osname}" == "darwin" ]]
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

if [[ $(pip show tensorflow) == *tensorflow* ]] ||
   [[ $(pip show tensorflow-macos) == *tensorflow-macos* ]] ||
   [[ $(pip show tf-nightly) == *tf-nightly* ]]; then
  echo 'Using installed tensorflow.'
else
  echo 'Installing tensorflow.'
  pip install tf-nightly
fi

# Copy the current bazelversion of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/.bazelversion -o .bazelversion

# Copy the building configuration of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/.bazelrc -o .bazelrc
# This line breaks Windows builds, so we remove it.
sed -i -e 's/build --noincompatible_remove_legacy_whole_archive//' .bazelrc

write_to_bazelrc "build:manylinux2014 --config=release_cpu_linux"
DIR="org_tensorflow"
# Check if the directory exists
if [ ! -d "$DIR" ]; then
  git clone https://github.com/tensorflow/tensorflow.git org_tensorflow
  cd org_tensorflow
  git checkout 735467e89ccfd7ace190363412bb5698164628b5
  cd ..
else
  echo "Directory $DIR already exists."
fi
write_to_bazelrc "common:linux --override_repository=org_tensorflow=./org_tensorflow"

if (which python3) | grep -q "python3"; then
  installed_python="python3"
elif (which python) | grep -q "python"; then
  installed_python="python"
fi

if [ -z "$HERMETIC_PYTHON_VERSION" ]; then
  if [ -n "$PY_VERSION" ]; then
    HERMETIC_PYTHON_VERSION=$PY_VERSION
  else
    HERMETIC_PYTHON_VERSION=$($installed_python  -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
  fi
fi
export HERMETIC_PYTHON_VERSION

echo "TF_VERSION=$TF_VERSION"
REQUIREMENTS_EXTRA_FLAGS="--upgrade"
if [[ "$TF_VERSION" == *"rc"* ]]; then
  REQUIREMENTS_EXTRA_FLAGS="$REQUIREMENTS_EXTRA_FLAGS --pre"
fi

# JUST FOR TESTING
wget https://storage.googleapis.com/ml-sysroot-testing/tensorflow_metadata-1.18.0.dev0-py3-none-any.whl
mv tensorflow_metadata-1.18.0.dev0-py3-none-any.whl /github/tensorflow_metadata-1.18.0.dev0-py3-none-any.whl
echo "tensorflow-metadata @ file:///github/tensorflow_metadata-1.18.0.dev0-py3-none-any.whl" >> oss_scripts/pip_package/requirements.in

bazel run //oss_scripts/pip_package:requirements.update -- $REQUIREMENTS_EXTRA_FLAGS

# JUST FOR TESTING
sed -i 's#file://tensorflow_metadata-1.18.0.dev0-py3-none-any.whl#file:///github/tensorflow_metadata-1.18.0.dev0-py3-none-any.whl#g' oss_scripts/pip_package/requirements_lock_3_9.txt

TF_ABIFLAG=$(bazel run //oss_scripts/pip_package:tensorflow_build_info -- abi)
SHARED_LIBRARY_NAME="libtensorflow_framework.so.2"
if is_macos; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.2.dylib"
fi

write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_ABIFLAG}
