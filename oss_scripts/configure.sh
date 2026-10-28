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

set -e  # fail and exit on any command erroring

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

# Remove .bazelrc if it already exists
rm -f .bazelrc

# Copy the current bazelversion of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/v2.22.0-rc0/.bazelversion -o .bazelversion

# Copy the building configuration of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/v2.22.0-rc0/.bazelrc -o .bazelrc
curl https://raw.githubusercontent.com/tensorflow/tensorflow/v2.22.0-rc0/bazel_downloader.cfg -o bazel_downloader.cfg
# This line breaks Windows builds, so we remove it.
sed -i -e 's/build --noincompatible_remove_legacy_whole_archive//' .bazelrc

# With --fetch_bazel_config_only, stop after downloading the TF Bazel config.
# The rest of this script runs `bazel run`, so scripts that have to install
# Bazel first (oss_scripts/install_bazel.sh reads .bazelversion) call this
# script once with the flag, then install Bazel, then run it again.
if [[ "${1:-}" == "--fetch_bazel_config_only" ]]; then
  exit 0
fi

write_to_bazelrc "build:manylinux2014 --config=release_cpu_linux"
write_to_bazelrc "test --test_env=TF_USE_LEGACY_KERAS=1"
write_to_bazelrc "common --features=-external_include_paths"
write_to_bazelrc "common --host_features=-external_include_paths"

if [ -z "$HERMETIC_PYTHON_VERSION" ]; then
  if [ -n "$PY_VERSION" ]; then
    HERMETIC_PYTHON_VERSION=$PY_VERSION
  else
    if (which python3) | grep -q "python3"; then
      installed_python="python3"
    elif (which python) | grep -q "python"; then
      installed_python="python"
    fi
    HERMETIC_PYTHON_VERSION=$($installed_python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
  fi
fi
export HERMETIC_PYTHON_VERSION

if [[ -n "$HERMETIC_PYTHON_VERSION" ]]; then
  write_to_bazelrc "build --repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION}"
fi
if [[ -n "$IS_NIGHTLY" ]]; then
  write_to_bazelrc "build --repo_env=IS_NIGHTLY=${IS_NIGHTLY}"
fi
if [[ -n "$TF_VERSION" ]]; then
  write_to_bazelrc "build --repo_env=TF_VERSION=${TF_VERSION}"
fi

# Re-lock the requirements for nightly builds, or when TF_VERSION is not in the
# lock file yet. Re-locking the current release would pick stale tf-keras and
# pre-release versions, so the committed lock file is used as is in that case.
# The tensorflow==${TF_VERSION} pin comes from @release_or_nightly's
# requirements.in, and pip allows pre-releases for such an explicit pin, so
# --pre is not needed.
if [[ "$IS_NIGHTLY" == "nightly" ]]; then
  bazel run //oss_scripts/pip_package:requirements.update -- --upgrade
elif [[ -n "$TF_VERSION" ]]; then
  requirements_lock="oss_scripts/pip_package/requirements_lock_${HERMETIC_PYTHON_VERSION//./_}.txt"
  if awk -v dep="tensorflow==${TF_VERSION//-/}" '$1 == dep {found = 1} END {exit !found}' "$requirements_lock"; then
    echo "tensorflow==${TF_VERSION} is already in ${requirements_lock}, using it as is."
  else
    bazel run //oss_scripts/pip_package:requirements.update -- --upgrade
  fi
fi

TF_ABIFLAG=$(bazel run //oss_scripts/pip_package:tensorflow_build_info -- abi)
SHARED_LIBRARY_NAME="libtensorflow_framework.so.2"
if is_macos; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.2.dylib"
fi

write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_ABIFLAG}
