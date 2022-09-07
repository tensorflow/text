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

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${osname}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

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
  if is_macos; then
    #  Only Apple Silicon will be installed with tensorflow-macos.
    if [[ x"$(arch)" == x"arm64" ]]; then
      pip install tensorflow-macos==2.9.0
    else
      pip install tensorflow==2.10.0
    fi
  else
    pip install tensorflow==2.10.0
  fi
fi

if is_windows; then
  # ICU must be built as a static library, so the external data must be built in
  sed -i -e 's/":headers",$/":headers", ":windows_static_link_data",/' third_party/icu/BUILD.bzl
fi

# Copy the current bazelversion of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/r2.10/.bazelversion -o .bazelversion

# Copy the building configuration of TF.
curl https://raw.githubusercontent.com/tensorflow/tensorflow/r2.10/.bazelrc -o .bazelrc
# This line breaks Windows builds, so we remove it.
sed -i -e 's/build --noincompatible_remove_legacy_whole_archive//' .bazelrc

# the next line is temporary to aid in transition
write_to_bazelrc "build:manylinux2010 --config=release_cpu_linux"
write_to_bazelrc "build:manylinux2014 --config=release_cpu_linux"

if (which python) | grep -q "python"; then
  installed_python="python"
elif (which python3) | grep -q "python3"; then
  installed_python="python3"
fi

TF_CFLAGS=( $($installed_python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))" | awk '{print $1}') )
TF_LFLAGS=( $($installed_python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" | awk '{print $1}') )
TF_LFLAGS_2=( $($installed_python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" | awk '{print $2}') )
TF_ABIFLAG=$($installed_python -c "import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)")

HEADER_DIR=${TF_CFLAGS:2}
SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS_2 | rev | cut -d":" -f1 | rev)
if is_macos; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
fi
if is_windows; then
  HEADER_DIR=$(echo "$HEADER_DIR" | tr '\\' '/')
  SHARED_LIBRARY_DIR="${HEADER_DIR:0:-7}python"
  SHARED_LIBRARY_NAME="_pywrap_tensorflow_internal.lib"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SHARED_LIBRARY_DIR
write_action_env_to_bazelrc "TF_HEADER_DIR" ${HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_ABIFLAG}
