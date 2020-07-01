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

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function is_linux() {
  [[ "${PLATFORM}" == "linux" ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

if [[ $(pip show tensorflow) == *tensorflow* ]] || [[ $(pip show tf-nightly) == *tf-nightly* ]] ; then
  echo 'Using installed tensorflow.'
else
  echo 'Installing tensorflow.'
  pip install tensorflow==2.2.0
fi

write_to_bazelrc "build:manylinux2010 --crosstool_top=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain"
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"
write_to_bazelrc "build --define=framework_shared_object=true"
write_to_bazelrc "build --experimental_repo_remote_exec"

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LFLAGS_2=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' | awk '{print $2}') )

if is_windows; then
  # Use pywrap_tensorflow instead of tensorflow_framework on Windows
  SHARED_LIBRARY_DIR=${TF_CFLAGS:2:-7}"python"
else
  SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
fi
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS_2 | rev | cut -d":" -f1 | rev)
if is_macos; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
elif is_windows; then
  # Use pywrap_tensorflow's import library on Windows. It is in the same dir as the dll/pyd.
  SHARED_LIBRARY_NAME="_pywrap_tensorflow_internal.lib"
fi


HEADER_DIR=${TF_CFLAGS:2}
if is_windows; then
  SHARED_LIBRARY_DIR=${SHARED_LIBRARY_DIR//\\//}
  SHARED_LIBRARY_NAME=${SHARED_LIBRARY_NAME//\\//}
  HEADER_DIR=${HEADER_DIR//\\//}
fi
write_action_env_to_bazelrc "TF_HEADER_DIR" ${HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
