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

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

if [[ $(pip show tensorflow) == *tensorflow* ]] || [[ $(pip show tf-nightly) == *tf-nightly* ]] ; then
  echo 'Using installed tensorflow.'
else
  echo 'Installing tensorflow.'
  pip install tensorflow==2.3.0
fi

write_to_bazelrc "build:manylinux2010 --crosstool_top=@org_tensorflow//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.1:toolchain"
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"
write_to_bazelrc "build --define=framework_shared_object=true"
write_to_bazelrc "build --experimental_repo_remote_exec"

# Config for Android build.
write_to_bazelrc "build:android --crosstool_top=//external:android/crosstool"
write_to_bazelrc "build:android --host_crosstool_top=@bazel_tools//tools/cpp:toolchain"
write_to_bazelrc "build:android --action_env TF_HEADER_DIR=\"\""
write_to_bazelrc "build:android --action_env TF_SHARED_LIBRARY_DIR=\"\""
write_to_bazelrc "build:android --action_env TF_SHARED_LIBRARY_NAME=\"\""
write_to_bazelrc "build:android_arm --config=android"
write_to_bazelrc "build:android_arm --cpu=armeabi-v7a"
write_to_bazelrc "build:android_arm --fat_apk_cpu=armeabi-v7a"
write_to_bazelrc "build:android_arm64 --config=android"
write_to_bazelrc "build:android_arm64 --cpu=arm64-v8a"
write_to_bazelrc "build:android_arm64 --fat_apk_cpu=arm64-v8a"
write_to_bazelrc "build:android_x86 --config=android"
write_to_bazelrc "build:android_x86 --cpu=x86"
write_to_bazelrc "build:android_x86 --fat_apk_cpu=x86"
write_to_bazelrc "build:android_x86_64 --config=android"
write_to_bazelrc "build:android_x86_64 --cpu=x86_64"
write_to_bazelrc "build:android_x86_64 --fat_apk_cpu=x86_64"

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LFLAGS_2=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' | awk '{print $2}') )

SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS_2 | rev | cut -d":" -f1 | rev)
if [[ "$(uname)" == "Darwin" ]]; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
fi
write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
