#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands
#
PY_VERSION=${1} #
#
echo $PATH #
ls /cygdrive/c/tools/msys64/usr/bin #
ls /cygdrive/c #
ls ${KOKORO_GFILE_DIR} #
ls / #
#
# Install the given bazel version on windows
function update_bazel_windows { #
  BAZEL_VERSION=$1 #
  curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-windows-x86_64.exe -O #
  ls #
  chmod +x bazel-*.exe #
  ./bazel-${BAZEL_VERSION}-windows-x86_64.exe --user #
  rm -f ./bazel-${BAZEL_VERSION}-windows-x86_64.exe #
  # Add new bazel installation to path
  PATH=/Users/kbuilder/bin;%PATH% #
} #
#
# Install recent bazel version if given
bazel version #
if [[ ! -z "$BAZEL_VERSION" ]]; then #
  update_bazel_windows ${BAZEL_VERSION} #
fi #
which bazel #
bazel version #
#
# Run configure.
./oss_scripts/configure.sh #
#
# Build the pip package
bazel build oss_scripts/pip_package:build_pip_package #
bazel-bin\oss_scripts\pip_package\build_pip_package.exe ${KOKORO_ARTIFACTS_DIR} #
#
ls ${KOKORO_ARTIFACTS_DIR} #
#
# Release
if [[ "$UPLOAD_TO_PYPI" == "upload" ]]; then #
  PYPI_PASSWD="$(cat "$KOKORO_KEYSTORE_DIR"/74641_tftext_pypi_automation_passwd)" #
  cat >~/.pypirc <<EOL #
[pypi] #
username = __token__ #
password = ${PYPI_PASSWD} #
EOL #
#
  # create virtual env
  python3 -m virtualenv /tmp/env #
  source /tmp/env/bin/activate #
  pip install twine #
 #
  cd ${KOKORO_ARTIFACTS_DIR} #
  twine upload *.whl #
fi #
 #
