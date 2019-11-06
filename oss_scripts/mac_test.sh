#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

# RELEASE_BRANCH=test_278952393,ALT_BUILD_SCRIPT=./oss_scripts/mac_test.sh

PY_VERSION=${1}

# Install the given bazel version on macos
function update_bazel_macos {
  BAZEL_VERSION=$1
  curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh -O
  ls
  chmod +x bazel-*.sh
  ./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh --user
  rm -f ./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh
  # Add new bazel installation to path
  PATH="/Users/kbuilder/bin:$PATH"
}

# Install bazel
update_bazel_macos ${BAZEL_VERSION:-0.29.1}
which bazel
bazel version

# Pick a more recent version of xcode
#sudo xcode-select --switch /Applications/Xcode_10.1.app/Contents/Developer

# create virtual env
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate

# Update to macos extensions
sed -i '' 's/".so"/".dylib"/' tensorflow_text/tftext.bzl
cat tensorflow_text/tftext.bzl
sed -i '' 's/*.so/*.dylib/' oss_scripts/pip_package/MANIFEST.in
cat oss_scripts/pip_package/MANIFEST.in
perl -pi -e "s/(load_library.load_op_library.*)\\.so'/\$1.dylib'/" $(find tensorflow_text/python -type f)

# Run configure.
export CC_OPT_FLAGS='-mavx'
./oss_scripts/configure.sh
cat .bazelrc

# Build the pip package
bazel build oss_scripts/pip_package:build_pip_package
./bazel-bin/oss_scripts/pip_package/build_pip_package ${KOKORO_ARTIFACTS_DIR}

ls tensorflow_text/python/ops

ls ${KOKORO_ARTIFACTS_DIR}
deactivate

# Release
if [[ "$UPLOAD_TO_PYPI" == "upload" ]]; then
  PYPI_PASSWD="$(cat "$KOKORO_KEYSTORE_DIR"/74641_tftext_pypi_automation_passwd)"
  cat >~/.pypirc <<EOL
[pypi]
username = __token__
password = ${PYPI_PASSWD}
EOL

  cd ${KOKORO_ARTIFACTS_DIR}
  python3 -m pip install -U twine
  twine upload *.whl
fi
