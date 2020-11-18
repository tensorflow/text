#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands change
# RELEASE_BRANCH=test_300197485,ALT_BUILD_SCRIPT=./oss_scripts/test_mac.sh

PY_VERSION=${1}

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# Checkout the release branch if specified.
git checkout -f "${RELEASE_BRANCH:-master}"

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

# Install recent bazel version
update_bazel_macos ${BAZEL_VERSION:-2.0.0}
which bazel
bazel version

# create virtual env
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate

# Update to macos extensions
sed -i '' 's/".so"/".dylib"/' tensorflow_text/tftext.bzl
sed -i '' 's/*.so/*.dylib/' oss_scripts/pip_package/MANIFEST.in
perl -pi -e "s/(load_library.load_op_library.*)\\.so'/\$1.dylib'/" $(find tensorflow_text/python -type f)

# Run configure.
export CC_OPT_FLAGS='-mavx'
./oss_scripts/configure.sh
cat .bazelrc

# Build the pip package
bazel build oss_scripts/pip_package:build_pip_package
./bazel-bin/oss_scripts/pip_package/build_pip_package ${KOKORO_ARTIFACTS_DIR}

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

  # create virtual env
  python3 -m virtualenv /tmp/env
  source /tmp/env/bin/activate
  pip install twine

  cd ${KOKORO_ARTIFACTS_DIR}
  twine upload *.whl
fi
