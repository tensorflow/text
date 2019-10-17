#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# Install the given bazel version on macos
function update_bazel_macos {
  if [[ "$BAZEL_VERSION" == "" ]]; then
    BAZEL_VERSION=${LATEST_BAZEL_VERSION}
  fi
  curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh -O
  ls
  chmod +x bazel-*.sh
  ./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh --user
  rm -f ./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh
  # Add new bazel installation to path
  PATH="/Users/kbuilder/bin:$PATH"
}

# Install bazel
update_bazel_macos 0.24.1
which bazel
bazel version

# Pick a more recent version of xcode
sudo xcode-select --switch /Applications/Xcode_10.1.app/Contents/Developer

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# create virtual env
"pip${PY_VERSION}" install --user --upgrade pip
"pip${PY_VERSION}" install --user --upgrade virtualenv
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate

# Checkout the release branch if specified.
git checkout -f "${RELEASE_BRANCH:-master}"

# Run configure.
export CC_OPT_FLAGS='-mavx'
./oss_scripts/configure.sh

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

  cd ${KOKORO_ARTIFACTS_DIR}
  python3 -m pip install -U twine
  twine upload *.whl
fi
