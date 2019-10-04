#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# Install the given bazel version on macos
function update_bazel_macos {
  if [[ -z "$1" ]]; then
    BAZEL_VERSION=${LATEST_BAZEL_VERSION}
  else
    BAZEL_VERSION=$1
  fi
  BAZEL_COMMAND="curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh -O && \
  chmod +x bazel-*.sh && ./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh --user && \
  rm -f bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh"
  # If the bazel update fails retry again in 60 seconds.
  run_with_retry "${BAZEL_COMMAND}"
  # Add new bazel installation to path
  PATH="/Users/kbuilder/bin:$PATH"
}

# Install bazel
update_bazel_macos 0.24.1
which bazel
bazel version

# Pick a more recent version of xcode
sudo xcode-select --switch /Applications/Xcode_9.2.app/Contents/Developer

# cd into the release branch in kokoro
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

# create virtual env
"pip${PY_VERSION}" install --user --upgrade pip
"pip${PY_VERSION}" install --user --upgrade virtualenv
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate

# Checkout the release branch if specified.
git checkout "${RELEASE_BRANCH:-master}"

# Run configure.
export CC_OPT_FLAGS='-mavx'
./oss_scripts/configure.sh

# Build the pip package
bazel build oss_scripts/pip_package:build_pip_package
./bazel-bin/oss_scripts/pip_package/build_pip_package .
