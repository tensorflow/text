#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

PY_VERSION=${1}

# cd into the base directory
cd "${KOKORO_ARTIFACTS_DIR}"/github/tensorflow_text/

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
rm -fr /var/tmp/_bazel_kbuilder/install/9be5dadb2a2b38082dbe665bf2db6464
bazel version
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

# Test
bazel test --test_output=errors tensorflow_text:all

deactivate
