#!/bin/bash

# RELEASE_BRANCH=test_278952393,ALT_BUILD_SCRIPT=./oss_scripts/build_test.sh

PY_VERSION=${1}

# Install the given bazel version on macos
function update_bazel_linux {
  BAZEL_VERSION=$1
  curl -L https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh -O
  ls
  chmod +x bazel-*.sh
  ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
  rm -f ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  # Add new bazel installation to path
  PATH="/home/kbuilder/bin:$PATH"
}

# Install bazel
#update_bazel_linux ${BAZEL_VERSION:-0.29.1}
bazel version

# create virtual env
"python${PY_VERSION}" -m virtualenv env
source env/bin/activate
python -V
pip install -U --user pip

# Run configure.
./oss_scripts/configure.sh

# Build the pip package
bazel build oss_scripts/pip_package:build_pip_package
./bazel-bin/oss_scripts/pip_package/build_pip_package ${KOKORO_ARTIFACTS_DIR}

ls ${KOKORO_ARTIFACTS_DIR}
deactivate

cd ${KOKORO_ARTIFACTS_DIR}
python3 -m pip install -U auditwheel==1.8.0
python3 -m pip install -U wheel==0.31.1
for f in *.whl; do auditwheel repair -w . --plat manylinux1_x86_64 $f; done

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
  twine upload tensorflow_text-*-manylinux1_x86_64.whl
fi
